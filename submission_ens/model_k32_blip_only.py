"""HAIKUTHON 팀 앙상블: 3단 깔때기 (A 뼈대 + C 재정렬기)

  [1] SigLIP (A가 학습한 Base, 선택: C의 SigLIP2를 질문별 z-가중합)  → 질문마다 상위 k1개
  [2] BLIP-2 ITM (A가 학습한 재정렬기)  → 1단계 점수와 질문별 z-가중합(A의 rerank_weight) → 순위
  [3] Qwen3-VL-Reranker-2B LoRA (C)  → 2단계 1·2위 차이가 작은 질문부터 상위 k3개를 다시 채점,
                                       2단계 점수와 질문별 z-가중합(qwen_weight)  → 최종 top-5

규정 관련: 모든 정규화·순위는 '질문 하나의 점수'만 사용. 다른 질문의 점수·예측 분포를 쓰지 않음.
           Qwen 처리 순서를 정할 때 쓰는 '2단계 1·2위 차이'도 질문 자신의 점수라 같은 원칙.
시간: predict() 안에서 stage2_deadline(2단계), time_limit(3단계)이 지나면 남은 작업은 앞 단계 순위로 대체.
이미지: PNG를 한 번만 디코딩해 224px(SigLIP/BLIP-2 공통)와 rr_res px(Qwen·SigLIP2 공통) 복사본을 만든다.

ZIP 구성
  model.py
  models/ens_config.json         설정 (아래 DEFAULT_CFG 덮어쓰기)
"""
import contextlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

DEFAULT_CFG = {
    "k1": 20,                 # 1단계 → 2단계 후보 수
    "s2_weight": 0.0,         # 1단계: z(SigLIP2) 비중 (0이면 SigLIP2 안 씀)
    "blip_weight": 0.5,       # 2단계: z(BLIP-2) 비중, 나머지는 z(1단계)  (A의 pipeline.json rerank_weight)
    "k3": 3,                  # 3단계에서 Qwen이 다시 볼 2단계 상위 수
    "qwen_weight": 0.5,       # 3단계: z(Qwen) 비중, 나머지는 z(2단계)
    "qwen_min_k": 2,          # 이 깊이까지 채점된 질문만 Qwen 결과 반영
    "stage2_deadline": 600.0,  # predict 시작 후 이 시간이 지나면 2단계 중단 (남은 질문은 1단계 순위)
    "time_limit": 780.0,      # 3단계 중단 시각 (서버 제한 900초)
    "rr_res": 256,            # Qwen·SigLIP2 입력 해상도
    "blip_text_tokens": 128,
    "blip_image_batch": 96,
    "blip_pair_batch": 512,
    "qwen_batch": 16,
    "qwen_system": "",        # C v4: 시스템/지시문 없이 LoRA 학습
    "qwen_instruct": "",
    "s1_img_batch": 128,
    "workers": 4,
    "verbose": True,
}

SIGLIP_MEAN, SIGLIP_STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)


def _log(cfg, msg):
    if cfg.get("verbose"):
        print(f"[ens] {msg}", flush=True)


def _pooled(o):
    if torch.is_tensor(o):
        return o
    po = getattr(o, "pooler_output", None)
    return po if po is not None else o[0]


def _zq(x):
    """질문(행)마다 표준화. NaN(채점 안 된 칸)은 제외하고 계산."""
    m = torch.nanmean(x, dim=1, keepdim=True)
    v = torch.nanmean((x - m) ** 2, dim=1, keepdim=True)
    return (x - m) / (v.sqrt() + 1e-6)


@contextlib.contextmanager
def _no_init():
    ctx = contextlib.nullcontext()
    for mod in ("transformers.initialization", "transformers.modeling_utils"):
        try:
            ctx = getattr(__import__(mod, fromlist=["no_init_weights"]), "no_init_weights")()
            break
        except Exception:
            continue
    with ctx:
        yield


# ------------------------------------------------------------------ 가중치 로딩
def _fill_q8(model, folder, sd_file="model_q8.safetensors"):
    """tools/quantize_vlm.py 형식(int8 + 행별 배율, q8_meta.json) → 모델에 채움."""
    from safetensors import safe_open
    meta = json.load(open(folder / "q8_meta.json"))
    msd = model.state_dict()
    with torch.no_grad(), safe_open(str(folder / sd_file), framework="pt") as f:
        for k in meta["quantized"]:
            q, sc = f.get_tensor(k), f.get_tensor(k + ".__scale")
            msd[k].copy_((q.float() * sc[:, None]).to(msd[k].dtype))
        for k in meta["kept"]:
            t = f.get_tensor(k)
            msd[k].copy_(t.to(msd[k].dtype) if t.is_floating_point() else t)
    for k, src in meta.get("tied", {}).items():
        if msd[k].data_ptr() != msd[src].data_ptr():
            with torch.no_grad():
                msd[k].copy_(msd[src])


def _load_blip(folder, device):
    from transformers import AutoConfig, AutoProcessor, Blip2ForImageTextRetrieval
    if (folder / "model_q8.safetensors").exists():
        cfg = AutoConfig.from_pretrained(folder, local_files_only=True)
        with _no_init():
            blip = Blip2ForImageTextRetrieval._from_config(cfg, dtype=torch.float16)
        for name in getattr(blip, "_keep_in_fp32_modules", None) or ["qformer", "query_tokens"]:
            mod = getattr(blip, name, None)       # from_pretrained와 같게 Q-Former 등은 fp32 유지
            if isinstance(mod, torch.nn.Module):
                mod.float()
            elif isinstance(mod, torch.nn.Parameter):
                mod.data = mod.data.float()
        _fill_q8(blip, folder)
    else:
        blip = Blip2ForImageTextRetrieval.from_pretrained(folder, dtype=torch.float16, local_files_only=True)
    blip = blip.to(device).eval()
    proc = AutoProcessor.from_pretrained(folder, local_files_only=True)
    return blip, proc


def _load_qwen(folder, device, dtype):
    """C 형식(int8.safetensors, per-row/per-group 배율) / 우리 형식(model_q8 + q8_meta) / HF 형식 모두 지원."""
    from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration
    if (folder / "int8.safetensors").exists():
        from safetensors.torch import load_file
        cfg = AutoConfig.from_pretrained(folder, local_files_only=True)
        prev = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            with _no_init():
                lm = Qwen3VLForConditionalGeneration._from_config(cfg)
        finally:
            torch.set_default_dtype(prev)
        lm.to(device)
        sd = load_file(str(folder / "int8.safetensors"))
        params = dict(lm.named_parameters(remove_duplicate=False))
        bufs = dict(lm.named_buffers(remove_duplicate=False))
        with torch.no_grad():
            for k, t in list(params.items()) + list(bufs.items()):
                if k not in sd:
                    continue
                v = sd[k]
                if v.dtype == torch.int8:
                    s = sd[k + ".__scale"].to(t.device).float()
                    s = s[:, None] if s.dim() == 1 else s
                    vq = v.to(t.device).float()
                    v = (vq.view(vq.shape[0], s.shape[1], -1) * s[:, :, None]).view_as(vq)
                t.data.copy_(v.to(device=t.device, dtype=t.dtype))
        del sd
        lm.tie_weights()
    elif (folder / "model_q8.safetensors").exists():
        cfg = AutoConfig.from_pretrained(folder, local_files_only=True)
        with _no_init():
            lm = Qwen3VLForConditionalGeneration._from_config(cfg, dtype=dtype)
        _fill_q8(lm, folder)
        lm.to(device)
    else:
        lm = Qwen3VLForConditionalGeneration.from_pretrained(folder, dtype=dtype, local_files_only=True).to(device)
    lm.eval()
    proc = AutoProcessor.from_pretrained(folder, local_files_only=True)
    proc.tokenizer.padding_side = "left"
    vocab = proc.tokenizer.get_vocab()
    W = lm.get_output_embeddings().weight
    w = (W[vocab["yes"]].float() - W[vocab["no"]].float()).to(device)
    return lm, proc, w


def load_model(model_dir, device):
    from transformers import AutoModel, AutoProcessor, AutoTokenizer, SiglipModel
    root = Path(model_dir) / "models"
    cfg = dict(DEFAULT_CFG)
    if (root / "ens_config.json").exists():
        cfg.update(json.load(open(root / "ens_config.json")))
    is_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    major = torch.cuda.get_device_capability(device)[0] if is_cuda else 0
    lm_dtype = torch.bfloat16 if major >= 8 else (torch.float16 if is_cuda else torch.float32)
    fp = torch.float16 if is_cuda else torch.float32
    t0 = time.time()
    m = {"cfg": cfg, "device": device}
    m["s1"] = SiglipModel.from_pretrained(root / "siglip", dtype=fp, local_files_only=True).to(device).eval()
    m["s1_proc"] = AutoProcessor.from_pretrained(root / "siglip", local_files_only=True)
    if cfg["s2_weight"] > 0 and (root / "siglip2").exists():
        m["s2"] = AutoModel.from_pretrained(root / "siglip2", dtype=fp, local_files_only=True).to(device).eval()
        m["s2_tok"] = AutoTokenizer.from_pretrained(root / "siglip2", local_files_only=True)
    m["blip"], m["blip_proc"] = _load_blip(root / "blip", device)
    if cfg["qwen_weight"] > 0 and (root / "qwen").exists():
        m["qwen"], m["qwen_proc"], m["qwen_w"] = _load_qwen(root / "qwen", device, lm_dtype)
    _log(cfg, f"load {time.time()-t0:.0f}s | s2={'s2' in m} qwen={'qwen' in m} | cfg={cfg}")
    return m


# ------------------------------------------------------------------ 이미지 (한 번만 디코딩)
def _decode_all(paths, rr, workers):
    n = len(paths)
    a224 = np.empty((n, 224, 224, 3), dtype=np.uint8)
    arr = np.empty((n, rr, rr, 3), dtype=np.uint8)
    bad = 0

    def one(i):
        try:
            with Image.open(paths[i]) as im:
                im = im.convert("RGB")
                a224[i] = np.asarray(im.resize((224, 224), Image.Resampling.BICUBIC))
                arr[i] = np.asarray(im.resize((rr, rr), Image.Resampling.BICUBIC))
            return 0
        except Exception:
            a224[i] = 0; arr[i] = 0
            return 1
    with ThreadPoolExecutor(workers) as ex:
        bad = sum(ex.map(one, range(n), chunksize=64))
    return a224, arr, bad


def _to_pixels(u8, mean, std, device, dtype):
    x = torch.from_numpy(u8).to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255.0)
    m = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=device).view(1, 3, 1, 1)
    return ((x - m) / s).to(dtype)


# ------------------------------------------------------------------ 1단계
@torch.inference_mode()
def _stage1(m, texts, a224, arr):
    cfg, dev = m["cfg"], m["device"]
    net, proc = m["s1"], m["s1_proc"]
    fi = []
    for s in range(0, len(a224), cfg["s1_img_batch"]):
        px = _to_pixels(a224[s:s + cfg["s1_img_batch"]], SIGLIP_MEAN, SIGLIP_STD, dev, net.dtype)
        fi.append(F.normalize(_pooled(net.get_image_features(pixel_values=px)).float(), dim=-1))
    I = torch.cat(fi)
    ft = []
    for s in range(0, len(texts), 256):
        tok = proc(text=texts[s:s + 256], padding="max_length", truncation=True, return_tensors="pt").to(dev)
        ft.append(F.normalize(_pooled(net.get_text_features(**tok)).float(), dim=-1))
    T = torch.cat(ft)
    I2 = T2 = None
    if "s2" in m:
        net2, tok2 = m["s2"], m["s2_tok"]
        native = int(net2.config.vision_config.image_size)
        kw = {"interpolate_pos_encoding": True} if cfg["rr_res"] != native else {}
        fi = []
        for s in range(0, len(arr), 64):
            px = _to_pixels(arr[s:s + 64], SIGLIP_MEAN, SIGLIP_STD, dev, net2.dtype)
            fi.append(F.normalize(_pooled(net2.get_image_features(pixel_values=px, **kw)).float(), dim=-1))
        I2 = torch.cat(fi)
        ft = []
        for s in range(0, len(texts), 256):
            enc = tok2([t.lower() for t in texts[s:s + 256]], padding="max_length", truncation=True, max_length=64, return_tensors="pt")
            ft.append(F.normalize(_pooled(net2.get_text_features(input_ids=enc["input_ids"].to(dev))).float(), dim=-1))
        T2 = torch.cat(ft)
    k1 = min(cfg["k1"], len(a224))
    tops, topi = [], []
    for s in range(0, len(T), 1024):
        sc = T[s:s + 1024] @ I.T
        if I2 is not None:
            w = cfg["s2_weight"]
            sc = (1 - w) * _zq(sc) + w * _zq(T2[s:s + 1024] @ I2.T)
        v, i = sc.topk(k1, dim=1)
        tops.append(v.cpu()); topi.append(i.cpu())
    return torch.cat(tops), torch.cat(topi)


# ------------------------------------------------------------------ 2단계 (BLIP-2 ITM, 이미지별 묶음 처리)
def _blip_image_embeds(blip, px):
    return blip.vision_model(pixel_values=px).last_hidden_state


def _blip_itm(blip, img_emb, ids, mask):
    qt = blip.query_tokens.expand(img_emb.shape[0], -1, -1)
    qmask = torch.ones(qt.shape[:2], dtype=mask.dtype, device=mask.device)
    imask = torch.ones(img_emb.shape[:2], dtype=torch.long, device=img_emb.device)
    emb = blip.embeddings(input_ids=ids, query_embeds=qt)
    h = blip.qformer(query_embeds=emb, query_length=qt.shape[1], attention_mask=torch.cat([qmask, mask], 1),
                     encoder_hidden_states=img_emb, encoder_attention_mask=imask, return_dict=True).last_hidden_state
    lg = blip.itm_head(h[:, :qt.shape[1]].to(blip.itm_head.weight.dtype)).mean(1)
    return (lg[:, 1] - lg[:, 0]).float()


@torch.inference_mode()
def _stage2(m, texts, a224, topi, deadline):
    cfg, dev = m["cfg"], m["device"]
    blip, bproc = m["blip"], m["blip_proc"]
    nq, k = topi.shape
    scores = torch.full((nq, k), float("nan"))
    tok = bproc.tokenizer(texts, padding="longest", truncation=True, max_length=cfg["blip_text_tokens"], return_tensors="pt")
    ids_all, mask_all = tok.input_ids.to(dev), tok.attention_mask.to(dev)
    qlen = mask_all.sum(1).tolist()
    fp16 = bool(cfg.get("itm_fp16", True)) and str(dev).startswith("cuda")
    # 이미지 → (질문, 칸) 목록. 상위 칸(순위가 높은 후보)을 먼저 처리해 시간이 모자라도 앞 순위부터 채워지게 정렬
    best_rank = {}
    pairs_by_img = {}
    for q, row in enumerate(topi.tolist()):
        for r, j in enumerate(row):
            pairs_by_img.setdefault(j, []).append((q, r))
            best_rank[j] = min(best_rank.get(j, 1 << 30), r)
    imgs = sorted(pairs_by_img, key=lambda j: (best_rank[j], j))
    vdt = blip.vision_model.embeddings.patch_embedding.weight.dtype
    ib, pb = cfg["blip_image_batch"], cfg["blip_pair_batch"]
    stopped = False
    for s in range(0, len(imgs), ib):
        if time.time() > deadline:
            stopped = True
            break
        chunk = imgs[s:s + ib]
        emb = _blip_image_embeds(blip, _to_pixels(a224[chunk], CLIP_MEAN, CLIP_STD, dev, vdt))
        pairs = [(li, q, r) for li, j in enumerate(chunk) for q, r in pairs_by_img[j]]
        pairs.sort(key=lambda t: qlen[t[1]])
        for p in range(0, len(pairs), pb):
            li, q, r = (torch.tensor(c) for c in zip(*pairs[p:p + pb]))
            qd = q.to(dev)
            L = int(max(qlen[i] for i in q.tolist()))
            with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
                v = _blip_itm(blip, emb[li.to(dev)], ids_all[qd, :L], mask_all[qd, :L])
            scores[q, r] = v.float().cpu()
    return scores, stopped


# ------------------------------------------------------------------ 3단계 (Qwen LoRA)
def _qwen_prompt(m, query):
    cfg = m["cfg"]
    user = [{"type": "text", "text": "<Instruct>: " + cfg["qwen_instruct"]}] if cfg["qwen_instruct"] else []
    user += [{"type": "text", "text": "<Query>:"}, {"type": "text", "text": query or ""},
             {"type": "text", "text": "\n<Document>:"}, {"type": "image"}]
    msgs = ([{"role": "system", "content": [{"type": "text", "text": cfg["qwen_system"]}]}] if cfg["qwen_system"] else [])
    msgs += [{"role": "user", "content": user}]
    return m["qwen_proc"].apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def _stage3(m, texts, arr, order_i, margin, deadline):
    """order_i: (Q, k3) 2단계 순위의 이미지 인덱스. margin 작은 질문부터, 칸 0·1 → 칸 2 … 순서로 채점."""
    cfg, dev = m["cfg"], m["device"]
    lm, proc, w, bs = m["qwen"], m["qwen_proc"], m["qwen_w"], cfg["qwen_batch"]
    nq, k3 = order_i.shape
    qs = sorted(range(nq), key=lambda q: margin[q])
    plan = [(q, r) for q in qs for r in range(min(2, k3))] + [(q, r) for r in range(2, k3) for q in qs]
    logits = torch.full((nq, k3), float("nan"))
    prompts = {}

    def prep(chunk):
        tx = []
        for q, _ in chunk:
            if q not in prompts:
                prompts[q] = _qwen_prompt(m, texts[q])
            tx.append(prompts[q])
        ims = [Image.fromarray(arr[order_i[q, r]]) for q, r in chunk]
        return proc(text=tx, images=ims, padding=True, return_tensors="pt", do_resize=False)

    chunks = [plan[s:s + bs] for s in range(0, len(plan), bs)]
    done = 0
    with ThreadPoolExecutor(1) as ex:
        fut = ex.submit(prep, chunks[0]) if chunks else None
        for ci, chunk in enumerate(chunks):
            if time.time() > deadline:
                break
            inputs = fut.result()
            if ci + 1 < len(chunks):
                fut = ex.submit(prep, chunks[ci + 1])
            h = lm.model(**inputs.to(dev)).last_hidden_state[:, -1].float()
            sc = (h @ w).cpu()
            for (q, r), v in zip(chunk, sc):
                logits[q, r] = v
            done += len(chunk)
    return logits, done, len(plan)


# ------------------------------------------------------------------ predict
def predict(model, queries, candidates, top_k=5):
    m, cfg = model, model["cfg"]
    t0 = time.time()
    qids = [q["prompt_id"] for q in queries]
    texts = [q["prompt"] for q in queries]
    img_ids = [c["image_id"] for c in candidates]
    paths = [c["image_path"] for c in candidates]
    a224, arr, bad = _decode_all(paths, cfg["rr_res"], cfg["workers"])
    t_dec = time.time()
    s1, topi = _stage1(m, texts, a224, arr)                       # (Q, k1)
    t1 = time.time()
    _log(cfg, f"decode {t_dec-t0:.0f}s (bad {bad}) | stage1 {t1-t_dec:.0f}s | k1={topi.shape[1]}")

    # 2단계: BLIP-2 + 1단계 z-가중합 (채점 못 한 칸이 있는 질문은 1단계 순위 유지)
    b, stopped = _stage2(m, texts, a224, topi, t0 + cfg["stage2_deadline"])
    wb = cfg["blip_weight"]
    nq, k1 = topi.shape
    # 질문마다 '앞에서부터 연속으로 채점된 칸 수(kd)'만큼만 융합·재정렬하고 나머지는 1단계 순서 유지.
    # (시간 가드로 일부만 채점돼도 그 질문의 앞 순위는 재정렬 혜택을 받음. A 원본은 하나라도 빠지면 질문 전체를 1단계로 되돌려
    #  K=20에서 시간 가드가 걸리면 거의 모든 질문이 1단계 순위가 되는 문제가 있었음)
    scored = ~torch.isnan(b)
    kd_all = torch.where(scored.all(1), torch.full((nq,), k1), (~scored).float().argmax(1))
    rank_i = topi.clone()
    rank_s = torch.full((nq, k1), float("-inf"))
    n_full = int((kd_all == k1).sum()); n_part = 0
    for kd in kd_all.unique().tolist():
        rows = (kd_all == kd).nonzero().flatten()
        if kd < 2:
            continue
        f = (1 - wb) * _zq(s1[rows, :kd]) + wb * _zq(b[rows, :kd])
        o = f.argsort(dim=1, descending=True)
        rank_i[rows, :kd] = topi[rows, :kd].gather(1, o)
        rank_s[rows, :kd] = f.gather(1, o)
        if kd < k1:
            n_part += len(rows)
    t2 = time.time()
    _log(cfg, f"stage2 {t2-t1:.0f}s | BLIP 전부 {n_full} / 일부 {n_part} / 미처리 {nq-n_full-n_part} {'(시간 가드)' if stopped else ''}")

    # 3단계: Qwen (2단계 상위 k3)
    final = rank_i.clone()                                       # (Q, k1) → 마지막에 top_k만 사용
    if "qwen" in m and cfg["k3"] >= 2:
        k3 = min(cfg["k3"], rank_i.shape[1])
        margin = torch.nan_to_num(rank_s[:, 0] - rank_s[:, 1], nan=0.0, posinf=0.0, neginf=0.0).tolist()
        lq, done, plan = _stage3(m, texts, arr, rank_i[:, :k3].numpy(), margin, t0 + cfg["time_limit"])
        wq = cfg["qwen_weight"]
        n_used = 0
        for q in range(len(qids)):
            row = lq[q]
            kd = 0
            while kd < k3 and not torch.isnan(row[kd]):
                kd += 1
            if kd < cfg["qwen_min_k"]:
                continue
            base = rank_s[q:q + 1, :kd]
            if not torch.isfinite(base).all():          # 2단계 미처리 질문: 1단계 점수 순위를 기준으로
                base = -torch.arange(kd, dtype=torch.float32)[None]
            sub = (1 - wq) * _zq(base) + wq * _zq(row[None, :kd])
            o = sub[0].argsort(descending=True)
            final[q, :kd] = rank_i[q, :kd][o]
            n_used += 1
        _log(cfg, f"stage3 {time.time()-t2:.0f}s | Qwen 쌍 {done}/{plan}, 반영 질문 {n_used}")
    preds = {qid: [img_ids[j] for j in row[:top_k]] for qid, row in zip(qids, final.tolist())}
    _log(cfg, f"predict {time.time()-t0:.0f}s")
    return preds                                                          

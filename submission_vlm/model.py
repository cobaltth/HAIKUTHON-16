"""HAIKUTHON 제출 모델: SigLIP 검색(1단계) + Qwen3-VL 재정렬(2단계, 상위 K).

평가 서버 인터페이스(스타터킷과 동일)
    load_model(model_dir, device) -> model
    predict(model, queries, candidates, top_k=5) -> {prompt_id: [image_id × top_k]}

규정 준수
- 외부 API 호출 없음 (모든 가중치는 zip 안의 models/ 에서 로드)
- 각 query의 순위는 그 query의 점수만으로 결정 (다른 query의 예측으로 보정하지 않음).
  시간 가드는 '경과 시간'만 보고 재정렬을 멈출 뿐, 다른 query의 결과를 참고하지 않는다.

설정: models/config_b.json (없으면 DEFAULT_CFG)
"""
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

DEFAULT_CFG = {
    "rerank": True,              # False면 SigLIP만
    "k": 5,                      # 재정렬할 상위 후보 수
    "vlm_size": 448,             # VLM 입력 이미지 한 변(px, 32의 배수). 작을수록 빠름
    "vlm_batch": 20,             # VLM 한 번에 처리할 (질문, 이미지) 쌍 수
    "margin_threshold": None,    # 숫자면: SigLIP 1·2위 점수 차가 이보다 큰 질문은 재정렬 생략 (자기 점수만 사용)
    "time_budget_sec": 780,      # predict 전체 예산(초). 넘으면 남은 질문은 SigLIP 순위 그대로
    "decode_workers": 4,         # 이미지 디코딩 병렬 스레드 (채점 서버 4 vCPU)
    "siglip_batch": 128,
    "instruction": ("Judge whether the image matches every detail of the scene description: "
                    "object types, object counts, colors, patterns, materials, poses, spatial relations "
                    "(left/right, above/below, on, inside, next to), rendering style, viewpoint and lighting."),
    "system": ("Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
               "Note that the answer can only be \"yes\" or \"no\"."),
    "fuse_alpha": 0.0,           # 최종 점수 = VLM logit + alpha * SigLIP 점수 (0이면 VLM만)
    "verbose": True,
}


def _log(cfg, *a):
    if cfg.get("verbose"):
        print("[model]", *a, flush=True)


def _pooled(o):
    return o.pooler_output if hasattr(o, "pooler_output") else o


# ------------------------------------------------------------------ load
def _load_q8(vdir, dtype):
    """tools/quantize_vlm.py로 만든 int8(행 단위) 가중치를 원래 dtype으로 복원해 모델에 채운다.
    텐서를 하나씩 복원해 바로 복사하므로 CPU 메모리 사용이 작다. (로딩 시간이라 추론 시간에 미포함)"""
    from safetensors import safe_open
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration
    meta = json.load(open(vdir / "q8_meta.json"))
    cfg = AutoConfig.from_pretrained(vdir, local_files_only=True)
    lm = Qwen3VLForConditionalGeneration._from_config(cfg, dtype=dtype)
    msd = lm.state_dict()
    covered = set(meta["quantized"]) | set(meta["kept"]) | set(meta["tied"])
    missing = [k for k in msd if k not in covered]
    if missing:
        raise RuntimeError(f"q8 가중치에 없는 키 {len(missing)}개: {missing[:5]}")
    with torch.no_grad(), safe_open(str(vdir / "model_q8.safetensors"), framework="pt") as f:
        for k in meta["quantized"]:
            q, sc = f.get_tensor(k), f.get_tensor(k + ".__scale")
            msd[k].copy_((q.float() * sc[:, None]).to(msd[k].dtype))
        for k in meta["kept"]:
            t = f.get_tensor(k)
            msd[k].copy_(t.to(msd[k].dtype) if t.is_floating_point() else t)
    for k, src in meta["tied"].items():          # 묶인 가중치가 실제로 공유되지 않는 경우 대비
        if msd[k].data_ptr() != msd[src].data_ptr():
            with torch.no_grad():
                msd[k].copy_(msd[src])
    return lm


def load_model(model_dir, device):
    from transformers import AutoProcessor, SiglipModel
    root = Path(model_dir) / "models"
    cfg = dict(DEFAULT_CFG)
    if (root / "config_b.json").exists():
        cfg.update(json.load(open(root / "config_b.json")))
    device = torch.device(device if isinstance(device, str) else device)
    cuda = device.type == "cuda"
    sig_dtype = torch.float16 if cuda else torch.float32
    sig = SiglipModel.from_pretrained(root / "siglip", local_files_only=True, dtype=sig_dtype).to(device).eval()
    sproc = AutoProcessor.from_pretrained(root / "siglip", local_files_only=True)
    m = {"cfg": cfg, "sig": sig, "sproc": sproc, "device": device, "vlm": None}
    if cfg["rerank"] and (root / "vlm").exists():
        from transformers import Qwen3VLForConditionalGeneration
        # bf16은 Ampere 이상(sm80+, L4=sm89)에서만 하드웨어 지원. T4(sm75)는 bf16이 매우 느려서 fp16 사용
        if cuda:
            major = torch.cuda.get_device_capability(device)[0]
            vdtype = torch.bfloat16 if major >= 8 else torch.float16
        else:
            vdtype = torch.float32
        if cfg.get("vlm_dtype"):
            vdtype = getattr(torch, cfg["vlm_dtype"])
        if (root / "vlm" / "model_q8.safetensors").exists():
            lm = _load_q8(root / "vlm", vdtype).to(device).eval()      # int8 압축본 → 로딩 때 복원
        else:
            lm = Qwen3VLForConditionalGeneration.from_pretrained(root / "vlm", local_files_only=True, dtype=vdtype).to(device).eval()
        vproc = AutoProcessor.from_pretrained(root / "vlm", local_files_only=True)
        vproc.tokenizer.padding_side = "left"
        vocab = vproc.tokenizer.get_vocab()
        w = (lm.lm_head.weight[vocab["yes"]] - lm.lm_head.weight[vocab["no"]]).detach().clone()
        m["vlm"] = {"model": lm.model, "proc": vproc, "w": w, "dtype": vdtype}
        _log(cfg, f"VLM 로드 ({vdtype}), k={cfg['k']}, size={cfg['vlm_size']}")
    else:
        _log(cfg, "SigLIP 단독 모드")
    return m


# ------------------------------------------------------------------ stage 1
def _decode(path, vlm_size):
    """PNG 한 번만 읽어서 (SigLIP용 PIL, VLM용 축소 JPEG 바이트) 반환. JPEG q95로 메모리 절약."""
    with Image.open(path) as im:
        im = im.convert("RGB")
    jpg = None
    if vlm_size:
        b = io.BytesIO()
        im.resize((vlm_size, vlm_size), Image.BICUBIC).save(b, "JPEG", quality=95)
        jpg = b.getvalue()
    return im, jpg


@torch.inference_mode()
def _stage1(m, queries, candidates, depth):
    cfg, sig, sproc, dev = m["cfg"], m["sig"], m["sproc"], m["device"]
    vlm_size = cfg["vlm_size"] if m["vlm"] is not None else 0
    jpgs, feats = [None] * len(candidates), []
    bs = cfg["siglip_batch"]
    paths = [str(c["image_path"]) for c in candidates]
    starts = list(range(0, len(paths), bs))
    with ThreadPoolExecutor(cfg["decode_workers"]) as ex:
        submit = lambda s: [ex.submit(_decode, p, vlm_size) for p in paths[s:s + bs]]
        pending = submit(starts[0]) if starts else []
        for n, s in enumerate(starts):
            # 다음 묶음을 미리 디코딩 시작 (메모리에는 최대 두 묶음만 유지)
            nxt = submit(starts[n + 1]) if n + 1 < len(starts) else []
            chunk = [f.result() for f in pending]
            pending = nxt
            for k, (_, j) in enumerate(chunk):
                jpgs[s + k] = j
            px = sproc(images=[im for im, _ in chunk], return_tensors="pt").pixel_values.to(dev, sig.dtype)
            feats.append(F.normalize(_pooled(sig.get_image_features(pixel_values=px)).float(), dim=-1))
    I = torch.cat(feats)
    top_s, top_i = [], []
    for s in range(0, len(queries), 256):
        tok = sproc(text=[q["prompt"] for q in queries[s:s + 256]], padding="max_length", truncation=True,
                    return_tensors="pt").to(dev)
        T = F.normalize(_pooled(sig.get_text_features(**tok)).float(), dim=-1)
        sc, ix = (T @ I.T).topk(depth, dim=1)          # 각 질문 행을 독립적으로 정렬
        top_s.append(sc.cpu()); top_i.append(ix.cpu())
    return torch.cat(top_s), torch.cat(top_i), jpgs


# ------------------------------------------------------------------ stage 2
def _messages(cfg, caption, image):
    return [
        {"role": "system", "content": [{"type": "text", "text": cfg["system"]}]},
        {"role": "user", "content": [
            {"type": "text", "text": "<Instruct>: " + cfg["instruction"]},
            {"type": "text", "text": "<Query>:"},
            {"type": "text", "text": caption},
            {"type": "text", "text": "\n<Document>:"},
            {"type": "image", "image": image},
        ]},
    ]


@torch.inference_mode()
def _vlm_scores(m, pairs):
    """pairs: [(caption, PIL)] → yes-no logit 리스트"""
    v, cfg = m["vlm"], m["cfg"]
    proc = v["proc"]
    texts = [proc.apply_chat_template(_messages(cfg, c, im), tokenize=False, add_generation_prompt=True) for c, im in pairs]
    inputs = proc(text=texts, images=[im for _, im in pairs], padding=True, return_tensors="pt").to(m["device"])
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(v["dtype"])
    h = v["model"](**inputs).last_hidden_state[:, -1]   # 왼쪽 패딩 → 마지막 위치가 실제 마지막 토큰
    return (h.float() @ v["w"].float()).tolist()


# ------------------------------------------------------------------ predict
def predict(model, queries, candidates, top_k=5):
    m, cfg = model, model["cfg"]
    t0 = time.time()
    ids = [c["image_id"] for c in candidates]
    use_vlm = m["vlm"] is not None and cfg["k"] > 1
    K = min(cfg["k"], len(ids)) if use_vlm else 1
    depth = min(len(ids), max(top_k, K))
    top_s, top_i, jpgs = _stage1(m, queries, candidates, depth)
    t1 = time.time()
    _log(cfg, f"1단계 SigLIP: 이미지 {len(ids)} / 질문 {len(queries)} → {t1 - t0:.1f}s")

    order = [list(row) for row in top_i.tolist()]       # 기본: SigLIP 순위
    n_rr = n_pairs = 0
    if use_vlm:
        budget = cfg["time_budget_sec"]
        thr = cfg["margin_threshold"]
        todo = [q for q in range(len(queries))
                if thr is None or float(top_s[q, 0] - top_s[q, 1]) <= thr]
        per_batch = max(1, cfg["vlm_batch"] // K)       # 한 배치에 들어갈 질문 수
        batch_t = None
        for s in range(0, len(todo), per_batch):
            # 시간 가드: 다음 배치를 돌려도 예산 안이면 진행 (경과 시간만 사용)
            elapsed = time.time() - t0
            if batch_t is not None and elapsed + 1.5 * batch_t > budget:
                _log(cfg, f"시간 가드 발동: {elapsed:.0f}s, 남은 질문 {len(todo) - s}개는 SigLIP 순위 유지")
                break
            tb = time.time()
            qs = todo[s:s + per_batch]
            pairs, owner = [], []
            for q in qs:
                for j in order[q][:K]:
                    pairs.append((queries[q]["prompt"], Image.open(io.BytesIO(jpgs[j])).convert("RGB")))
                    owner.append(q)
            vs = _vlm_scores(m, pairs)
            for n, q in enumerate(qs):
                cand = order[q][:K]
                sv = vs[n * K:(n + 1) * K]
                ss = top_s[q, :K].tolist()
                fused = [a + cfg["fuse_alpha"] * b for a, b in zip(sv, ss)]
                rer = [c for _, _, c in sorted(zip(fused, range(K), cand), key=lambda x: (-x[0], x[1]))]
                order[q] = rer + order[q][K:]
            n_rr += len(qs); n_pairs += len(pairs)
            batch_t = time.time() - tb
    t2 = time.time()
    if use_vlm:
        rate = n_pairs / max(t2 - t1, 1e-6)
        _log(cfg, f"2단계 VLM: 질문 {n_rr}/{len(queries)} 재정렬, {n_pairs}쌍, {t2 - t1:.1f}s ({rate:.1f}쌍/s)")
    _log(cfg, f"총 {t2 - t0:.1f}s")
    return {q["prompt_id"]: [ids[j] for j in order[k][:top_k]] for k, q in enumerate(queries)}

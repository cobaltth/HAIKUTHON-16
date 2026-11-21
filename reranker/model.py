"""HAIKUTHON v1: SigLIP2 retrieval -> Qwen3-VL-Reranker-2B reranking of the top-k.

Every query is predicted independently: its top-5 depends only on its own text, the candidate images and the
fixed settings below. No statistics across queries, no cross-query deduplication or matching.

ZIP layout
  model.py
  models/siglip2/    SigLIP2 so400m (fp16) + tokenizer + haiku_preprocess.json
  models/reranker/   config + processor files + int8.safetensors (int8 weights, per-row scales; restored to bf16 at load)
  models/v1_config.json   (optional) overrides for CFG
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import contextlib
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

SYSTEM = ('Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
          'Note that the answer can only be "yes" or "no".')
INSTRUCT_BINDING = ("Verify attribute binding and relation direction: every color, count, material, pattern and pose must belong "
                    "to the object named in the query, and every spatial relation must point the stated way. "
                    "A swapped or reversed detail means no.")

CFG = {
    "time_limit": 780.0,      # seconds inside predict(); no new reranking after this (server limit is 900)
    "rr_res": 256,            # reranker image side (multiple of 32)
    "k_min": 1,               # per-query k starts here (1 = no reranking when SigLIP top-1 is clearly ahead)
    "k_max": 6,               # ... and grows to at most k_max
    "gap": 0.05,              # add candidate r while stage1[0] - stage1[r] <= gap (tuned on valA/4000-pool under guard budgets)
    "rr_batch": 16,
    "s1_img_batch": 64,
    "num_workers": int(os.environ.get("HAIKU_NUM_WORKERS", "4")),
    "instruct": "",           # v4: no instruct / no system prompt (LoRA-trained that way; ~25% faster per pair)
    "system": "",
    "s1_text_max_len": 64,
    "s1_res": 256,            # SigLIP2 input side (native 384; 256 = same recall, 2.4x faster). 0 = native
}


def _log(msg):
    print(f"[haiku] {msg}", flush=True)


def _as_tensor(out):
    if torch.is_tensor(out):
        return out
    for key in ("pooler_output", "text_embeds", "image_embeds"):
        val = getattr(out, key, None)
        if val is not None:
            return val
    return out[0]


def _to_rgb(img):
    if img.mode == "RGB":
        return img
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, rgba).convert("RGB")
    return img.convert("RGB")


class _ImageDS(Dataset):
    """Decodes each image once and returns both the SigLIP2 input and the reranker copy."""

    def __init__(self, paths, s1_size, s1_resample, rr_size):
        self.paths, self.s1_size, self.s1_resample, self.rr_size = paths, s1_size, s1_resample, rr_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        ok = True
        try:
            img = Image.open(self.paths[i])
            if img.format == "JPEG":
                img.draft("RGB", (self.s1_size * 2, self.s1_size * 2))
            img = _to_rgb(img)
            a = img.resize((self.s1_size, self.s1_size), resample=self.s1_resample)
            b = img.resize((self.rr_size, self.rr_size), resample=Image.Resampling.BICUBIC)
        except Exception:
            ok = False
            a = Image.new("RGB", (self.s1_size, self.s1_size))
            b = Image.new("RGB", (self.rr_size, self.rr_size))
        return (torch.from_numpy(np.asarray(a, dtype=np.uint8).copy()),
                torch.from_numpy(np.asarray(b, dtype=np.uint8).copy()), i, ok)


def _worker_init(_):
    torch.set_num_threads(1)


@contextlib.contextmanager
def _no_init():
    ctx = contextlib.nullcontext()
    for mod in ("transformers.modeling_utils", "transformers.initialization"):
        try:
            ctx = getattr(__import__(mod, fromlist=["no_init_weights"]), "no_init_weights")()
            break
        except Exception:
            continue
    with ctx:
        yield


def _load_reranker(rdir, device):
    q = os.path.join(rdir, "int8.safetensors")
    if not os.path.exists(q):
        lm = Qwen3VLForConditionalGeneration.from_pretrained(rdir, dtype=torch.bfloat16, local_files_only=True)
    else:
        from safetensors.torch import load_file
        cfg = AutoConfig.from_pretrained(rdir, local_files_only=True)
        prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with _no_init():
                lm = Qwen3VLForConditionalGeneration._from_config(cfg)
        finally:
            torch.set_default_dtype(prev)
        lm.to(device)
        sd = load_file(q)  # int8 weights on CPU (~2.2GB); restored to bf16 one tensor at a time on the GPU
        params = dict(lm.named_parameters(remove_duplicate=False))
        bufs = dict(lm.named_buffers(remove_duplicate=False))
        missing, n_set = [], 0
        with torch.no_grad():
            for k, t in list(params.items()) + list(bufs.items()):
                if k not in sd:
                    if k in params and not k.startswith("lm_head"):
                        missing.append(k)
                    continue
                v = sd[k]
                if v.dtype == torch.int8:
                    s = sd[k + ".__scale"].to(t.device).float()
                    s = s[:, None] if s.dim() == 1 else s  # per-row (out,) or per-group (out, G)
                    vq = v.to(t.device).float()
                    v = (vq.view(vq.shape[0], s.shape[1], -1) * s[:, :, None]).view_as(vq)
                t.data.copy_(v.to(device=t.device, dtype=t.dtype))
                n_set += 1
        del sd
        lm.tie_weights()
        if missing:
            _log(f"WARNING reranker params not in checkpoint: {missing[:5]} ({len(missing)})")
        _log(f"reranker restored {n_set} tensors from int8 checkpoint")
    lm.eval()
    lm.to(device)
    proc = AutoProcessor.from_pretrained(rdir, local_files_only=True)
    proc.tokenizer.padding_side = "left"
    vocab = proc.tokenizer.get_vocab()
    W = lm.get_output_embeddings().weight
    w = (W[vocab["yes"]].float() - W[vocab["no"]].float()).to(device)
    return lm, proc, w


def load_model(model_dir, device):
    t0 = time.perf_counter()
    cfg = dict(CFG)
    over = os.path.join(model_dir, "models", "v1_config.json")
    if os.path.exists(over):
        with open(over) as f:
            cfg.update(json.load(f))
    is_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    if is_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
    sdir = os.path.join(model_dir, "models", "siglip2")
    s1 = AutoModel.from_pretrained(sdir, dtype=torch.float16 if is_cuda else torch.float32, local_files_only=True).to(device).eval()
    tok = AutoTokenizer.from_pretrained(sdir, local_files_only=True)
    pp = {"image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5], "resample": 3}
    ppath = os.path.join(sdir, "haiku_preprocess.json")
    if os.path.exists(ppath):
        with open(ppath) as f:
            pp.update(json.load(f))
    lm, proc, w = _load_reranker(os.path.join(model_dir, "models", "reranker"), device)
    m = {
        "cfg": cfg, "device": device, "s1": s1, "tok": tok, "s1_dtype": s1.dtype,
        "s1_native": int(s1.config.vision_config.image_size),
        "s1_resample": Image.Resampling(int(pp["resample"])),
        "mean": torch.tensor(pp["image_mean"], device=device).view(1, 3, 1, 1),
        "std": torch.tensor(pp["image_std"], device=device).view(1, 3, 1, 1),
        "lm": lm, "proc": proc, "w": w,
    }
    _log(f"loaded in {time.perf_counter() - t0:.1f}s cfg={ {k: v for k, v in cfg.items() if k != 'instruct'} }")
    return m


@torch.inference_mode()
def _encode_texts(m, texts):
    out = []
    for s in range(0, len(texts), 256):
        batch = [(t or "").lower() for t in texts[s:s + 256]]
        enc = m["tok"](batch, padding="max_length", truncation=True, max_length=m["cfg"]["s1_text_max_len"], return_tensors="pt")
        f = _as_tensor(m["s1"].get_text_features(input_ids=enc["input_ids"].to(m["device"])))
        out.append(F.normalize(f.float(), dim=-1))
    return torch.cat(out)


@torch.inference_mode()
def _encode_images(m, paths):
    cfg, device = m["cfg"], m["device"]
    n, rr = len(paths), cfg["rr_res"]
    s1_size = int(cfg.get("s1_res") or m["s1_native"])  # read at predict time so cfg overrides apply
    interp = s1_size != m["s1_native"]
    ds = _ImageDS(paths, s1_size, m["s1_resample"], rr)
    nw = cfg["num_workers"]
    loader = DataLoader(ds, batch_size=cfg["s1_img_batch"], shuffle=False, num_workers=nw,
                        pin_memory=str(device).startswith("cuda"), worker_init_fn=_worker_init if nw else None,
                        prefetch_factor=4 if nw else None)
    feats, rr_imgs, bad, t0 = None, np.empty((n, rr, rr, 3), dtype=np.uint8), 0, time.perf_counter()
    for bi, (x, xr, idx, ok) in enumerate(loader):
        bad += int((~ok).sum())
        rr_imgs[idx.numpy()] = xr.numpy()
        x = x.to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255.0)
        x = ((x - m["mean"]) / m["std"]).to(m["s1_dtype"])
        kw = {"interpolate_pos_encoding": True} if interp else {}
        f = F.normalize(_as_tensor(m["s1"].get_image_features(pixel_values=x, **kw)).float(), dim=-1)
        if feats is None:
            feats = torch.zeros(n, f.shape[1], device=device)
        feats[idx.to(device)] = f
        if bi % 50 == 0:
            _log(f"images {min((bi + 1) * cfg['s1_img_batch'], n)}/{n} ({time.perf_counter() - t0:.0f}s)")
    if bad:
        _log(f"WARNING: {bad} images failed to decode")
    return feats, rr_imgs


def _prompt(m, query):
    """Official reranker format. cfg["system"] / cfg["instruct"] = "" drop those parts (shorter = faster; LoRA-trained)."""
    system, instruct = m["cfg"].get("system", SYSTEM), m["cfg"].get("instruct", INSTRUCT_BINDING)
    user = [{"type": "text", "text": "<Instruct>: " + instruct}] if instruct else []
    user += [{"type": "text", "text": "<Query>:"}, {"type": "text", "text": query or ""},
             {"type": "text", "text": "\n<Document>:"}, {"type": "image"}]
    msgs = ([{"role": "system", "content": [{"type": "text", "text": system}]}] if system else []) + [{"role": "user", "content": user}]
    return m["proc"].apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def predict(model, queries, candidates, top_k=5):
    m, cfg = model, model["cfg"]
    t0 = time.perf_counter()
    qids = [q["prompt_id"] for q in queries]
    texts = [q["prompt"] for q in queries]
    img_ids = [c["image_id"] for c in candidates]
    paths = [c["image_path"] for c in candidates]
    nq, ni = len(qids), len(img_ids)
    kout = min(top_k, ni)
    kmax = min(max(cfg["k_max"], cfg["k_min"], kout), ni)

    # ---- stage 1: SigLIP2 (each query's ranking uses only its own text embedding)
    T = _encode_texts(m, texts)
    I, rr_imgs = _encode_images(m, paths)
    bad = ~torch.isfinite(I).all(1)
    if bad.any():
        I[bad] = 0.0
    top_s, top_i = [], []
    for s in range(0, nq, 2048):
        v, i = (T[s:s + 2048] @ I.T).topk(kmax, dim=1)
        top_s.append(v.cpu()); top_i.append(i.cpu())
    top_s, top_i = torch.cat(top_s).numpy(), torch.cat(top_i).numpy()
    t1 = time.perf_counter()
    _log(f"stage 1 done: {nq} queries x {ni} images in {t1 - t0:.1f}s")

    # ---- per-query k: a fixed rule of the query's own stage-1 scores. Start at k_min and add candidate r while
    # it is within `gap` of the query's top-1 score (k=1 means stage 1 is confident: no reranking).
    ks = np.full(nq, min(cfg["k_min"], kmax), dtype=int)
    if cfg["k_max"] > cfg["k_min"] and cfg["gap"] > 0:
        for q in range(nq):
            k = ks[q]
            while k < kmax and top_s[q, 0] - top_s[q, k] <= cfg["gap"]:
                k += 1
            ks[q] = k
    # work order: ranks 0-1 of every reranked query first, then rank 2 of queries with k>=3, then rank 3, ...
    # so if the time guard fires, the queries it touches still get their top-2 reranked.
    pairs = [(q, r) for q in range(nq) if ks[q] >= 2 for r in range(2)]
    pairs += [(q, r) for r in range(2, kmax) for q in range(nq) if ks[q] > r]
    _log(f"rerank plan: {len(pairs)} pairs, k distribution {np.bincount(ks, minlength=kmax + 1)[1:].tolist()}")

    # ---- stage 2: VLM reranking with a time guard (unfinished queries keep the stage-1 order)
    prompts = {}
    logits = np.full((nq, kmax), -np.inf, dtype=np.float32)
    scored = np.zeros((nq, kmax), dtype=bool)
    lm, proc, w, bs = m["lm"], m["proc"], m["w"], cfg["rr_batch"]

    def prep(chunk):
        tx = []
        for q, _ in chunk:
            if q not in prompts:
                prompts[q] = _prompt(m, texts[q])
            tx.append(prompts[q])
        ims = [Image.fromarray(rr_imgs[top_i[q, r]]) for q, r in chunk]
        return proc(text=tx, images=ims, padding=True, return_tensors="pt", do_resize=False)

    chunks = [pairs[s:s + bs] for s in range(0, len(pairs), bs)]
    stopped = False
    with ThreadPoolExecutor(1) as ex:
        fut = ex.submit(prep, chunks[0]) if chunks else None
        for ci, chunk in enumerate(chunks):
            inputs = fut.result()
            if ci + 1 < len(chunks):
                if time.perf_counter() - t0 > cfg["time_limit"]:
                    stopped = True
                    break
                fut = ex.submit(prep, chunks[ci + 1])
            inputs = inputs.to(m["device"])
            h = lm.model(**inputs).last_hidden_state[:, -1].float()
            sc = (h @ w).cpu().numpy()
            for (q, r), v in zip(chunk, sc):
                logits[q, r] = v
                scored[q, r] = True
            if ci % 200 == 0:
                _log(f"rerank {min((ci + 1) * bs, len(pairs))}/{len(pairs)} pairs ({time.perf_counter() - t1:.0f}s)")
    if stopped:
        _log(f"TIME GUARD: stopped at {time.perf_counter() - t0:.0f}s")

    # ---- output
    preds, n_rr, n_full = {}, 0, 0
    for q in range(nq):
        cand = list(top_i[q])
        kd = 0  # leading ranks that were scored
        while kd < min(ks[q], kmax) and scored[q, kd]:
            kd += 1
        if kd >= 2:
            order = list(np.argsort(-logits[q, :kd], kind="stable"))
            cand = [cand[i] for i in order] + cand[kd:]
            n_rr += 1
            n_full += int(kd == ks[q])
        preds[qids[q]] = [img_ids[j] for j in cand[:kout]]
    t2 = time.perf_counter()
    m["_last"] = {"top_i": top_i, "top_s": top_s, "logits": logits, "scored": scored, "ks": ks}  # for offline analysis
    m["_timing"] = {"stage1": t1 - t0, "rerank": t2 - t1, "pairs_done": int(scored.sum()), "pairs_planned": len(pairs),
                    "reranked_queries": n_rr, "full_k_queries": n_full}
    _log(f"predict done in {t2 - t0:.1f}s; reranked {n_rr}/{nq} queries")
    return preds

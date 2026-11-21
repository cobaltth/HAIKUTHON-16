"""HAIKUTHON v10: SigLIP2 retrieval -> Qwen3-VL-Reranker-2B reranking of the top-k (vision features cached per image).

v9 (cfg "fam_k" > k_max): the query's stage-1 top-fam_k images are treated as one family. The first fam_n0
candidates are scored in stage-1 order; the remaining reranker slots are filled adaptively: the reranker logits
already obtained for THIS query are extrapolated to its unscored family members through image-image similarity
(family-centred SigLIP2 image embeddings), and the best-looking members are scored next.

v10 (cfg "cube": true, needs fam_k = 32): if the family's image embeddings split cleanly into m binary axes
(m = 5, 4 or 3; every combination present equally often), the reranker is used axis by axis: score a start image,
then one image per axis that differs from the start on that axis only; flip every axis whose neighbour scored
higher; score the image(s) of the resulting combination. Families without a clean split use the v9 rule.
All of it uses only the query's own text, its own candidate images and its own reranker scores.

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
from types import SimpleNamespace

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
    "vcache": True,           # v7: run the vision tower once per image and reuse its output for every pair
    "vc_batch": 32,           # pairs per forward in the cached path (text + cached image tokens only)
    "vc_img_batch": 64,       # images per vision-tower forward
    "vc_tol": 0.5,            # self-check: max |cached - reference| logit difference before falling back
    "len_sort": True,         # inside a pass, batch queries of similar token length together (less padding)
    "fam_k": 0,               # v9: family size taken from stage 1 (0 = off -> identical to v7/v8)
    "fam_n0": 4,              # v9: candidates scored in stage-1 order before adaptive selection starts
    "fam_step": 1,            # v9: candidates chosen per adaptive round
    "fam_beta": 0.25,         # v9: weight of the extrapolated reranker preference vs the stage-1 score
    "cube": False,            # v10: axis-by-axis judging when the family is a clean binary grid
    "cube_tau": 0.25,         # v10: minimum |coordinate| (ideal 1.0) for an axis to count as cleanly split
    "cube_fit_w": 1.0,        # v10: start image = argmax z(stage1) + w * z(axis-additive fit of stage1); 0 = stage-1 top-1
    "cube_margin": 0.0,       # v10: flip an axis when neighbour logit - start logit > margin
    "cube_mode": "nbr_kernel",  # v10: "flip" | "nbr_kernel" (start + axis neighbours, then grid-distance extrapolation) | "kernel"
    "cube_beta": 1.0,         # v10: weight of the grid-distance extrapolation for families with a clean grid
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


def _fam_pick(S, fam_s, ordq, lg, r, beta):
    """Re-order the unscored family members of every query (pure numpy, one query per row, no cross-query terms).

    S     (nq, K, K) image-image similarity inside the query's family (family-centred cosine)
    fam_s (nq, K)    stage-1 scores of the family members
    ordq  (nq, K)    current order (indices into the family); the first r are already scored by the reranker
    lg    (nq, r)    reranker logits of ordq[:, :r]
    Unscored member j gets  z(stage1_j) + beta * z(sum_i z(logit_i) * S[j, i]).
    Rows whose logits are not all finite are left unchanged.
    """
    nq, K = fam_s.shape
    new = ordq.copy()
    if r <= 1 or r >= K:
        return new
    good = np.isfinite(lg).all(1)
    if not good.any():
        return new
    g = np.where(good)[0]
    sc, un = ordq[g, :r], ordq[g, r:]
    z = lg[g].astype(np.float64)
    z = z - z.mean(1, keepdims=True)
    z = z / (z.std(1, keepdims=True) + 1e-6)
    sub = S[g[:, None, None], un[:, :, None], sc[:, None, :]].astype(np.float64)      # (n, K-r, r)
    pred = (sub * z[:, None, :]).sum(-1)
    pz = (pred - pred.mean(1, keepdims=True)) / (pred.std(1, keepdims=True) + 1e-6)
    fs = fam_s[g].astype(np.float64)
    fz = (np.take_along_axis(fs, un, 1) - fs.mean(1, keepdims=True)) / (fs.std(1, keepdims=True) + 1e-6)
    b = np.asarray(beta, dtype=np.float64)
    b = b[g][:, None] if b.ndim == 1 else b
    o = np.argsort(-(fz + b * pz), axis=1, kind="stable")
    new[g, r:] = np.take_along_axis(un, o, 1)
    return new


def _cube_axes(G, r_list=(5, 4, 3), tau=0.25, sweeps=6, n_ang=32):
    """Binary axes of each family from its centred Gram matrix G (f, n, n) of image embeddings.

    If the n images are all combinations of r two-valued factors and each factor shifts the embedding by a fixed
    vector, the top-r principal components span the factors, and the factors are the rotation of those components
    whose entries all have equal magnitude (minimum sum of 4th powers). Returns code (f, n) with the r-bit
    combination of every image (-1 if no clean grid) and m (f,) = r used (0 if none). A grid is accepted only if every
    axis splits n/2 : n/2, every |coordinate| >= tau and every combination occurs exactly n / 2**r times.
    """
    f, n, _ = G.shape
    _, U = np.linalg.eigh(G.astype(np.float64))
    U = U[:, :, ::-1]
    codes = np.full((f, n), -1, dtype=np.int64)
    m_of = np.zeros(f, dtype=np.int64)
    ang = np.linspace(0.0, np.pi / 2, n_ang, endpoint=False)
    ca, sa = np.cos(ang)[None, :, None], np.sin(ang)[None, :, None]
    todo = np.ones(f, dtype=bool)
    for r in r_list:
        if n % (1 << r) or (n >> r) > 4 or not todo.any():
            continue
        idx = np.where(todo)[0]
        ar = np.arange(len(idx))
        Y = U[idx][:, :, :r] * np.sqrt(n)
        for _ in range(sweeps):
            for a in range(r):
                for b in range(a + 1, r):
                    ya, yb = Y[:, None, :, a], Y[:, None, :, b]
                    A, B = ya * ca + yb * sa, yb * ca - ya * sa          # (f, n_ang, n)
                    k = ((A ** 4).sum(-1) + (B ** 4).sum(-1)).argmin(1)
                    Y[:, :, a], Y[:, :, b] = A[ar, k], B[ar, k]
        bits = Y > 0
        code = (bits * (1 << np.arange(r))).sum(-1)
        cnt = np.stack([(code == c).sum(1) for c in range(1 << r)], 1)
        ok = (bits.sum(1) == n // 2).all(1) & (np.abs(Y).min(1) >= tau).all(1) & (cnt == (n >> r)).all(1)
        g = idx[ok]
        codes[g], m_of[g] = code[ok], r
        todo[g] = False
    return codes, m_of


def _cube_setup(I, fam_i, fam_s, S, cfg, n0):
    """Per query: grid coordinates of its family, start image, one neighbour per axis. Returns the plan."""
    nq, K = fam_i.shape
    order = np.argsort(fam_i, axis=1)
    srt = np.take_along_axis(fam_i, order, 1)
    uq, inv = np.unique(srt, axis=0, return_inverse=True)   # same candidate set -> same (deterministic) coordinates
    inv = np.asarray(inv).reshape(-1)
    G = np.empty((len(uq), K, K), dtype=np.float64)
    for s in range(0, len(uq), 256):
        E = I[torch.as_tensor(uq[s:s + 256], device=I.device)].double()
        E = E - E.mean(1, keepdim=True)
        G[s:s + 256] = (E @ E.transpose(1, 2)).cpu().numpy()
    cu, mu = _cube_axes(G, tau=float(cfg.get("cube_tau", 0.25)))
    code = np.full((nq, K), -1, dtype=np.int64)
    np.put_along_axis(code, order, cu[inv], 1)
    cm = mu[inv].copy()
    ordq = np.tile(np.arange(K), (nq, 1))
    lock = np.full(nq, n0, dtype=np.int64)
    cax = np.full((nq, 5), -1, dtype=np.int64)
    w = float(cfg.get("cube_fit_w", 1.0))
    mode = str(cfg.get("cube_mode", "nbr_kernel"))
    phase = np.zeros(nq, dtype=np.int64)
    pc = np.array([bin(i).count("1") for i in range(32)], dtype=np.float64)

    def z(x):
        return (x - x.mean()) / (x.std() + 1e-9)

    for q in np.where(cm > 0)[0]:
        m_, c, fs = int(cm[q]), code[q], fam_s[q].astype(np.float64)
        bits = (c[:, None] >> np.arange(m_)) & 1
        d = np.array([fs[bits[:, a] == 1].mean() - fs[bits[:, a] == 0].mean() for a in range(m_)])
        fit = ((bits * 2 - 1) * d).sum(1) / 2
        v = int(np.argmax(z(fs) + w * z(fit)))
        axo = np.argsort(np.abs(d), kind="stable")           # least certain stage-1 axes are tested first
        head = [v]
        for a in axo:
            memb = np.where(c == (c[v] ^ (1 << int(a))))[0]
            head.append(int(memb[np.argmax(S[q, v, memb])]))  # the member most similar to the start image
        if mode != "kernel" and m_ >= int(cfg.get("cube_nbr_min_axes", 4)):   # with only 3 clean axes: keep stage-1 start
            hs = set(head)
            ordq[q] = head + [j for j in range(K) if j not in hs]
            cax[q, :m_] = axo
            lock[q] = m_ + 1
        if mode != "flip" or lock[q] != m_ + 1:
            phase[q] = 1
        # similarity used for extrapolation: grid distance (number of differing axes), image similarity as tie-break
        S[q] = -np.sqrt(pc[c[:, None] ^ c[None, :]]) + 0.1 * S[q]
    return {"ord": ordq, "lock": lock, "cm": cm, "ccode": code, "cax": cax, "phase": phase,
            "cube_m": np.bincount(cm, minlength=6).tolist(), "cube_families": int(len(uq)), "flips": 0, "cube_done": 0}


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
    fam_k = min(int(cfg.get("fam_k", 0) or 0), ni)
    fam_on = fam_k > kmax and kmax >= 3
    K = fam_k if fam_on else kmax
    top_s, top_i = [], []
    for s in range(0, nq, 2048):
        v, i = (T[s:s + 2048] @ I.T).topk(K, dim=1)
        top_s.append(v.cpu()); top_i.append(i.cpu())
    fam_s, fam_i = torch.cat(top_s).numpy(), torch.cat(top_i).numpy()
    top_s, top_i = fam_s[:, :kmax].copy(), fam_i[:, :kmax].copy()   # columns >= fam_n0 are re-chosen later when fam_on
    fam = {"on": fam_on, "K": K, "rounds": 0, "moved": 0}
    if fam_on:
        try:   # image-image similarity inside each query's own family (centred on the family mean)
            S = np.empty((nq, K, K), dtype=np.float32)
            for s in range(0, nq, 256):
                E = I[torch.as_tensor(fam_i[s:s + 256], device=I.device)]
                E = F.normalize(E - E.mean(1, keepdim=True), dim=-1)
                S[s:s + 256] = (E @ E.transpose(1, 2)).cpu().numpy()
            fam["S"] = S
            fam["ord"] = np.tile(np.arange(K), (nq, 1))
            fam["n0"] = min(max(2, int(cfg.get("fam_n0", 4))), kmax)
            fam["step"] = max(1, int(cfg.get("fam_step", 1)))
            if cfg.get("cube", False) and kmax >= 8:
                try:
                    fam.update(_cube_setup(I, fam_i, fam_s, S, cfg, fam["n0"]))
                    fam["n0"], fam["step"] = min(fam["n0"], 4), 1
                    top_i = np.take_along_axis(fam_i, fam["ord"][:, :kmax], 1)
                    top_s = np.take_along_axis(fam_s, fam["ord"][:, :kmax], 1)
                    _log(f"cube: families {fam['cube_families']}, queries by axes found (0..5) {fam['cube_m']}")
                except Exception as e:
                    _log(f"WARNING cube setup failed ({e!r}); adaptive family selection only")
                    for k_ in ("lock", "cm", "ccode", "cax", "phase"):
                        fam.pop(k_, None)
                    fam["ord"] = np.tile(np.arange(K), (nq, 1))
                    top_s, top_i = fam_s[:, :kmax].copy(), fam_i[:, :kmax].copy()
                    fam["cube_error"] = repr(e)
        except Exception as e:
            _log(f"WARNING family setup failed ({e!r}); plain top-k reranking")
            fam["on"] = fam_on = False
            fam["error"] = repr(e)
    t1 = time.perf_counter()
    _log(f"stage 1 done: {nq} queries x {ni} images in {t1 - t0:.1f}s")

    # ---- per-query k: a fixed rule of the query's own stage-1 scores. Start at k_min and add candidate r while
    # it is within `gap` of the query's top-1 score (k=1 means stage 1 is confident: no reranking).
    ks = np.full(nq, min(cfg["k_min"], kmax), dtype=int)
    if fam_on:
        ks[:] = kmax          # family mode: every query gets all kmax reranker slots
    elif cfg["k_max"] > cfg["k_min"] and cfg["gap"] > 0:
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
    dev = m["device"]
    vc_info = {"used": False}

    def out_of_time():
        return time.perf_counter() - t0 > cfg["time_limit"]

    def fam_select(r):
        """Before slot r is scored: re-choose slots r.. of every query from its own family, using its own logits
        for slots < r. On any problem the stage-1 order is kept."""
        try:
            lock = fam.get("lock")
            if lock is not None:   # v10: queries whose start + axis neighbours are all scored -> decide the axes
                ordq, margin = fam["ord"], float(cfg.get("cube_margin", 0.0))
                for q in np.where((fam["cm"] > 0) & (fam["phase"] == 0) & (fam["cm"] + 1 <= r))[0]:
                    m_ = int(fam["cm"][q])
                    lg = logits[q, :m_ + 1]
                    fam["phase"][q] = 1
                    lock[q] = r
                    if not np.isfinite(lg).all():
                        continue
                    code = fam["ccode"][q]
                    tgt = int(code[ordq[q, 0]])
                    for k_ in range(m_):
                        if lg[1 + k_] - lg[0] > margin:
                            tgt ^= 1 << int(fam["cax"][q, k_])
                            fam["flips"] += 1
                    seen = set(ordq[q, :r].tolist())
                    mem = sorted((int(j) for j in np.where(code == tgt)[0] if int(j) not in seen), key=lambda j: -fam_s[q, j])
                    if mem:
                        ms = set(mem)
                        ordq[q, r:] = mem + [int(j) for j in ordq[q, r:] if int(j) not in ms]
                    lock[q] = r + len(mem)
                    fam["cube_done"] += 1
            beta = float(cfg.get("fam_beta", 0.25))
            if lock is not None:
                beta = np.where(fam["cm"] > 0, float(cfg.get("cube_beta", 1.0)), beta)
            new = _fam_pick(fam["S"], fam_s, fam["ord"], logits[:, :r], r, beta)
            if lock is not None:
                keep = lock > r      # slots fixed by the cube plan are not re-chosen
                new[keep] = fam["ord"][keep]
            fam["moved"] += int((new[:, r:kmax] != fam["ord"][:, r:kmax]).sum())
            fam["ord"] = new
            top_i[:, r:kmax] = np.take_along_axis(fam_i, new[:, r:kmax], 1)
            top_s[:, r:kmax] = np.take_along_axis(fam_s, new[:, r:kmax], 1)
            fam["rounds"] += 1
        except Exception as e:
            _log(f"WARNING family selection failed at slot {r} ({e!r}); keeping current order")
            fam["error"] = repr(e)

    def prep(chunk):
        tx = []
        for q, _ in chunk:
            if q not in prompts:
                prompts[q] = _prompt(m, texts[q])
            tx.append(prompts[q])
        ims = [Image.fromarray(rr_imgs[top_i[q, r]]) for q, r in chunk]
        return proc(text=tx, images=ims, padding=True, return_tensors="pt", do_resize=False)

    def score_ref(inputs):
        h = lm.model(**inputs.to(dev)).last_hidden_state[:, -1].float()
        return (h @ w).cpu().numpy()

    def store(chunk, sc):
        for (q, r), v in zip(chunk, sc):
            logits[q, r] = v
            scored[q, r] = True

    def run_reference(todo):
        """v4-v6 path: every pair goes through the processor and the vision tower."""
        chunks = [todo[s:s + bs] for s in range(0, len(todo), bs)]
        with ThreadPoolExecutor(1) as ex:
            fut = ex.submit(prep, chunks[0]) if chunks else None
            for ci, chunk in enumerate(chunks):
                inputs = fut.result()
                if ci + 1 < len(chunks):
                    if out_of_time():
                        return True
                    fut = ex.submit(prep, chunks[ci + 1])
                store(chunk, score_ref(inputs))
                if ci % 200 == 0:
                    _log(f"rerank(ref) {min((ci + 1) * bs, len(todo))}/{len(todo)} pairs ({time.perf_counter() - t1:.0f}s)")
        return False

    def run_cached():
        """v7 path. The vision tower output of an image does not depend on the query, so it is computed once per
        image (lazily, pass by pass) and re-used; each pair then only costs the language model. Returns
        (stopped_by_guard, ok). ok=False means the self-check failed and nothing was stored."""
        ip, tk = proc.image_processor, proc.tokenizer
        img_tok, pad_id = lm.config.image_token_id, tk.pad_token_id
        image_token = getattr(proc, "image_token", "<|image_pad|>")
        vb, ib = int(cfg["vc_batch"]), int(cfg["vc_img_batch"])
        # passes: ranks 0-1 of every reranked query, then rank 2, rank 3, ...
        n01 = 2 * int((ks >= 2).sum())
        rest = pairs[n01:]

        def seg_iter():
            yield pairs[:n01]
            for r in range(2, kmax):
                if fam_on and r >= fam["n0"] and (r - fam["n0"]) % fam["step"] == 0:
                    fam_select(r)     # slots < r of every query are scored at this point
                seg = [p for p in rest if p[1] == r]
                if seg:
                    yield seg

        if fam_on:   # any family member may be chosen later
            all_imgs = np.unique(fam_i.astype(np.int64))
        else:
            all_imgs = np.unique(np.array([top_i[q, r] for q, r in pairs], dtype=np.int64))
        pos = np.full(ni, -1, dtype=np.int64)
        st = {"E": None, "D": None, "n": 0, "grid": None, "ntok": None, "t_vis": 0.0, "t_lm": 0.0, "ids": {}}

        def img_prep(js):
            return ip(images=[Image.fromarray(rr_imgs[j]) for j in js], do_resize=False, return_tensors="pt")

        def build(need):
            """Vision features for the images in `need` (not cached yet). Returns True if the guard fired."""
            tb = time.perf_counter()
            groups = [need[s:s + ib] for s in range(0, len(need), ib)]
            with ThreadPoolExecutor(1) as ex:
                fut = ex.submit(img_prep, groups[0]) if groups else None
                for gi, js in enumerate(groups):
                    enc = fut.result()
                    if gi + 1 < len(groups):
                        if out_of_time():
                            st["t_vis"] += time.perf_counter() - tb
                            return True
                        fut = ex.submit(img_prep, groups[gi + 1])
                    grid = enc["image_grid_thw"].to(dev)
                    out = lm.model.get_image_features(enc["pixel_values"].to(dev), grid, return_dict=True)
                    e = torch.stack(list(out.pooler_output))                                  # (b, ntok, D)
                    d = torch.stack([torch.stack(list(f)) for f in out.deepstack_features])   # (L, b, ntok, D)
                    if st["E"] is None:
                        st["E"] = torch.empty((len(all_imgs),) + tuple(e.shape[1:]), dtype=e.dtype, device=dev)
                        st["D"] = torch.empty((d.shape[0], len(all_imgs)) + tuple(d.shape[2:]), dtype=d.dtype, device=dev)
                        st["grid"], st["ntok"] = grid[:1].clone(), int(e.shape[1])
                        _log(f"vcache: {len(all_imgs)} images x {st['ntok']} tokens x {e.shape[2]} dims, {d.shape[0]} deepstack levels, "
                             f"{(st['E'].numel() + st['D'].numel()) * st['E'].element_size() / 1e9:.1f} GB")
                    n0 = st["n"]
                    st["E"][n0:n0 + len(js)] = e
                    st["D"][:, n0:n0 + len(js)] = d
                    pos[np.asarray(js)] = np.arange(n0, n0 + len(js))
                    st["n"] = n0 + len(js)
            st["t_vis"] += time.perf_counter() - tb
            return False

        def tok_ids(q):
            if q not in st["ids"]:
                if q not in prompts:
                    prompts[q] = _prompt(m, texts[q])
                tx = prompts[q].replace(image_token, image_token * st["ntok"])
                st["ids"][q] = tk(tx, add_special_tokens=False)["input_ids"]
            return st["ids"][q]

        def make_ids(chunk):
            seqs = [tok_ids(q) for q, _ in chunk]
            L = max(len(x) for x in seqs)
            ids = torch.full((len(chunk), L), pad_id, dtype=torch.long)
            am = torch.zeros((len(chunk), L), dtype=torch.long)
            for i, x in enumerate(seqs):          # left padding: the last position is always the real last token
                ids[i, L - len(x):] = torch.tensor(x, dtype=torch.long)
                am[i, L - len(x):] = 1
            return ids, am

        def score_cached(chunk):
            ids, am = make_ids(chunk)
            ids, am = ids.to(dev), am.to(dev)
            j = torch.as_tensor(pos[[top_i[q, r] for q, r in chunk]], device=dev)
            E, D = st["E"][j], st["D"][:, j]
            feats = SimpleNamespace(pooler_output=(E.reshape(-1, E.shape[-1]),),
                                    deepstack_features=[(D[l].reshape(-1, D.shape[-1]),) for l in range(D.shape[0])])
            out = lm.model(input_ids=ids, attention_mask=am, image_grid_thw=st["grid"].expand(len(chunk), -1),
                           mm_token_type_ids=(ids == img_tok).to(torch.int32), mm_encoder_outputs={"image": feats})
            return (out.last_hidden_state[:, -1].float() @ w).cpu().numpy()

        def fin(stopped_, ok_=True):
            vc_info.update({"images": st["n"], "t_vis": round(st["t_vis"], 2), "t_lm": round(st["t_lm"], 2)})
            return stopped_, ok_

        checked = False
        for si, seg in enumerate(seg_iter()):
            need = [int(j) for j in np.unique(np.array([top_i[q, r] for q, r in seg], dtype=np.int64)) if pos[j] < 0]
            if not checked:
                # self-check on the first pairs: cached path must reproduce the reference path
                probe = seg[:min(8, len(seg))]
                if build([int(j) for j in dict.fromkeys(int(top_i[q, r]) for q, r in probe)]):
                    return fin(True)
                ref_in = prep(probe)
                my_ids, _ = make_ids(probe)
                same = tuple(ref_in["input_ids"].shape) == tuple(my_ids.shape) and bool((ref_in["input_ids"] == my_ids).all())
                vc_info["ids_match"] = same
                a_ref = score_ref(ref_in)
                a_vc = score_cached(probe)
                diff = float(np.abs(a_ref - a_vc).max())
                vc_info["check_diff"] = diff
                _log(f"vcache self-check: token ids match={same}, max |cached - reference| = {diff:.4f} (tol {cfg['vc_tol']})")
                if not np.isfinite(diff) or diff > cfg["vc_tol"]:
                    return fin(False, False)
                checked = True
                need = [j for j in need if pos[j] < 0]
            if build(need):
                return fin(True)
            if cfg.get("len_sort", True):
                if si == 0:   # keep (q,0),(q,1) together, queries ordered by token length
                    qs = sorted(dict.fromkeys(q for q, _ in seg), key=lambda q: len(tok_ids(q)))
                    seg = [(q, r) for q in qs for r in range(2)]
                else:
                    seg = sorted(seg, key=lambda p: len(tok_ids(p[0])))
            tl = time.perf_counter()
            for ci, s in enumerate(range(0, len(seg), vb)):
                if out_of_time():
                    st["t_lm"] += time.perf_counter() - tl
                    return fin(True)
                chunk = seg[s:s + vb]
                store(chunk, score_cached(chunk))
                if ci % 200 == 0:
                    _log(f"rerank pass {si} {min(s + vb, len(seg))}/{len(seg)} pairs ({time.perf_counter() - t1:.0f}s)")
            st["t_lm"] += time.perf_counter() - tl
        return fin(False)

    stopped = False
    if pairs and cfg.get("vcache", True):
        try:
            stopped, ok = run_cached()
            vc_info["used"] = ok
        except Exception as e:  # any API mismatch -> reference path (still a valid submission)
            _log(f"WARNING vcache failed ({e!r}); using the reference path")
            ok = False
            vc_info["error"] = repr(e)
        if not ok:   # keep whatever the cached path already scored, do the rest the old way
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            stopped = run_reference([p for p in pairs if not scored[p[0], p[1]]])
    elif pairs:
        stopped = run_reference(pairs)
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
    m["_last"] = {"top_i": top_i, "top_s": top_s, "logits": logits, "scored": scored, "ks": ks,
                  "fam_i": fam_i, "fam_s": fam_s, "fam_ord": fam.get("ord"), "cube_m": fam.get("cm"), "cube_code": fam.get("ccode")}  # for offline analysis
    m["_timing"] = {"stage1": t1 - t0, "rerank": t2 - t1, "pairs_done": int(scored.sum()), "pairs_planned": len(pairs),
                    "reranked_queries": n_rr, "full_k_queries": n_full, "vcache": vc_info,
                    "fam": {k: v for k, v in fam.items() if k not in ("S", "ord", "lock", "cm", "ccode", "cax", "phase")}}
    _log(f"predict done in {t2 - t0:.1f}s; reranked {n_rr}/{nq} queries")
    return preds

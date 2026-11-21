#!/usr/bin/env python
"""LoRA fine-tuning of Qwen3-VL-Reranker-2B on sibling / hard negatives (team train split only).

  python train_lora.py --sub sub_v1 --groups groups.csv --out lora_r16 --epochs 2
  python prepare_v1.py --reranker lora_r16/merged --out sub_v2 ...     # pack the merged model like v1

Per training query: the positive image, `negs` negatives drawn from its siblings (same scene template, groups.csv)
plus SigLIP2 top-8 within the train pool, and optionally one text negative (attribute binding swap / relation
flip / count change applied to the query, scored against the positive image).
Loss = CE over [pos, negs] + lam_txt * CE over [query, text-negative] on the positive image.
Scores use exactly the inference path of model.py: same prompt, 256px, h_last @ (W_yes - W_no).
valA (800 queries, pool valA+valB) is scored before training and after every epoch with SigLIP top-5 + rerank top-3.
"""
import argparse
import csv
import importlib.util
import json
import math
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


# ----------------------------------------------------------------------------- LoRA
class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha, dropout):
        super().__init__()
        self.base, self.scale = base, alpha / r
        dev = base.weight.device
        self.A = nn.Parameter(torch.empty(r, base.in_features, device=dev, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + (self.drop(x) @ self.A.t().to(x.dtype)) @ self.B.t().to(x.dtype) * self.scale

    @torch.no_grad()
    def merged(self):
        W = self.base.weight
        W.add_((self.B @ self.A * self.scale).to(W.dtype))
        return self.base


def add_lora(lm, r, alpha, dropout):
    reps = []
    for name, mod in lm.named_modules():
        for cname, child in mod.named_children():
            full = f"{name}.{cname}"
            if isinstance(child, nn.Linear) and cname in TARGETS and "language_model" in full and "visual" not in full:
                reps.append((mod, cname, child))
    for mod, cname, child in reps:
        setattr(mod, cname, LoRALinear(child, r, alpha, dropout))
    return len(reps)


def lora_modules(lm):
    return {n: m for n, m in lm.named_modules() if isinstance(m, LoRALinear)}


def merge_lora(lm):
    reps = [(mod, cname, child) for _, mod in lm.named_modules() for cname, child in mod.named_children()
            if isinstance(child, LoRALinear)]
    for mod, cname, child in reps:
        setattr(mod, cname, child.merged())
    return len(reps)


# ----------------------------------------------------------------------------- text negatives
COLORS = ("red orange yellow green blue purple pink brown black white gray grey silver golden gold beige teal cyan "
          "magenta navy maroon turquoise violet lime olive tan cream").split()
PATTERNS = "striped polka-dotted checkered plaid spotted dotted floral zigzag patterned solid".split()
MATERIALS = "wooden metal metallic glass plastic leather stone ceramic paper fabric rubber woven knitted".split()
COUNTS = ["one", "two", "three", "four", "five"]
FLIPS = [("left", "right"), ("above", "below"), ("in front of", "behind")]


def _keep_case(src, new):
    return new[:1].upper() + new[1:] if src[:1].isupper() else new


def _swap_attr(q, vocab, rng):
    ms = [m for m in re.finditer(r"\b(" + "|".join(map(re.escape, vocab)) + r")\b", q, flags=re.I)]
    pairs = [(a, b) for i, a in enumerate(ms) for b in ms[i + 1:] if a.group(0).lower() != b.group(0).lower()]
    if not pairs:
        return None
    a, b = rng.choice(pairs)
    return (q[:a.start()] + _keep_case(a.group(0), b.group(0).lower()) + q[a.end():b.start()]
            + _keep_case(b.group(0), a.group(0).lower()) + q[b.end():])


def _flip_rel(q, rng):
    opts = []
    for x, y in FLIPS:
        for s, t in ((x, y), (y, x)):
            for m in re.finditer(r"\b" + re.escape(s) + r"\b", q, flags=re.I):
                opts.append((m, t))
    if not opts:
        return None
    m, t = rng.choice(opts)
    return q[:m.start()] + _keep_case(m.group(0), t) + q[m.end():]


def _change_count(q, rng):
    ms = list(re.finditer(r"\b(" + "|".join(COUNTS) + r")\b", q, flags=re.I))
    if not ms:
        return None
    m = rng.choice(ms)
    i = COUNTS.index(m.group(0).lower())
    j = rng.choice([k for k in (i - 1, i + 1) if 0 <= k < len(COUNTS)])
    return q[:m.start()] + _keep_case(m.group(0), COUNTS[j]) + q[m.end():]


def text_negative(q, rng):
    fns = [lambda: _swap_attr(q, COLORS, rng), lambda: _swap_attr(q, PATTERNS, rng),
           lambda: _swap_attr(q, MATERIALS, rng), lambda: _flip_rel(q, rng), lambda: _change_count(q, rng)]
    rng.shuffle(fns)
    for f in fns:
        out = f()
        if out and out != q:
            return out
    return None


# ----------------------------------------------------------------------------- data / stage 1
def load_sub(sub):
    sys.path.insert(0, sub)
    spec = importlib.util.spec_from_file_location("sub_model_lora", os.path.join(sub, "model.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def siglip_m(mod, sub, device, rr_res, workers):
    cfg = dict(mod.CFG)
    cfg.update(rr_res=rr_res, num_workers=workers, s1_res=0)
    sdir = os.path.join(sub, "models", "siglip2")
    s1 = AutoModel.from_pretrained(sdir, dtype=torch.float16, local_files_only=True).to(device).eval()
    pp = {"image_mean": [0.5] * 3, "image_std": [0.5] * 3, "resample": 3}
    if os.path.exists(os.path.join(sdir, "haiku_preprocess.json")):
        pp.update(json.load(open(os.path.join(sdir, "haiku_preprocess.json"))))
    return {"cfg": cfg, "device": device, "s1": s1, "tok": AutoTokenizer.from_pretrained(sdir, local_files_only=True),
            "s1_dtype": s1.dtype, "s1_size": int(s1.config.vision_config.image_size),
            "s1_native": int(s1.config.vision_config.image_size),
            "s1_resample": Image.Resampling(int(pp["resample"])),
            "mean": torch.tensor(pp["image_mean"], device=device).view(1, 3, 1, 1),
            "std": torch.tensor(pp["image_std"], device=device).view(1, 3, 1, 1)}


def score_lists(top5, gt):
    r1 = nd = 0.0
    for p, g in zip(top5, gt):
        p = list(p)
        r1 += float(p[0] == g)
        if g in p[:5]:
            nd += 1.0 / math.log2(p.index(g) + 2)
    n = len(gt)
    return r1 / n, nd / n, 0.5 * (r1 + nd) / n


# ----------------------------------------------------------------------------- scoring
def make_inputs(proc, prompt_fn, texts, imgs, device):
    return proc(text=[prompt_fn(t) for t in texts], images=[Image.fromarray(x) for x in imgs],
                padding=True, return_tensors="pt", do_resize=False).to(device)


def scores(lm, w, inputs):
    h = lm.model(**inputs, use_cache=False).last_hidden_state[:, -1].float()
    return h @ w


@torch.no_grad()
def evaluate(lm, w, proc, prompt_fn, texts, top5, rr_imgs, gt, k, bs, device):
    was = lm.training
    lm.eval()
    pairs = [(q, r) for q in range(len(texts)) for r in range(k)]
    logits = np.zeros((len(texts), k), np.float32)
    for s in range(0, len(pairs), bs):
        ch = pairs[s:s + bs]
        inp = make_inputs(proc, prompt_fn, [texts[q] for q, _ in ch], [rr_imgs[top5[q, r]] for q, r in ch], device)
        for (q, r), v in zip(ch, scores(lm, w, inp).cpu().numpy()):
            logits[q, r] = v
    out = []
    for q in range(len(texts)):
        order = np.argsort(-logits[q], kind="stable")
        out.append([top5[q, i] for i in order] + list(top5[q, k:]))
    lm.train(was)
    return score_lists(out, gt), logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="sub_v1", help="v1 submission folder (SigLIP2 weights + model.py)")
    ap.add_argument("--reranker", default="Qwen/Qwen3-VL-Reranker-2B")
    ap.add_argument("--train_dir", default="haikuthon-train")
    ap.add_argument("--split", default="splits/split_group_seed42.json")
    ap.add_argument("--groups", default="groups.csv")
    ap.add_argument("--train_on", default="train", help="comma list of split names to train on, e.g. train or train,valA")
    ap.add_argument("--final", action="store_true", help="final model: train on every listed split even if it overlaps the eval split; keep the last epoch")
    ap.add_argument("--eval_on", default="valA", choices=["valA", "valB"], help="held-out split (pool = it + the other val split)")
    ap.add_argument("--out", default="lora_r16")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--qbs", type=int, default=4, help="queries per step")
    ap.add_argument("--negs", type=int, default=3)
    ap.add_argument("--lam_txt", type=float, default=0.5)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--eval_bs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_merge", action="store_true")
    ap.add_argument("--instruct", default=None, help="override the instruct ('' = none); saved to <out>/prompt.json")
    ap.add_argument("--system", default=None, help="override the system prompt ('' = none)")
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    dev = "cuda:0"
    os.makedirs(a.out, exist_ok=True)
    t_all = time.perf_counter()

    root = Path(a.train_dir).resolve()
    recs = {r["id"]: r for r in (json.loads(l) for l in open(root / "prompts.jsonl", encoding="utf-8") if l.strip())}
    sp = json.loads(Path(a.split).read_text())
    tr_ids = [i for s in a.train_on.split(",") for i in sp[s]]
    other = {"valA": "valB", "valB": "valA"}[a.eval_on]
    va_ids, pool_ids = sp[a.eval_on], sp[a.eval_on] + sp[other]
    assert a.final or not (set(tr_ids) & set(va_ids)), f"eval split {a.eval_on} overlaps the training splits {a.train_on}"
    if set(tr_ids) & set(pool_ids):
        print(f"[data] note: {len(set(tr_ids) & set(pool_ids))} pool images are training images (eval slightly optimistic)")
    gid = {}
    if os.path.exists(a.groups):
        gid = {r["id"]: r["group_id"] for r in csv.DictReader(open(a.groups))}
    else:  # proxy for the scene-template groups: prompts sharing >= 70% of their word set (within the train ids)
        from scipy.sparse import csr_matrix
        from sklearn.feature_extraction.text import CountVectorizer
        X = CountVectorizer(binary=True, token_pattern=r"[a-zA-Z-]+").fit_transform([recs[i]["prompts"].lower() for i in tr_ids])
        X = csr_matrix(X, dtype=np.float32)
        inter = (X @ X.T).toarray()
        sz = np.asarray(X.sum(1)).ravel()
        jac = inter / (sz[:, None] + sz[None, :] - inter)
        np.fill_diagonal(jac, 0)
        parent = list(range(len(tr_ids)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for x, y in zip(*np.where(jac >= 0.7)):
            parent[find(x)] = find(y)
        gid = {i: find(n) for n, i in enumerate(tr_ids)}
        print(f"[data] {a.groups} not found: word-Jaccard>=0.7 proxy groups, {len(set(gid.values()))} groups")

    # ---- stage 1 on train pool (mining) and on the valA pool (evaluation)
    mod = load_sub(os.path.abspath(a.sub))
    sm = siglip_m(mod, os.path.abspath(a.sub), dev, a.res, a.workers)
    path = lambda i: str(root / "images" / f"{i}.png")
    tr_texts = [recs[i]["prompts"] for i in tr_ids]
    T = mod._encode_texts(sm, tr_texts)
    I, tr_imgs = mod._encode_images(sm, [path(i) for i in tr_ids])
    S = T @ I.T
    S.fill_diagonal_(-1e4)
    hard = S.topk(8, dim=1).indices.cpu().numpy()
    tpos = {i: n for n, i in enumerate(tr_ids)}
    sib = [[tpos[j] for j in tr_ids if j != i and gid.get(j) is not None and gid.get(j) == gid.get(i)] for i in tr_ids]
    neg_pool = [list(dict.fromkeys(sib[n] + list(hard[n]))) for n in range(len(tr_ids))]
    va_texts = [recs[i]["prompts"] for i in va_ids]
    Tv = mod._encode_texts(sm, va_texts)
    Iv, va_imgs = mod._encode_images(sm, [path(i) for i in pool_ids])
    va_top5 = (Tv @ Iv.T).topk(5, dim=1).indices.cpu().numpy()
    va_gt = list(range(len(va_ids)))  # valA images come first in the pool
    del sm, T, I, S, Tv, Iv
    torch.cuda.empty_cache()
    n_sib = sum(1 for s in sib if s)
    print(f"[data] train {len(tr_ids)} (with siblings {n_sib}), mean neg pool {np.mean([len(p) for p in neg_pool]):.1f}; "
          f"valA {len(va_ids)} x pool {len(pool_ids)}, SigLIP R@3 {np.mean([g in va_top5[q, :a.k] for q, g in enumerate(va_gt)]):.3f}")
    txt_neg_rate = np.mean([text_negative(q, random.Random(0)) is not None for q in tr_texts])
    ex = [(q, text_negative(q, random.Random(1))) for q in tr_texts[:3]]
    print(f"[data] text negatives available for {txt_neg_rate:.0%} of queries; e.g. {ex[0]}")

    # ---- model
    lm = Qwen3VLForConditionalGeneration.from_pretrained(a.reranker, dtype=torch.bfloat16).to(dev)
    proc = AutoProcessor.from_pretrained(a.reranker)
    proc.tokenizer.padding_side = "left"
    voc = proc.tokenizer.get_vocab()
    W = lm.get_output_embeddings().weight
    w = (W[voc["yes"]].float() - W[voc["no"]].float()).detach()
    for p in lm.parameters():
        p.requires_grad_(False)
    pcfg = {"instruct": mod.CFG["instruct"] if a.instruct is None else a.instruct,
            "system": mod.CFG.get("system", mod.SYSTEM) if a.system is None else a.system}
    json.dump(pcfg, open(os.path.join(a.out, "prompt.json"), "w"))
    fake = {"cfg": pcfg, "proc": proc}
    print(f"[prompt] {mod._prompt(fake, 'QUERY')!r} -> {len(proc.tokenizer(mod._prompt(fake, 'QUERY'))['input_ids'])} text tokens + image")
    cache = {}

    def prompt_fn(t):
        if t not in cache:
            cache[t] = mod._prompt(fake, t)
        return cache[t]

    (r1, nd, sc0), _ = evaluate(lm, w, proc, prompt_fn, va_texts, va_top5, va_imgs, va_gt, a.k, a.eval_bs, dev)
    print(f"[eval] before LoRA (bf16): {a.eval_on} R@1 {r1:.4f} NDCG@5 {nd:.4f} score {sc0:.4f}")
    log = [{"epoch": 0, "R@1": r1, "NDCG@5": nd, "score": sc0}]

    n_l = add_lora(lm, a.rank, a.alpha, a.dropout)
    params = [p for m in lora_modules(lm).values() for p in (m.A, m.B)]
    print(f"[lora] {n_l} linear layers, {sum(p.numel() for p in params) / 1e6:.1f}M trainable params")
    lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lm.config.use_cache = False
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    steps_ep = math.ceil(len(tr_ids) / a.qbs)
    total = steps_ep * a.epochs
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    def build(batch_q):
        texts, imgs, n_txt = [], [], []
        for n in batch_q:
            pool = neg_pool[n]
            wts = [2.0 if j in sib[n] else 1.0 for j in pool]
            negs = []
            cand = list(pool)
            while cand and len(negs) < a.negs:
                j = rng.choices(range(len(cand)), weights=[wts[pool.index(c)] for c in cand])[0]
                negs.append(cand.pop(j))
            while len(negs) < a.negs:  # tiny pools: random train images
                j = rng.randrange(len(tr_ids))
                if j != n and j not in negs:
                    negs.append(j)
            for j in [n] + negs:
                texts.append(tr_texts[n]); imgs.append(tr_imgs[j])
        for n in batch_q:
            tn = text_negative(tr_texts[n], rng) if a.lam_txt > 0 else None
            n_txt.append(tn is not None)
            if tn is not None:
                texts.append(tn); imgs.append(tr_imgs[n])
        return make_inputs(proc, prompt_fn, texts, imgs, "cpu"), n_txt

    best = (sc0, None, 0)
    lm.train()
    step = 0
    for ep in range(1, a.epochs + 1):
        order = list(range(len(tr_ids)))
        rng.shuffle(order)
        batches = [order[s:s + a.qbs] for s in range(0, len(order), a.qbs)]
        t_ep, run_loss, run_acc, run_n = time.perf_counter(), 0.0, 0.0, 0
        with ThreadPoolExecutor(1) as ex:
            fut = ex.submit(build, batches[0])
            for bi, bq in enumerate(batches):
                inputs, has_txt = fut.result()
                if bi + 1 < len(batches):
                    fut = ex.submit(build, batches[bi + 1])
                s = scores(lm, w, inputs.to(dev))
                nb, g = len(bq), 1 + a.negs
                s_img = s[:nb * g].view(nb, g)
                loss = F.cross_entropy(s_img, torch.zeros(nb, dtype=torch.long, device=dev))
                if any(has_txt):
                    pos = s_img[[i for i, h in enumerate(has_txt) if h], 0]
                    neg = s[nb * g:]
                    loss = loss + a.lam_txt * F.softplus(neg - pos).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); step += 1
                run_loss += float(loss); run_acc += float((s_img.argmax(1) == 0).float().mean()); run_n += 1
                if step % 50 == 0 or bi == 0:
                    el = time.perf_counter() - t_ep
                    print(f"[train] ep {ep} step {bi + 1}/{len(batches)} loss {run_loss / run_n:.4f} img-acc {run_acc / run_n:.3f} "
                          f"lr {sched.get_last_lr()[0]:.2e} {el:.0f}s (ETA epoch {el / (bi + 1) * (len(batches) - bi - 1):.0f}s)", flush=True)
                    run_loss = run_acc = 0.0; run_n = 0
        (r1, nd, sc), _ = evaluate(lm, w, proc, prompt_fn, va_texts, va_top5, va_imgs, va_gt, a.k, a.eval_bs, dev)
        print(f"[eval] epoch {ep}: {a.eval_on} R@1 {r1:.4f} NDCG@5 {nd:.4f} score {sc:.4f} (before {sc0:.4f}, {sc - sc0:+.4f}) "
              f"epoch time {time.perf_counter() - t_ep:.0f}s", flush=True)
        log.append({"epoch": ep, "R@1": r1, "NDCG@5": nd, "score": sc})
        state = {n: (m.A.detach().cpu().clone(), m.B.detach().cpu().clone()) for n, m in lora_modules(lm).items()}
        torch.save(state, os.path.join(a.out, f"lora_ep{ep}.pt"))
        if sc > best[0] or a.final:
            best = (sc, state, ep)

    json.dump({"args": vars(a), "log": log, "best_epoch": best[2]}, open(os.path.join(a.out, "log.json"), "w"), indent=1)
    print(f"[done] best epoch {best[2]} valA score {best[0]:.4f} (before {sc0:.4f}); total {(time.perf_counter() - t_all) / 60:.1f} min")
    if a.no_merge or best[1] is None:
        print("[done] not merging" + (" (no epoch beat the base model)" if best[1] is None else ""))
        return
    for n, m in lora_modules(lm).items():
        m.A.data.copy_(best[1][n][0]); m.B.data.copy_(best[1][n][1])
    lm.gradient_checkpointing_disable()
    print(f"[merge] {merge_lora(lm)} layers merged")
    lm.config.use_cache = True
    md = os.path.join(a.out, "merged")
    lm.save_pretrained(md)
    proc.save_pretrained(md)
    print(f"[merge] saved {md}")


if __name__ == "__main__":
    main()

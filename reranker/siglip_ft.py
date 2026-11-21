#!/usr/bin/env python
"""Stage-1 fine-tuning: LoRA on both SigLIP2 towers with sibling-heavy batches (team train split only).

  python siglip_ft.py --siglip sub_v1/models/siglip2 --out siglip_ft --epochs 4 --res 256
  -> siglip_ft/merged/  (drop-in replacement for <submission>/models/siglip2, fp16, same tokenizer/preprocess)

Batches are built from whole scene-template groups (siblings together), so most in-batch negatives are hard.
Extra text rows: rule-based caption swaps (colour/pattern/material swap, relation flip, count change) that must not
match ANY image in the batch (including their own). Horizontal flips swap left/right in the caption.
Loss = SigLIP sigmoid loss with the model's own logit scale/bias (frozen).
Eval each epoch on valA queries x (valA+valB) images at the inference resolution: R@1/3/4/5/8 + stage-1 score.
"""
import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_lora import LoRALinear, text_negative  # noqa: E402

TARGETS = ("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2")


def add_lora(model, r, alpha, dropout):
    reps = [(mod, cn, ch) for _, mod in model.named_modules() for cn, ch in mod.named_children()
            if isinstance(ch, nn.Linear) and cn in TARGETS and not isinstance(mod, nn.MultiheadAttention)]
    for mod, cn, ch in reps:
        setattr(mod, cn, LoRALinear(ch, r, alpha, dropout))
    return len(reps)


def merge_lora(model):
    reps = [(mod, cn, ch) for _, mod in model.named_modules() for cn, ch in mod.named_children() if isinstance(ch, LoRALinear)]
    for mod, cn, ch in reps:
        setattr(mod, cn, ch.merged())
    return len(reps)


def feats(out):
    if torch.is_tensor(out):
        return out
    return out.pooler_output if getattr(out, "pooler_output", None) is not None else out[0]


def flip_lr(text):
    return re.sub(r"\b(left|right)\b", lambda m: {"left": "right", "right": "left"}[m.group(1).lower()], text, flags=re.I)


class ImgDS(Dataset):
    def __init__(self, paths, res, resample):
        self.paths, self.res, self.resample = paths, res, resample

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert("RGB").resize((self.res, self.res), resample=self.resample)
        return torch.from_numpy(np.asarray(im, dtype=np.uint8).copy()), i


def load_images(paths, res, resample, workers):
    out = np.empty((len(paths), res, res, 3), dtype=np.uint8)
    for x, idx in DataLoader(ImgDS(paths, res, resample), batch_size=64, num_workers=workers):
        out[idx.numpy()] = x.numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--siglip", default="sub_v1/models/siglip2")
    ap.add_argument("--train_dir", default="haikuthon-train")
    ap.add_argument("--split", default="splits/split_group_seed42.json")
    ap.add_argument("--train_on", default="train")
    ap.add_argument("--eval_on", default="valA")
    ap.add_argument("--out", default="siglip_ft")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--txt_neg", type=float, default=0.5, help="fraction of batch rows that also get a text negative")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--family_json", default="", help="runs/family_of.json: true families (id -> key). Used for batches; eval only on queries whose family has no training member")
    ap.add_argument("--select", default="recall", choices=["recall", "score"], help="epoch selection: mean(R@3,R@4,R@5) or stage-1 score")
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    dev = "cuda:0"
    os.makedirs(a.out, exist_ok=True)

    root = Path(a.train_dir).resolve()
    recs = {r["id"]: r for r in (json.loads(l) for l in open(root / "prompts.jsonl", encoding="utf-8") if l.strip())}
    sp = json.loads(Path(a.split).read_text())
    tr_ids = [i for s in a.train_on.split(",") for i in sp[s]]
    other = {"valA": "valB", "valB": "valA"}[a.eval_on]
    va_ids, pool_ids = sp[a.eval_on], sp[a.eval_on] + sp[other]
    assert not set(tr_ids) & set(va_ids)

    fam_of = json.load(open(a.family_json)) if a.family_json else None
    # sibling groups: true families if given, else proxy (word-set Jaccard >= 0.7) within the training ids
    from scipy.sparse import csr_matrix
    from sklearn.feature_extraction.text import CountVectorizer
    X = csr_matrix(CountVectorizer(binary=True, token_pattern=r"[a-zA-Z-]+").fit_transform(
        [recs[i]["prompts"].lower() for i in tr_ids]), dtype=np.float32)
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
    groups = {}
    for n in range(len(tr_ids)):
        groups.setdefault(fam_of[tr_ids[n]] if fam_of else find(n), []).append(n)
    groups = list(groups.values())
    print(f"[data] train {len(tr_ids)} in {len(groups)} proxy groups (multi-member {sum(len(g) > 1 for g in groups)})")

    m = AutoModel.from_pretrained(a.siglip, dtype=torch.float32).to(dev)
    tok = AutoTokenizer.from_pretrained(a.siglip)
    pp = {"image_mean": [0.5] * 3, "image_std": [0.5] * 3, "resample": 3}
    if os.path.exists(os.path.join(a.siglip, "haiku_preprocess.json")):
        pp.update(json.load(open(os.path.join(a.siglip, "haiku_preprocess.json"))))
    mean = torch.tensor(pp["image_mean"], device=dev).view(1, 3, 1, 1)
    std = torch.tensor(pp["image_std"], device=dev).view(1, 3, 1, 1)
    native = int(m.config.vision_config.image_size)
    kw = {"interpolate_pos_encoding": True} if a.res != native else {}
    resample = Image.Resampling(int(pp["resample"]))
    path = lambda i: str(root / "images" / f"{i}.png")
    t0 = time.perf_counter()
    tr_imgs = load_images([path(i) for i in tr_ids], a.res, resample, a.workers)
    pool_imgs = load_images([path(i) for i in pool_ids], a.res, resample, a.workers)
    print(f"[data] decoded {len(tr_ids) + len(pool_ids)} images in {time.perf_counter() - t0:.0f}s")
    tr_texts = [recs[i]["prompts"] for i in tr_ids]
    va_texts = [recs[i]["prompts"] for i in va_ids]
    tr_set = set(tr_ids)
    eval_mask = np.ones(len(va_ids), bool)
    if fam_of:
        fam_members = {}
        for i, f in fam_of.items():
            fam_members.setdefault(f, []).append(i)
        eval_mask = np.array([not any(j in tr_set for j in fam_members[fam_of[i]]) for i in va_ids])
        print(f"[data] eval on {eval_mask.sum()} {a.eval_on} queries whose family has no training member (families complete in the pool)")

    def enc_text(texts):
        e = tok([t.lower() for t in texts], padding="max_length", truncation=True, max_length=64, return_tensors="pt")
        return F.normalize(feats(m.get_text_features(input_ids=e["input_ids"].to(dev))).float(), dim=-1)

    def enc_img(arr):
        x = torch.from_numpy(arr).to(dev).permute(0, 3, 1, 2).float().div_(255.0)
        x = (x - mean) / std
        return F.normalize(feats(m.get_image_features(pixel_values=x, **kw)).float(), dim=-1)

    @torch.no_grad()
    def evaluate():
        m.eval()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            T = torch.cat([enc_text(va_texts[s:s + 256]) for s in range(0, len(va_texts), 256)])
            I = torch.cat([enc_img(pool_imgs[s:s + 128]) for s in range(0, len(pool_imgs), 128)])
        S = T.float() @ I.float().T
        gt = torch.arange(len(va_ids), device=dev)
        rank = ((S > S[gt, gt][:, None]).sum(1) + 1).cpu().numpy()[eval_mask]
        score = np.mean([0.5 * ((k == 1) + (1 / math.log2(k + 1) if k <= 5 else 0)) for k in rank])
        m.train()
        return {f"R@{k}": float((rank <= k).mean()) for k in (1, 3, 4, 5, 8)} | {"score": float(score)}

    base = evaluate()
    print(f"[eval] before FT: {base}", flush=True)
    log = [{"epoch": 0, **base}]

    for p in m.parameters():
        p.requires_grad_(False)
    n_l = add_lora(m, a.rank, a.alpha, 0.0)
    params = [p for mod in m.modules() if isinstance(mod, LoRALinear) for p in (mod.A, mod.B)]
    print(f"[lora] {n_l} linear layers, {sum(p.numel() for p in params) / 1e6:.1f}M params")
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m.train()
    scale, bias = m.logit_scale.exp().detach().float(), m.logit_bias.detach().float()
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)

    def batches():
        gs = groups[:]
        rng.shuffle(gs)
        cur = []
        for g in gs:
            g = g[:]
            rng.shuffle(g)
            for n in g[:8]:
                cur.append(n)
                if len(cur) == a.batch:
                    yield cur
                    cur = []
        if len(cur) >= 8:
            yield cur

    steps_ep = sum(1 for _ in batches())
    total = steps_ep * a.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, int(0.1 * total))) *
                                              0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    crit = (lambda r: (r["R@3"] + r["R@4"] + r["R@5"]) / 3) if a.select == "recall" else (lambda r: r["score"])
    best = (crit(base), None, 0)
    for ep in range(1, a.epochs + 1):
        t_ep, run = time.perf_counter(), []
        for b in batches():
            flip = [rng.random() < 0.5 for _ in b]
            imgs = np.stack([tr_imgs[n][:, ::-1] if f else tr_imgs[n] for n, f in zip(b, flip)])
            texts = [flip_lr(tr_texts[n]) if f else tr_texts[n] for n, f in zip(b, flip)]
            negs = [text_negative(t, rng) for t in texts if rng.random() < a.txt_neg]
            negs = [t for t in negs if t]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                T = enc_text(texts + negs)
                I = enc_img(np.ascontiguousarray(imgs))
            logits = scale * (T.float() @ I.float().T) + bias
            labels = -torch.ones_like(logits)
            labels[torch.arange(len(b)), torch.arange(len(b))] = 1.0
            loss = -F.logsigmoid(labels * logits).sum() / len(b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step()
            run.append(float(loss))
        res = evaluate()
        print(f"[epoch {ep}] loss {np.mean(run):.4f} | {res} | {time.perf_counter() - t_ep:.0f}s", flush=True)
        log.append({"epoch": ep, **res})
        if crit(res) > best[0]:
            best = (crit(res), {n: (mod.A.detach().cpu().clone(), mod.B.detach().cpu().clone())
                                   for n, mod in m.named_modules() if isinstance(mod, LoRALinear)}, ep)
    json.dump({"args": vars(a), "log": log, "best_epoch": best[2]}, open(os.path.join(a.out, "log.json"), "w"), indent=1)
    if best[1] is None:
        print("[done] no epoch beat the base model; not saving")
        return
    for n, mod in m.named_modules():
        if isinstance(mod, LoRALinear):
            mod.A.data.copy_(best[1][n][0]); mod.B.data.copy_(best[1][n][1])
    m.gradient_checkpointing_disable()
    merge_lora(m)
    md = os.path.join(a.out, "merged")
    m.to(torch.float16).save_pretrained(md)
    tok.save_pretrained(md)
    for f in ("haiku_preprocess.json", "preprocessor_config.json"):
        if os.path.exists(os.path.join(a.siglip, f)):
            import shutil
            shutil.copy(os.path.join(a.siglip, f), os.path.join(md, f))
    print(f"[done] best epoch {best[2]} {a.select} {best[0]:.4f} (base {crit(base):.4f}); saved {md}")


if __name__ == "__main__":
    main()

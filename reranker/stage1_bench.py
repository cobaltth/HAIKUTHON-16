#!/usr/bin/env python
"""Stage-1 speed/recall trade-off: SigLIP2 checkpoints and input resolutions on the team valA pool (800 x 1600).

  python stage1_bench.py --models sub_v1/models/siglip2 google/siglip2-so400m-patch16-256 --res 384 320 288 256

For each (model, res): R@1, recall@3/5/8, stage-1 score, GPU ms/image (fp16, batch 64) and the 16000-image estimate.
A resolution different from the checkpoint's native size uses interpolate_pos_encoding.
Also saves runs/s1_<tag>.npz (top-8 indices + scores) for adaptive-k re-tuning.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


class DS(Dataset):
    def __init__(self, paths, res, resample):
        self.paths, self.res, self.resample = paths, res, resample

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert("RGB").resize((self.res, self.res), resample=self.resample)
        return torch.from_numpy(np.asarray(im, dtype=np.uint8).copy()), i


def feats(out):
    if torch.is_tensor(out):
        return out
    return out.pooler_output if getattr(out, "pooler_output", None) is not None else out[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["sub_v1/models/siglip2"])
    ap.add_argument("--res", nargs="+", type=int, default=[384, 320, 288, 256])
    ap.add_argument("--native_only_for_hub", action="store_true", help="hub checkpoints only at native res")
    ap.add_argument("--train_dir", default="haikuthon-train")
    ap.add_argument("--split", default="splits/split_group_seed42.json")
    ap.add_argument("--which", default="valA")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    root = Path(a.train_dir).resolve()
    recs = {r["id"]: r for r in (json.loads(l) for l in open(root / "prompts.jsonl", encoding="utf-8") if l.strip())}
    sp = json.loads(Path(a.split).read_text())
    other = {"valA": "valB", "valB": "valA"}[a.which]
    q_ids, pool = sp[a.which], sp[a.which] + sp[other]
    texts = [recs[i]["prompts"].lower() for i in q_ids]
    paths = [str(root / "images" / f"{i}.png") for i in pool]
    gt = torch.arange(len(q_ids))
    os.makedirs("runs", exist_ok=True)
    rows = []
    for mp in a.models:
        m = AutoModel.from_pretrained(mp, dtype=torch.float16).cuda().eval()
        tok = AutoTokenizer.from_pretrained(mp)
        native = int(m.config.vision_config.image_size)
        pp = {"image_mean": [0.5] * 3, "image_std": [0.5] * 3, "resample": 3}
        for f in ("haiku_preprocess.json", "preprocessor_config.json"):
            p = os.path.join(mp, f) if os.path.isdir(mp) else None
            if p and os.path.exists(p):
                d = json.load(open(p))
                pp.update({k: d[k] for k in ("image_mean", "image_std", "resample") if k in d})
                break
        mean = torch.tensor(pp["image_mean"], device="cuda").view(1, 3, 1, 1)
        std = torch.tensor(pp["image_std"], device="cuda").view(1, 3, 1, 1)
        with torch.inference_mode():
            T = []
            for s in range(0, len(texts), 256):
                enc = tok(texts[s:s + 256], padding="max_length", truncation=True, max_length=64, return_tensors="pt")
                T.append(F.normalize(feats(m.get_text_features(input_ids=enc["input_ids"].cuda())).float(), dim=-1))
            T = torch.cat(T)
        res_list = [native] if (a.native_only_for_hub and not os.path.isdir(mp)) else a.res
        for res in res_list:
            interp = res != native
            dl = DataLoader(DS(paths, res, Image.Resampling(int(pp["resample"]))), batch_size=64, num_workers=a.workers)
            I = torch.zeros(len(paths), T.shape[1], device="cuda")
            with torch.inference_mode():
                for x, idx in dl:
                    x = ((x.cuda().permute(0, 3, 1, 2).float() / 255.0 - mean) / std).half()
                    kw = {"interpolate_pos_encoding": True} if interp else {}
                    I[idx.cuda()] = F.normalize(feats(m.get_image_features(pixel_values=x, **kw)).float(), dim=-1)
                S = T @ I.T
                top_s, top_i = S.topk(8, dim=1)
                rank = (S > S[gt, gt][:, None]).sum(1) + 1
                xb = torch.randn(64, 3, res, res, device="cuda", dtype=torch.float16)
                kw = {"interpolate_pos_encoding": True} if interp else {}
                m.get_image_features(pixel_values=xb, **kw); torch.cuda.synchronize()
                t = time.perf_counter()
                for _ in range(5):
                    m.get_image_features(pixel_values=xb, **kw)
                torch.cuda.synchronize()
                ms = (time.perf_counter() - t) / 320 * 1000
            r = rank.cpu().numpy()
            score = np.mean([0.5 * ((k == 1) + (1 / math.log2(k + 1) if k <= 5 else 0)) for k in r])
            tag = f"{os.path.basename(mp.rstrip('/'))}_{res}"
            np.savez(f"runs/s1_{tag}.npz", top_i=top_i.cpu().numpy(), top_s=top_s.cpu().numpy())
            row = dict(model=tag, R1=float((r == 1).mean()), R3=float((r <= 3).mean()), R5=float((r <= 5).mean()),
                       R8=float((r <= 8).mean()), score=float(score), ms=ms, min16k=16000 * ms / 60000)
            rows.append(row)
            print(f"{tag:40s} R@1 {row['R1']:.3f} R@3 {row['R3']:.3f} R@5 {row['R5']:.3f} R@8 {row['R8']:.3f} "
                  f"score {row['score']:.4f} | {ms:.1f} ms/img -> 16000 imgs {row['min16k']:.1f} min", flush=True)
        del m
        torch.cuda.empty_cache()
    json.dump(rows, open("runs/stage1_bench.json", "w"), indent=1)


if __name__ == "__main__":
    main()

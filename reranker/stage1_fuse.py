"""Stage-1 fusion check: SigLIP2-so400m (ours, 256 px) + google/siglip-base-patch16-224 (team B) [+ SigLIP2 at 384].

Zero-shot encoders only, so every one of the 4000 train captions can be a query against all 4000 images
(families complete, nothing was trained on them). Reports recall@k per encoder, per-query z-score fusion
(each query normalised by its own scores only = query independent), and union-of-top-k recall.

    python stage1_fuse.py --siglip2 sub_v4/models/siglip2 --base google/siglip-base-patch16-224
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def feats(out):
    if torch.is_tensor(out):
        return out
    return out.pooler_output if getattr(out, "pooler_output", None) is not None else out[0]


class DS(Dataset):
    def __init__(self, paths, sizes):
        self.paths, self.sizes = paths, sizes

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert("RGB")
        return [torch.from_numpy(np.asarray(im.resize((s, s), Image.Resampling.BICUBIC), dtype=np.uint8).copy()) for s in self.sizes] + [i]


def recall(S, ks=(1, 2, 3, 4, 5, 6, 8, 10, 16)):
    gt = S.diagonal()[:, None]
    rank = (S > gt).sum(1) + 1
    return {k: round(float((rank <= k).float().mean()), 4) for k in ks}, rank


def zrow(S):
    return (S - S.mean(1, keepdim=True)) / S.std(1, keepdim=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--siglip2", default="sub_v4/models/siglip2")
    ap.add_argument("--base", default="google/siglip-base-patch16-224")
    ap.add_argument("--train_dir", default="haikuthon-train")
    ap.add_argument("--with384", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="runs/stage1_fuse.npz")
    a = ap.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "0"
    from transformers import AutoModel, AutoTokenizer
    dev = "cuda:0"
    root = Path(a.train_dir)
    recs = [json.loads(l) for l in open(root / "prompts.jsonl", encoding="utf-8") if l.strip()]
    ids, texts = [r["id"] for r in recs], [r["prompts"] for r in recs]
    sizes = [256, 224] + ([384] if a.with384 else [])
    bufs = {s: np.empty((len(ids), s, s, 3), np.uint8) for s in sizes}
    t0 = time.perf_counter()
    for batch in DataLoader(DS([str(root / "images" / f"{i}.png") for i in ids], sizes), batch_size=64, num_workers=a.workers):
        idx = batch[-1].numpy()
        for s, x in zip(sizes, batch[:-1]):
            bufs[s][idx] = x.numpy()
    print(f"[data] {len(ids)} captions x {len(ids)} images, decoded in {time.perf_counter() - t0:.0f}s", flush=True)

    @torch.no_grad()
    def encode(m, tok, res, lower, kw, max_len=64):
        m.eval()
        t = time.perf_counter()
        I = []
        for s in range(0, len(ids), 128):
            x = torch.from_numpy(bufs[res][s:s + 128]).to(dev).permute(0, 3, 1, 2).half().div_(255).sub_(0.5).div_(0.5)
            I.append(F.normalize(feats(m.get_image_features(pixel_values=x, **kw)).float(), dim=-1))
        torch.cuda.synchronize(); ti = time.perf_counter() - t
        T = []
        for s in range(0, len(texts), 256):
            e = tok([q.lower() if lower else q for q in texts[s:s + 256]], padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
            T.append(F.normalize(feats(m.get_text_features(input_ids=e["input_ids"].to(dev))).float(), dim=-1))
        I, T = torch.cat(I), torch.cat(T)
        return T @ I.T, ti

    S = {}
    m = AutoModel.from_pretrained(a.siglip2, dtype=torch.float16).to(dev)
    tok = AutoTokenizer.from_pretrained(a.siglip2)
    native = int(m.config.vision_config.image_size)
    S["s2_256"], ti = encode(m, tok, 256, True, {"interpolate_pos_encoding": True} if native != 256 else {})
    print(f"[enc] siglip2 @256: image encode {ti:.1f}s / {len(ids)} imgs", flush=True)
    if a.with384:
        S["s2_384"], ti = encode(m, tok, 384, True, {})
        print(f"[enc] siglip2 @384: image encode {ti:.1f}s", flush=True)
    del m; torch.cuda.empty_cache()
    m = AutoModel.from_pretrained(a.base, dtype=torch.float16).to(dev)
    tok = AutoTokenizer.from_pretrained(a.base)
    S["base"], ti = encode(m, tok, 224, False, {})
    S["base_lc"], _ = encode(m, tok, 224, True, {})
    print(f"[enc] siglip-base @224: image encode {ti:.1f}s", flush=True)
    del m; torch.cuda.empty_cache()

    print("\nrecall@k (4000 captions vs 4000 images)")
    ranks = {}
    for n, s in S.items():
        r, ranks[n] = recall(s)
        print(f"  {n:10s} {r}")
    b = "base" if recall(S["base"])[0][5] >= recall(S["base_lc"])[0][5] else "base_lc"
    names = ["s2_256", b] + (["s2_384"] if a.with384 else [])
    Z = {n: zrow(S[n]) for n in names}
    print(f"\nfusion (per-query z-score, w * s2_256 + (1-w) * {b})")
    for w in (0.3, 0.4, 0.5, 0.6, 0.7):
        print(f"  w={w}: {recall(w * Z['s2_256'] + (1 - w) * Z[b])[0]}")
    if a.with384:
        print("  s2_256+s2_384 (0.5/0.5):", recall(0.5 * Z["s2_256"] + 0.5 * Z["s2_384"])[0])
        print("  all three (1/3 each):   ", recall(sum(Z.values()) / 3)[0])
    print("\nunion of top-k lists (ours, base): mean unique size, recall")
    ta = S["s2_256"].topk(10, 1).indices.cpu().numpy(); tb = S[b].topk(10, 1).indices.cpu().numpy()
    q = np.arange(len(ids))
    for k in (2, 3, 4, 5):
        u = [set(ta[i, :k]) | set(tb[i, :k]) for i in q]
        print(f"  top-{k} each: size {np.mean([len(x) for x in u]):.2f}, recall {np.mean([i in x for i, x in zip(q, u)]):.4f} "
              f"(ours top-{k} {np.mean([i in ta[i, :k] for i in q]):.4f}, base top-{k} {np.mean([i in tb[i, :k] for i in q]):.4f})")
    print(f"  top-5 overlap (ours vs base): {np.mean([len(set(ta[i, :5]) & set(tb[i, :5])) for i in q]):.2f} of 5   [test: 1.9]")
    np.savez_compressed(a.out, **{n: s.half().cpu().numpy() for n, s in S.items()})
    print(f"[saved] {a.out}")


if __name__ == "__main__":
    main()

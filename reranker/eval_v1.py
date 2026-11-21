#!/usr/bin/env python
"""Run the submission's own load_model/predict on the team split and score it like the leaderboard.

  python eval_v1.py --sub sub_v1 --which valA            # pool = valA queries x (valA+valB) images = 1600
  python eval_v1.py --sub sub_v1 --which valA --config '{"k_max":8,"gap":0.02}'

Score = (R@1 + NDCG@5)/2 with one relevant image per query. Also runs stage 1 alone (time_limit=0 -> the time
guard falls back to SigLIP order for every query) and extrapolates server time for 8000 queries x 16000 images.
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np


def score(preds, gt):
    r1 = nd = 0.0
    for q, g in gt.items():
        p = preds[q]
        r1 += float(p[0] == g)
        if g in p[:5]:
            nd += 1.0 / math.log2(p.index(g) + 2)
    n = len(gt)
    return r1 / n, nd / n, 0.5 * (r1 + nd) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="sub_v1")
    ap.add_argument("--train_dir", default="haikuthon-train")
    ap.add_argument("--split", default="splits/split_group_seed42.json")
    ap.add_argument("--which", default="valA", choices=["valA", "valB"])
    ap.add_argument("--config", default="{}")
    ap.add_argument("--no_stage1_only", action="store_true")
    ap.add_argument("--pool_all", action="store_true", help="candidates = all images of train_dir (4000) instead of valA+valB")
    ap.add_argument("--pool_train", action="store_true", help="candidates = split images + train images (fair for models trained on train+other)")
    ap.add_argument("--out", default="runs/v1_eval.jsonl")
    ap.add_argument("--dump", default="", help="save top-k indices/scores/logits to this .npz (query q's answer is candidate q)")
    a = ap.parse_args()

    root = Path(a.train_dir).resolve()
    recs = [json.loads(l) for l in open(root / "prompts.jsonl", encoding="utf-8") if l.strip()]
    by_id = {r["id"]: r for r in recs}
    sp = json.loads(Path(a.split).read_text())
    other = {"valA": "valB", "valB": "valA"}[a.which]
    queries = [{"prompt_id": f"p_{i}", "prompt": by_id[i]["prompts"]} for i in sp[a.which]]
    pool = sp[a.which] + sp[other]
    if a.pool_train:
        pool = sp[a.which] + sp["train"]
    if a.pool_all:
        pool = pool + [i for i in sorted(by_id) if i not in set(pool)]
    cands = [{"image_id": f"i_{i}", "image_path": str(root / "images" / f"{i}.png")} for i in pool]
    gt = {f"p_{i}": f"i_{i}" for i in sp[a.which]}

    sub = os.path.abspath(a.sub)
    sys.path.insert(0, sub)
    spec = importlib.util.spec_from_file_location("sub_model", os.path.join(sub, "model.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    t = time.perf_counter()
    m = mod.load_model(sub, "cuda:0")
    t_load = time.perf_counter() - t
    m["cfg"].update(json.loads(a.config))
    base_cfg = dict(m["cfg"])

    rows = []
    if not a.no_stage1_only:
        m["cfg"]["time_limit"] = 0.0
        p = mod.predict(m, queries, cands, top_k=5)
        r1, nd, sc = score(p, gt)
        rows.append({"run": "stage1_only", "pool": len(cands), "R@1": r1, "NDCG@5": nd, "score": sc, **m["_timing"]})
        m["cfg"] = dict(base_cfg)

    import torch
    torch.cuda.synchronize()
    t = time.perf_counter()
    p = mod.predict(m, queries, cands, top_k=5)
    total = time.perf_counter() - t
    r1, nd, sc = score(p, gt)
    tm = m["_timing"]
    if a.dump:
        L = m["_last"]
        np.savez(a.dump, top_i=L["top_i"], top_s=L["top_s"], logits=L["logits"], scored=L["scored"], ks=L["ks"])
        print(f"dumped {a.dump}")
    rows.append({"run": "v1", "pool": len(cands), "R@1": r1, "NDCG@5": nd, "score": sc, "total_s": total, **tm,
                 "cfg": {k: v for k, v in base_cfg.items() if k != "instruct"}})

    nq, ni = len(queries), len(cands)
    s1_server = tm["stage1"] * 16000 / ni
    per_pair = tm["rerank"] / max(tm["pairs_done"], 1)
    pairs_server = tm["pairs_planned"] / nq * 8000
    print("\n==== results ({} queries x {} images) ====".format(nq, ni))
    for r in rows:
        print(f"{r['run']:12s} R@1 {r['R@1']:.4f}  NDCG@5 {r['NDCG@5']:.4f}  score {r['score']:.4f}")
    print(f"pairs planned {tm['pairs_planned']} done {tm['pairs_done']} | reranked queries {tm['reranked_queries']} (full k {tm['full_k_queries']})")
    print(f"load_model {t_load:.0f}s | stage1 {tm['stage1']:.1f}s | rerank {tm['rerank']:.1f}s for {tm['pairs_done']} pairs "
          f"({per_pair * 1000:.1f} ms/pair) | peak GPU {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    print(f"server estimate (8000 x 16000): stage1 ~{s1_server / 60:.1f} min + rerank {pairs_server:.0f} pairs "
          f"~{pairs_server * per_pair / 60:.1f} min = ~{(s1_server + pairs_server * per_pair) / 60:.1f} min "
          f"(guard at {base_cfg['time_limit'] / 60:.1f} min; Colab CPU/IO differ from the server)")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()

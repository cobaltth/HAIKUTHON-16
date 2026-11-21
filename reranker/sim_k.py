#!/usr/bin/env python
"""Tune the per-query k rule offline from an eval_v1 --dump made with fixed k (e.g. {"k_min":8,"k_max":8}).

  python sim_k.py runs/v2_k8.npz [runs/v2_k8_pool4000.npz ...] --ms 26.4 --budget_min 7.0

Rule: k starts at k_min and grows while top_s[0] - top_s[k] <= gap, up to k_max (k=1 means no reranking).
Prints score and pairs/query per rule, the 8000-query rerank minutes at --ms per pair, and the best rule
that fits --budget_min. Halves (even/odd queries) show how stable a choice is.
"""
import argparse
import math

import numpy as np


def run(d, kmin, kmax, gap):
    top_i, top_s, L = d["top_i"], d["top_s"], d["logits"]
    n = len(top_i)
    gt = np.arange(n)
    sc = np.zeros(n)
    pairs = 0
    for q in range(n):
        k = kmin
        while k < kmax and top_s[q, 0] - top_s[q, k] <= gap:
            k += 1
        lst = list(top_i[q])
        if k >= 2:
            o = np.argsort(-L[q, :k], kind="stable")
            lst = [lst[i] for i in o] + lst[k:]
            pairs += k
        g = gt[q]
        sc[q] = 0.5 * ((lst[0] == g) + (1 / math.log2(lst.index(g) + 2) if g in lst[:5] else 0))
    return sc, pairs / n


def run_budget(d, kmin, kmax, gap, budget):
    """Same rule, executed like model.py: ranks 0-1 of all reranked queries first, then rank 2, 3, ...; stop after
    budget * n pairs (time guard). Queries keep their stage-1 order beyond the leading scored ranks."""
    top_i, top_s, L = d["top_i"], d["top_s"], d["logits"]
    n = len(top_i)
    ks = np.full(n, kmin)
    for q in range(n):
        while ks[q] < kmax and top_s[q, 0] - top_s[q, ks[q]] <= gap:
            ks[q] += 1
    order = [(q, r) for q in range(n) if ks[q] >= 2 for r in range(2)]
    order += [(q, r) for r in range(2, kmax) for q in range(n) if ks[q] > r]
    done = np.zeros((n, kmax), bool)
    for q, r in order[: int(budget * n)]:
        done[q, r] = True
    sc = np.zeros(n)
    for q in range(n):
        kd = 0
        while kd < ks[q] and done[q, kd]:
            kd += 1
        lst = list(top_i[q])
        if kd >= 2:
            o = np.argsort(-L[q, :kd], kind="stable")
            lst = [lst[i] for i in o] + lst[kd:]
        sc[q] = 0.5 * ((lst[0] == q) + (1 / math.log2(lst.index(q) + 2) if q in lst[:5] else 0))
    return sc, min(len(order), int(budget * n)) / n, len(order) / n


def rank_table(d, rules):
    top_i, top_s, L = d["top_i"], d["top_s"], d["logits"]
    n = len(top_i)
    print("    rule                           rank1  rank2  rank3  rank4  rank5  6-8    >8")
    for name, (kmin, kmax, gap) in rules.items():
        r = []
        for q in range(n):
            k = kmin
            while k < kmax and top_s[q, 0] - top_s[q, k] <= gap:
                k += 1
            lst = list(top_i[q])
            if k >= 2:
                o = np.argsort(-L[q, :k], kind="stable")
                lst = [lst[i] for i in o] + lst[k:]
            r.append(lst.index(q) + 1 if q in lst else 99)
        r = np.array(r)
        dist = [np.mean(r == i) for i in range(1, 6)] + [np.mean((r > 5) & (r <= 8)), np.mean(r > 8)]
        print(f"    {name:30s} " + "  ".join(f"{x:.3f}" for x in dist))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--ms", type=float, default=26.4)
    ap.add_argument("--budget_min", type=float, default=7.0)
    ap.add_argument("--budgets", default="2.0,2.5,3.0,3.5,4.0", help="pairs/query actually executed (guard) to evaluate")
    a = ap.parse_args()
    for f in a.dumps:
        d = dict(np.load(f))
        K = int(np.isfinite(d["logits"]).all(0).sum())
        n = len(d["top_i"])
        half = np.arange(n) % 2 == 0
        gt_in = [(d["top_i"][:, :k] == np.arange(n)[:, None]).any(1).mean() for k in range(1, K + 1)]
        print(f"\n== {f}: {n} queries, logits for top-{K}; answer in top-k: " + " ".join(f"{k + 1}:{v:.3f}" for k, v in enumerate(gt_in)))
        rows = []
        for k in range(1, K + 1):
            s, p = run(d, k, k, 0.0)
            rows.append((f"fixed k={k}", s, p))
        for kmin in (1, 2):
            for kmax in range(3, K + 1):
                for gap in (0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05):
                    s, p = run(d, kmin, kmax, gap)
                    rows.append((f"k{kmin}->{kmax} gap {gap}", s, p))
        for name, s, p in rows[:K]:
            print(f"  {name:22s} score {s.mean():.4f} (even {s[half].mean():.4f} odd {s[~half].mean():.4f}) pairs/q {p:.2f} -> {8000 * p * a.ms / 60000:.1f} min")
        fit = [r for r in rows if 8000 * r[2] * a.ms / 60000 <= a.budget_min]
        print(f"  best rules within {a.budget_min} min of reranking at {a.ms} ms/pair:")
        for name, s, p in sorted(fit, key=lambda r: -r[1].mean())[:8]:
            print(f"    {name:22s} score {s.mean():.4f} (even {s[half].mean():.4f} odd {s[~half].mean():.4f}) pairs/q {p:.2f} -> {8000 * p * a.ms / 60000:.1f} min")
        rank_table(d, {"SigLIP only": (1, 1, 0), "v2 submitted k1->5 gap .025": (1, 5, .025), "fixed k=3": (3, 3, 0),
                       "fixed k=5": (5, 5, 0), "fixed k=8": (8, 8, 0)})
        print("  under a time guard (pass-ordered, budget = executed pairs/q): best planned rules")
        cands = [(1, kmax, g) for kmax in (3, 4, 5, 6, 8) for g in (0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.08)]
        cands += [(2, kmax, g) for kmax in (4, 6, 8) for g in (0.02, 0.03, 0.05)] + [(k, k, 0) for k in (3, 4, 5, 8)]
        for B in [float(x) for x in a.budgets.split(",")]:
            res = sorted(((run_budget(d, *c, B), c) for c in cands), key=lambda t: -t[0][0].mean())
            (s1, used, plan), c = res[0]
            cur = run_budget(d, 1, 5, 0.025, B)
            print(f"    budget {B:.1f}: best k{c[0]}->{c[1]} gap {c[2]} score {s1.mean():.4f} (even {s1[half].mean():.4f} odd {s1[~half].mean():.4f}, plans {plan:.2f}/q)"
                  f" | v2 rule {cur[0].mean():.4f} | 2nd k{res[1][1][0]}->{res[1][1][1]} gap {res[1][1][2]} {res[1][0][0].mean():.4f}")
        print("  frontier (best score per pairs/q bucket):")
        for lo in np.arange(1.0, K + 0.5, 0.5):
            b = [r for r in rows if lo <= r[2] < lo + 0.5]
            if b:
                name, s, p = max(b, key=lambda r: r[1].mean())
                print(f"    pairs/q {lo:.1f}-{lo + 0.5:.1f}: {name:22s} score {s.mean():.4f} pairs/q {p:.2f}")


if __name__ == "__main__":
    main()

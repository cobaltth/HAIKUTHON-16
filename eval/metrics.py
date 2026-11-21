"""대회 점수식: s(q) = (Recall@1 + NDCG@5) / 2. 순위 r은 1부터, 상위 5 밖이면 0점."""
import math

RANK_SCORE = {r: (float(r == 1) + 1 / math.log2(r + 1)) / 2 for r in range(1, 6)}
BUCKETS = [(1, 1), (2, 5), (6, 10), (11, 20), (21, 50), (51, 10**9)]


def rank_of(target, ranking):
    """ranking(이미지 id 리스트)에서 정답 순위(1부터). 없으면 None."""
    try:
        return ranking.index(target) + 1
    except ValueError:
        return None


def evaluate(predictions, gt, ks=(1, 5, 10, 20, 50)):
    """predictions: {query_id: [image_id, ...] (길이 ≥5, 길수록 순위분포가 자세함)}
    gt: {query_id: 정답 image_id}"""
    ranks = [rank_of(gt[q], predictions[q]) for q in gt]
    n = len(ranks)
    depth = min(len(v) for v in predictions.values())
    r1 = sum(r == 1 for r in ranks) / n
    ndcg = sum(1 / math.log2(r + 1) for r in ranks if r and r <= 5) / n
    out = {"n": n, "R@1": r1, "NDCG@5": ndcg, "score": (r1 + ndcg) / 2}
    for k in ks:
        if k <= depth:
            out[f"Recall@{k}"] = sum(r is not None and r <= k for r in ranks) / n
    dist = {}
    for lo, hi in BUCKETS:
        if lo > depth:
            break
        label = f"{lo}" if lo == hi else (f"{lo}-{min(hi, depth)}" if hi < 10**9 else f"{lo}+")
        dist[label] = sum(r is not None and lo <= r <= hi for r in ranks) / n
    dist[f">{depth}(miss)"] = sum(r is None for r in ranks) / n
    out["rank_dist"] = dist
    return out, ranks


def format_report(res):
    lines = [f"n={res['n']}  score={res['score']:.4f}  R@1={res['R@1']:.4f}  NDCG@5={res['NDCG@5']:.4f}"]
    lines.append("  " + "  ".join(f"{k}={v:.3f}" for k, v in res.items() if k.startswith("Recall@")))
    lines.append("  정답 순위 분포: " + "  ".join(f"[{k}] {v:.1%}" for k, v in res["rank_dist"].items()))
    return "\n".join(lines)

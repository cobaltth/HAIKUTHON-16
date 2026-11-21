"""채택 판정: 같은 split/pool로 --save 한 결과 두 개를 query 단위로 짝지어 비교 (paired bootstrap).

예) python eval/compare.py runs/base.json runs/groupbatch.json
출력: 점수 차(B−A), 95% 신뢰구간, B가 더 나을 확률, query별 승/패/무.
판정 규칙(권장): 95% 구간 하한 > 0 이면 채택, 구간이 0을 걸치면 '차이 불명확' → 비용(시간·용량) 적은 쪽.
"""
import json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import RANK_SCORE


def per_query(path):
    d = json.load(open(path))
    return {q: (RANK_SCORE.get(r, 0.0) if r else 0.0) for q, r in d["ranks"].items()}


def main(a_path, b_path, n_boot=10000, seed=0):
    A, B = per_query(a_path), per_query(b_path)
    if set(A) != set(B):
        sys.exit(f"query 집합이 다릅니다 (A {len(A)}, B {len(B)}). 같은 split/pool로 다시 저장하세요.")
    qs = sorted(A); diff = [B[q] - A[q] for q in qs]; n = len(qs)
    mean = sum(diff) / n
    rng = random.Random(seed); boots = []
    for _ in range(n_boot):
        boots.append(sum(diff[rng.randrange(n)] for _ in range(n)) / n)
    boots.sort()
    lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]
    p_better = sum(b > 0 for b in boots) / n_boot
    win = sum(d > 0 for d in diff); lose = sum(d < 0 for d in diff)
    print(f"A={sum(A.values())/n:.4f}  B={sum(B.values())/n:.4f}  (n={n})")
    print(f"B−A = {mean:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  P(B>A)={p_better:.3f}")
    print(f"query별: B 승 {win} / 패 {lose} / 무 {n-win-lose}")
    verdict = "채택 (B가 유의하게 나음)" if lo > 0 else ("기각 (B가 유의하게 나쁨)" if hi < 0 else "차이 불명확 → 비용 적은 쪽")
    print("판정:", verdict)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])

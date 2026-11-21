"""후보 수(pool)가 늘 때 정답 순위가 얼마나 밀리는지 측정하고 테스트 규모(16,000장)로 외삽.

학습하지 않은 사전학습 모델(zero-shot)만 대상 → train 이미지까지 4,000장 전부를 후보로 써도 공정.
1) 캡션 4,000 × 이미지 4,000 유사도 행렬
2) 질문마다 정답보다 점수가 높은 오답 수 = 형제(같은 그룹) / 일반 으로 분리
3) 후보 N장(그룹 단위 샘플링: 형제는 함께 들어가거나 함께 빠짐)에서 실측 Recall@K
4) 16,000장 외삽:  낙관 = 형제 수는 그대로, 일반 오답만 ×f  /  비관 = 형제·일반 모두 ×4
   검증: 4,000장에서 같은 방식으로 2,000장을 예측해 실측과 비교
5) 시간: 이미지 디코딩(단일 스레드/병렬) / 전처리 / GPU 인코딩 분리 측정
"""
import csv, json, math, random, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, SiglipModel

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "/kaggle/working/haikuthon-train")
ROOT = Path(__file__).resolve().parents[1]
MODEL = sys.argv[2] if len(sys.argv) > 2 else "google/siglip-base-patch16-224"
TEST_N = 16000
KS = [1, 5, 10, 20, 30, 50, 100]

rows = [json.loads(l) for l in open(DATA / "prompts.jsonl", encoding="utf-8") if l.strip()]
rows.sort(key=lambda r: int(r["id"]))
ids = [r["id"] for r in rows]
caps = [r["prompts"] for r in rows]
grp = {r["id"]: int(r["group_id"]) for r in csv.DictReader(open(ROOT / "eda/groups.csv"))}
G = torch.tensor([grp[i] for i in ids])
n = len(ids)

dev = "cuda" if torch.cuda.is_available() else "cpu"
net = SiglipModel.from_pretrained(MODEL).to(dev).eval()
DT = torch.float16 if dev == "cuda" else torch.float32
net = net.to(DT)
proc = AutoProcessor.from_pretrained(MODEL)


def pooled(o):
    return o.pooler_output if hasattr(o, "pooler_output") else o


def load(i):
    with Image.open(DATA / "images" / f"{i}.png") as im:
        return im.convert("RGB")


# ---------- 시간: 단일 스레드 디코딩 (스타터킷 predict 방식) 샘플 200장
t = time.time(); [load(i) for i in ids[:200]]; dec_single = (time.time() - t) / 200

# ---------- 전체 인코딩 (병렬 디코딩) + 단계별 시간
t_dec = t_pre = t_gpu = 0.0
feats = []
pool_ex = ThreadPoolExecutor(8)
with torch.inference_mode():
    for s in range(0, n, 64):
        t0 = time.time(); imgs = list(pool_ex.map(load, ids[s:s + 64])); t1 = time.time()
        px = proc(images=imgs, return_tensors="pt").pixel_values.to(dev, DT); t2 = time.time()
        f = pooled(net.get_image_features(pixel_values=px)); (torch.cuda.synchronize() if dev == "cuda" else None); t3 = time.time()
        feats.append(F.normalize(f.float(), dim=-1)); t_dec += t1 - t0; t_pre += t2 - t1; t_gpu += t3 - t2
    I = torch.cat(feats)
    T = torch.cat([F.normalize(pooled(net.get_text_features(**proc(text=caps[s:s + 256], padding="max_length",
                  truncation=True, return_tensors="pt").to(dev))).float(), dim=-1) for s in range(0, n, 256)])
S = (T @ I.T).cpu()                                  # (캡션, 이미지)
correct = S.diag()
higher = S > correct[:, None]                        # 정답보다 높은 이미지
same = G[:, None] == G[None, :]
c_sib = (higher & same).sum(1).float()
c_non = (higher & ~same).sum(1).float()
n_sib = same.sum(1).float() - 1


def recall_from_rank(rank, ks=KS):
    return {k: float((rank <= k).float().mean()) for k in ks}


def score_from_rank(rank):
    r1 = (rank == 1).float().mean()
    nd = torch.where(rank <= 5, 1 / torch.log2(rank + 1), torch.zeros_like(rank)).mean()
    return float((r1 + nd) / 2)


def extrapolate(N, base_n=n):
    f_non = (N - 1 - n_sib) / (base_n - 1 - n_sib)   # 일반 오답 증가 배율
    opt = 1 + c_sib + c_non * f_non
    pes = 1 + (c_sib + c_non) * (N - 1) / (base_n - 1)
    return opt, pes


# ---------- 실측: 그룹 단위 샘플링으로 후보 N장
groups = {}
for k, i in enumerate(ids):
    groups.setdefault(grp[i], []).append(k)
glist = list(groups.values())


def measured(N, trials=5):
    acc = {k: 0.0 for k in KS}; sc = 0.0; ranks_all = []
    for t in range(trials):
        rng = random.Random(t); gs = glist[:]; rng.shuffle(gs); pick = []
        for g in gs:
            if len(pick) + len(g) <= N:
                pick += g
        idx = torch.tensor(sorted(pick))
        sub = S[idx][:, idx]
        rank = 1 + (sub > sub.diag()[:, None]).sum(1).float()
        for k, v in recall_from_rank(rank).items():
            acc[k] += v / trials
        sc += score_from_rank(rank) / trials
    return acc, sc, len(pick)


out = {"model": MODEL, "timing": {}, "measured": {}, "extrapolated": {}, "check": {}}
print(f"\n=== 실측 (캡션 {n}개 전부 질문, 그룹 단위 샘플링 5회 평균) ===")
print("후보수   score   " + "  ".join(f"R@{k:<3}" for k in KS))
for N in [800, 1600, 2000, 4000]:
    acc, sc, real = measured(N, trials=1 if N >= n else 5)
    out["measured"][N] = {"score": sc, "recall": acc}
    print(f"{real:5d}   {sc:.3f}   " + "  ".join(f"{acc[k]:.3f}" for k in KS))

# 외삽 검증: 2000장을 4000장 통계로 예측 vs 실측
rng = random.Random(0); gs = glist[:]; rng.shuffle(gs); pick = []
for g in gs:
    if len(pick) + len(g) <= 2000:
        pick += g
idx = torch.tensor(sorted(pick)); sub = S[idx][:, idx]
real2k = 1 + (sub > sub.diag()[:, None]).sum(1).float()
# 같은 질문들에 대해 4000장 통계로 2000장 예측 (축소 방향: f<1)
opt2k, pes2k = (x[idx] for x in extrapolate(len(pick)))
out["check"] = {"real": recall_from_rank(real2k), "pred_opt": recall_from_rank(opt2k), "pred_pes": recall_from_rank(pes2k)}
print("\n=== 외삽 방식 검증: 4,000장 통계로 2,000장 예측 vs 실측 ===")
for name, v in out["check"].items():
    print(f"{name:9s} " + "  ".join(f"R@{k}={v[k]:.3f}" for k in KS))

print(f"\n=== 테스트 규모 외삽 (후보 {TEST_N}장) ===")
opt, pes = extrapolate(TEST_N)
for name, rank in [("낙관(형제 수 그대로)", opt), ("비관(형제도 4배)", pes)]:
    rc = recall_from_rank(rank); out["extrapolated"][name] = {"score": score_from_rank(rank), "recall": rc}
    print(f"{name:14s} score≈{score_from_rank(rank):.3f}  " + "  ".join(f"R@{k}={rc[k]:.3f}" for k in KS))
need = {}
for name, rank in [("opt", opt), ("pes", pes)]:
    s = torch.sort(rank).values
    need[name] = {p: int(math.ceil(float(s[int(p / 100 * (n - 1))]))) for p in (90, 95, 98, 99)}
out["K_needed"] = need
print("\n정답을 p% 질문에서 포함하려면 필요한 K (16,000장 추정):")
for p in (90, 95, 98, 99):
    print(f"  {p}%: 낙관 K≥{need['opt'][p]}  /  비관 K≥{need['pes'][p]}")

tot_img = n
out["timing"] = {"decode_single_per_img": dec_single, "decode_parallel_per_img": t_dec / tot_img,
                 "preprocess_per_img": t_pre / tot_img, "gpu_per_img": t_gpu / tot_img}
print("\n=== 시간 (Kaggle T4, 이미지 1장당) ===")
for k, v in out["timing"].items():
    print(f"  {k:26s} {v*1000:6.1f} ms   → 16,000장 {v*TEST_N/60:5.1f}분")
print(f"  (참고: 스타터킷 predict는 디코딩을 단일 스레드로 함)")
Path(ROOT / "runs").mkdir(exist_ok=True)
json.dump(out, open(ROOT / "runs/pool_scaling.json", "w"), ensure_ascii=False, indent=1)
print("저장: runs/pool_scaling.json")

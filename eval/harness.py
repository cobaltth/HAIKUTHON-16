"""로컬 검증 하네스: 제출 폴더의 model.py(load_model/predict)를 서버와 같은 방식으로 호출해 채점.

예) python eval/harness.py --submission ../starterkit/sample_submission --data ../haikuthon-train
    --split valA --pool pair --topk 50

--pool own  : 후보 = 해당 검증셋 이미지 800장 (스타터킷 검증과 같은 조건)
--pool pair : 후보 = valA+valB 이미지 1,600장 → 테스트처럼 query 1개당 이미지 2장 비율
              (상대 세트는 이미지만 후보로 쓰고 그 캡션은 안 봄)
"""
import argparse, importlib.util, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import evaluate, format_report


def load_submission(folder):
    spec = importlib.util.spec_from_file_location("submission_model", Path(folder) / "model.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(folder)))
    spec.loader.exec_module(mod)
    return mod


def build(data_dir, split, which, pool):
    rows = {}
    with open(Path(data_dir) / "prompts.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line); rows[r["id"]] = r["prompts"]
    q_ids = split[which]
    other = {"valA": "valB", "valB": "valA"}.get(which)
    img_ids = q_ids + (split[other] if pool == "pair" and other else [])
    queries = [{"prompt_id": i, "prompt": rows[i]} for i in q_ids]
    # 서버와 같이 '문자열 절대 경로'로 넘긴다 (Path 전용 문법을 쓴 model.py가 로컬에서만 통과하는 일 방지)
    candidates = [{"image_id": i, "image_path": str((Path(data_dir) / "images" / f"{i}.png").resolve())}
                  for i in sorted(img_ids, key=int)]
    return queries, candidates, {i: i for i in q_ids}


def check_predictions(preds, queries, candidates, top_k):
    """서버 제출 조건 검사: 모든 query에 대해 서로 다른 후보 id를 정확히 top_k개 (문자열)."""
    valid_ids = {c["image_id"] for c in candidates}
    errors = []
    missing = [q["prompt_id"] for q in queries if q["prompt_id"] not in preds]
    if missing: errors.append(f"예측 없는 query {len(missing)}개 (예: {missing[:3]})")
    for q in queries:
        r = preds.get(q["prompt_id"])
        if r is None: continue
        if len(r) != top_k: errors.append(f"{q['prompt_id']}: 길이 {len(r)} != top_k {top_k}"); break
        if len(set(r)) != len(r): errors.append(f"{q['prompt_id']}: 중복 id {r}"); break
        if any(not isinstance(x, str) for x in r): errors.append(f"{q['prompt_id']}: 문자열 아닌 id {r}"); break
        if any(x not in valid_ids for x in r): errors.append(f"{q['prompt_id']}: 후보에 없는 id {r}"); break
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True, help="model.py가 있는 폴더")
    ap.add_argument("--data", required=True, help="haikuthon-train 폴더")
    ap.add_argument("--split-file", default=str(Path(__file__).resolve().parents[1] / "splits/split_group_seed42.json"))
    ap.add_argument("--split", default="valA", choices=["valA", "valB", "train"])
    ap.add_argument("--pool", default="pair", choices=["own", "pair"])
    ap.add_argument("--topk", type=int, default=50, help="순위 분포용으로 predict에 요청할 깊이(≥5)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--save", default=None, help="query별 순위를 json으로 저장 (eval/compare.py 비교, hard negative 마이닝용)")
    ap.add_argument("--server-check", type=int, default=64,
                    help="채점 전에 query N개로 top_k=5 호출해 제출 형식 검사 (0이면 생략)")
    a = ap.parse_args()

    split = json.load(open(a.split_file))
    queries, candidates, gt = build(a.data, split, a.split, a.pool)
    mod = load_submission(a.submission)
    device = a.device
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    if a.split_file.endswith("split_group_seed42.json") and Path(a.submission).resolve().parent.name == "starterkit":
        print("⚠ 스타터킷 예제 가중치는 valA의 82%를 학습에서 봤습니다. --split-file splits/split_random_starterkit.json --pool own 으로 재세요.")
    t0 = time.time(); model = mod.load_model(a.submission, device); t1 = time.time()
    if a.server_check:
        sub = queries[:a.server_check]
        errs = check_predictions(mod.predict(model, sub, candidates, top_k=5), sub, candidates, 5)
        print("[서버 형식 검사 top_k=5]", "통과" if not errs else "실패: " + " / ".join(errs))
        if errs:
            sys.exit(1)
    t1 = time.time()
    preds = mod.predict(model, queries, candidates, top_k=max(5, a.topk)); t2 = time.time()
    errs = check_predictions(preds, queries, candidates, max(5, a.topk))
    if errs:
        print("[top_k 형식 경고]", " / ".join(errs))
    res, ranks = evaluate(preds, gt)
    print(f"[{a.split} / pool={a.pool}: query {len(queries)}, 후보 이미지 {len(candidates)}]")
    print(format_report(res))
    per_q = (t2 - t1) / len(queries)
    print(f"  로딩 {t1-t0:.1f}s / 추론 {t2-t1:.1f}s → 테스트(8,000 query·16,000장) 단순 환산 ≈ {per_q*8000/60:.1f}분 (이미지 수 비례 부분은 과소추정될 수 있음)")
    if a.save:
        json.dump({"result": res, "ranks": dict(zip(gt, ranks)), "preds": preds}, open(a.save, "w"))


if __name__ == "__main__":
    main()

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
    candidates = [{"image_id": i, "image_path": Path(data_dir) / "images" / f"{i}.png"} for i in sorted(img_ids, key=int)]
    return queries, candidates, {i: i for i in q_ids}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True, help="model.py가 있는 폴더")
    ap.add_argument("--data", required=True, help="haikuthon-train 폴더")
    ap.add_argument("--split-file", default=str(Path(__file__).resolve().parents[1] / "splits/split_group_seed42.json"))
    ap.add_argument("--split", default="valA", choices=["valA", "valB", "train"])
    ap.add_argument("--pool", default="pair", choices=["own", "pair"])
    ap.add_argument("--topk", type=int, default=50, help="순위 분포용으로 predict에 요청할 깊이(≥5)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--save", default=None, help="query별 순위를 json으로 저장 (hard negative 마이닝용)")
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
    t0 = time.time(); model = mod.load_model(a.submission, device); t1 = time.time()
    preds = mod.predict(model, queries, candidates, top_k=max(5, a.topk)); t2 = time.time()
    res, ranks = evaluate(preds, gt)
    print(f"[{a.split} / pool={a.pool}: query {len(queries)}, 후보 이미지 {len(candidates)}]")
    print(format_report(res))
    per_q = (t2 - t1) / len(queries)
    print(f"  로딩 {t1-t0:.1f}s / 추론 {t2-t1:.1f}s → 테스트(8,000 query·16,000장) 단순 환산 ≈ {per_q*8000/60:.1f}분 (이미지 수 비례 부분은 과소추정될 수 있음)")
    if a.save:
        json.dump({"result": res, "ranks": dict(zip(gt, ranks)), "preds": preds}, open(a.save, "w"))


if __name__ == "__main__":
    main()

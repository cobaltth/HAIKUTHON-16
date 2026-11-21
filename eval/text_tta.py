"""텍스트 TTA 실험: 질문 문장 하나를 '문장 순서 섞기' 규칙으로 여러 버전 만들어 임베딩을 평균.

- 질문 자기 자신만 변형 → 다른 테스트 질문/예측 분포를 쓰지 않음 (규정 준수)
- 학습 없는 SigLIP으로 측정 (TTA는 학습과 독립이라 zero-shot에서 효과를 먼저 확인)
- 두 가지 조건: (1) valA 800 질문 / 후보 1,600장(valA+valB)  (2) 캡션 4,000개 전부 / 후보 4,000장 (테스트에 더 가까움)

예) python eval/text_tta.py --data /kaggle/working/haikuthon-train --variants 1,2,4,8
출력: 버전 수별 score/R@1/R@5/R@10/R@20 표 + runs/tta_<조건>_v<n>.json (compare.py로 v1 대비 비교)
"""
import argparse, json, random, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "eval"))
from aug.text_aug import shuffle_sentences  # noqa: E402
from metrics import evaluate  # noqa: E402


def variants(qid, text, n):
    """원문 + 서로 다른 섞기 버전 최대 n-1개 (질문 id로 시드 고정 → 재현 가능)."""
    out = [text]
    rng = random.Random(f"tta-{qid}")
    for _ in range(8 * n):
        if len(out) >= n:
            break
        v = shuffle_sentences(text, rng)
        if v not in out:
            out.append(v)
    return out


@torch.inference_mode()
def encode_images(model, proc, paths, device, workers=4, bs=64):
    def load(p):
        with Image.open(p) as im:
            return im.convert("RGB")
    feats = []
    with ThreadPoolExecutor(workers) as ex:
        for s in range(0, len(paths), bs):
            ims = list(ex.map(load, paths[s:s + bs]))
            px = proc(images=ims, return_tensors="pt").pixel_values.to(device, torch.float16)
            feats.append(F.normalize(model.get_image_features(pixel_values=px).pooler_output.float(), dim=-1))
    return torch.cat(feats)


@torch.inference_mode()
def encode_texts(model, proc, texts, device, bs=256):
    feats = []
    for s in range(0, len(texts), bs):
        tok = proc(text=texts[s:s + bs], padding="max_length", truncation=True, return_tensors="pt").to(device)
        feats.append(F.normalize(model.get_text_features(**tok).pooler_output.float(), dim=-1))
    return torch.cat(feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="google/siglip-base-patch16-224")
    ap.add_argument("--variants", default="1,2,4,8")
    ap.add_argument("--split-file", default=str(ROOT / "splits/split_group_seed42.json"))
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--topk", type=int, default=50)
    a = ap.parse_args()
    from transformers import AutoProcessor, SiglipModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SiglipModel.from_pretrained(a.model, dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
    proc = AutoProcessor.from_pretrained(a.model)

    data = Path(a.data)
    caps = {}
    for line in open(data / "prompts.jsonl", encoding="utf-8"):
        if line.strip():
            r = json.loads(line); caps[r["id"]] = r["prompts"]
    split = json.load(open(a.split_file))
    conds = {"valA": (list(split["valA"]), list(split["valA"]) + list(split["valB"])),
             "all4000": (sorted(caps), sorted(caps))}

    all_ids = sorted(caps)
    t = time.time()
    img = encode_images(model, proc, [str(data / "images" / f"{i}.png") for i in all_ids], dev)
    print(f"이미지 {len(all_ids)}장 인코딩 {time.time()-t:.0f}s", flush=True)
    col = {i: k for k, i in enumerate(all_ids)}
    ns = [int(x) for x in a.variants.split(",")]
    Path(a.runs).mkdir(exist_ok=True)

    for cname, (qids, pool) in conds.items():
        pool_feat = img[[col[i] for i in pool]]
        print(f"\n== {cname}: 질문 {len(qids)} / 후보 {len(pool)}")
        print(f"{'버전':>4} {'score':>7} {'R@1':>6} {'NDCG5':>6} {'R@5':>6} {'R@10':>6} {'R@20':>6} {'평균버전':>6} {'텍스트s':>7}")
        for n in ns:
            t = time.time()
            texts, owner = [], []
            for qi, q in enumerate(qids):
                vs = variants(q, caps[q], n)
                texts += vs; owner += [qi] * len(vs)
            tf = encode_texts(model, proc, texts, dev)
            own = torch.tensor(owner, device=dev)
            agg = torch.zeros(len(qids), tf.shape[1], device=dev).index_add_(0, own, tf)
            agg = F.normalize(agg, dim=-1)
            top = (agg @ pool_feat.T).topk(min(a.topk, len(pool)), dim=1).indices.tolist()
            preds = {q: [pool[j] for j in row] for q, row in zip(qids, top)}
            res, ranks = evaluate(preds, {q: q for q in qids})
            sec = time.time() - t
            print(f"{n:>4} {res['score']:7.4f} {res['R@1']:6.3f} {res['NDCG@5']:6.3f} {res['Recall@5']:6.3f} "
                  f"{res.get('Recall@10', float('nan')):6.3f} {res.get('Recall@20', float('nan')):6.3f} {len(texts)/len(qids):6.2f} {sec:7.1f}", flush=True)
            json.dump({"name": f"tta_{cname}_v{n}", "result": res, "ranks": dict(zip(qids, ranks))},
                      open(Path(a.runs) / f"tta_{cname}_v{n}.json", "w"))


if __name__ == "__main__":
    main()

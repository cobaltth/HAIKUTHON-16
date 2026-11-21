"""B 트랙 학습 스크립트: 그룹 분할 + 증강 옵션 + 스타터킷과 같은 predict로 검증.

예) 증강 없이 기본 FT (ablation 기준)
    python train/train_b.py --data ../haikuthon-train --model google/siglip-base-patch16-224 --name r1_base
    + 그룹 배치            --group-batch                --name r2_group
    + 좌우 반전·교환        --flip-swap-p 0.5            --name r3_flip
    + 텍스트 hard negative  --n-text-neg 2               --name r4_textneg
    + 약한 크롭            --crop-p 0.5                 --name r5_crop
    사전학습 그대로(학습 0회) --epochs 0                  --name r0_zeroshot

결과: runs/<name>.json (eval/compare.py로 비교), out/<name>/submission/ (제출 폴더 형식, --save-model 시)
규정: 학습은 Colab·Kaggle 무료 GPU에서만. 증강은 규칙 기반(외부 생성 모델 X).
"""
import argparse, json, random, shutil, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]
from aug.dataset import GroupBatchSampler, load_groups
from aug.image_aug import hflip, weak_square_crop
from aug.loss import clip_loss, siglip_loss
from aug.text_aug import hard_negatives, swap_left_right
from harness import build, check_predictions, load_submission
from metrics import evaluate, format_report
from train.adapters import get_adapter


# ---------------------------------------------------------------- 데이터
class PairDataset(torch.utils.data.Dataset):
    """id → (증강된 이미지, 캡션, 가짜 캡션 목록). 이미지는 처음에 한 번 디코딩해 cache_size로 줄여 메모리에 둔다."""

    def __init__(self, ids, captions, image_dir, cfg, cache_size, seed, workers=8):
        self.ids, self.caps, self.cfg, self.seed, self.epoch = list(ids), captions, cfg, seed, 0
        self.index = {i: k for k, i in enumerate(self.ids)}

        def load(i):
            with Image.open(Path(image_dir) / f"{i}.png") as im:
                im = im.convert("RGB")
                return im.resize((cache_size, cache_size), Image.BICUBIC) if cache_size else im.copy()

        t = time.time()
        with ThreadPoolExecutor(workers) as ex:
            self.cache = dict(zip(self.ids, ex.map(load, self.ids)))
        print(f"이미지 {len(self.ids)}장 캐시 ({cache_size or '원본'}px) {time.time()-t:.0f}s", flush=True)

    def set_epoch(self, e):
        self.epoch = e

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, k):
        i = self.ids[k]
        rng = random.Random(f"{self.seed}-{self.epoch}-{i}")
        img, cap = self.cache[i], self.caps[i]
        c = self.cfg
        if c["crop_p"] and rng.random() < c["crop_p"]:
            img = weak_square_crop(img, rng, c["crop_min"])
        if c["flip_swap_p"] and rng.random() < c["flip_swap_p"]:
            img, cap = hflip(img), swap_left_right(cap)
        negs = []
        if c["n_text_neg"]:
            cands = hard_negatives(cap, rng, kinds=c["neg_kinds"])
            negs = [t for _, t in rng.sample(cands, min(c["n_text_neg"], len(cands)))]
        return img, cap, negs


class RandomBatchSampler:
    """기준용 무작위 배치 (에포크마다 섞음)."""

    def __init__(self, ids, batch_size, seed):
        self.ids, self.bs, self.seed, self.epoch = list(ids), batch_size, seed, 0

    def set_epoch(self, e):
        self.epoch = e

    def __iter__(self):
        ids = self.ids[:]
        random.Random(self.seed + self.epoch).shuffle(ids)
        for k in range(0, len(ids), self.bs):
            yield ids[k:k + self.bs]

    def __len__(self):
        return (len(self.ids) + self.bs - 1) // self.bs


class IdToIndex:
    """id 배치를 Dataset 인덱스 배치로 바꿔 DataLoader에 넘긴다."""

    def __init__(self, sampler, index):
        self.s, self.index = sampler, index

    def __iter__(self):
        for b in self.s:
            yield [self.index[i] for i in b]

    def __len__(self):
        return len(self.s)


def collate(batch):
    imgs, caps, negs = zip(*batch)
    return list(imgs), list(caps), list(negs)


# ---------------------------------------------------------------- 검증
def run_eval(adapter, template, data, split, which, pool, topk):
    """스타터킷 model.py의 predict를 그대로 써서 채점 (서버와 같은 경로)."""
    queries, candidates, gt = build(data, split, which, pool)
    model = adapter.as_submission_model()
    adapter.net.eval()
    t = time.time()
    errs = check_predictions(template.predict(model, queries[:32], candidates, top_k=5), queries[:32], candidates, 5)
    if errs:
        raise RuntimeError("서버 형식 검사 실패: " + " / ".join(errs))
    preds = template.predict(model, queries, candidates, top_k=max(5, topk))
    sec = time.time() - t
    res, ranks = evaluate(preds, gt)
    adapter.net.train()
    return res, dict(zip(gt, ranks)), preds, sec


# ---------------------------------------------------------------- 메인
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="haikuthon-train 폴더 (prompts.jsonl, images/)")
    ap.add_argument("--model", default="google/siglip-base-patch16-224",
                    help="사전학습 모델. ⚠ 스타터킷 예제 가중치는 valA를 이미 봤으므로 쓰지 말 것")
    ap.add_argument("--adapter", default="siglip")
    ap.add_argument("--name", required=True, help="실험 이름 → runs/<name>.json")
    ap.add_argument("--split-file", default=str(ROOT / "splits/split_group_seed42.json"))
    ap.add_argument("--groups", default=str(ROOT / "eda/groups.csv"))
    ap.add_argument("--train-on", default="train", help="train | train+valB | all (최종 제출용)")
    ap.add_argument("--eval-split", default="valA")
    ap.add_argument("--pool", default="pair", choices=["own", "pair"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--trainable", default="heads", help="heads | lastK (예: last2) | all")
    ap.add_argument("--train-logit", action="store_true", help="logit_scale/bias도 학습")
    # 증강·샘플링 (전부 기본 꺼짐 → 하나씩 켜며 ablation)
    ap.add_argument("--group-batch", action="store_true", help="형제 캡션을 같은 배치에")
    ap.add_argument("--flip-swap-p", type=float, default=0.0)
    ap.add_argument("--crop-p", type=float, default=0.0)
    ap.add_argument("--crop-min", type=float, default=0.9)
    ap.add_argument("--n-text-neg", type=int, default=0)
    ap.add_argument("--neg-kinds", default="color,bind,rel,pattern,material")
    ap.add_argument("--neg-weight", type=float, default=1.0)
    # 기타
    ap.add_argument("--cache-size", type=int, default=256, help="학습 이미지 캐시 해상도 (0=원본). 모델 입력(224)보다 크게")
    ap.add_argument("--eval-every", type=int, default=1, help="N 에포크마다 검증 (0=마지막에만)")
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--save-model", action="store_true", help="best 가중치를 제출 폴더 형식으로 저장")
    ap.add_argument("--out", default=str(ROOT / "out"))
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--limit", type=int, default=0, help="(테스트용) 세트당 id 개수 제한")
    a = ap.parse_args()

    random.seed(a.seed); torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda" and not a.no_amp
    split = json.load(open(a.split_file))
    if a.limit:
        split = {k: (v[:a.limit] if isinstance(v, list) else v) for k, v in split.items()}
    train_ids = {"train": split["train"], "train+valB": split["train"] + split["valB"],
                 "all": split["train"] + split["valA"] + split["valB"]}[a.train_on]
    if a.eval_split in train_ids[:0] or set(split[a.eval_split]) & set(train_ids):
        print(f"⚠ 학습 데이터({a.train_on})에 {a.eval_split}가 포함됨 → 검증 점수는 의미 없음 (최종 제출용 학습일 때만 사용)")
    caps = {}
    with open(Path(a.data) / "prompts.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line); caps[r["id"]] = r["prompts"]

    cfg = {"flip_swap_p": a.flip_swap_p, "crop_p": a.crop_p, "crop_min": a.crop_min,
           "n_text_neg": a.n_text_neg, "neg_kinds": tuple(a.neg_kinds.split(","))}
    template = load_submission(ROOT / "submission_template")
    adapter = get_adapter(a.adapter, a.model, device)
    n_train = adapter.set_trainable(a.trainable, a.train_logit)
    print(f"장치 {device} | 학습 {len(train_ids)}쌍 | 학습 파라미터 {n_train/1e6:.1f}M ({a.trainable}) | cfg {cfg} | group_batch={a.group_batch}", flush=True)

    history, best = [], {"score": -1}
    t_start = time.time()
    if a.epochs > 0:
        ds = PairDataset(train_ids, caps, Path(a.data) / "images", cfg, a.cache_size, a.seed)
        base = (GroupBatchSampler(train_ids, load_groups(a.groups), a.batch_size, seed=a.seed) if a.group_batch
                else RandomBatchSampler(train_ids, a.batch_size, a.seed))
        loader = torch.utils.data.DataLoader(ds, batch_sampler=IdToIndex(base, ds.index), collate_fn=collate,
                                             num_workers=a.workers)
        params = [p for p in adapter.net.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        adapter.net.train()
        for ep in range(1, a.epochs + 1):
            base.set_epoch(ep); ds.set_epoch(ep)
            t, tot, n, n_neg = time.time(), 0.0, 0, 0
            for imgs, texts, negs in loader:
                flat = [x for ns in negs for x in ns]
                owner = torch.tensor([k for k, ns in enumerate(negs) for _ in ns], device=device, dtype=torch.long)
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    ie = adapter.encode_image(adapter.pixels(imgs))
                    te = adapter.encode_text(adapter.tokens(texts))
                    ne = adapter.encode_text(adapter.tokens(flat)) if flat else None
                lp = adapter.loss_params()
                if adapter.loss_type == "sigmoid":
                    loss = siglip_loss(ie, te, lp["logit_scale"].float(), lp["logit_bias"].float(), ne, owner, a.neg_weight)
                else:
                    loss = clip_loss(ie, te, lp["logit_scale"].float(), ne, owner)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                tot += loss.item() * len(imgs); n += len(imgs); n_neg += len(flat)
            rec = {"epoch": ep, "loss": tot / n, "train_sec": round(time.time() - t, 1), "text_negs": n_neg}
            if a.eval_every and (ep % a.eval_every == 0 or ep == a.epochs):
                res, ranks, preds, sec = run_eval(adapter, template, a.data, split, a.eval_split, a.pool, a.topk)
                rec.update(score=res["score"], eval_sec=round(sec, 1))
                print(f"[ep {ep}] loss {rec['loss']:.4f} ({rec['train_sec']}s)\n" + format_report(res), flush=True)
                if res["score"] > best["score"]:
                    best = {"score": res["score"], "epoch": ep, "result": res, "ranks": ranks, "preds": preds, "eval_sec": sec,
                            "state": {k: v.detach().cpu().clone() for k, v in adapter.net.state_dict().items()} if a.save_model else None}
            else:
                print(f"[ep {ep}] loss {rec['loss']:.4f} ({rec['train_sec']}s)", flush=True)
            history.append(rec)

    if best["score"] < 0:  # 학습 0회(zero-shot) 또는 마지막에만 평가
        res, ranks, preds, sec = run_eval(adapter, template, a.data, split, a.eval_split, a.pool, a.topk)
        print("[final]\n" + format_report(res), flush=True)
        best = {"score": res["score"], "epoch": a.epochs, "result": res, "ranks": ranks, "preds": preds, "eval_sec": sec,
                "state": None}

    n_q = len(split[a.eval_split])
    print(f"best epoch {best['epoch']} score {best['score']:.4f} | 검증 추론 {best['eval_sec']:.1f}s "
          f"(query {n_q}) → 테스트 규모 단순 환산 ≈ {best['eval_sec'] / n_q * 8000 / 60:.1f}분 | 총 {time.time()-t_start:.0f}s")

    Path(a.runs).mkdir(parents=True, exist_ok=True)
    run = {"name": a.name, "args": vars(a), "best_epoch": best["epoch"], "result": best["result"],
           "ranks": best["ranks"], "preds": best["preds"], "history": history}
    json.dump(run, open(Path(a.runs) / f"{a.name}.json", "w"), ensure_ascii=False)
    print(f"저장: {Path(a.runs) / (a.name + '.json')}")

    if a.save_model:
        if best["state"] is not None:
            adapter.net.load_state_dict(best["state"])
        sub = Path(a.out) / a.name / "submission"
        (sub / "models").mkdir(parents=True, exist_ok=True)
        adapter.save(sub / "models")
        for f in ("model.py", "requirements.txt"):
            shutil.copy(ROOT / "submission_template" / f, sub / f)
        print(f"제출 폴더: {sub}")


if __name__ == "__main__":
    main()

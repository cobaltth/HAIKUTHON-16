"""1단계 앙상블 측정: SigLIP(Base, A 계열) + SigLIP2 so400m@256(C 계열) 유사도를 질문별 z-정규화 후 가중합.

- 질문 하나의 점수만으로 정규화 → 다른 질문 정보 안 씀 (규정 준수)
- 조건: valA 800 질문 × 후보 4,000장 전체 (테스트에 더 가까운 큰 후보 풀)
- 출력: 모델별 / 가중치별 R@1, Recall@3/5/8/10/20/30, 점수(½(R@1+NDCG@5)), 이미지 인코딩 ms/장

예) python eval/stage1_fusion.py --data /kaggle/working/haikuthon-train \
      --base google/siglip-base-patch16-224 --base out/a1_save/submission/models \
      --s2 google/siglip2-so400m-patch14-384
"""
import argparse, json, math, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
KS = (1, 3, 5, 8, 10, 20, 30)


def metrics(sim, gt_col):
    """sim: (Q, N) 점수, gt_col: (Q,) 정답 열. 정답 순위로 지표 계산."""
    gt = sim.gather(1, gt_col[:, None])
    rank = (sim > gt).sum(1) + 1  # 1부터
    r = rank.float()
    out = {f"R@{k}": (rank <= k).float().mean().item() for k in KS}
    ndcg = torch.where(rank <= 5, 1 / torch.log2(r + 1), torch.zeros_like(r)).mean().item()
    out["score"] = (out["R@1"] + ndcg) / 2
    return out


def zq(s):
    return (s - s.mean(1, keepdim=True)) / (s.std(1, keepdim=True) + 1e-6)


@torch.inference_mode()
def run_base(path, caps_q, imgs, dev):
    from transformers import AutoProcessor, SiglipModel
    m = SiglipModel.from_pretrained(path, dtype=torch.float16).to(dev).eval()
    p = AutoProcessor.from_pretrained(path)
    t = time.time(); fi = []
    for s in range(0, len(imgs), 64):
        px = p(images=imgs[s:s + 64], return_tensors="pt").pixel_values.to(dev, torch.float16)
        fi.append(F.normalize(m.get_image_features(pixel_values=px).pooler_output.float(), dim=-1))
    torch.cuda.synchronize(); ms = (time.time() - t) / len(imgs) * 1000
    ft = []
    for s in range(0, len(caps_q), 256):
        tok = p(text=caps_q[s:s + 256], padding="max_length", truncation=True, return_tensors="pt").to(dev)
        ft.append(F.normalize(m.get_text_features(**tok).pooler_output.float(), dim=-1))
    del m; torch.cuda.empty_cache()
    return torch.cat(ft) @ torch.cat(fi).T, ms


@torch.inference_mode()
def run_s2(path, caps_q, imgs, dev, res=256):
    """C의 model.py와 같은 전처리: 소문자, max_len 64, bicubic res px, mean/std 0.5, 위치 임베딩 보간."""
    from transformers import AutoModel, AutoTokenizer
    m = AutoModel.from_pretrained(path, dtype=torch.float16).to(dev).eval()
    tok = AutoTokenizer.from_pretrained(path)
    native = int(m.config.vision_config.image_size)
    kw = {"interpolate_pos_encoding": True} if res != native else {}

    def as_t(o):
        if torch.is_tensor(o):
            return o
        po = getattr(o, "pooler_output", None)
        return po if po is not None else o[0]
    t = time.time(); fi = []
    for s in range(0, len(imgs), 64):
        arr = np.stack([np.asarray(im.resize((res, res), Image.Resampling.BICUBIC), dtype=np.uint8) for im in imgs[s:s + 64]])
        x = torch.from_numpy(arr).to(dev).permute(0, 3, 1, 2).float().div_(255.0)
        x = ((x - 0.5) / 0.5).half()
        fi.append(F.normalize(as_t(m.get_image_features(pixel_values=x, **kw)).float(), dim=-1))
    torch.cuda.synchronize(); ms = (time.time() - t) / len(imgs) * 1000
    ft = []
    for s in range(0, len(caps_q), 256):
        enc = tok([c.lower() for c in caps_q[s:s + 256]], padding="max_length", truncation=True, max_length=64, return_tensors="pt")
        ft.append(F.normalize(as_t(m.get_text_features(input_ids=enc["input_ids"].to(dev))).float(), dim=-1))
    del m; torch.cuda.empty_cache()
    return torch.cat(ft) @ torch.cat(fi).T, ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", action="append", default=[], help="SigLIP Base 계열 (여러 개 가능)")
    ap.add_argument("--s2", action="append", default=[], help="SigLIP2 계열 (C 전처리, 256px)")
    ap.add_argument("--s2-res", type=int, default=256)
    ap.add_argument("--split-file", default=str(ROOT / "splits/split_group_seed42.json"))
    ap.add_argument("--save", default=str(ROOT / "runs/stage1_fusion.json"))
    a = ap.parse_args()
    dev = "cuda"
    data = Path(a.data)
    caps = {}
    for line in open(data / "prompts.jsonl", encoding="utf-8"):
        if line.strip():
            r = json.loads(line); caps[r["id"]] = r["prompts"]
    pool = sorted(caps)
    qids = list(json.load(open(a.split_file))["valA"])
    col = {i: k for k, i in enumerate(pool)}
    gt = torch.tensor([col[q] for q in qids], device=dev)

    t = time.time()
    def load(i):
        with Image.open(data / "images" / f"{i}.png") as im:
            return im.convert("RGB").resize((384, 384), Image.Resampling.BICUBIC)  # 공통 캐시 (메모리 절약)
    with ThreadPoolExecutor(4) as ex:
        imgs = list(ex.map(load, pool))
    print(f"이미지 {len(imgs)}장 디코딩 {time.time()-t:.0f}s", flush=True)
    cq = [caps[q] for q in qids]

    sims, report = {}, {}
    for pth in a.base:
        name = "base:" + ("/".join(Path(pth).parts[-3:]) if Path(pth).exists() else pth.split("/")[-1])
        s, ms = run_base(pth, cq, imgs, dev); sims[name] = s
        report[name] = {**metrics(s, gt), "ms_per_img": ms}
    for pth in a.s2:
        name = "s2:" + pth.split("/")[-1]
        s, ms = run_s2(pth, cq, imgs, dev, a.s2_res); sims[name] = s
        report[name] = {**metrics(s, gt), "ms_per_img": ms}

    names = list(sims)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            za, zb = zq(sims[names[i]]), zq(sims[names[j]])
            for w in (0.3, 0.4, 0.5, 0.6, 0.7):
                report[f"fuse[{names[i]}×{w} + {names[j]}×{1-w:.1f}]"] = metrics(w * za + (1 - w) * zb, gt)

    hdr = f"{'모델':58s} {'score':>6} " + " ".join(f"{'R@'+str(k):>6}" for k in KS) + f" {'ms/장':>6}"
    print(f"\n== valA {len(qids)} 질문 × 후보 {len(pool)}장\n" + hdr)
    for n, r in report.items():
        print(f"{n[:58]:58s} {r['score']:6.4f} " + " ".join(f"{r[f'R@{k}']:6.3f}" for k in KS)
              + (f" {r['ms_per_img']:6.1f}" if "ms_per_img" in r else ""), flush=True)
    Path(a.save).parent.mkdir(exist_ok=True)
    json.dump(report, open(a.save, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

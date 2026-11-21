"""문장 분해 채점 실험: 캡션을 문장 단위로 나눠 각각 채점한 뒤 합치면 형제 구분이 나아지는가.
valA 800 질문 × 후보 4,000장.
 1단계 SigLIP: whole(전체 캡션) vs 문장별 점수의 mean / min / z-mean / whole+z-mean
 2단계 BLIP-2 ITM (a1 top-20 안에서): whole vs 문장별 mean / min / z-mean / whole+z-mean
"""
import argparse, json, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def split_sent(c):
    return [s for s in re.split(r"(?<=\.)\s+", c.strip()) if s]


def metr(rank):
    r = rank.float()
    ndcg = torch.where(rank <= 5, 1 / torch.log2(r + 1), torch.zeros_like(r)).mean().item()
    r1 = (rank <= 1).float().mean().item(); r5 = (rank <= 5).float().mean().item()
    return f"score {(r1+ndcg)/2:.4f} R@1 {r1:.3f} R@5 {r5:.3f}"


def rank_of(score, gt_col):
    g = score.gather(1, gt_col[:, None])
    return (score > g).sum(1) + 1


def zrow(x):
    return (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--siglip", action="append", required=True)
    ap.add_argument("--blip", default="")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--split-file", default=str(ROOT / "splits/split_group_seed42.json"))
    a = ap.parse_args()
    dev = "cuda"
    data = Path(a.data)
    caps = {}
    for line in open(data / "prompts.jsonl", encoding="utf-8"):
        if line.strip():
            r = json.loads(line); caps[r["id"]] = r["prompts"]
    pool = sorted(caps); col = {i: k for k, i in enumerate(pool)}
    qids = list(json.load(open(a.split_file))["valA"])
    gt = torch.tensor([col[q] for q in qids], device=dev)
    whole = [caps[q] for q in qids]
    sents, owner = [], []
    for qi, c in enumerate(whole):
        for s in split_sent(c):
            sents.append(s); owner.append(qi)
    owner_t = torch.tensor(owner, device=dev)
    nq = len(qids)
    print(f"sentences {len(sents)} (avg {len(sents)/nq:.2f}/caption)", flush=True)

    def agg(S_sent, how):
        """S_sent: (n_sent, N) → (nq, N)"""
        out = torch.zeros(nq, S_sent.shape[1], device=dev)
        if how == "mean":
            out.index_add_(0, owner_t, S_sent)
            cnt = torch.bincount(owner_t, minlength=nq).float()[:, None]
            return out / cnt
        if how == "zmean":
            return agg(zrow(S_sent), "mean")
        if how == "min":
            out.fill_(float("inf"))
            return out.scatter_reduce(0, owner_t[:, None].expand_as(S_sent), S_sent, reduce="amin")

    t = time.time()
    def load(i):
        with Image.open(data / "images" / f"{i}.png") as im:
            return im.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
    with ThreadPoolExecutor(4) as ex:
        imgs = list(ex.map(load, pool))
    print(f"decode {time.time()-t:.0f}s", flush=True)

    from transformers import AutoProcessor, SiglipModel
    lines = []
    top_for_blip = None
    for path in a.siglip:
        name = Path(path).parts[-3] if Path(path).exists() else path.split("/")[-1]
        m = SiglipModel.from_pretrained(path, dtype=torch.float16).to(dev).eval()
        p = AutoProcessor.from_pretrained(path)
        po = lambda o: o.pooler_output if hasattr(o, "pooler_output") else o
        with torch.inference_mode():
            I = torch.cat([F.normalize(po(m.get_image_features(pixel_values=p(images=imgs[s:s+128], return_tensors="pt").pixel_values.to(dev, torch.float16))).float(), dim=-1) for s in range(0, len(imgs), 128)])
            enc = lambda tx: torch.cat([F.normalize(po(m.get_text_features(**p(text=tx[s:s+256], padding="max_length", truncation=True, return_tensors="pt").to(dev))).float(), dim=-1) for s in range(0, len(tx), 256)])
            Tw, Ts = enc(whole), enc(sents)
        del m; torch.cuda.empty_cache()
        Sw = Tw @ I.T; Ss = Ts @ I.T
        res = {"whole": Sw, "s_mean": agg(Ss, "mean"), "s_min": agg(Ss, "min"), "s_zmean": agg(Ss, "zmean")}
        res["whole+s_zmean"] = zrow(Sw) + agg(Ss, "zmean")
        for k, v in res.items():
            lines.append(f"S1 {name[:14]:14s} {k:14s} " + metr(rank_of(v, gt)))
            print(lines[-1], flush=True)
        if top_for_blip is None or "a1" in name or "save" in path:
            top_for_blip = Sw.topk(a.k, dim=1).indices
            s1_top = Sw.gather(1, top_for_blip)

    if a.blip:
        from submission_ens.model import _load_blip, _blip_image_embeds, _blip_itm, _to_pixels, CLIP_MEAN, CLIP_STD
        blip, bproc = _load_blip(Path(a.blip), dev)
        vdt = blip.vision_model.embeddings.patch_embedding.weight.dtype
        uniq = sorted(set(top_for_blip.flatten().tolist()))
        a224 = np.stack([np.asarray(imgs[j]) for j in uniq])
        pos = {j: i for i, j in enumerate(uniq)}
        t = time.time()
        embs = []
        with torch.inference_mode():
            for s in range(0, len(uniq), 48):
                embs.append(_blip_image_embeds(blip, _to_pixels(a224[s:s+48], CLIP_MEAN, CLIP_STD, dev, vdt)))
        E = torch.cat(embs)
        print(f"blip vit {len(uniq)} imgs {time.time()-t:.0f}s", flush=True)

        def itm_texts(texts, rows):
            tok = bproc.tokenizer(texts, padding="longest", truncation=True, max_length=128, return_tensors="pt")
            ids, mask = tok.input_ids.to(dev), tok.attention_mask.to(dev)
            out = torch.zeros(len(texts), a.k, device=dev)
            with torch.inference_mode():
                for ti in range(len(texts)):
                    cand = [pos[j] for j in top_for_blip[rows[ti]].tolist()]
                    out[ti] = _blip_itm(blip, E[cand], ids[ti:ti+1].expand(a.k, -1), mask[ti:ti+1].expand(a.k, -1))
            return out
        t = time.time()
        Bw = itm_texts(whole, list(range(nq)))
        Bs = itm_texts(sents, owner)
        print(f"blip itm {time.time()-t:.0f}s", flush=True)
        gt_in = (top_for_blip == gt[:, None])
        def rank_k(score):  # top-k 안 순위, 밖이면 999
            g = torch.where(gt_in, score, torch.full_like(score, float("-inf"))).max(1).values
            r = (score > g[:, None]).sum(1) + 1
            return torch.where(gt_in.any(1), r, torch.full_like(r, 999))
        # 문장 → 질문 집계 (k칸)
        def agg_k(S, how):
            out = torch.zeros(nq, a.k, device=dev)
            if how == "mean":
                out.index_add_(0, owner_t, S); return out / torch.bincount(owner_t, minlength=nq).float()[:, None]
            if how == "min":
                out.fill_(float("inf")); return out.scatter_reduce(0, owner_t[:, None].expand_as(S), S, reduce="amin")
            if how == "zmean":
                return agg_k(zrow(S), "mean")
        z1 = zrow(s1_top)
        variants = {"stage1 only": s1_top, "whole": Bw, "s_mean": agg_k(Bs, "mean"), "s_min": agg_k(Bs, "min"),
                    "s_zmean": agg_k(Bs, "zmean")}
        variants["whole+s_zmean"] = zrow(Bw) + agg_k(Bs, "zmean")
        for k, v in list(variants.items()):
            lines.append(f"S2 blip k{a.k} {k:14s} " + metr(rank_k(v)))
            if k != "stage1 only":
                lines.append(f"S2 blip k{a.k} {k+'+.1s1':14s} " + metr(rank_k(0.9 * zrow(v) + 0.1 * z1)))
    (ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / "runs/decomp.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()

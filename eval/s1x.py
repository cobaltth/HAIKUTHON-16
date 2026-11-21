"""1단계 강화·채점 속도·반사실 대조 실험 (valA 800 × 후보 4,000).
 A. SigLIP(a1) 기준 + 반사실 캡션 대조 (규칙 기반 hard_negatives)
 B. BLIP-2 ITC 단독 / SigLIP과 z-합
 C. ITM 속도: Q-Former fp32 vs autocast fp16 (쌍/초, 점수 일치)
 D. ITM 반사실 대조 (a1 top-20 안)
 E. (선택) zero-shot BLIP-2 ITC
"""
import argparse, json, random, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = []


def say(s):
    OUT.append(s); print("R> " + s, flush=True)
    (ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / "runs/s1x.txt").write_text("\n".join(OUT) + "\n")


def metr_rank(rank):
    r = rank.float()
    ndcg = torch.where(rank <= 5, 1 / torch.log2(r + 1), torch.zeros_like(r)).mean().item()
    f = lambda k: (rank <= k).float().mean().item()
    return f"score {(f(1)+ndcg)/2:.4f} R@1 {f(1):.3f} R@5 {f(5):.3f} R@10 {f(10):.3f} R@20 {f(20):.3f}"


def metr(S, gt):
    return metr_rank((S > S.gather(1, gt[:, None])).sum(1) + 1)


def zrow(x):
    return (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--siglip", required=True)
    ap.add_argument("--blip", required=True)
    ap.add_argument("--blip0", default="")
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
    nq, N, K = len(qids), len(pool), a.k
    from aug.text_aug import hard_negatives
    cf, own = [], []
    for i, w in enumerate(whole):
        for _, c in hard_negatives(w, random.Random(i)):
            cf.append(c); own.append(i)
    own_t = torch.tensor(own, device=dev)
    cnt = torch.bincount(own_t, minlength=nq).float()[:, None]
    say(f"cf captions {len(cf)} (avg {len(cf)/nq:.2f}/q, none {(cnt==0).sum().item()})")

    def agg(Sc, base, how):
        if how == "mean":
            o = torch.zeros_like(base).index_add_(0, own_t, Sc) / cnt.clamp(min=1)
        else:
            o = torch.full_like(base, float("-inf")).scatter_reduce(0, own_t[:, None].expand_as(Sc), Sc, reduce="amax")
        return torch.where(cnt > 0, o, base)

    t = time.time()
    def load(i):
        with Image.open(data / "images" / f"{i}.png") as im:
            return im.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
    with ThreadPoolExecutor(4) as ex:
        imgs = list(ex.map(load, pool))
    print(f"decode {time.time()-t:.0f}s", flush=True)

    # ---------- A. SigLIP
    from transformers import AutoProcessor, SiglipModel
    m = SiglipModel.from_pretrained(a.siglip, dtype=torch.float16).to(dev).eval()
    p = AutoProcessor.from_pretrained(a.siglip)
    po = lambda o: o.pooler_output if hasattr(o, "pooler_output") else o
    with torch.inference_mode():
        I = torch.cat([F.normalize(po(m.get_image_features(pixel_values=p(images=imgs[s:s+128], return_tensors="pt").pixel_values.to(dev, torch.float16))).float(), dim=-1) for s in range(0, N, 128)])
        enc = lambda tx: torch.cat([F.normalize(po(m.get_text_features(**p(text=tx[s:s+256], padding="max_length", truncation=True, return_tensors="pt").to(dev))).float(), dim=-1) for s in range(0, len(tx), 256)])
        S = enc(whole) @ I.T
        Sc = enc(cf) @ I.T
    del m; torch.cuda.empty_cache()
    S = S.clone(); Sc = Sc.clone()
    say("A siglip whole            " + metr(S, gt))
    for how in ("mean", "max"):
        A = agg(Sc, S, how)
        for al in (0.25, 0.5, 1.0):
            say(f"A siglip - {al:.2f}*cf_{how:4s}   " + metr(S - al * A, gt))
    top = S.topk(K, 1).indices
    s1_top = S.gather(1, top)
    gt_in = top == gt[:, None]

    def rank_k(score):
        g = torch.where(gt_in, score, torch.full_like(score, float("-inf"))).max(1).values
        r = (score > g[:, None]).sum(1) + 1
        return torch.where(gt_in.any(1), r, torch.full_like(r, 999))

    # ---------- B. BLIP ITC
    from submission_ens.model import _load_blip, _blip_image_embeds, _blip_itm, _to_pixels, CLIP_MEAN, CLIP_STD
    a224 = np.stack([np.asarray(im) for im in imgs])

    def itc(blip, tokenizer, tag):
        vdt = blip.vision_model.embeddings.patch_embedding.weight.dtype
        t = time.time()
        with torch.inference_mode():
            E = torch.cat([_blip_image_embeds(blip, _to_pixels(a224[s:s+48], CLIP_MEAN, CLIP_STD, dev, vdt)) for s in range(0, N, 48)])
            print(f"{tag} vit {time.time()-t:.0f}s E {E.dtype} qformer {next(blip.qformer.parameters()).dtype}", flush=True)
            qd = next(blip.qformer.parameters()).dtype
            t = time.time(); IQ = []
            for s in range(0, N, 64):
                e = E[s:s+64].to(qd)
                qt = blip.query_tokens.expand(e.shape[0], -1, -1)
                im = torch.ones(e.shape[:2], dtype=torch.long, device=dev)
                h = blip.qformer(query_embeds=qt, encoder_hidden_states=e, encoder_attention_mask=im, return_dict=True).last_hidden_state
                IQ.append(F.normalize(blip.vision_projection(h.to(blip.vision_projection.weight.dtype)).float(), dim=-1))
            IQ = torch.cat(IQ)
            tok = tokenizer(whole, padding="longest", truncation=True, max_length=128, return_tensors="pt")
            ids, mask = tok.input_ids.to(dev), tok.attention_mask.to(dev)
            T = []
            for s in range(0, nq, 128):
                qe = blip.embeddings(input_ids=ids[s:s+128]).to(qd)
                h = blip.qformer(query_embeds=qe, query_length=0, attention_mask=mask[s:s+128], return_dict=True).last_hidden_state
                T.append(F.normalize(blip.text_projection(h[:, 0].to(blip.text_projection.weight.dtype)).float(), dim=-1))
            T = torch.cat(T)
            Si = torch.cat([torch.einsum("qd,nkd->qnk", T[s:s+100], IQ).max(-1).values for s in range(0, nq, 100)])
        say(f"B {tag} ITC alone ({time.time()-t:.0f}s)   " + metr(Si, gt))
        for w in (0.5, 1.0, 2.0):
            say(f"B siglip + {w:.1f}*{tag} ITC (z)   " + metr(zrow(S) + w * zrow(Si), gt))
        return E, Si

    blip, bproc = _load_blip(Path(a.blip), dev)
    try:
        E, Si = itc(blip, bproc.tokenizer, "A-blip")
    except Exception as ex:
        say(f"B failed: {type(ex).__name__}: {str(ex)[:200]}"); raise

    # ---------- C/D. ITM 속도 + 반사실 대조
    def itm(texts, rows, autocast, pb=384):
        tok = bproc.tokenizer(texts, padding="longest", truncation=True, max_length=128, return_tensors="pt")
        ids, mask = tok.input_ids.to(dev), tok.attention_mask.to(dev)
        lens = mask.sum(1)
        order = lens.argsort()
        pt = order.repeat_interleave(K)
        pi = top[torch.tensor(rows, device=dev)][order].reshape(-1)
        out = torch.empty(len(texts) * K, device=dev)
        torch.cuda.synchronize(); t = time.time()
        with torch.inference_mode():
            for s in range(0, len(pt), pb):
                t_, i_ = pt[s:s+pb], pi[s:s+pb]
                L = int(lens[t_].max())
                with torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
                    v = _blip_itm(blip, E[i_], ids[t_, :L], mask[t_, :L])
                out[s:s+pb] = v.float()
        torch.cuda.synchronize(); dt = time.time() - t
        res = torch.empty(len(texts), K, device=dev)
        res[order] = out.view(len(texts), K)
        return res, len(pt) / dt

    try:
        B32, sp32 = itm(whole, list(range(nq)), False)
        say(f"C ITM fp32   {sp32:6.0f} pairs/s  " + metr_rank(rank_k(B32)))
        B16, sp16 = itm(whole, list(range(nq)), True)
        agree = (B32.argmax(1) == B16.argmax(1)).float().mean().item()
        say(f"C ITM fp16ac {sp16:6.0f} pairs/s  " + metr_rank(rank_k(B16)) + f" | top1 agree {agree:.3f} maxdiff {(B32-B16).abs().max().item():.3f} nan {torch.isnan(B16).sum().item()}")
        use16 = agree > 0.97 and not torch.isnan(B16).any()
        Bw = B16 if use16 else B32
        Bc, _ = itm(cf, own, use16)
        say("D ITM whole                " + metr_rank(rank_k(Bw)))
        for how in ("mean", "max"):
            if how == "mean":
                A = torch.zeros(nq, K, device=dev).index_add_(0, own_t, Bc) / cnt.clamp(min=1)
            else:
                A = torch.full((nq, K), float("-inf"), device=dev).scatter_reduce(0, own_t[:, None].expand_as(Bc), Bc, reduce="amax")
            A = torch.where(cnt > 0, A, Bw)
            for al in (0.25, 0.5, 1.0):
                say(f"D ITM - {al:.2f}*cf_{how:4s}      " + metr_rank(rank_k(Bw - al * A)))
        say("D ITM + ITC(z,0.5)          " + metr_rank(rank_k(zrow(Bw) + 0.5 * zrow(Si.gather(1, top)))))
        say("D ITM*.9 + siglip*.1 (z)    " + metr_rank(rank_k(0.9 * zrow(Bw) + 0.1 * zrow(s1_top))))
    except Exception as ex:
        say(f"C/D failed: {type(ex).__name__}: {str(ex)[:200]}")

    # ---------- E. zero-shot BLIP ITC
    if a.blip0:
        try:
            del blip, E; torch.cuda.empty_cache()
            from transformers import AutoProcessor as AP, Blip2ForImageTextRetrieval
            b0 = Blip2ForImageTextRetrieval.from_pretrained(a.blip0, dtype=torch.float16).to(dev).eval()
            for name in ("qformer",):
                getattr(b0, name).float()
            b0.query_tokens.data = b0.query_tokens.data.float()
            b0.embeddings.float(); b0.vision_projection.float(); b0.text_projection.float()
            itc(b0, AP.from_pretrained(a.blip0).tokenizer, "zs-blip")
        except Exception as ex:
            say(f"E failed: {type(ex).__name__}: {str(ex)[:200]}")
    say("S1XDONE")


if __name__ == "__main__":
    main()

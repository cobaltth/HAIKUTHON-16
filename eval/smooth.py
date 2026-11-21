"""이웃 점수 합산(조합표 구조 활용) 실험: valA 800 × 후보 4,000.
 질문의 후보(top-K) 안에서, 각 이미지 점수에 '이미지끼리 닮은 이웃'의 점수를 더하면 정답이 더 잘 올라오는가.
 x~_i = x_i + beta * sum_j w_ij x_j,  w = softmax(g_ij / tau) (j != i, 후보 안)
 base x: SigLIP(a1) 점수 / BLIP ITM 점수.  g: SigLIP 이미지 임베딩 / BLIP ViT-g 평균 / BLIP Q-Former ITC / DINOv2
"""
import argparse, json, sys, time
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
    (ROOT / "runs/smooth.txt").write_text("\n".join(OUT) + "\n")


def zrow(x):
    return (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--siglip", required=True)
    ap.add_argument("--blip", required=True)
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
    nq, N = len(qids), len(pool)

    def load(i):
        with Image.open(data / "images" / f"{i}.png") as im:
            return im.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
    with ThreadPoolExecutor(4) as ex:
        imgs = list(ex.map(load, pool))

    from transformers import AutoProcessor, SiglipModel
    m = SiglipModel.from_pretrained(a.siglip, dtype=torch.float16).to(dev).eval()
    p = AutoProcessor.from_pretrained(a.siglip)
    po = lambda o: o.pooler_output if hasattr(o, "pooler_output") else o
    with torch.inference_mode():
        I = torch.cat([F.normalize(po(m.get_image_features(pixel_values=p(images=imgs[s:s+128], return_tensors="pt").pixel_values.to(dev, torch.float16))).float(), dim=-1) for s in range(0, N, 128)])
        T = torch.cat([F.normalize(po(m.get_text_features(**p(text=whole[s:s+256], padding="max_length", truncation=True, return_tensors="pt").to(dev))).float(), dim=-1) for s in range(0, nq, 256)])
    del m; torch.cuda.empty_cache()
    S = (T @ I.T).clone(); I = I.clone()

    from submission_ens.model import _load_blip, _blip_image_embeds, _blip_itm, _to_pixels, CLIP_MEAN, CLIP_STD
    blip, bproc = _load_blip(Path(a.blip), dev)
    a224 = np.stack([np.asarray(im) for im in imgs])
    vdt = blip.vision_model.embeddings.patch_embedding.weight.dtype
    qd = next(blip.qformer.parameters()).dtype
    with torch.inference_mode():
        E = torch.cat([_blip_image_embeds(blip, _to_pixels(a224[s:s+48], CLIP_MEAN, CLIP_STD, dev, vdt)) for s in range(0, N, 48)])
        Gv = F.normalize(E[:, 1:].float().mean(1), dim=-1).clone()
        IQ = []
        for s in range(0, N, 64):
            e = E[s:s+64].to(qd)
            qt = blip.query_tokens.expand(e.shape[0], -1, -1)
            h = blip.qformer(query_embeds=qt, encoder_hidden_states=e, encoder_attention_mask=torch.ones(e.shape[:2], dtype=torch.long, device=dev), return_dict=True).last_hidden_state
            hp = F.normalize(blip.vision_projection(h.to(blip.vision_projection.weight.dtype)).float(), dim=-1)
            IQ.append(F.normalize(hp.reshape(hp.shape[0], -1), dim=-1))
        Gq = torch.cat(IQ).clone()
    feats = {"siglip": I, "vitg": Gv, "qformer": Gq}
    try:
        from transformers import AutoImageProcessor, AutoModel
        dm = AutoModel.from_pretrained("facebook/dinov2-small", dtype=torch.float16).to(dev).eval()
        dp = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        with torch.inference_mode():
            o = []
            for s in range(0, N, 128):
                h = dm(pixel_values=dp(images=imgs[s:s+128], return_tensors="pt").pixel_values.to(dev, torch.float16)).last_hidden_state
                o.append(F.normalize(torch.cat([h[:, 0], h[:, 1:].mean(1)], -1).float(), dim=-1))
        feats["dino"] = torch.cat(o).clone(); del dm
    except Exception as ex:
        print("dino skip", ex)
    torch.cuda.empty_cache()

    for K in (8, 16, 32):
        top = S.topk(K, 1).indices
        gt_in = top == gt[:, None]
        tok = bproc.tokenizer(whole, padding="longest", truncation=True, max_length=128, return_tensors="pt")
        ids, mask = tok.input_ids.to(dev), tok.attention_mask.to(dev)
        B = torch.empty(nq, K, device=dev)
        with torch.inference_mode():
            for s in range(0, nq, 12):
                qi = torch.arange(s, min(s + 12, nq), device=dev)
                t_ = qi.repeat_interleave(K); i_ = top[qi].reshape(-1)
                with torch.autocast("cuda", dtype=torch.float16):
                    B[qi] = _blip_itm(blip, E[i_], ids[t_], mask[t_]).float().view(len(qi), K)

        def met(score):
            g = torch.where(gt_in, score, torch.full_like(score, float("-inf"))).max(1).values
            r = torch.where(gt_in.any(1), (score > g[:, None]).sum(1) + 1, torch.full((nq,), 999, device=dev))
            rf = r.float()
            ndcg = torch.where(r <= 5, 1 / torch.log2(rf + 1), torch.zeros_like(rf)).mean().item()
            r1 = (r <= 1).float().mean().item()
            return (r1 + ndcg) / 2, r1, (r <= 5).float().mean().item()

        for bname, X in (("siglip", zrow(S.gather(1, top))), ("blipITM", zrow(B.clone()))):
            b0 = met(X)
            say(f"K{K} {bname:8s} base            score {b0[0]:.4f} R@1 {b0[1]:.3f} R@5 {b0[2]:.3f}")
            for gname, Fm in feats.items():
                f = Fm[top]                                   # (nq, K, d)
                g = torch.einsum("qkd,qjd->qkj", f, f)
                g = g - torch.eye(K, device=dev)[None] * 1e4  # 자기 자신 제외
                best = None
                for tau in (0.01, 0.03, 0.1):
                    w = torch.softmax(g / tau, dim=-1)
                    nb = torch.einsum("qkj,qj->qk", w, X)
                    for beta in (0.25, 0.5, 1.0, 2.0):
                        r = met(X + beta * nb)
                        if best is None or r[0] > best[0][0]:
                            best = (r, tau, beta)
                r, tau, beta = best
                say(f"K{K} {bname:8s} + nbr[{gname:7s}] best score {r[0]:.4f} R@1 {r[1]:.3f} R@5 {r[2]:.3f} (d {r[0]-b0[0]:+.4f}, tau {tau}, beta {beta})")
    say("SMDONE")


if __name__ == "__main__":
    main()

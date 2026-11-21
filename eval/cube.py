"""조합표(cube) 좌표 + 축별 평균 실험 (valA 800 × 4,000, 세트 = 1단계 top-8, 축 3개).
 C의 v10 _cube_axes(세트 평균을 뺀 이미지 임베딩의 주성분 → 이진 축)를 그대로 써서
 (1) 깔끔한 조합표로 인정되는 세트 비율, (2) BLIP/SigLIP 점수를 축별 평균(가법 적합)으로 바꿨을 때 R@1 변화를 잰다.
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
    (ROOT / "runs/cube.txt").write_text("\n".join(OUT) + "\n")


def _cube_axes(G, r_list=(5, 4, 3), tau=0.25, sweeps=6, n_ang=32):
    """Binary axes of each family from its centred Gram matrix G (f, n, n) of image embeddings.

    If the n images are all combinations of r two-valued factors and each factor shifts the embedding by a fixed
    vector, the top-r principal components span the factors, and the factors are the rotation of those components
    whose entries all have equal magnitude (minimum sum of 4th powers). Returns code (f, n) with the r-bit
    combination of every image (-1 if no clean grid) and m (f,) = r used (0 if none). A grid is accepted only if every
    axis splits n/2 : n/2, every |coordinate| >= tau and every combination occurs exactly n / 2**r times.
    """
    f, n, _ = G.shape
    _, U = np.linalg.eigh(G.astype(np.float64))
    U = U[:, :, ::-1]
    codes = np.full((f, n), -1, dtype=np.int64)
    m_of = np.zeros(f, dtype=np.int64)
    ang = np.linspace(0.0, np.pi / 2, n_ang, endpoint=False)
    ca, sa = np.cos(ang)[None, :, None], np.sin(ang)[None, :, None]
    todo = np.ones(f, dtype=bool)
    for r in r_list:
        if n % (1 << r) or (n >> r) > 4 or not todo.any():
            continue
        idx = np.where(todo)[0]
        ar = np.arange(len(idx))
        Y = U[idx][:, :, :r] * np.sqrt(n)
        for _ in range(sweeps):
            for a in range(r):
                for b in range(a + 1, r):
                    ya, yb = Y[:, None, :, a], Y[:, None, :, b]
                    A, B = ya * ca + yb * sa, yb * ca - ya * sa          # (f, n_ang, n)
                    k = ((A ** 4).sum(-1) + (B ** 4).sum(-1)).argmin(1)
                    Y[:, :, a], Y[:, :, b] = A[ar, k], B[ar, k]
        bits = Y > 0
        code = (bits * (1 << np.arange(r))).sum(-1)
        cnt = np.stack([(code == c).sum(1) for c in range(1 << r)], 1)
        ok = (bits.sum(1) == n // 2).all(1) & (np.abs(Y).min(1) >= tau).all(1) & (cnt == (n >> r)).all(1)
        g = idx[ok]
        codes[g], m_of[g] = code[ok], r
        todo[g] = False
    return codes, m_of


def zr(x):
    return (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--siglip", required=True)
    ap.add_argument("--blip", required=True)
    ap.add_argument("--s2", default="")
    ap.add_argument("--k", type=int, default=8)
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

    def load(i):
        with Image.open(data / "images" / f"{i}.png") as im:
            return im.convert("RGB").resize((256, 256), Image.Resampling.BICUBIC)
    with ThreadPoolExecutor(4) as ex:
        im256 = list(ex.map(load, pool))
    imgs = [im.resize((224, 224), Image.Resampling.BICUBIC) for im in im256]

    from transformers import AutoProcessor, SiglipModel
    m = SiglipModel.from_pretrained(a.siglip, dtype=torch.float16).to(dev).eval()
    p = AutoProcessor.from_pretrained(a.siglip)
    po = lambda o: o.pooler_output if hasattr(o, "pooler_output") else o
    with torch.inference_mode():
        I = torch.cat([F.normalize(po(m.get_image_features(pixel_values=p(images=imgs[s:s+128], return_tensors="pt").pixel_values.to(dev, torch.float16))).float(), dim=-1) for s in range(0, N, 128)])
        T = torch.cat([F.normalize(po(m.get_text_features(**p(text=whole[s:s+256], padding="max_length", truncation=True, return_tensors="pt").to(dev))).float(), dim=-1) for s in range(0, nq, 256)])
    del m; torch.cuda.empty_cache()
    S = (T @ I.T).clone(); I = I.clone()
    feats = {"siglip": I}

    if a.s2 and Path(a.s2).exists():
        try:
            from transformers import AutoModel
            m2 = AutoModel.from_pretrained(a.s2, dtype=torch.float16).to(dev).eval()
            native = int(m2.config.vision_config.image_size)
            kw = {"interpolate_pos_encoding": True} if 256 != native else {}
            o = []
            with torch.inference_mode():
                for s in range(0, N, 64):
                    arr = np.stack([np.asarray(im, dtype=np.uint8) for im in im256[s:s+64]])
                    x = torch.from_numpy(arr).to(dev).permute(0, 3, 1, 2).float().div_(255.0)
                    x = ((x - 0.5) / 0.5).half()
                    e = m2.get_image_features(pixel_values=x, **kw)
                    e = e if torch.is_tensor(e) else (e.pooler_output if getattr(e, "pooler_output", None) is not None else e[0])
                    o.append(F.normalize(e.float(), dim=-1))
            feats["siglip2"] = torch.cat(o).clone(); del m2; torch.cuda.empty_cache()
        except Exception as ex:
            say(f"siglip2 skip: {type(ex).__name__}: {str(ex)[:120]}")

    from submission_ens.model import _load_blip, _blip_image_embeds, _blip_itm, _to_pixels, CLIP_MEAN, CLIP_STD
    blip, bproc = _load_blip(Path(a.blip), dev)
    a224 = np.stack([np.asarray(im) for im in imgs])
    vdt = blip.vision_model.embeddings.patch_embedding.weight.dtype
    with torch.inference_mode():
        E = torch.cat([_blip_image_embeds(blip, _to_pixels(a224[s:s+48], CLIP_MEAN, CLIP_STD, dev, vdt)) for s in range(0, N, 48)])
        feats["vitg_mean"] = F.normalize(E[:, 1:].float().mean(1), dim=-1).clone()
        feats["vitg_cls"] = F.normalize(E[:, 0].float(), dim=-1).clone()
    top = S.topk(K, 1).indices
    gt_in = (top == gt[:, None])
    tok = bproc.tokenizer(whole, padding="longest", truncation=True, max_length=128, return_tensors="pt")
    ids, mask = tok.input_ids.to(dev), tok.attention_mask.to(dev)
    B = torch.empty(nq, K, device=dev)
    with torch.inference_mode():
        for s in range(0, nq, 32):
            qi = torch.arange(s, min(s + 32, nq), device=dev)
            t_ = qi.repeat_interleave(K); i_ = top[qi].reshape(-1)
            with torch.autocast("cuda", dtype=torch.float16):
                B[qi] = _blip_itm(blip, E[i_], ids[t_], mask[t_]).float().view(len(qi), K)
    Bn = B.cpu().numpy().astype(np.float64); Sn = S.gather(1, top).cpu().numpy().astype(np.float64)
    gin = gt_in.cpu().numpy()
    has = gin.any(1); gpos = gin.argmax(1)

    def met(score, sel=None):
        g = np.where(gin, score, -1e18).max(1)
        r = np.where(has, (score > g[:, None]).sum(1) + 1, 999)
        if sel is not None:
            r = r[sel]
        ndcg = np.where(r <= 5, 1 / np.log2(r + 1.0), 0).mean()
        return f"score {((r<=1).mean()+ndcg)/2:.4f} R@1 {(r<=1).mean():.3f}"

    say(f"K={K} 정답이 세트 안 {has.mean():.3f} | BLIP 기준 {met(zr(Bn))} | SigLIP 기준 {met(zr(Sn))}")
    for fname, Fm in feats.items():
        f = Fm[top].double()
        f = f - f.mean(1, keepdim=True)
        G = (f @ f.transpose(1, 2)).cpu().numpy()
        for tau in (0.25, 0.1, 0.0):
            code, mo = _cube_axes(G, r_list=(3, 2), tau=tau)
            say(f"[{fname:9s} tau {tau}] 축 3개 {np.mean(mo==3):.3f} / 축 2개 {np.mean(mo==2):.3f} / 실패 {np.mean(mo==0):.3f}")
            for bname, X in (("BLIP", Bn), ("SigLIP", Sn)):
                for mmin in (3, 2):
                    sel = mo >= mmin
                    if sel.sum() < 20:
                        continue
                    res = []
                    for w in (1.0, 3.0, 1e6):
                        sc = zr(X).copy()
                        for q in np.where(sel)[0]:
                            m_ = int(mo[q]); bits = (code[q][:, None] >> np.arange(m_)) & 1
                            d = np.array([X[q][bits[:, a_] == 1].mean() - X[q][bits[:, a_] == 0].mean() for a_ in range(m_)])
                            fit = ((bits * 2 - 1) * d).sum(1) / 2
                            sc[q] = zr(X[q]) + w * zr(fit) if w < 1e5 else zr(fit) + 1e-3 * zr(X[q])
                        res.append(f"w{w:g}: 전체 {met(sc)} | 해당 {met(sc, sel)}")
                    say(f"   {bname:6s} 축≥{mmin} (n={int(sel.sum())}) 기준 해당 {met(zr(X), sel)} || " + " || ".join(res))
    say("CUBEDONE")


if __name__ == "__main__":
    main()

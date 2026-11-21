"""형제 그룹 찾기 도구 비교: 문장 순서는 a1 SigLIP, 이미지끼리 유사도는 여러 특징으로.
특징: siglip(a1 이미지 임베딩) / thumb(32x32 RGB 픽셀) / thumb16g(16x16 흑백) / hist(RGB 8x8x8 색 히스토그램) / dino(facebook/dinov2-small CLS)
지표: 형제 정밀도@(g-1), 후보 B장(3,5,8)에서 plain vs 확장(gallery, local40) 정답 포함률, 오라클 그룹.
"""
import argparse, csv, json, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def expand(order_row, m, nbr_fn, budget):
    out, seen = [], set()
    for s in order_row[:m]:
        for j in [s] + nbr_fn(s):
            if j not in seen:
                seen.add(j); out.append(j)
                if len(out) >= budget:
                    return out
    for j in order_row:
        if j not in seen:
            seen.add(j); out.append(j)
            if len(out) >= budget:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="out/a1_save/submission/models")
    ap.add_argument("--feats", default="siglip,thumb,thumb16g,hist,dino")
    ap.add_argument("--split-file", default=str(ROOT / "splits/split_group_seed42.json"))
    ap.add_argument("--groups", default=str(ROOT / "eda/groups.csv"))
    a = ap.parse_args()
    dev = "cuda"
    data = Path(a.data)
    caps = {}
    for line in open(data / "prompts.jsonl", encoding="utf-8"):
        if line.strip():
            r = json.loads(line); caps[r["id"]] = r["prompts"]
    pool = sorted(caps)
    col = {i: k for k, i in enumerate(pool)}
    qids = list(json.load(open(a.split_file))["valA"])
    gt = [col[q] for q in qids]
    grp = {r["id"]: int(r["group_id"]) for r in csv.DictReader(open(a.groups))}
    g_of = [grp[i] for i in pool]
    members = {}
    for k, g in enumerate(g_of):
        members.setdefault(g, []).append(k)

    t = time.time()
    def load(i):
        with Image.open(data / "images" / f"{i}.png") as im:
            return im.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
    with ThreadPoolExecutor(4) as ex:
        imgs = list(ex.map(load, pool))
    print(f"decode {time.time()-t:.0f}s", flush=True)

    from transformers import AutoProcessor, SiglipModel
    m = SiglipModel.from_pretrained(a.model, dtype=torch.float16).to(dev).eval()
    p = AutoProcessor.from_pretrained(a.model)
    with torch.inference_mode():
        fi, ft = [], []
        for s in range(0, len(imgs), 128):
            px = p(images=imgs[s:s + 128], return_tensors="pt").pixel_values.to(dev, torch.float16)
            o = m.get_image_features(pixel_values=px)
            fi.append(F.normalize((o.pooler_output if hasattr(o, "pooler_output") else o).float(), dim=-1))
        cq = [caps[q] for q in qids]
        for s in range(0, len(cq), 256):
            tok = p(text=cq[s:s + 256], padding="max_length", truncation=True, return_tensors="pt").to(dev)
            o = m.get_text_features(**tok)
            ft.append(F.normalize((o.pooler_output if hasattr(o, "pooler_output") else o).float(), dim=-1))
    del m; torch.cuda.empty_cache()
    fi = torch.cat(fi); ft = torch.cat(ft)
    order = (ft @ fi.T).argsort(1, descending=True)[:, :200].tolist()

    feats = {}
    want = a.feats.split(",")
    if "siglip" in want:
        feats["siglip"] = fi
    if "thumb" in want:
        x = torch.tensor(np.stack([np.asarray(im.resize((32, 32), Image.Resampling.BOX), dtype=np.float32) for im in imgs])).reshape(len(imgs), -1)
        feats["thumb"] = F.normalize(x - x.mean(1, keepdim=True), dim=-1).to(dev)
    if "thumb16g" in want:
        x = torch.tensor(np.stack([np.asarray(im.convert("L").resize((16, 16), Image.Resampling.BOX), dtype=np.float32) for im in imgs])).reshape(len(imgs), -1)
        feats["thumb16g"] = F.normalize(x - x.mean(1, keepdim=True), dim=-1).to(dev)
    if "hist" in want:
        hs = []
        for im in imgs:
            q = (np.asarray(im.resize((64, 64)), dtype=np.int32) // 32)
            idx = q[..., 0] * 64 + q[..., 1] * 8 + q[..., 2]
            hs.append(np.bincount(idx.ravel(), minlength=512).astype(np.float32))
        feats["hist"] = F.normalize(torch.tensor(np.stack(hs)).sqrt(), dim=-1).to(dev)
    if "dino" in want:
        from transformers import AutoImageProcessor, AutoModel
        dm = AutoModel.from_pretrained("facebook/dinov2-small", dtype=torch.float16).to(dev).eval()
        dp = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        out = []
        with torch.inference_mode():
            for s in range(0, len(imgs), 128):
                px = dp(images=imgs[s:s + 128], return_tensors="pt").pixel_values.to(dev, torch.float16)
                h = dm(pixel_values=px).last_hidden_state
                out.append(F.normalize(torch.cat([h[:, 0], h[:, 1:].mean(1)], -1).float(), dim=-1))
        feats["dino"] = torch.cat(out)
        del dm; torch.cuda.empty_cache()

    lines = []
    plain = {B: sum(gt[i] in order[i][:B] for i in range(len(qids))) / len(qids) for B in (3, 5, 8)}
    orc = {B: sum((gt[i] in order[i][:B]) or any(g_of[j] == g_of[gt[i]] for j in order[i][:B]) for i in range(len(qids))) / len(qids) for B in (3, 5, 8)}
    lines.append("plain " + " ".join(f"B{B} {plain[B]:.3f}" for B in plain) + " | orc " + " ".join(f"B{B} {orc[B]:.3f}" for B in orc))
    for name, f in feats.items():
        ii = f @ f.T
        ii.fill_diagonal_(-9)
        nn = ii.argsort(1, descending=True)[:, :16].tolist()
        prec = []
        for k in range(len(pool)):
            g = len(members[g_of[k]])
            if g < 2:
                continue
            sib = set(members[g_of[k]]) - {k}
            prec.append(len(sib & set(nn[k][:g - 1])) / (g - 1))
        best = {}
        for B in (3, 5, 8):
            for mm in (1, 2):
                for nb in (2, 4, 7):
                    gal = sum(gt[i] in expand(order[i], mm, lambda s: nn[s][:nb], B) for i in range(len(qids))) / len(qids)
                    hits = 0
                    for i in range(len(qids)):
                        cand = torch.tensor(order[i][:40], device=dev)
                        def loc(s, cand=cand, nb=nb):
                            return cand[ii[s, cand].argsort(descending=True)[:nb]].tolist()
                        hits += gt[i] in expand(order[i], mm, loc, B)
                    for kind, v in (("g", gal), ("l", hits / len(qids))):
                        key = (B, kind)
                        if key not in best or v > best[key][0]:
                            best[key] = (v, f"m{mm}n{nb}")
        lines.append(f"{name:8s} sibP {sum(prec)/len(prec):.3f} " + " ".join(
            f"B{B} g{best[(B,'g')][0]:.3f} l{best[(B,'l')][0]:.3f}" for B in (3, 5, 8)))
        print(lines[-1], flush=True)
    Path(ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / "runs/group_expand2.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()

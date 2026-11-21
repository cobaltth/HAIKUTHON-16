"""형제 그룹 확장 효과 측정 (학습 데이터, valA 800 질문 × 후보 4,000장).

질문마다 같은 후보 예산 B장 안에 정답이 들어오는 비율(재정렬 상한)을 비교:
  plain   : 문장-이미지 유사도 상위 B장
  gallery : (원본) 상위 m장을 씨앗으로, 갤러리 전체에서 이미지끼리 닮은 이웃을 붙임
  local   : (안전) 씨앗의 이웃을 그 질문의 상위 L장 안에서만 찾음
  oracle  : 정답 그룹(groups.csv)의 이미지가 상위 B장 안에 하나라도 있으면 그룹 전체를 넣는다고 가정한 상한
또 이미지끼리 유사도만으로 형제 그룹이 얼마나 복원되는지(이웃 정밀도) 잰다.

예) python eval/group_expand.py --data /kaggle/working/haikuthon-train \
      --model google/siglip-base-patch16-224 --model out/a1_save/submission/models
"""
import argparse, csv, json, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


@torch.inference_mode()
def encode(path, caps, imgs, dev):
    from transformers import AutoProcessor, SiglipModel
    m = SiglipModel.from_pretrained(path, dtype=torch.float16).to(dev).eval()
    p = AutoProcessor.from_pretrained(path)
    fi, ft = [], []
    for s in range(0, len(imgs), 128):
        px = p(images=imgs[s:s + 128], return_tensors="pt").pixel_values.to(dev, torch.float16)
        o = m.get_image_features(pixel_values=px)
        fi.append(F.normalize((o.pooler_output if hasattr(o, "pooler_output") else o).float(), dim=-1))
    for s in range(0, len(caps), 256):
        tok = p(text=caps[s:s + 256], padding="max_length", truncation=True, return_tensors="pt").to(dev)
        o = m.get_text_features(**tok)
        ft.append(F.normalize((o.pooler_output if hasattr(o, "pooler_output") else o).float(), dim=-1))
    del m; torch.cuda.empty_cache()
    return torch.cat(ft), torch.cat(fi)


def expand(order_row, seeds_m, nbr_fn, budget):
    """order_row: 문장 유사도 순 이미지 번호. 씨앗 m장마다 [씨앗, 이웃들...]을 붙이고, 남으면 plain 순서로 채움."""
    out, seen = [], set()
    for s in order_row[:seeds_m]:
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
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--split-file", default=str(ROOT / "splits/split_group_seed42.json"))
    ap.add_argument("--groups", default=str(ROOT / "eda/groups.csv"))
    ap.add_argument("--save", default=str(ROOT / "runs/group_expand.json"))
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
    print(f"decode {len(imgs)} {time.time()-t:.0f}s", flush=True)

    report = {}
    for path in a.model:
        name = "/".join(Path(path).parts[-3:]) if Path(path).exists() else path.split("/")[-1]
        ft, fi = encode(path, [caps[q] for q in qids], imgs, dev)
        sim = ft @ fi.T                       # (Q, N)
        order = sim.argsort(1, descending=True)[:, :200].tolist()
        ii = fi @ fi.T                        # 이미지끼리
        ii.fill_diagonal_(-1)
        nn = ii.argsort(1, descending=True)[:, :16].tolist()
        nn_s = ii.sort(1, descending=True).values[:, :16].tolist()

        # 1) 형제 복원 정밀도: 그룹 크기 g>1인 이미지의 이웃 상위 (g-1)개 중 진짜 형제 비율
        prec, hit7 = [], []
        for k in range(len(pool)):
            g = len(members[g_of[k]])
            if g < 2:
                continue
            sib = set(members[g_of[k]]) - {k}
            prec.append(len(sib & set(nn[k][:g - 1])) / (g - 1))
            hit7.append(len(sib & set(nn[k][:7])) / len(sib))
        r = {"sib_precision@(g-1)": sum(prec) / len(prec), "sib_recall@7nbr": sum(hit7) / len(hit7)}
        # 형제 vs 비형제 이웃 유사도 (임계값 고르기용)
        sib_s, non_s = [], []
        for k in range(len(pool)):
            sib = set(members[g_of[k]])
            for j, s in zip(nn[k][:8], nn_s[k][:8]):
                (sib_s if j in sib else non_s).append(s)
        sib_s.sort(); non_s.sort()
        q = lambda v, p: v[int(p * (len(v) - 1))] if v else float("nan")
        r["nbr_sim_sib_p10/p50"] = (round(q(sib_s, .1), 3), round(q(sib_s, .5), 3))
        r["nbr_sim_non_p50/p90"] = (round(q(non_s, .5), 3), round(q(non_s, .9), 3))

        # 2) 같은 후보 예산 B 안의 정답 포함률
        for B in (3, 5, 8, 10, 15, 20):
            plain = sum(gt[i] in order[i][:B] for i in range(len(qids))) / len(qids)
            orc = 0
            for i in range(len(qids)):
                top = order[i][:B]
                orc += (gt[i] in top) or any(g_of[j] == g_of[gt[i]] for j in top)
            r[f"B{B}_plain"] = plain
            r[f"B{B}_oracle_group"] = orc / len(qids)
            for m in (1, 2, 3):
                for nb in (4, 7):
                    gal = sum(gt[i] in expand(order[i], m, lambda s: nn[s][:nb], B) for i in range(len(qids))) / len(qids)
                    r[f"B{B}_gallery_m{m}_n{nb}"] = gal
                    hits = 0
                    for i in range(len(qids)):
                        cand = torch.tensor(order[i][:40], device=dev)
                        def loc(s, cand=cand, nb=nb):
                            v = ii[s, cand]
                            return cand[v.argsort(descending=True)[:nb]].tolist()
                        hits += gt[i] in expand(order[i], m, loc, B)
                    r[f"B{B}_local40_m{m}_n{nb}"] = hits / len(qids)
        report[name] = r
        print(f"\n== {name}", flush=True)
        for k, v in r.items():
            print(f"  {k}: {v if not isinstance(v, float) else round(v, 4)}", flush=True)
    Path(a.save).parent.mkdir(exist_ok=True)
    json.dump(report, open(a.save, "w"), ensure_ascii=False, indent=1)
    lines = []
    for name, r in report.items():
        lines.append(f"{name[-28:]} sibP {r['sib_precision@(g-1)']:.3f} sibR7 {r['sib_recall@7nbr']:.3f} "
                     f"sim sib {r['nbr_sim_sib_p10/p50']} non {r['nbr_sim_non_p50/p90']}")
        for B in (3, 5, 8, 10, 15, 20):
            gk = max((k for k in r if k.startswith(f"B{B}_gallery")), key=lambda k: r[k])
            lk = max((k for k in r if k.startswith(f"B{B}_local40")), key=lambda k: r[k])
            lines.append(f"B{B} plain {r[f'B{B}_plain']:.3f} orc {r[f'B{B}_oracle_group']:.3f} "
                         f"gal {r[gk]:.3f}({gk.split('_',2)[2]}) loc {r[lk]:.3f}({lk.split('_',2)[2]})")
    Path(a.save).with_suffix(".txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

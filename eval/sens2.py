"""장면 단어(조명·시점·화풍) 민감도 검사: 정답 이미지 I 고정, 질의 Q의 한 군데만 틀리게 바꾼 Q'와 비교.
 모델이 그 속성을 읽으면 s(Q,I) > s(Q',I). 비율이 0.5면 못 읽는 것. (C의 reranker/sens.py와 같은 변형 규칙)
 대상: SigLIP (zero-shot / A 학습본 / a1), BLIP-2 ITM (A 학습본)
"""
import argparse, json, random, re, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LIGHT = ["side lighting", "direct sunlight", "soft diffuse lighting", "warm indoor lighting"]
VIEWS = ["overhead", "front", "side", "oblique"]
STYLES = ["low-poly 3D render", "clay-style 3D render", "watercolor painting", "oil painting", "flat illustration",
          "colored-pencil drawing", "ink drawing", "pencil sketch", "3D render", "photograph"]
STY_TO = ["photograph", "3D render", "watercolor painting", "oil painting", "flat illustration", "pencil sketch"]
OUT = []


def say(s):
    OUT.append(s); print("R> " + s, flush=True)
    (ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / "runs/sens2.txt").write_text("\n".join(OUT) + "\n")


def fix_articles(t):
    t = re.sub(r"\b([Aa])n (?=(?:front|side|photograph|3D|watercolor|flat|colored|pencil|low|clay))", r"\1 ", t)
    t = re.sub(r"\b([Aa]) (?=(?:overhead|oblique|oil|ink))", r"\1n ", t)
    return t


def var_light(q):
    for i, x in enumerate(LIGHT):
        m = re.search(re.escape(x), q, flags=re.I)
        if m:
            return q[:m.start()] + LIGHT[(i + 1) % 4] + q[m.end():]


def var_view(q):
    m = re.search(r"\b(overhead|front|side|oblique)(?=[- ](?:view|angle))", q, flags=re.I) or \
        re.search(r"(?<=from the )(front|side)\b", q, flags=re.I) or re.search(r"(?<=from )(overhead)\b", q, flags=re.I)
    if m:
        i = VIEWS.index(m.group(1).lower())
        new = VIEWS[(i + 1) % 4]
        new = new.capitalize() if m.group(1)[0].isupper() else new
        return fix_articles(q[:m.start(1)] + new + q[m.end(1):])


def var_style(q):
    for x in STYLES:
        m = re.search(re.escape(x), q, flags=re.I)
        if m:
            cand = [s for s in STY_TO if s.lower() not in x.lower() and x.lower() not in s.lower()]
            return fix_articles(q[:m.start()] + cand[len(q) % len(cand)] + q[m.end():])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--siglip", action="append", default=[])
    ap.add_argument("--blip", default="")
    ap.add_argument("--split-file", default=str(ROOT / "splits/split_group_seed42.json"))
    a = ap.parse_args()
    dev = "cuda"
    data = Path(a.data)
    caps = {}
    for line in open(data / "prompts.jsonl", encoding="utf-8"):
        if line.strip():
            r = json.loads(line); caps[r["id"]] = r["prompts"]
    qids = list(json.load(open(a.split_file))["valA"])
    whole = [caps[q] for q in qids]
    from aug.text_aug import hard_negatives
    var, own, kind = [], [], []
    for i, w in enumerate(whole):
        for k, f in (("조명", var_light), ("시점", var_view), ("화풍", var_style)):
            v = f(w)
            if v and v != w:
                var.append(v); own.append(i); kind.append(k)
        for k, c in hard_negatives(w, random.Random(i)):
            var.append(c); own.append(i); kind.append({"color": "색 바꿈", "bind": "색 맞바꿈", "pattern": "무늬", "material": "재질"}.get(k, "관계 방향"))
    own_t = torch.tensor(own, device=dev)
    kinds = ["조명", "시점", "화풍", "색 바꿈", "색 맞바꿈", "관계 방향", "무늬", "재질"]
    say("변형 수 " + " ".join(f"{k} {kind.count(k)}" for k in kinds))

    def report(name, sq, sv):
        """sq: (nq,) 원 질의 점수, sv: (nvar,) 변형 질의 점수"""
        win = (sq[own_t] > sv).float(); drop = sq[own_t] - sv
        scale = sq.std().item() + 1e-9
        parts = []
        for k in kinds:
            idx = torch.tensor([j for j, kk in enumerate(kind) if kk == k], device=dev)
            if len(idx):
                parts.append(f"{k} {win[idx].mean().item():.3f}({drop[idx].mean().item()/scale:+.2f})")
        say(f"{name:14s} " + " | ".join(parts))

    def load(i):
        with Image.open(data / "images" / f"{i}.png") as im:
            return im.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
    with ThreadPoolExecutor(4) as ex:
        imgs = list(ex.map(load, qids))

    from transformers import AutoProcessor, SiglipModel
    for path in a.siglip:
        name = "siglip:" + (Path(path).parts[-3] if Path(path).exists() and len(Path(path).parts) > 3 else path.split("/")[-1])[:7]
        try:
            m = SiglipModel.from_pretrained(path, dtype=torch.float16).to(dev).eval()
            p = AutoProcessor.from_pretrained(path)
            po = lambda o: o.pooler_output if hasattr(o, "pooler_output") else o
            with torch.inference_mode():
                I = torch.cat([F.normalize(po(m.get_image_features(pixel_values=p(images=imgs[s:s+128], return_tensors="pt").pixel_values.to(dev, torch.float16))).float(), dim=-1) for s in range(0, len(imgs), 128)])
                enc = lambda tx: torch.cat([F.normalize(po(m.get_text_features(**p(text=tx[s:s+256], padding="max_length", truncation=True, return_tensors="pt").to(dev))).float(), dim=-1) for s in range(0, len(tx), 256)])
                Tq, Tv = enc(whole), enc(var)
            report(name, (Tq * I).sum(-1).clone(), (Tv * I[own_t]).sum(-1).clone())
            del m; torch.cuda.empty_cache()
        except Exception as ex:
            say(f"{name} failed: {type(ex).__name__}: {str(ex)[:160]}")

    if a.blip:
        from submission_ens.model import _load_blip, _blip_image_embeds, _blip_itm, _to_pixels, CLIP_MEAN, CLIP_STD
        blip, bproc = _load_blip(Path(a.blip), dev)
        a224 = np.stack([np.asarray(im) for im in imgs])
        vdt = blip.vision_model.embeddings.patch_embedding.weight.dtype

        def itm(texts, img_idx):
            tok = bproc.tokenizer(texts, padding="longest", truncation=True, max_length=128, return_tensors="pt")
            ids, mask = tok.input_ids.to(dev), tok.attention_mask.to(dev)
            out = torch.empty(len(texts), device=dev)
            for s in range(0, len(texts), 256):
                with torch.autocast("cuda", dtype=torch.float16):
                    out[s:s+256] = _blip_itm(blip, E[img_idx[s:s+256]], ids[s:s+256], mask[s:s+256]).float()
            return out
        with torch.inference_mode():
            E = torch.cat([_blip_image_embeds(blip, _to_pixels(a224[s:s+48], CLIP_MEAN, CLIP_STD, dev, vdt)) for s in range(0, len(imgs), 48)])
            sq = itm(whole, torch.arange(len(whole), device=dev)); sv = itm(var, own_t)
        report("blip-ITM(A v1)", sq.clone(), sv.clone())
    say("SENSDONE")


if __name__ == "__main__":
    main()

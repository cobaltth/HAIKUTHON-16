#!/usr/bin/env python
"""Is the reranker sensitive to scene-level words (lighting / viewpoint / style)? For each valA query Q and its TRUE image I,
compare s(Q, I) with s(Q', I) where Q' changes exactly one thing. If the model reads that attribute, s(Q') << s(Q)."""
import importlib.util, json, os, re, sys, time
from types import SimpleNamespace
import numpy as np, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cqm_val import counterfactuals

SUB = sys.argv[1] if len(sys.argv) > 1 else "sub_v7"
LIGHT = ["side lighting", "direct sunlight", "soft diffuse lighting", "warm indoor lighting"]
VIEWS = ["overhead", "front", "side", "oblique"]
STYLES = ["low-poly 3D render", "clay-style 3D render", "watercolor painting", "oil painting", "flat illustration",
          "colored-pencil drawing", "ink drawing", "pencil sketch", "3D render", "photograph"]
STY_TO = ["photograph", "3D render", "watercolor painting", "oil painting", "flat illustration", "pencil sketch"]


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
    root = "haikuthon-train"
    recs = [json.loads(l) for l in open(f"{root}/prompts.jsonl", encoding="utf-8") if l.strip()]
    by_id = {r["id"]: r for r in recs}
    sp = json.load(open("splits/split_group_seed42.json"))
    ids = sp["valA"]
    texts = [by_id[i]["prompts"] for i in ids]
    sub = os.path.abspath(SUB)
    spec = importlib.util.spec_from_file_location("sub_model", os.path.join(sub, "model.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    m = mod.load_model(sub, "cuda:0")
    lm, proc, w, dev = m["lm"], m["proc"], m["w"], m["device"]
    rr = int(m["cfg"]["rr_res"]); ip, tk = proc.image_processor, proc.tokenizer
    img_tok, pad_id = lm.config.image_token_id, tk.pad_token_id
    image_token = getattr(proc, "image_token", "<|image_pad|>")
    E = D = grid = None
    with torch.inference_mode():
        for s in range(0, len(ids), 64):
            ims = [mod._to_rgb(Image.open(f"{root}/images/{i}.png")).resize((rr, rr), resample=Image.Resampling.BICUBIC) for i in ids[s:s + 64]]
            enc = ip(images=ims, do_resize=False, return_tensors="pt"); g = enc["image_grid_thw"].to(dev)
            out = lm.model.get_image_features(enc["pixel_values"].to(dev), g, return_dict=True)
            e = torch.stack(list(out.pooler_output)); dd = torch.stack([torch.stack(list(f)) for f in out.deepstack_features])
            if E is None:
                E = torch.empty((len(ids),) + tuple(e.shape[1:]), dtype=e.dtype, device=dev)
                D = torch.empty((dd.shape[0], len(ids)) + tuple(dd.shape[2:]), dtype=dd.dtype, device=dev); grid = g[:1].clone()
            E[s:s + len(ims)] = e; D[:, s:s + len(ims)] = dd
    ntok = E.shape[1]

    def score(text_list, img_idx):
        ids_l = [tk(mod._prompt(m, t).replace(image_token, image_token * ntok), add_special_tokens=False)["input_ids"] for t in text_list]
        order = np.argsort([len(x) for x in ids_l], kind="stable"); res = np.zeros(len(ids_l), dtype=np.float32)
        with torch.inference_mode():
            for s in range(0, len(order), 48):
                b = order[s:s + 48]; L = max(len(ids_l[i]) for i in b)
                idt = torch.full((len(b), L), pad_id, dtype=torch.long); am = torch.zeros((len(b), L), dtype=torch.long)
                for r, i in enumerate(b):
                    x = ids_l[i]; idt[r, L - len(x):] = torch.tensor(x); am[r, L - len(x):] = 1
                idt, am = idt.to(dev), am.to(dev)
                j = torch.as_tensor(np.asarray(img_idx)[b], device=dev); Ej, Dj = E[j], D[:, j]
                feats = SimpleNamespace(pooler_output=(Ej.reshape(-1, Ej.shape[-1]),),
                                        deepstack_features=[(Dj[l].reshape(-1, Dj.shape[-1]),) for l in range(Dj.shape[0])])
                o = lm.model(input_ids=idt, attention_mask=am, image_grid_thw=grid.expand(len(b), -1),
                             mm_token_type_ids=(idt == img_tok).to(torch.int32), mm_encoder_outputs={"image": feats})
                res[b] = (o.last_hidden_state[:, -1].float() @ w).cpu().numpy()
        return res

    base = score(texts, list(range(len(ids))))
    print(f"s(Q, true image): mean {base.mean():.2f} median {np.median(base):.2f}  P(>0) {float((base > 0).mean()):.3f}")
    variants = {"lighting": var_light, "viewpoint": var_view, "style": var_style}
    for kind in ("color", "flip", "count", "pattern", "material"):
        variants[kind] = (lambda q, kind=kind: next((t for k2, t in counterfactuals(q) if k2 == kind), None))
    shown = 0
    for name, fn in variants.items():
        tl, il = [], []
        for qi, q in enumerate(texts):
            v = fn(q)
            if v and v != q:
                tl.append(v); il.append(qi)
        if not tl:
            print(name, "no variants"); continue
        sc = score(tl, il); b = base[il]
        print(f"{name:10s} n={len(tl):4d}  P(s(Q)>s(Q')) {float((b > sc).mean()):.3f}  mean drop {float((b - sc).mean()):6.2f}  "
              f"median s(Q') {np.median(sc):6.2f}  P(s(Q')<0) {float((sc < 0).mean()):.3f}")
        if name in ("lighting", "viewpoint", "style"):
            print("    e.g.", texts[il[0]][-90:], "->", tl[0][-90:])


if __name__ == "__main__":
    main()

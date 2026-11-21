#!/usr/bin/env python
"""Counterfactual Query Margin (CQM) check on valA x 4000 pool with the v7 cached path.

For each query Q and its stage-1 top-K candidates (taken from an eval_v1 dump), score the original query and every
rule-based counterfactual Q- (colour/pattern/material swap between two mentions, relation flip, count +-1).
CQM score = s(Q, I) - lam * agg_m s(Q-_m, I). Also sweeps a stage-1 prior: s(Q, I) + alpha * z(stage1).
Uses only the query's own text and its own candidates.
"""
import importlib.util, json, math, os, re, sys, time
from types import SimpleNamespace
import numpy as np, torch
from PIL import Image

SUB, DUMP, K = "sub_v7", "runs/v4_r192_k8_p4000.npz", 8
COLORS = ("red orange yellow green blue purple pink brown black white gray grey silver golden gold beige teal cyan "
          "magenta navy maroon turquoise violet lime olive tan cream").split()
PATTERNS = "striped polka-dotted checkered plaid spotted dotted floral zigzag patterned solid".split()
MATERIALS = "wooden metal metallic glass plastic leather stone ceramic paper fabric rubber woven knitted".split()
COUNTS = ["one", "two", "three", "four", "five"]
FLIPS = [("left", "right"), ("above", "below"), ("in front of", "behind")]


def kc(src, new):
    return new[:1].upper() + new[1:] if src[:1].isupper() else new


def counterfactuals(q):
    out = []
    for name, vocab in (("color", COLORS), ("pattern", PATTERNS), ("material", MATERIALS)):
        ms = list(re.finditer(r"\b(" + "|".join(map(re.escape, vocab)) + r")\b", q, flags=re.I))
        for i, a in enumerate(ms):
            for b in ms[i + 1:]:
                if a.group(0).lower() != b.group(0).lower():
                    out.append((name, q[:a.start()] + kc(a.group(0), b.group(0).lower()) + q[a.end():b.start()]
                                + kc(b.group(0), a.group(0).lower()) + q[b.end():]))
    for x, y in FLIPS:
        for s, t in ((x, y), (y, x)):
            for m in re.finditer(r"\b" + re.escape(s) + r"\b", q, flags=re.I):
                out.append(("flip", q[:m.start()] + kc(m.group(0), t) + q[m.end():]))
    for m in re.finditer(r"\b(" + "|".join(COUNTS) + r")\b", q, flags=re.I):
        i = COUNTS.index(m.group(0).lower())
        for j in (i - 1, i + 1):
            if 0 <= j < len(COUNTS):
                out.append(("count", q[:m.start()] + kc(m.group(0), COUNTS[j]) + q[m.end():]))
    seen, res = {q}, []
    for kind, t in out:
        if t not in seen:
            seen.add(t); res.append((kind, t))
    return res


def metrics(S, k):
    """S: (nq, K) scores, answer = candidate index where top_i == q (ans_pos), -1 if absent."""
    r1 = nd = hit = 0.0
    for q in range(S.shape[0]):
        order = np.argsort(-S[q, :k], kind="stable")
        a = ANS[q]
        if 0 <= a < k:
            hit += 1
            r = int(np.where(order == a)[0][0])
            r1 += r == 0
            if r < 5:
                nd += 1 / math.log2(r + 2)
    n = S.shape[0]
    return r1 / n, nd / n, 0.5 * (r1 + nd) / n, r1 / max(hit, 1)


def main():
    global ANS
    root = "haikuthon-train"
    recs = [json.loads(l) for l in open(f"{root}/prompts.jsonl", encoding="utf-8") if l.strip()]
    by_id = {r["id"]: r for r in recs}
    sp = json.load(open("splits/split_group_seed42.json"))
    pool = sp["valA"] + sp["valB"]
    pool = pool + [i for i in sorted(by_id) if i not in set(pool)]
    texts = [by_id[i]["prompts"] for i in sp["valA"]]
    d = np.load(DUMP)
    top_i, top_s, dump_logits = d["top_i"][:, :K], d["top_s"][:, :K], d["logits"][:, :K]
    nq = len(texts)
    assert top_i.shape[0] == nq, (top_i.shape, nq)
    ANS = np.array([int(np.where(top_i[q] == q)[0][0]) if (top_i[q] == q).any() else -1 for q in range(nq)])
    print("queries", nq, "pool", len(pool), "answer in top-K", float((ANS >= 0).mean()))

    sub = os.path.abspath(SUB)
    sys.path.insert(0, sub)
    spec = importlib.util.spec_from_file_location("sub_model", os.path.join(sub, "model.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    m = mod.load_model(sub, "cuda:0")
    lm, proc, w, dev = m["lm"], m["proc"], m["w"], m["device"]
    rr = int(m["cfg"]["rr_res"])
    ip, tk = proc.image_processor, proc.tokenizer
    img_tok, pad_id = lm.config.image_token_id, tk.pad_token_id
    image_token = getattr(proc, "image_token", "<|image_pad|>")

    # ---- vision cache for every candidate image
    uniq = np.unique(top_i)
    pos = np.full(len(pool), -1); pos[uniq] = np.arange(len(uniq))
    E = D = grid = None
    t0 = time.perf_counter()
    with torch.inference_mode():
        for s in range(0, len(uniq), 64):
            ims = []
            for j in uniq[s:s + 64]:
                im = mod._to_rgb(Image.open(f"{root}/images/{pool[j]}.png")).resize((rr, rr), resample=Image.Resampling.BICUBIC)
                ims.append(im)
            enc = ip(images=ims, do_resize=False, return_tensors="pt")
            g = enc["image_grid_thw"].to(dev)
            out = lm.model.get_image_features(enc["pixel_values"].to(dev), g, return_dict=True)
            e = torch.stack(list(out.pooler_output)); dd = torch.stack([torch.stack(list(f)) for f in out.deepstack_features])
            if E is None:
                E = torch.empty((len(uniq),) + tuple(e.shape[1:]), dtype=e.dtype, device=dev)
                D = torch.empty((dd.shape[0], len(uniq)) + tuple(dd.shape[2:]), dtype=dd.dtype, device=dev)
                grid = g[:1].clone()
            E[s:s + len(ims)] = e; D[:, s:s + len(ims)] = dd
    ntok = E.shape[1]
    print(f"vision cache {len(uniq)} images in {time.perf_counter() - t0:.1f}s")

    def score(text_list, img_idx):
        """text_list[i] scored against pool image img_idx[i]; returns np array."""
        ids_l = [tk(mod._prompt(m, t).replace(image_token, image_token * ntok), add_special_tokens=False)["input_ids"] for t in text_list]
        order = np.argsort([len(x) for x in ids_l], kind="stable")
        res = np.zeros(len(ids_l), dtype=np.float32)
        with torch.inference_mode():
            for s in range(0, len(order), 48):
                b = order[s:s + 48]
                L = max(len(ids_l[i]) for i in b)
                ids = torch.full((len(b), L), pad_id, dtype=torch.long); am = torch.zeros((len(b), L), dtype=torch.long)
                for r, i in enumerate(b):
                    x = ids_l[i]; ids[r, L - len(x):] = torch.tensor(x); am[r, L - len(x):] = 1
                ids, am = ids.to(dev), am.to(dev)
                j = torch.as_tensor(pos[np.asarray(img_idx)[b]], device=dev)
                Ej, Dj = E[j], D[:, j]
                feats = SimpleNamespace(pooler_output=(Ej.reshape(-1, Ej.shape[-1]),),
                                        deepstack_features=[(Dj[l].reshape(-1, Dj.shape[-1]),) for l in range(Dj.shape[0])])
                o = lm.model(input_ids=ids, attention_mask=am, image_grid_thw=grid.expand(len(b), -1),
                             mm_token_type_ids=(ids == img_tok).to(torch.int32), mm_encoder_outputs={"image": feats})
                res[b] = (o.last_hidden_state[:, -1].float() @ w).cpu().numpy()
        return res

    # ---- base scores
    t0 = time.perf_counter()
    base = score([texts[q] for q in range(nq) for _ in range(K)], top_i.reshape(-1)).reshape(nq, K)
    print(f"base {nq * K} pairs in {time.perf_counter() - t0:.1f}s; max |base - dump| = {np.abs(base - dump_logits).max():.3f}, "
          f"mean {np.abs(base - dump_logits).mean():.3f}")

    # ---- counterfactual scores
    cfs = [counterfactuals(t) for t in texts]
    ncf = np.array([len(c) for c in cfs])
    kinds = {}
    for c in cfs:
        for kind, _ in c:
            kinds[kind] = kinds.get(kind, 0) + 1
    print("queries with >=1 counterfactual", float((ncf > 0).mean()), "mean count", float(ncf.mean()), "max", int(ncf.max()), kinds)
    for q in (0, 1, 2):
        print("  Q :", texts[q]); [print("  Q-:", kind, "|", t) for kind, t in cfs[q][:3]]
    tl, il, own = [], [], []
    for q in range(nq):
        for ci, (_, t) in enumerate(cfs[q]):
            for r in range(K):
                tl.append(t); il.append(top_i[q, r]); own.append((q, ci, r))
    t0 = time.perf_counter()
    sc = score(tl, il)
    print(f"counterfactual {len(tl)} pairs in {time.perf_counter() - t0:.1f}s")
    M = int(ncf.max()) if len(ncf) else 0
    CF = np.full((nq, max(M, 1), K), -np.inf, dtype=np.float32)
    for (q, ci, r), v in zip(own, sc):
        CF[q, ci, r] = v
    np.savez("runs/cqm_val.npz", base=base, CF=CF, ncf=ncf, top_i=top_i, top_s=top_s, ans=ANS)

    cf_max = np.where(ncf[:, None] > 0, CF.max(1), 0.0)
    CFm = np.where(np.isfinite(CF), CF, np.nan)
    with np.errstate(all="ignore"):
        cf_mean = np.where(ncf[:, None] > 0, np.nanmean(CFm, 1), 0.0)
    has = ncf > 0

    def show(name, S):
        row = []
        for k in (3, 5, 6, 8):
            r1, nd, s_, acc = metrics(S, k)
            row.append(f"k{k}: R@1 {r1:.4f} sc {s_:.4f} acc {acc:.4f}")
        print(f"{name:28s} " + " | ".join(row))

    print("\n==== all queries ====")
    show("base (Qwen logit)", base)
    for lam in (0.25, 0.5, 0.75, 1.0):
        show(f"CQM max lam={lam}", base - lam * cf_max)
    for lam in (0.5, 1.0):
        show(f"CQM mean lam={lam}", base - lam * cf_mean)
    # soft: log-odds of "matches Q and none of Q-"
    sig = lambda x: 1 / (1 + np.exp(-x))
    show("log sig(s) + log(1-sig(max))", np.log(sig(base) + 1e-9) + np.where(has[:, None], np.log(1 - sig(cf_max) + 1e-9), 0.0))
    print("\n==== stage-1 prior: logit + alpha * z(stage1 within top-k) ====")
    for alpha in (0.5, 1.0, 2.0, 4.0):
        row = []
        for k in (3, 5, 6, 8):
            z = (top_s[:, :k] - top_s[:, :k].mean(1, keepdims=True)) / (top_s[:, :k].std(1, keepdims=True) + 1e-6)
            S = base.copy(); S[:, :k] = base[:, :k] + alpha * z
            r1, nd, s_, acc = metrics(S, k)
            row.append(f"k{k}: R@1 {r1:.4f} sc {s_:.4f}")
        print(f"alpha={alpha:<4}                   " + " | ".join(row))
    # error breakdown at k=8: how many base errors are fixed / how many correct ones are broken by CQM max lam=.5
    k = 8
    b1 = np.array([np.argsort(-base[q, :k], kind="stable")[0] == ANS[q] for q in range(nq)])
    S = base - 0.5 * cf_max
    c1 = np.array([np.argsort(-S[q, :k], kind="stable")[0] == ANS[q] for q in range(nq)])
    inn = (ANS >= 0) & (ANS < k)
    print(f"\nk=8, lam=.5: answer in top-8 {int(inn.sum())}; base wrong {int((inn & ~b1).sum())} (with counterfactual {int((inn & ~b1 & has).sum())}); "
          f"fixed {int((inn & ~b1 & c1).sum())}; broken {int((inn & b1 & ~c1).sum())}")


if __name__ == "__main__":
    main()

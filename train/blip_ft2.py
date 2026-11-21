"""BLIP-2 ITM 추가 학습 (Q-Former LoRA): 학습 데이터의 8장 쌍둥이 세트(같은 장면, 한 속성씩만 다른 이미지)를 서로의 오답으로 쓴다.
 - 세트는 캡션 단어 집합의 유사도로 찾는다(8명이 서로를 가장 가까운 7개로 꼽는 세트만 사용).
 - 손실: 캡션→세트 8장 중 정답 고르기 + 이미지→세트 8캡션(+조명·시점·화풍만 바꾼 캡션) 중 정답 고르기.
 - 결과: LoRA 가중치(safetensors). 제출 코드는 로딩 직후 W += scale * B @ A 로 합친다(추론 시간 변화 없음).
"""
import argparse, collections, json, math, random, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LIGHT = ["side lighting", "direct sunlight", "soft diffuse lighting", "warm indoor lighting"]
VIEWS = ["overhead", "front", "side", "oblique"]
STYLES = ["low-poly 3D render", "clay-style 3D render", "watercolor painting", "oil painting", "flat illustration",
          "colored-pencil drawing", "ink drawing", "pencil sketch", "3D render", "photograph"]
STY_TO = ["photograph", "3D render", "watercolor painting", "oil painting", "flat illustration", "pencil sketch"]
REL = {'left', 'right', 'above', 'below', 'under', 'beneath', 'over', 'top', 'front', 'behind'}
STOP = set("a an the is are of to in on at with and it its this that there one two three both which who as by from for be shown shows show viewed view scene image rendered render set setting contains contain appears appear alongside also present positioned taken while other each them their has have holds hold".split())
OUT = []
TAG = "x"


def say(s):
    OUT.append(s); print("R> " + s, flush=True)
    (ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / f"runs/blip_ft2_{TAG}.txt").write_text("\n".join(OUT) + "\n")


def fix_articles(t):
    t = re.sub(r"\b([Aa])n (?=(?:front|side|photograph|3D|watercolor|flat|colored|pencil|low|clay))", r"\1 ", t)
    t = re.sub(r"\b([Aa]) (?=(?:overhead|oblique|oil|ink))", r"\1n ", t)
    return t


def var_light(q, rng):
    for i, x in enumerate(LIGHT):
        m = re.search(re.escape(x), q, flags=re.I)
        if m:
            return q[:m.start()] + LIGHT[(i + rng.randint(1, 3)) % 4] + q[m.end():]


def var_view(q, rng):
    m = re.search(r"\b(overhead|front|side|oblique)(?=[- ](?:view|angle))", q, flags=re.I) or \
        re.search(r"(?<=from the )(front|side)\b", q, flags=re.I) or re.search(r"(?<=from )(overhead)\b", q, flags=re.I)
    if m:
        i = VIEWS.index(m.group(1).lower())
        new = VIEWS[(i + rng.randint(1, 3)) % 4]
        new = new.capitalize() if m.group(1)[0].isupper() else new
        return fix_articles(q[:m.start(1)] + new + q[m.end(1):])


def var_style(q, rng):
    for x in STYLES:
        m = re.search(re.escape(x), q, flags=re.I)
        if m:
            cand = [s for s in STY_TO if s.lower() not in x.lower() and x.lower() not in s.lower()]
            return fix_articles(q[:m.start()] + rng.choice(cand) + q[m.end():])


def scenes_of(caps, with4=False):
    ids = sorted(caps)
    S = {i: frozenset(x for x in re.findall(r"[a-z]+(?:-[a-z]+)*", caps[i].lower()) if x not in STOP and x not in REL) for i in ids}
    inv = collections.defaultdict(set)
    for i in ids:
        for w in S[i]:
            inv[w].add(i)
    cnt = collections.Counter()
    for i in ids:
        c = collections.Counter()
        for w in S[i]:
            if len(inv[w]) < 400:
                for j in inv[w]:
                    c[j] += 1
        sc = sorted(((len(S[i] & S[j]) / len(S[i] | S[j]), j) for j in c if j != i), reverse=True)
        cnt[frozenset([i] + [j for _, j in sc[:7]])] += 1
    eight = sorted(sorted(g) for g, c in cnt.items() if c == 8 and len(g) == 8)
    if not with4:
        return eight
    used = {i for g in eight for i in g}
    left = [i for i in ids if i not in used]
    ls = set(left)
    c4 = collections.Counter()
    for i in left:
        c = collections.Counter()
        for w in S[i]:
            if len(inv[w]) < 400:
                for j in inv[w]:
                    if j in ls:
                        c[j] += 1
        sc = sorted(((len(S[i] & S[j]) / len(S[i] | S[j]), j) for j in c if j != i), reverse=True)
        c4[frozenset([i] + [j for _, j in sc[:3]])] += 1
    four = sorted(sorted(g) for g, c in c4.items() if c == 4 and len(g) == 4)
    return eight, four


class LoRA(nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.s = alpha / r

    def forward(self, x):
        y = self.base(x)
        if self.s:
            y = y + F.linear(F.linear(x.to(self.A.dtype), self.A), self.B).to(y.dtype) * self.s
        return y


def main():
    global TAG
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--blip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="a")
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--spb", type=int, default=2, help="scenes per step")
    ap.add_argument("--n-val", type=int, default=40)
    ap.add_argument("--max-min", type=float, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use4", type=int, default=1, help="4장 세트도 학습에 사용")
    a = ap.parse_args()
    TAG = a.tag
    dev = "cuda"
    rng = random.Random(a.seed); torch.manual_seed(a.seed)
    data = Path(a.data)
    caps = {}
    for line in open(data / "prompts.jsonl", encoding="utf-8"):
        if line.strip():
            r = json.loads(line); caps[r["id"]] = r["prompts"]
    scenes, four = scenes_of(caps, with4=True)
    random.Random(0).shuffle(scenes)
    val, trn = scenes[:a.n_val], scenes[a.n_val:]
    if a.n_val == 0:
        val = scenes[:40]
    if a.use4:
        trn = trn + four
    used = sorted({i for g in scenes + (four if a.use4 else []) for i in g}); col = {i: k for k, i in enumerate(used)}
    say(f"[{TAG}] scenes8 {len(scenes)} scenes4 {len(four)} train {len(trn)} val {len(val)} images {len(used)} | r {a.r} alpha {a.alpha} lr {a.lr} ep {a.epochs}")

    def load(i):
        with Image.open(data / "images" / f"{i}.png") as im:
            return np.asarray(im.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC))
    with ThreadPoolExecutor(4) as ex:
        a224 = np.stack(list(ex.map(load, used)))

    from submission_ens.model import _load_blip, _blip_image_embeds, _blip_itm, _to_pixels, CLIP_MEAN, CLIP_STD
    blip, bproc = _load_blip(Path(a.blip), dev)
    vdt = blip.vision_model.embeddings.patch_embedding.weight.dtype
    with torch.inference_mode():
        E = torch.cat([_blip_image_embeds(blip, _to_pixels(a224[s:s + 48], CLIP_MEAN, CLIP_STD, dev, vdt)) for s in range(0, len(used), 48)])
    E = E.cpu().clone()
    blip.vision_model.to("cpu"); torch.cuda.empty_cache()
    for p in blip.parameters():
        p.requires_grad_(False)
    names = [n for n, m in blip.qformer.encoder.named_modules() if isinstance(m, nn.Linear)]
    wrapped = {}
    for n in names:
        parent = blip.qformer.encoder
        parts = n.split(".")
        for p_ in parts[:-1]:
            parent = getattr(parent, p_)
        w = LoRA(getattr(parent, parts[-1]), a.r, a.alpha).to(dev)
        setattr(parent, parts[-1], w)
        wrapped["qformer.encoder." + n] = w
    blip.itm_head.float()
    for p in blip.itm_head.parameters():
        p.requires_grad_(True)
    params = [p for w in wrapped.values() for p in (w.A, w.B)] + list(blip.itm_head.parameters())
    say(f"[{TAG}] lora modules {len(wrapped)} trainable {sum(p.numel() for p in params)/1e6:.2f}M")
    tokz = lambda texts: bproc.tokenizer(texts, padding="longest", truncation=True, max_length=128, return_tensors="pt").to(dev)

    def fwd(texts, ti, ii):
        tok = tokz(texts)
        with torch.autocast("cuda", dtype=torch.float16):
            return _blip_itm(blip, E[ii.cpu()].to(dev), tok.input_ids[ti], tok.attention_mask[ti])

    @torch.no_grad()
    def evaluate():
        r1 = nd = n = 0; sens = collections.defaultdict(list); vr = random.Random(1)
        for g in val:
            texts = [caps[i] for i in g]
            alts = []
            for b, i in enumerate(g):
                for nm, fn in (("light", var_light), ("view", var_view), ("style", var_style)):
                    v = fn(caps[i], vr)
                    if v:
                        alts.append((nm, b, v))
            ti = torch.arange(8, device=dev).repeat_interleave(8).tolist() + [8 + k for k in range(len(alts))]
            ii = [col[i] for i in g] * 8 + [col[g[b]] for _, b, _ in alts]
            lg = fwd(texts + [v for _, _, v in alts], torch.tensor(ti, device=dev), torch.tensor(ii, device=dev)).float()
            L = lg[:64].view(8, 8)
            rank = (L > L.diag()[:, None]).sum(1) + 1
            r1 += (rank == 1).sum().item(); nd += torch.where(rank <= 5, 1 / torch.log2(rank + 1.0), torch.zeros(8, device=dev)).sum().item(); n += 8
            for k, (nm, b, _) in enumerate(alts):
                sens[nm].append(float(L[b, b] > lg[64 + k]))
        return (f"세트 내 8지선다 score {(r1/n + nd/n)/2:.4f} R@1 {r1/n:.3f} | 민감도 " +
                " ".join(f"{k} {np.mean(v):.3f}" for k, v in sens.items())), (r1 / n + nd / n) / 2

    msg, best = evaluate()
    say(f"[{TAG}] ep 0 (학습 전) {msg}")
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    steps_ep = math.ceil(len(trn) / a.spb); total = steps_ep * a.epochs; warm = max(10, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    scaler = torch.amp.GradScaler("cuda")
    t0 = time.time(); step = 0; stop = False
    ar8 = torch.arange(8, device=dev)

    def save(path):
        from safetensors.torch import save_file
        sd = {}
        for n, w in wrapped.items():
            sd[n + ".A"] = w.A.detach().half().cpu(); sd[n + ".B"] = w.B.detach().half().cpu()
        sd["itm_head.weight"] = blip.itm_head.weight.detach().float().cpu(); sd["itm_head.bias"] = blip.itm_head.bias.detach().float().cpu()
        sd["scale"] = torch.tensor([a.alpha / a.r], dtype=torch.float32)
        save_file(sd, path)

    for ep in range(1, a.epochs + 1):
        rng.shuffle(trn); run = 0.0; nrun = 0
        for s in range(0, len(trn), a.spb):
            texts, ti, ii, spans = [], [], [], []
            for g in trn[s:s + a.spb]:
                n_ = len(g)
                base = len(texts); texts += [caps[i] for i in g]
                ti += [base + x for x in range(n_) for _ in range(n_)]; ii += [col[i] for i in g] * n_
                alt = []
                for b, i in enumerate(g):
                    for fn in (var_light, var_view, var_style):
                        v = fn(caps[i], rng)
                        if v:
                            alt.append(b); ti.append(len(texts)); ii.append(col[i]); texts.append(v)
                spans.append((n_, alt))
            lg = fwd(texts, torch.tensor(ti, device=dev), torch.tensor(ii, device=dev)).float()
            loss = 0.0; p = 0
            for n_, alt in spans:
                L = lg[p:p + n_ * n_].view(n_, n_); p += n_ * n_
                M = torch.full((n_, n_ + 3), -1e4, device=dev)
                M[:, :n_] = L.t()
                c = [0] * n_
                for b in alt:
                    M[b, n_ + c[b]] = lg[p]; c[b] += 1; p += 1
                loss = loss + (F.cross_entropy(L, ar8[:n_]) + F.cross_entropy(M, ar8[:n_])) * (n_ / 8)
            loss = loss / len(spans)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            run += loss.item(); nrun += 1; step += 1
            if step == 20:
                say(f"[{TAG}] 20 steps {time.time()-t0:.0f}s → 예상 총 {(time.time()-t0)/20*total/60:.1f}분, loss {run/nrun:.3f}")
            if (time.time() - t0) / 60 > a.max_min:
                stop = True; break
        msg, sc = evaluate()
        say(f"[{TAG}] ep {ep} loss {run/max(nrun,1):.3f} {time.time()-t0:.0f}s | {msg}")
        if sc >= best or a.n_val == 0:
            best = sc; save(a.out)
            say(f"[{TAG}] saved ep {ep} → {a.out}")
        if stop:
            break
    # 합치기 검증: 저장본을 원본 가중치에 더한 뒤(LoRA 끔) 같은 점수가 나오는지
    from safetensors.torch import load_file
    sd = load_file(a.out)
    with torch.no_grad():
        for n, w in wrapped.items():
            w.base.weight.data += (sd[n + ".B"].float().to(dev) @ sd[n + ".A"].float().to(dev)).to(w.base.weight.dtype) * float(sd["scale"][0])
            w.s = 0.0
        blip.itm_head.weight.data.copy_(sd["itm_head.weight"].to(dev)); blip.itm_head.bias.data.copy_(sd["itm_head.bias"].to(dev))
    msg, _ = evaluate()
    say(f"[{TAG}] 합친 뒤(저장본) {msg} | 파일 {Path(a.out).stat().st_size/1e6:.1f}MB")
    say(f"[{TAG}] FTDONE")


if __name__ == "__main__":
    main()

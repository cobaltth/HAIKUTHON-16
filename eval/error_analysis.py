"""1등 오답 전수 분석: 정답 캡션 vs 모델 1등 캡션을 비교해 '무엇이 달랐는지' 자동 라벨링.

python eval/error_analysis.py runs/r0_zeroshot.json <haikuthon-train> docs/r0_errors_all.md [eda/groups.csv]

라벨 (여러 개 동시 가능)
  color / pattern / material / state(자세·열림) / count(개수·단수복수) / relation(관계어 구성)
  style(화풍) / view(시점) / light(조명) / setting(장소)
  binding_or_order : 위 속성 어휘가 전혀 다르지 않음 = 같은 단어로 결합·관계 방향만 바뀜
  object           : 등장 물체 자체가 다름 (형제 아님)
"""
import collections, csv, json, re, sys
from pathlib import Path

COLORS = "brown white black yellow green red beige orange gray blue purple pink".split()
PATTERNS = "solid-colored striped polka-dotted checkered".split()
MATERIALS = "metal wood wooden plastic glass rubber ceramic fabric stone leather cardboard wicker".split()
STATES = "open closed standing sitting lying crouching kneeling".split()
COUNTS = {"one": 1, "single": 1, "two": 2, "three": 3, "four": 4}
REL = ["left", "right", "above", "below", "next to", "beside", "on", "inside", "holds", "holding", "wears", "wearing",
       "rides", "riding", "rests", "resting", "sits", "under"]
STYLES = {"photograph": r"photograph", "oil": r"oil painting", "watercolor": r"watercolor", "gouache": r"gouache",
          "colored-pencil": r"colored-pencil|colored pencil", "pencil sketch": r"pencil sketch", "ink": r"ink drawing",
          "flat": r"flat illustration", "low-poly": r"low-poly", "3d": r"(?<!low-poly )3d", "paper-cut": r"paper-cut",
          "clay": r"clay-style"}
VIEWS = {"front": r"front", "side": r"side[- ]view|(?<!lit )from the side|in side view", "overhead": r"overhead",
         "oblique": r"oblique"}
LIGHTS = {"direct": r"direct sunlight", "side": r"side lighting|side-lit|lit from the side", "soft": r"soft diffuse", "warm": r"warm indoor"}
SETTINGS = ["studio", "park", "farm", "living room", "garden", "bedroom", "dining room", "office", "patio", "meadow",
            "kitchen", "street", "airport", "pasture", "classroom", "cafe", "beach", "schoolyard", "workshop"]
STOP = set("""a an the is are of in on to with and one two three four single which that it its scene shown shows show
from view viewed lighting lit rendered render as present also there by both under set setting contains appears appear
image photographed uses has made color pattern angle plain backdrop studio while who this be being alongside
left right above below next beside inside holds holding wears wearing rides riding rests resting sits sit stands
taken positioned depicted captured shot seen placed located sitting""".split())


def words(t):
    return re.findall(r"[a-z\-]+", t.lower())


IRREG = {"cacti": "cactus", "knives": "knife", "shelves": "shelf", "loaves": "loaf", "scarves": "scarf",
         "people": "person", "leaves": "leaf", "mice": "mouse", "geese": "goose"}


def singular(w):
    if w in IRREG:
        return IRREG[w]
    if w.endswith(("us", "ss", "is")):
        return w
    if w.endswith(("uses", "oes")):
        return w[:-2]
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("ches", "shes", "sses", "xes")):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


MODS = set("""rectangular star-shaped square round floral small large tall oval zigzag-patterned inflated triangular
hexagonal folded unfolded leather stone cardboard wicker orange-colored colored unlit lit solid""".split())
ATTR = set(COLORS + PATTERNS + MATERIALS + STATES) | MODS
STYLE_WORDS = set("photograph oil painting watercolor gouache colored-pencil pencil sketch ink drawing flat illustration low-poly d paper-cut collage clay-style front front-view side side-view overhead overhead-view oblique direct sunlight soft diffuse warm indoor".split())


NOUNS = set()
JUNK = set("of which other is and in inside to the under sits appear bedroom farm library photograph living clothing studio plain one".split())


def build_nouns(captions):
    """개수어 바로 뒤(수식어 건너뜀)에 2회 이상 나온 단어 = 물체 명사 사전"""
    c = collections.Counter()
    for t in captions:
        w = words(t)
        for i, x in enumerate(w):
            if x in COUNTS:
                rest = [y for y in w[i + 1:i + 7] if y not in ATTR]
                if rest and rest[0] not in STOP and rest[0] not in STYLE_WORDS and rest[0] not in JUNK and rest[0] not in COUNTS:
                    c[singular(rest[0])] += 1
    NOUNS.clear(); NOUNS.update(k for k, v in c.items() if v >= 2 and k not in COLORS)  # orange(과일/색) 등 색 단어 제외


def nouns(t):
    return frozenset(singular(w) for w in words(t) if singular(w) in NOUNS)


def objects(t):
    """속성·관계·화풍·기능어를 뺀 나머지 = 물체/장소 어휘 (단수형)"""
    return collections.Counter(singular(w) for w in words(t)
                               if w not in STOP and w not in ATTR and w not in STYLE_WORDS and w not in COUNTS)


def bag(t, vocab):
    tl = re.sub(r"\bsolid\b(?!-)", "solid-colored", t.lower().replace("wooden", "wood"))
    for a, b in ((r"\bstands?\b", "standing"), (r"\blies\b", "lying"), (r"\bkneels\b", "kneeling"), (r"\bcrouches\b", "crouching")):
        tl = re.sub(a, b, tl)
    return collections.Counter(m.group(0) for m in re.finditer(r"\b(" + "|".join(map(re.escape, vocab)) + r")\b", tl))


def pick(t, table):
    tl = t.lower()
    return frozenset(k for k, p in table.items() if re.search(p, tl))


def count_sig(t):
    """(개수, 물체) 묶음. 개수어 바로 뒤(색·무늬 등 속성어는 건너뜀)의 첫 단어를 물체로 본다.
    그 단어가 기능어/배경어(예: one plain backdrop)면 세지 않는다."""
    sig = collections.Counter()
    for clause in re.split(r"[.;]", t.lower()):
        w = re.findall(r"[a-z\-]+", clause)
        for i, x in enumerate(w):
            if x in COUNTS:
                rest = [y for y in w[i + 1:i + 7] if y not in ATTR and y != "single"]
                if rest and rest[0] not in STOP and rest[0] not in STYLE_WORDS and rest[0] not in COUNTS:
                    sig[(COUNTS[x], singular(rest[0]))] += 1
    return sig


def compare(a, b):
    """a=정답 캡션, b=모델 1등 캡션 → (라벨 목록, 차이 설명 dict)"""
    labels, detail = [], {}
    na, nb = nouns(a), nouns(b)
    if na != nb:   # 등장 물체 종류가 다름 = 형제 아님
        labels.append("object")
        detail["objects_only_correct"], detail["objects_only_wrong"] = sorted(na - nb), sorted(nb - na)
    for name, vocab in (("color", COLORS), ("pattern", PATTERNS), ("material", MATERIALS), ("state", STATES)):
        ba, bb = bag(a, vocab), bag(b, vocab)
        if set(ba) != set(bb):
            labels.append(name)
            detail[name] = (sorted(set(ba) - set(bb)), sorted(set(bb) - set(ba)))
    ca, cb = count_sig(a), count_sig(b)
    shared = na & nb                                  # 양쪽에 다 있는 물체의 개수만 비교
    ca = {k for k in ca if k[1] in shared or not NOUNS}
    cb = {k for k in cb if k[1] in shared or not NOUNS}
    if ca != cb:
        labels.append("count")
        detail["count"] = (sorted(ca - cb), sorted(cb - ca))
    # 관계어: 방향어 + 동사(형태 통일). 'on'은 장소 표현(on a farm)과 섞여 제외
    norm = {"holding": "holds", "wearing": "wears", "riding": "rides", "resting": "rests", "beside": "next to"}
    def rel_set(t):
        r = {norm.get(k, k) for k in bag(t, REL) if k not in ("on", "sits", "under", "rests", "resting")}
        tl = t.lower()
        if re.search(r"\bcontaining one\b", tl):   # 'the scene contains one ...'는 관계가 아님
            r.add("inside")
        if re.search(r"\b(rests?|resting|sits?|sitting|lies|placed) on\b", tl):
            r.add("on")
        return r
    ra, rb = rel_set(a), rel_set(b)
    if ra != rb:
        labels.append("relation")
        detail["relation"] = (sorted(ra - rb), sorted(rb - ra))
    for name, table in (("style", STYLES), ("view", VIEWS), ("light", LIGHTS)):
        pa, pb = pick(a, table), pick(b, table)
        if pa and pb and pa != pb:
            labels.append(name)
            detail[name] = (sorted(pa), sorted(pb))
    sa, sb = bag(a, SETTINGS), bag(b, SETTINGS)
    if set(sa) != set(sb):
        labels.append("setting")
        detail["setting"] = (sorted(sa), sorted(sb))
    if not labels:
        labels = ["binding_or_order"]
    return labels, detail


LABEL_KO = {"binding_or_order": "같은 단어·결합/관계 방향만 다름", "color": "색", "pattern": "무늬", "material": "재질",
            "state": "자세·열림 상태", "count": "개수", "relation": "관계어", "style": "화풍", "view": "시점",
            "light": "조명", "setting": "장소", "object": "다른 물체(형제 아님)"}


def main(run_path, data_dir, out_md, groups_path=None):
    run = json.load(open(run_path))
    caps = {}
    for l in open(Path(data_dir) / "prompts.jsonl", encoding="utf-8"):
        if l.strip():
            r = json.loads(l); caps[r["id"]] = r["prompts"]
    build_nouns(caps.values())
    grp = {}
    if groups_path and Path(groups_path).exists():
        grp = {r["id"]: r["group_id"] for r in csv.DictReader(open(groups_path))}
    rows = []
    for q, rank in run["ranks"].items():
        if rank == 1:
            continue
        top1 = run["preds"][q][0]
        labels, detail = compare(caps[q], caps[top1])
        rows.append({"q": q, "top1": top1, "rank": rank if rank else ">50", "labels": labels, "detail": detail,
                     "sibling_group": bool(grp) and grp.get(q) == grp.get(top1),
                     "sibling_norm": nouns(caps[q]) == nouns(caps[top1])})
    n = len(rows); nq = len(run["ranks"])
    lab = collections.Counter(l for r in rows for l in r["labels"])
    only = collections.Counter(r["labels"][0] for r in rows if len(r["labels"]) == 1)
    nlab = collections.Counter(len(r["labels"]) for r in rows)
    rank_c = collections.Counter(str(r["rank"]) for r in rows)
    L = []
    L.append(f"# r0 1등 오답 전수 분석 ({n}개)\n")
    L.append(f"- 질문 {nq}개 중 1등 오답 {n}개. 정답 캡션과 모델 1등 캡션을 규칙 기반으로 비교해 자동 라벨링 (한 사례에 여러 라벨 가능)")
    L.append(f"- 형제(같은 물체 구성): 기존 그룹 규칙 {sum(r['sibling_group'] for r in rows)}개 → 단수/복수·수식어 정규화 후 {sum(r['sibling_norm'] for r in rows)}개 / 물체 자체가 다른 경우(object): {lab['object']}개")
    L.append(f"- 정답 순위: " + ", ".join(f"{k}위 {v}개" for k, v in sorted(rank_c.items(), key=lambda x: (len(x[0]), x[0]))))
    L.append("\n## 1. 유형별 빈도\n")
    L.append("| 유형 | 등장 사례 수 | 비율 | 이 유형 '하나만' 다른 사례 |")
    L.append("|---|---|---|---|")
    for k, v in lab.most_common():
        L.append(f"| {LABEL_KO[k]} (`{k}`) | {v} | {v/n:.1%} | {only.get(k, 0)} |")
    L.append(f"\n차이 유형 개수별 사례 수: " + ", ".join(f"{k}개 {v}" for k, v in sorted(nlab.items())))
    L.append("\n## 2. 전체 사례 표\n")
    L.append("| # | 정답 | 모델 1등 | 정답 순위 | 형제 | 유형 | 차이 (정답 → 오답) |")
    L.append("|---|---|---|---|---|---|---|")
    for i, r in enumerate(sorted(rows, key=lambda r: r["q"]), 1):
        d = []
        for k in r["labels"]:
            if k in r["detail"]:
                x = r["detail"][k]
                if isinstance(x, tuple):
                    a, b = x
                    fa = ", ".join(map(lambda t: f"{t[0]} {t[1]}" if isinstance(t, tuple) else str(t), a)) or "-"
                    fb = ", ".join(map(lambda t: f"{t[0]} {t[1]}" if isinstance(t, tuple) else str(t), b)) or "-"
                    d.append(f"{k}: {fa} → {fb}")
        if "object" in r["labels"]:
            d.append(f"물체: {', '.join(r['detail'].get('objects_only_correct', [])) or '-'} → {', '.join(r['detail'].get('objects_only_wrong', [])) or '-'}")
        L.append(f"| {i} | {r['q']} | {r['top1']} | {r['rank']} | {'O' if r['sibling_norm'] else ''} | "
                 f"{', '.join(LABEL_KO[l] for l in r['labels'])} | {'; '.join(d) or '단어 구성 동일'} |")
    Path(out_md).write_text("\n".join(L), encoding="utf-8")
    json.dump(rows, open(Path(out_md).with_suffix(".json"), "w"), ensure_ascii=False, default=list)
    print(f"{n}개 분석 → {out_md}")
    for k, v in lab.most_common():
        print(f"  {LABEL_KO[k]:24s} {v:4d} ({v/n:.1%})  단독 {only.get(k, 0)}")


if __name__ == "__main__":
    main(*sys.argv[1:4], sys.argv[4] if len(sys.argv) > 4 else None)

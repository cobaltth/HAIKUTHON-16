"""규칙 기반 캡션 변형 (외부 생성 모델 사용 X → 대회 규정 1번 준수).

- swap_left_right(caption): left↔right 교환. 좌우 반전 이미지와 짝지으면 '새 정답 쌍'
- hard_negatives(caption):  원래 이미지와 '안 맞는' 캡션 목록 (색 바꾸기 / 색-물체 결합 교환 /
                            관계 뒤집기 / 무늬·재질 바꾸기). 각 항목 = (종류, 캡션)
"""
import random, re
from .vocab import COLORS, PATTERNS, MATERIALS, REL_SWAPS


def _word_re(words):
    return re.compile(r"\b(" + "|".join(map(re.escape, words)) + r")\b", re.IGNORECASE)

_COLOR_RE, _PAT_RE, _MAT_RE = _word_re(COLORS), _word_re(PATTERNS), _word_re(MATERIALS)


def _keep_case(src, dst):
    return dst.capitalize() if src[:1].isupper() else dst


def swap_pair(caption, a, b):
    """단어 a와 b를 서로 맞바꾼다 (대소문자 유지)."""
    def rep(m):
        w = m.group(0)
        return _keep_case(w, b if w.lower() == a else a)
    return _word_re([a, b]).sub(rep, caption)


def swap_left_right(caption):
    return swap_pair(caption, "left", "right")


def has_left_right(caption):
    return bool(_word_re(["left", "right"]).search(caption))


def _replace_one(caption, regex, pool, rng):
    """regex에 걸리는 단어 중 하나를 골라, 캡션에 없는 다른 값으로 바꾼다. 같은 단어의 모든 등장을 함께 바꿈."""
    found = sorted({m.group(0).lower() for m in regex.finditer(caption)})
    if not found:
        return None
    src = rng.choice(found)
    present = {m.group(0).lower() for m in regex.finditer(caption)}
    cands = [w for w in pool if w not in present and not (src in ("wood", "wooden") and w in ("wood", "wooden"))]
    if not cands:
        return None
    dst = rng.choice(cands)
    return _word_re([src]).sub(lambda m: _keep_case(m.group(0), dst), caption)


def hard_negatives(caption, rng=None, kinds=("color", "bind", "rel", "pattern", "material")):
    rng = rng or random.Random(0)
    out = []
    if "color" in kinds:
        c = _replace_one(caption, _COLOR_RE, COLORS, rng)
        if c: out.append(("color", c))
    if "bind" in kinds:  # 서로 다른 색 두 개를 맞바꿈 = 색-물체 결합 교환 (0849↔1958 유형)
        cols = sorted({m.group(0).lower() for m in _COLOR_RE.finditer(caption)})
        if len(cols) >= 2:
            a, b = rng.sample(cols, 2)
            out.append(("bind", swap_pair(caption, a, b)))
    if "rel" in kinds:   # 관계 방향 뒤집기 (0002↔0652 유형)
        for a, b in REL_SWAPS:
            if _word_re([a, b]).search(caption):
                out.append((f"rel_{a}", swap_pair(caption, a, b)))
    if "pattern" in kinds:
        c = _replace_one(caption, _PAT_RE, PATTERNS, rng)
        if c: out.append(("pattern", c))
    if "material" in kinds:
        c = _replace_one(caption, _MAT_RE, MATERIALS, rng)
        if c: out.append(("material", c))
    return [(k, c) for k, c in out if c.lower() != caption.lower()]

"""그룹 단위 검증 분할: 형제 캡션(같은 물체 구성)이 같은 세트에 들어가도록 나눈다.
출력: split_group_seed{seed}.json = {"train": [...], "valA": [...], "valB": [...], "meta": {...}}
"""
import argparse, csv, json, random, collections
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--groups", default="eda/groups.csv")
ap.add_argument("--out", default="splits/split_group_seed42.json")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--n-val", type=int, default=800)
a = ap.parse_args()

members = collections.defaultdict(list)
with open(a.groups, newline="") as f:
    for r in csv.DictReader(f):
        members[int(r["group_id"])].append(r["id"])
gids = sorted(members)
random.Random(a.seed).shuffle(gids)

sets = {"valA": [], "valB": [], "train": []}
for g in gids:
    ids = members[g]
    for name in ("valA", "valB"):
        if len(sets[name]) + len(ids) <= a.n_val:
            sets[name] += ids
            break
    else:
        sets["train"] += ids
for k in sets:
    sets[k].sort(key=int)

# 검증: 겹침 없음, 형제가 갈라지지 않음
all_ids = sum(sets.values(), [])
assert len(all_ids) == len(set(all_ids)) == sum(len(v) for v in members.values())
where = {i: k for k, v in sets.items() for i in v}
assert all(len({where[i] for i in ids}) == 1 for ids in members.values())

def gsize_dist(ids):
    c = collections.Counter(len(members[g]) for g in members if members[g][0] in set(ids))
    return dict(sorted(c.items()))
meta = {"seed": a.seed, "sizes": {k: len(v) for k, v in sets.items()},
        "group_size_dist": {k: gsize_dist(v) for k, v in sets.items()}}
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
json.dump({**sets, "meta": meta}, open(a.out, "w"), indent=1)
print(json.dumps(meta, indent=1))

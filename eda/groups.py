# 같은 물체 구성(object signature)을 공유하는 캡션끼리 그룹 id 부여 -> groups.csv (id,group_id,group_size)
import json, sys, csv, collections
sys.path.insert(0, '.')
from eda3_sig import objsig
rows=[json.loads(l) for l in open(sys.argv[1])]
g={}; members=collections.defaultdict(list)
for r in rows:
    s=objsig(r['prompts']); g.setdefault(s,len(g)); members[g[s]].append(r['id'])
with open(sys.argv[2],'w',newline='') as f:
    w=csv.writer(f); w.writerow(['id','group_id','group_size'])
    for r in rows:
        gid=g[objsig(r['prompts'])]; w.writerow([r['id'],gid,len(members[gid])])
print('groups',len(g))

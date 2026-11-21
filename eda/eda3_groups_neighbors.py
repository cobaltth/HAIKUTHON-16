import json, re, collections, sys, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
rows=[json.loads(l) for l in open(sys.argv[1])]
ids=[r['id'] for r in rows]; caps=[r['prompts'] for r in rows]
ATTR=set('brown white black yellow green red beige orange gray blue purple pink solid-colored striped polka-dotted checkered metal wood wooden plastic glass rubber fabric ceramic open closed standing sitting lying crouching kneeling one two three four single a an the'.split())
STY=set('photograph photographed oil painting watercolor gouache colored-pencil pencil sketch ink drawing flat illustration low-poly d render rendered paper-cut collage clay-style front front-view side side-view overhead oblique view viewed angle lighting lit direct sunlight soft diffuse warm indoor scene image shows shown shot uses is are in with from under and of to left right next above below on rests resting sits inside holds holding wears wearing rides riding which that who also present appears appear set setting contains there it both by alongside has made color pattern studio plain backdrop as'.split())
def objsig(t):
    w=re.findall(r"[a-z\-]+",t.lower())
    return frozenset(x for x in w if x not in ATTR and x not in STY)
sig=collections.defaultdict(list)
for i,c in zip(ids,caps): sig[objsig(c)].append(i)
sizes=collections.Counter(len(v) for v in sig.values())
print('object-signature groups: size->count',dict(sorted(sizes.items())))
multi=[v for v in sig.values() if len(v)>1]
print('captions sharing object-set with another caption:',sum(len(v) for v in multi))
d=dict(zip(ids,caps))
for v in sorted(multi,key=len,reverse=True)[:4]:
    print('\n--- group',v[:6]); [print(' ',i,'|',d[i]) for i in v[:3]]
# tf-idf nearest neighbours
X=TfidfVectorizer(ngram_range=(1,2)).fit_transform(caps)
S=(X@X.T).toarray(); np.fill_diagonal(S,0)
nn=S.max(1); print('\nTF-IDF NN cosine quantiles 50/90/99/max',np.quantile(nn,[.5,.9,.99,1]).round(3))
top=np.dstack(np.unravel_index(np.argsort(-np.triu(S).ravel())[:6],S.shape))[0]
for a,b in top: print(f'\n{S[a,b]:.3f}\n  {ids[a]} | {caps[a]}\n  {ids[b]} | {caps[b]}')

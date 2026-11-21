import json, re, collections, sys
rows=[json.loads(l) for l in open(sys.argv[1])]
caps={r['id']:r['prompts'] for r in rows}
low={k:v.lower() for k,v in caps.items()}
C=collections.Counter
def cnt(name, pats):
    c=C()
    for t in low.values():
        hit=[p for p in pats if re.search(pats[p],t)]
        for h in hit: c[h]+=1
        if not hit: c['(none)']+=1
    print(f'\n[{name}]',dict(c.most_common()))
cnt('style',{'photograph':r'photograph','oil':r'oil painting','watercolor':r'watercolor','gouache':r'gouache','colored-pencil':r'colored-pencil|colored pencil','pencil sketch':r'pencil sketch','ink':r'ink drawing','flat illus':r'flat illustration','low-poly':r'low-poly','3d(non-lowpoly)':r'(?<!low-poly )3d render','paper-cut':r'paper-cut','clay':r'clay-style'})
cnt('view',{'front':r'front','side':r'side view|side-view|from the side|in side','overhead':r'overhead','oblique':r'oblique'})
cnt('light',{'direct sun':r'direct sunlight','side light':r'side lighting|side-lit|lit from the side','soft diffuse':r'soft diffuse','warm indoor':r'warm indoor'})
cnt('relation',{'left':r'\bleft\b','right':r'\bright\b','next to':r'next to|beside','above':r'\babove\b','below':r'\bbelow\b','on/rests':r'\brests? on|resting on|sits on|\bon top|\bon one\b|\bon the\b','inside':r'\binside\b|\bin one (box|suitcase|container|cabinet|bowl)','holds':r'\bhold','wears':r'\bwear','rides':r'\brid(es|ing)'})
cnt('count words',{'two':r'\btwo\b','three':r'\bthree\b','four':r'\bfour\b','single':r'\bsingle\b'})
cnt('pattern',{p:r'\b'+p+r'\b' for p in ['solid-colored','striped','polka-dotted','checkered']})
cnt('material',{m:r'\b'+m+r'\b' for m in ['metal','wood','wooden','plastic','glass','rubber','paper','fabric','ceramic']})
cnt('state',{m:r'\b'+m+r'\b' for m in ['open','closed','standing','sitting','lying','crouching','kneeling']})
cnt('setting',{s:r'\b'+s+r'\b' for s in ['studio','park','farm','living room','garden','bedroom','dining room','office','patio','meadow','kitchen','street','airport','pasture','workshop','cafe','schoolyard','beach','classroom']})
cols=['brown','white','black','yellow','green','red','beige','orange','gray','blue','purple','pink']
print('\n[colors]',dict(C(w for t in low.values() for w in re.findall(r'\b('+'|'.join(cols)+r')\b',t)).most_common()))
print('[#distinct colors per caption]',dict(sorted(C(len(set(re.findall(r'\b('+'|'.join(cols)+r')\b',t))) for t in low.values()).items())))
print('[person present]',sum('person' in t for t in low.values()))
# repeated object type within caption (two X of same category distinguished only by attributes), e.g., "one scarf ... one scarf"
rep=0;ex=[]
for k,t in low.items():
    objs=re.findall(r'\bone (?:[a-z\-]+ ){0,2}?([a-z]+)\b',t)

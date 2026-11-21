import json, re, collections, statistics as st, sys
from transformers import SiglipTokenizer
P=sys.argv[1]; M=sys.argv[2]
rows=[json.loads(l) for l in open(P)]
caps=[r['prompts'] for r in rows]
tok=SiglipTokenizer(vocab_file=M+'/spiece.model')
cfg=json.load(open(M+'/tokenizer_config.json')); print('tokcfg',{k:v for k,v in cfg.items() if k!='added_tokens_decoder'})
lens=[len(tok(c)['input_ids']) for c in caps]
ch=[len(c) for c in caps]
def q(x): s=sorted(x); return [s[int(len(s)*p)] for p in (0,.25,.5,.75,.9,.99)]+[s[-1]]
print('chars min/25/50/75/90/99/max',q(ch))
print('tokens min/25/50/75/90/99/max',q(lens))
for L in (64,77,128): print(f'>{L} tokens: {sum(l>L for l in lens)/len(lens):.1%}')
sent=[len(re.split(r'(?<=\.)\s+',c.strip())) for c in caps]; print('sentences per caption',collections.Counter(sent).most_common())
# what gets cut at 64
cut=[]
for c in caps:
    ids=tok(c)['input_ids']
    if len(ids)>64: cut.append(tok.decode(ids[63:]))
print('example truncated tails:'); [print('  ...',t[:150]) for t in cut[:5]]
words=collections.Counter(w for c in caps for w in re.findall(r"[a-z\-]+",c.lower()))
print('vocab size',len(words)); print(words.most_common(250))
json.dump(lens,open('toklens.json','w'))

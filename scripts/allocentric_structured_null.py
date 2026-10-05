import json, numpy as np, gzip, re
from pathlib import Path
R=Path('results/perspective-taking-n5000-20261004'); D=Path('examples/omnispatial-perspective-taking')
L='ABCD'
def parse(t):
    m=re.match(r'\s*(\S)',t); c=m.group(1).upper() if m else ''; return c if c in L else ''
search=[json.loads(l) for l in (D/'search.jsonl').read_text(encoding='utf-8').splitlines()]
allo=[i for i,r in enumerate(search) if r['sub_task_type']=='Allocentric']; gold=np.array([L[search[i]['answer']] for i in allo])
bo=json.loads(gzip.decompress((R/'baseline/base.json.gz').read_bytes()))['search']['outputs']
base=np.array([parse(bo[i]['text']) for i in allo])==gold
vec,sig=[],[]
for k in range(50):
    for f in sorted((R/f'search-shard-{k:03d}'/'candidates').glob('*.json.gz')):
        raw=json.loads(gzip.decompress(f.read_bytes())); o=raw['splits']['search']['outputs']
        vec.append(np.array([parse(o[i]['text']) for i in allo])==gold); sig.append(raw['candidate']['sigma'])
V=np.array(vec); S=np.array(sig); g=(V.sum(1)-base.sum())/65*100
rng=np.random.default_rng(1); res={}
sims={t:[] for t in (3,5,7)}; mx=[]
for _ in range(1000):
    G=[]
    for s in (.00025,.0005,.001,.002):
        flip=(V[S==s]!=base).mean(0)          # per-question flip rate for this sigma
        F=rng.random((1250,65))<flip
        C=np.where(F,~base,base)
        G.append((C.sum(1)-base.sum())/65*100)
    G=np.concatenate(G); mx.append(G.max())
    for t in sims: sims[t].append(int((G>=t-1e-9).sum()))
res={'observed':{t:int((g>=t-1e-9).sum()) for t in (3,5,7)},'observed_max':float(g.max()),
     'structured_null':{t:{'mean':float(np.mean(v)),'p95':float(np.quantile(v,.95)),'p_obs_ge':float(np.mean(np.array(v)>=int((g>=t-1e-9).sum())))} for t,v in sims.items()},
     'structured_null_max':{'mean':float(np.mean(mx)),'p95':float(np.quantile(mx,.95)),'P_max_ge_observed':float(np.mean(np.array(mx)>=g.max()-1e-9))},
     'fragile_questions':{'n_questions_flip_rate_ge_20pct':int(((V!=base).mean(0)>=.2).sum()),'mean_flip_rate':float((V!=base).mean())}}
print(json.dumps(res,indent=1))
out=R/'allocentric-followup/structured_null.json'
if not out.exists(): json.dump(res,open(out,'w'),indent=1)

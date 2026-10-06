"""Post-hoc, CPU-only: winner vs sigma=0.002 pool, accuracy change on SEARCH/RERANK split by whether gold contains 'front'."""
import json
from pathlib import Path

import pandas as pd

from answer_prior_analysis import OUT, PA, WIN, feats

ex = {}
for s in ('search', 'validation'):
    for l in Path(f'examples/omnispatial-perspective-taking/{s}.jsonl').read_text(encoding='utf-8').splitlines():
        r = json.loads(l); ex[r['uid']] = r
d = pd.read_parquet(PA / 'paper_master_predictions.parquet', columns=['candidate_id', 'sigma', 'phase', 'example_id', 'base_correct', 'candidate_correct'])
d = d[d.phase.isin(['SEARCH', 'RERANK']) & (d.sigma == 0.002)].copy()
d['example_id'] = d.example_id.astype(str); d['candidate_id'] = d.candidate_id.astype(str)
d['gold_front'] = [feats(ex[u]['options'][ex[u]['answer']])['front'] for u in d.example_id]
d['rep'] = (~d.base_correct.astype(bool)) & d.candidate_correct.astype(bool); d['reg'] = d.base_correct.astype(bool) & ~d.candidate_correct.astype(bool)
g = d.groupby(['phase', 'gold_front', 'candidate_id'], observed=True).agg(n=('rep', 'size'), rep=('rep', 'sum'), reg=('reg', 'sum')).reset_index()
g['net_pp'] = 100 * (g.rep - g.reg) / g.n
out = {}
for (ph, gf), x in g[g.n > 0].groupby(['phase', 'gold_front'], observed=True):
    w = x[x.candidate_id == WIN].iloc[0]; pool = x[x.candidate_id != WIN]
    out[f'{ph}|gold_front={gf}'] = {'n_items': int(w.n), 'winner_repairs': int(w.rep), 'winner_regressions': int(w.reg), 'winner_net_pp': float(w.net_pp),
                                     'pool_n': int(len(pool)), 'pool_mean_net_pp': float(pool.net_pp.mean()), 'winner_percentile': float(100 * (pool.net_pp < w.net_pp).mean())}
(OUT / 'answer_prior_goldsplit.json').write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))

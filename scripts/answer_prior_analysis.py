"""Post-hoc, CPU-only: do random perturbations act as global answer-prior shifts (toward option *content* such as
'front' / 'left' / 'can not determine'), and does that account for SEARCH/RERANK gains that fail on TEST?

Inputs: master predictions (letters per candidate x example), the three example files (option texts), and the
random-control vLLM outputs (controls' RERANK/TEST answers). No model runs.
"""
import gzip
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

PA = Path('results/paper-analysis')
OUT = PA / 'answer-prior'
WIN = '25a60f0b59f103ebf782a31f94bb569d897a652a0bc1bd95c5cdc5521418b843'
FEATS = ('front', 'back', 'left', 'right', 'cnd')
LET = 'ABCD'


def feats(text):
    t = text.lower()
    return {'front': 'front' in t or 'forward' in t, 'back': 'back' in t, 'left': 'left' in t, 'right': 'right' in t,
            'cnd': 'not determine' in t}


def fmt(options):
    o = [x.lower() for x in options]
    if sum('not determine' in x for x in o) >= 1 and all(any(k in x for k in ('left', 'right', 'not determine')) for x in o):
        return 'left_right_cnd'
    if any(('front' in x or 'back' in x) for x in o) and all(any(k in x for k in ('front', 'back', 'left', 'right')) for x in o):
        return 'compass'
    if any(re.search(r'\b(zero|one|two|three|four|five|six|seven|eight|nine)\b', x) for x in o):
        return 'count'
    return 'other'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ex = {}
    for s in ('search', 'validation', 'test'):
        for l in Path(f'examples/omnispatial-perspective-taking/{s}.jsonl').read_text(encoding='utf-8').splitlines():
            r = json.loads(l); ex[r['uid']] = r
    d = pd.read_parquet(PA / 'paper_master_predictions.parquet', columns=['candidate_id', 'sigma', 'phase', 'example_id', 'base_prediction',
                                                                          'candidate_prediction', 'base_correct', 'candidate_correct'])
    d['example_id'] = d.example_id.astype(str); d['candidate_id'] = d.candidate_id.astype(str)
    # controls' TEST/RERANK answers from the random-control run (vLLM, same protocol)
    ctrl = json.loads((PA / 'random-control-transfer/frozen_controls.json').read_text())['candidates']
    have = set(zip(d.candidate_id, d.phase))
    rows = []
    for c in ctrl:
        rr = json.loads(gzip.decompress((PA / f"random-control-transfer/gpu/control_{c['seed']}.json.gz").read_bytes()))
        for ph in ('RERANK', 'TEST'):
            if (c['candidate_id'], ph) in have: continue
            for q in rr[ph]:
                rows.append({'candidate_id': c['candidate_id'], 'sigma': c['sigma'], 'phase': ph, 'example_id': q['uid'], 'candidate_prediction': q['parsed']})
    if rows:
        add = pd.DataFrame(rows)
        base = d[d.candidate_id == 'BASE'][['phase', 'example_id', 'base_prediction', 'base_correct']].drop_duplicates(['phase', 'example_id'])
        add = add.merge(base, on=['phase', 'example_id'])
        add['candidate_correct'] = [p == LET[ex[u]['answer']] for p, u in zip(add.candidate_prediction, add.example_id)]
        d = pd.concat([d, add], ignore_index=True)
    ctrl_ids = {c['candidate_id'] for c in ctrl}
    d = d[d.candidate_id != 'BASE']
    # option-content features of the predicted option, base vs candidate
    def f_of(u, letter):
        if letter not in LET: return {k: False for k in FEATS}
        return feats(ex[u]['options'][LET.index(letter)])
    uniq = d[['example_id', 'base_prediction']].drop_duplicates()
    for k in FEATS:
        d[f'c_{k}'] = [f_of(u, p)[k] for u, p in zip(d.example_id, d.candidate_prediction)]
    bf = {(u, p): f_of(u, p) for u, p in zip(uniq.example_id, uniq.base_prediction)}
    for k in FEATS:
        d[f'b_{k}'] = [bf[(u, p)][k] for u, p in zip(d.example_id, d.base_prediction)]
    d['fmt'] = [fmt(ex[u]['options']) for u in d.example_id]
    d['gold_front'] = [feats(ex[u]['options'][ex[u]['answer']])['front'] for u in d.example_id]
    R = {'post_hoc': True}
    # set composition
    R['composition'] = {}
    for s, ph in (('search', 'SEARCH'), ('validation', 'RERANK'), ('test', 'TEST')):
        sub = d[(d.phase == ph)].drop_duplicates('example_id')
        R['composition'][ph] = {'n': int(len(sub)), 'format': sub.fmt.value_counts().to_dict(),
                                'gold_contains': {k: int(sum(feats(ex[u]['options'][ex[u]['answer']])[k] for u in sub.example_id)) for k in FEATS},
                                'base_pred_contains': {k: int(sub[f'b_{k}'].sum()) for k in FEATS}}
    # per candidate x phase summary
    g = d.groupby(['candidate_id', 'phase'])
    S = g.agg(sigma=('sigma', 'first'), n=('example_id', 'size'), gain=('candidate_correct', 'mean'), base=('base_correct', 'mean'),
              **{f'dc_{k}': (f'c_{k}', 'mean') for k in FEATS}, **{f'db_{k}': (f'b_{k}', 'mean') for k in FEATS}).reset_index()
    S['gain'] = 100 * (S.gain - S.base)
    for k in FEATS:
        S[f'shift_{k}'] = 100 * (S[f'dc_{k}'] - S[f'db_{k}'])  # pp change in share of answers whose option contains k
    S['is_winner'] = S.candidate_id == WIN; S['is_control'] = S.candidate_id.isin(ctrl_ids)
    S.to_csv(OUT / 'candidate_phase_answer_prior.csv', index=False)
    # across candidates: how much of gain is explained by answer-prior shifts (linear, per phase, sigma=0.002 and all)
    R['gain_vs_prior_shift'] = {}
    for ph in ('SEARCH', 'RERANK', 'TEST'):
        for sg in ('all', 0.002):
            T = S[(S.phase == ph) & ((S.sigma == sg) if sg != 'all' else (S.sigma > 0))]
            if len(T) < 10: continue
            X = np.column_stack([np.ones(len(T))] + [T[f'shift_{k}'] for k in FEATS]); y = T.gain.to_numpy()
            b, *_ = np.linalg.lstsq(X, y, rcond=None); r2 = 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            # 5-fold CV R^2
            idx = np.random.default_rng(0).permutation(len(y)); pr = np.zeros(len(y))
            for f in np.array_split(idx, 5):
                m = np.ones(len(y), bool); m[f] = False; bb, *_ = np.linalg.lstsq(X[m], y[m], rcond=None); pr[f] = X[f] @ bb
            R['gain_vs_prior_shift'][f'{ph}|sigma={sg}'] = {'n': int(len(T)), 'r_gain_shift': {k: float(np.corrcoef(T[f'shift_{k}'], y)[0, 1]) for k in FEATS},
                                                           'coef': dict(zip(['icpt', *FEATS], map(float, b))), 'r2_in': float(r2),
                                                           'r2_cv5': float(1 - ((y - pr) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
                                                           'sd_shift': {k: float(T[f'shift_{k}'].std()) for k in FEATS}}
    # winner vs controls vs RERANK pool percentiles
    W = S[S.is_winner].set_index('phase'); out = {}
    for ph in ('SEARCH', 'RERANK', 'TEST'):
        if ph not in W.index: continue
        pool = S[(S.phase == ph) & (S.sigma == 0.002) & ~S.is_winner]; cc = S[(S.phase == ph) & S.is_control]
        out[ph] = {'winner': {c: float(W.loc[ph, c]) for c in ['gain'] + [f'shift_{k}' for k in FEATS]},
                   'pool_sigma0.002_n': int(len(pool)),
                   'pool_percentile': {c: float(100 * (pool[c] < W.loc[ph, c]).mean()) for c in ['gain'] + [f'shift_{k}' for k in FEATS]},
                   'controls': {c: [float(x) for x in cc[c]] for c in ['gain'] + [f'shift_{k}' for k in FEATS]}}
    R['winner'] = out
    # winner: gain by format within TEST (and RERANK), with controls for reference
    fm = {}
    for ph in ('RERANK', 'TEST'):
        sub = d[(d.phase == ph)]
        for f, gg in sub.groupby('fmt'):
            w = gg[gg.candidate_id == WIN]; cs = gg[gg.candidate_id.isin(ctrl_ids)]
            if len(w) == 0: continue
            fm[f'{ph}|{f}'] = {'n': int(len(w)), 'winner_gain_pp': float(100 * (w.candidate_correct.mean() - w.base_correct.mean())),
                               'winner_repairs': int(((~w.base_correct.astype(bool)) & w.candidate_correct.astype(bool)).sum()),
                               'winner_regressions': int((w.base_correct.astype(bool) & ~w.candidate_correct.astype(bool)).sum()),
                               'controls_mean_gain_pp': float(100 * (cs.candidate_correct.astype(float).mean() - cs.base_correct.astype(float).mean())) if len(cs) else None}
    R['winner_by_format'] = fm
    (OUT / 'answer_prior_results.json').write_text(json.dumps(R, indent=1, default=float))
    return R


if __name__ == '__main__':
    R = main()
    print(json.dumps(R, indent=1, default=float)[:9000])

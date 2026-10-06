"""Post-hoc, CPU-only mechanism checks for the answer-content ('front') prior shift (see answer_prior_analysis.py).

1. Score-level bias: does 9504111 tilt letter log-probs toward options containing a direction word on every item,
   regardless of the gold answer (a global readout prior), and is that unusual against the 12 controls?
2. Reproducibility: across the 543 RERANK candidates, does a perturbation's answer-content shift on SEARCH predict its
   shift on RERANK (independent items, same format)? Does the gain left after the shift model replicate?
3. Switch targets: do the winner's changed answers move into 'front' options even where 'front' is wrong?
"""
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from answer_prior_analysis import FEATS, LET, OUT, PA, WIN, feats


def examples():
    ex = {}
    for s in ('search', 'validation', 'test'):
        for l in Path(f'examples/omnispatial-perspective-taking/{s}.jsonl').read_text(encoding='utf-8').splitlines():
            r = json.loads(l); ex[r['uid']] = r
    return ex


def score_level_bias(ex):
    ld = lambda f: json.loads(gzip.decompress(Path(f).read_bytes()))
    vec = lambda q: np.array([q['letter_logprobs'][a] for a in LET], float)
    base = {q['uid']: vec(q) for q in ld(PA / 'causal-diagnostic/session1/base_full.json.gz')}
    cands = {'9504111': {q['uid']: vec(q) for q in ld(PA / 'causal-diagnostic/session1/candidate_full.json.gz')}}
    for c in json.loads((PA / 'random-control-transfer/frozen_controls.json').read_text())['candidates']:
        rr = ld(PA / f"random-control-transfer/gpu/control_{c['seed']}.json.gz")
        cands[str(c['seed'])] = {q['uid']: vec(q) for ph in ('RERANK', 'TEST') for q in rr[ph]}
    rows = []
    for name, Z in cands.items():
        for u, z in Z.items():
            r = ex[u]; F = [feats(o) for o in r['options']]; gf = F[r['answer']]
            zb = base[u] - base[u].mean(); zc = z - z.mean()
            for f in ('front', 'back', 'left', 'right'):
                has = np.array([x[f] for x in F])
                if has.all() or not has.any():
                    continue
                db = (zc[has].mean() - zc[~has].mean()) - (zb[has].mean() - zb[~has].mean())
                rows.append({'cand': name, 'uid': u, 'phase': 'TEST' if u.startswith('test:') else 'RERANK', 'feat': f,
                             'gold_has': bool(gf[f]), 'd_bias': float(db)})
    D = pd.DataFrame(rows); D.to_csv(OUT / 'score_level_bias_per_item.csv', index=False)
    agg = D.groupby(['phase', 'feat', 'cand']).d_bias.agg(['mean', 'count', 'std']).reset_index()
    agg['t'] = agg['mean'] / (agg['std'] / np.sqrt(agg['count']))
    out = {}
    for (ph, f), g in agg.groupby(['phase', 'feat']):
        w = g[g.cand == '9504111'].iloc[0]; c = g[g.cand != '9504111']
        sub = D[(D.phase == ph) & (D.feat == f) & (D.cand == '9504111')]
        out[f'{ph}|{f}'] = {'n_items': int(w['count']), 'winner_mean_d_bias': float(w['mean']), 'winner_t': float(w['t']),
                            'controls_mean_d_bias': sorted(round(float(x), 3) for x in c['mean']),
                            'controls_abs_t_max': float(c['t'].abs().max()),
                            'winner_rank_desc_of_13': int(1 + (c['mean'] > w['mean']).sum()),
                            'winner_mean_if_gold_has': float(sub[sub.gold_has].d_bias.mean()) if sub.gold_has.any() else None,
                            'winner_mean_if_gold_lacks': float(sub[~sub.gold_has].d_bias.mean()) if (~sub.gold_has).any() else None,
                            'winner_frac_items_positive': float((sub.d_bias > 0).mean())}
    return out


def reproducibility():
    S = pd.read_csv(OUT / 'candidate_phase_answer_prior.csv')
    P = S.pivot(index='candidate_id', columns='phase')
    both = P.dropna(subset=[('gain', 'SEARCH'), ('gain', 'RERANK')])
    out = {'n_candidates': int(len(both))}
    for k in ('gain', 'shift_front', 'shift_back', 'shift_left', 'shift_right'):
        out[f'SEARCH_vs_RERANK_r_{k}'] = float(np.corrcoef(both[(k, 'SEARCH')], both[(k, 'RERANK')])[0, 1])

    def split(ph):
        X = np.column_stack([np.ones(len(both))] + [both[(f'shift_{k}', ph)] for k in FEATS[:4]]); y = both[('gain', ph)].to_numpy()
        b, *_ = np.linalg.lstsq(X, y, rcond=None); return y - X @ b, X @ b
    rS, pS = split('SEARCH'); rR, pR = split('RERANK')
    out['SEARCH_vs_RERANK_r_shift_explained_gain'] = float(np.corrcoef(pS, pR)[0, 1])
    out['SEARCH_vs_RERANK_r_residual_gain'] = float(np.corrcoef(rS, rR)[0, 1])
    w = list(both.index).index(WIN)
    out.update({'winner_RERANK_gain': float(both[('gain', 'RERANK')].iloc[w]), 'winner_RERANK_shift_explained': float(pR[w]),
                'winner_RERANK_residual': float(rR[w]), 'winner_RERANK_residual_percentile': float(100 * (rR < rR[w]).mean()),
                'winner_RERANK_shift_explained_percentile': float(100 * (pR < pR[w]).mean()),
                'winner_SEARCH_gain': float(both[('gain', 'SEARCH')].iloc[w]), 'winner_SEARCH_shift_explained': float(pS[w]),
                'winner_SEARCH_residual': float(rS[w])})
    tt = P.dropna(subset=[('gain', 'TEST'), ('gain', 'RERANK')])
    out['n_TEST_measured'] = int(len(tt))
    for k in ('gain', 'shift_front', 'shift_right', 'shift_left'):
        out[f'RERANK_{k}_vs_TEST_gain_r'] = float(np.corrcoef(tt[(k, 'RERANK')], tt[('gain', 'TEST')])[0, 1])
    out['RERANK_abs_front_shift_vs_TEST_gain_r'] = float(np.corrcoef(tt[('shift_front', 'RERANK')].abs(), tt[('gain', 'TEST')])[0, 1])
    return out


def switch_targets(ex):
    d = pd.read_parquet(PA / 'paper_master_predictions.parquet', columns=['candidate_id', 'sigma', 'phase', 'example_id', 'base_prediction', 'candidate_prediction'])
    d = d[d.phase.isin(['SEARCH', 'RERANK']) & (d.sigma == 0.002)].copy()
    d['example_id'] = d.example_id.astype(str); d['candidate_id'] = d.candidate_id.astype(str)
    d = d[(d.base_prediction != d.candidate_prediction) & d.candidate_prediction.isin(list(LET)) & d.base_prediction.isin(list(LET))]
    opt = lambda u, p: feats(ex[u]['options'][LET.index(p)])['front']
    d['net_front'] = [int(opt(u, c)) - int(opt(u, b)) for u, b, c in zip(d.example_id, d.base_prediction, d.candidate_prediction)]
    d['gold_front'] = [feats(ex[u]['options'][ex[u]['answer']])['front'] for u in d.example_id]
    per = pd.DataFrame({'n_changed': d.groupby('candidate_id').size(),
                        'net_front_gold_front': d[d.gold_front].groupby('candidate_id').net_front.sum(),
                        'net_front_gold_not_front': d[~d.gold_front].groupby('candidate_id').net_front.sum()}).fillna(0)
    w = per.loc[WIN]; pool = per.drop(WIN)
    return {'winner': {k: int(v) for k, v in w.items()}, 'pool_n': int(len(pool)),
            'pool_mean': {k: float(v) for k, v in pool.mean().items()},
            'winner_percentile': {k: float(100 * (pool[k] < w[k]).mean()) for k in per.columns},
            'note': 'SEARCH+RERANK, sigma=0.002 candidates with any SEARCH or RERANK record. net_front = changes into a front '
                    'option minus changes out of one. A global readout bias also moves answers into front where front is wrong.'}


if __name__ == '__main__':
    ex = examples()
    R = {'post_hoc': True, 'score_level_bias': score_level_bias(ex), 'reproducibility': reproducibility(), 'switch_targets': switch_targets(ex)}
    (OUT / 'answer_prior_mechanism_checks.json').write_text(json.dumps(R, indent=1, default=float))
    print(json.dumps(R, indent=1, default=float))

"""Locked analysis for M1 (results/paper-analysis/m1/plan_lock.md)."""
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from answer_prior_analysis import feats  # noqa: E402

PA = Path('results/paper-analysis')
D = PA / 'm1'
SRC = D / 'pod'
WIN = '25a60f0b59f103ebf782a31f94bb569d897a652a0bc1bd95c5cdc5521418b843'
LET = 'ABCD'
B = 2000


def ld(f):
    return json.loads(gzip.decompress(Path(f).read_bytes()))


def fisher_ci(r, n):
    z = np.arctanh(r); se = 1 / np.sqrt(n - 3)
    return [float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))]


def main():
    rows = [json.loads(l) for l in (SRC / 'holdout.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    gold = np.array([LET[r['answer']] for r in rows]); img = np.array([r['image_sha256'] for r in rows])
    front = np.array([feats(r['options'][r['answer']])['front'] for r in rows])
    base = ld(SRC / 'base.json.gz')['HOLDOUT']; cb = np.array([x['parsed'] for x in base]) == gold
    cands = json.loads((D / 'frozen_candidates.json').read_text())['candidates']
    C = {}
    for c in cands:
        f = SRC / f"cand_{c['seed']}.json.gz"
        if f.exists():
            C[c['candidate_id']] = (c, np.array([x['parsed'] for x in ld(f)['HOLDOUT']]) == gold)
    ui = np.unique(img); groups = [np.flatnonzero(img == u) for u in ui]; rng = np.random.default_rng(20261006)
    draws = [np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))]) for _ in range(B)]

    def gain(cc, mask=None):
        m = np.ones(len(cb), bool) if mask is None else mask
        return 100 * (cc[m].mean() - cb[m].mean())

    def boot(fn):
        v = np.array([fn(ix) for ix in draws]); return [float(np.quantile(v, .025)), float(np.quantile(v, .975))]
    R = {'n_items': len(rows), 'n_images': int(len(ui)), 'base_accuracy': float(100 * cb.mean()), 'n_candidates_measured': len(C),
         'parse_failures_base': int(sum(x['parsed'] == '' for x in base))}
    if WIN in C:
        w = C[WIN][1]; g = gain(w); ci = boot(lambda ix: 100 * (w[ix].mean() - cb[ix].mean()))
        R['M1_1_winner_gain'] = float(g); R['M1_1_ci95'] = ci
        R['M1_1_verdict'] = ('FULL TRANSFER' if ci[0] > 0 and g >= 6 else 'TRANSFERS' if ci[0] > 0 else 'DOES NOT TRANSFER')
        R['winner_repairs_regressions'] = [int((w & ~cb).sum()), int((~w & cb).sum())]
        d = lambda ix: (100 * (w[ix][front[ix]].mean() - cb[ix][front[ix]].mean())) - (100 * (w[ix][~front[ix]].mean() - cb[ix][~front[ix]].mean()))
        R['M1_2_front_minus_nonfront'] = float(gain(w, front) - gain(w, ~front)); R['M1_2_ci95'] = boot(d)
        R['M1_2_gain_front'] = float(gain(w, front)); R['M1_2_gain_nonfront'] = float(gain(w, ~front))
        R['M1_2_supports_prior_shift'] = R['M1_2_ci95'][0] > 0
    g_all = {cid: gain(v[1]) for cid, v in C.items()}
    if WIN in g_all:
        R['winner_rank_among_measured'] = int(1 + sum(x > g_all[WIN] for x in g_all.values()))
    if len(C) >= 40:
        d = pd.read_parquet(PA / 'paper_master_predictions.parquet', columns=['candidate_id', 'phase', 'candidate_correct', 'base_correct'])
        d['candidate_id'] = d.candidate_id.astype(str); d = d[d.phase == 'RERANK']
        gR = d.groupby('candidate_id').apply(lambda x: 100 * (x.candidate_correct.astype(float).mean() - x.base_correct.astype(float).mean()))
        meta = json.loads((PA / 'stage3/pod/meta.json').read_text()); idx = {c[0]: k for k, c in enumerate(meta['candidates'])}
        nv = [i for i, gname in enumerate(meta['groups']) if gname != 'vision']
        tau = np.load(PA / 'stage3/pod/proj_S.npy')[:, nv].sum(1) + np.load(PA / 'stage3/pod/proj_R.npy')[:, nv].sum(1)
        ids = [c for c in C if c in gR.index]
        x1 = np.array([gR[c] for c in ids]); x2 = np.array([tau[idx[c]] for c in ids]); y = np.array([g_all[c] for c in ids])
        r1 = float(np.corrcoef(x1, y)[0, 1]); r2 = float(np.corrcoef(x2, y)[0, 1])
        R['M1_3'] = {'n': len(ids), 'r_RERANKgain_H': r1, 'ci_RERANKgain_H': fisher_ci(r1, len(ids)), 'r_tau_H': r2, 'ci_tau_H': fisher_ci(r2, len(ids)),
                     'tau_supports': r2 >= 0.25}
        top = np.array([g_all[c] for c, v in C.items() if v[0]['role'] in ('top50', 'winner')]); ctl = np.array([g_all[c] for c, v in C.items() if v[0]['role'] == 'control'])
        bs = [rng.choice(top, len(top)).mean() - rng.choice(ctl, len(ctl)).mean() for _ in range(B)]
        R['M1_4'] = {'top50_mean_gain': float(top.mean()), 'controls_mean_gain': float(ctl.mean()), 'difference': float(top.mean() - ctl.mean()),
                     'ci95': [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], 'n_top': int(len(top)), 'n_controls': int(len(ctl))}
    else:
        R['M1_3'] = R['M1_4'] = 'INCONCLUSIVE (n<40)'
    (D / 'm1_results.json').write_text(json.dumps(R, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
    pd.DataFrame([{'candidate_id': c, 'role': v[0]['role'], 'gain_H': g_all[c]} for c, v in C.items()]).to_csv(D / 'm1_per_candidate.csv', index=False)
    return R


if __name__ == '__main__':
    print(json.dumps(main(), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))

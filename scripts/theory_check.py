"""Theory check (results/paper-analysis/theory/plan_lock.md). CPU, deterministic."""
import json
from math import erf, sqrt
from pathlib import Path

import numpy as np

P0 = Path('results/paper-analysis/p0'); OUT = Path('results/paper-analysis/theory/theory_results.json')
Phi = np.vectorize(lambda x: 0.5 * (1 + erf(x / sqrt(2))))


def auc(score, y):
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0: return float('nan')
    r = np.argsort(np.argsort(np.concatenate([pos, neg]), kind='mergesort')) + 1.0
    # average ranks for ties
    allv = np.concatenate([pos, neg]); u, inv = np.unique(allv, return_inverse=True)
    ranks = np.zeros(len(u)); np.add.at(ranks, inv, r); cnt = np.bincount(inv); r = (ranks / cnt)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    pred, meas = np.load(P0 / 'pod/pred.npy').astype(float), np.load(P0 / 'pod/meas.npy').astype(float)
    info = json.loads((P0 / 'pod/p0_info.json').read_text()); c = np.array([t['c_base'] for t in info['contrast_tokens']])
    cands = json.loads((P0 / 'candidates.json').read_text())['candidates']; sig = np.array([x['sigma'] for x in cands])
    assert pred.shape == (len(c), len(sig))
    s_hat = np.sqrt(np.mean((pred / sig) ** 2, axis=1))
    flip = (np.sign(c[:, None] + meas) != np.sign(c[:, None])).astype(int)
    p = Phi(-np.abs(c)[:, None] / (sig[None, :] * s_hat[:, None]))
    res = {'n_items': len(c), 'cv_s_hat': float(s_hat.std() / s_hat.mean())}
    for name, m in (('le_0.002', sig <= 0.002), ('0.005', sig == 0.005)):
        f, q = flip[:, m].ravel(), p[:, m].ravel(); a = auc(q, f); ratio = float(q.sum() / max(f.sum(), 1e-9))
        d = {'n_pairs': int(f.size), 'n_flips': int(f.sum()), 'auc': a, 'pred_over_obs': ratio, 'pred_flips': float(q.sum())}
        if name == 'le_0.002':
            d['T1a'] = 'UNDERPOWERED' if f.sum() < 20 else ('SUPPORTS' if a >= 0.80 else ('NOT SUPPORTED' if a < 0.65 else 'PARTIAL'))
            d['T1b'] = 'CALIBRATED' if 0.67 <= ratio <= 1.5 else ('MISCALIBRATED (over-predicts)' if ratio > 1.5 else 'MISCALIBRATED (under-predicts)')
        maj = np.sign(np.sum(np.sign(c[:, None] + meas[:, m]), axis=1)); maj[maj == 0] = np.sign(c[maj == 0])
        d['vote_equals_base_share'] = float(np.mean(maj == np.sign(c)))
        if name == 'le_0.002':
            v = d['vote_equals_base_share']; d['T2'] = 'SUPPORTS' if v >= 0.95 else ('NOT SUPPORTED' if v < 0.85 else 'PARTIAL')
        res[name] = d
    rng = np.random.default_rng(0)
    for s in sorted(set(sig)):
        mm = meas[:, sig == s].mean(axis=1); bs = [mm[rng.integers(0, len(mm), len(mm))].mean() for _ in range(5000)]
        res[f'mean_dc_sigma_{s}'] = [float(mm.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    pp = Phi(-np.abs(c) / (0.002 * s_hat)); target = pp.mean()
    Ts = np.linspace(0.01, 20, 4000); rates = [np.mean(1 / (1 + np.exp(np.abs(c) / T))) for T in Ts]
    T = float(Ts[int(np.argmin(np.abs(np.array(rates) - target)))]); ps = 1 / (1 + np.exp(np.abs(c) / T))
    from scipy.stats import spearmanr
    res['T4'] = {'matched_T': T, 'spearman_pert_vs_sampling': float(spearmanr(pp, ps).correlation), 'mean_flip_prob_sigma_0.002': float(target)}
    OUT.write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()

"""Locked analysis for R2 (results/paper-analysis/r2/plan_lock.md); same computations as r1_analysis.py."""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from r2_common import MID  # noqa: E402
from stage12_gradient import tilt_weights  # noqa: E402
import causal_diag_9504111 as cd  # noqa: E402


def examples():
    ex = {}
    for s in ('search', 'validation'):
        for l in (cd.DATA / f'{s}.jsonl').read_text(encoding='utf-8').splitlines():
            r = json.loads(l); ex[r['uid']] = r
    return ex

D = Path('results/paper-analysis/r2')
SRC = D / 'pod'


def r(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def main():
    m = json.loads((SRC / 'measurements.json').read_text()); meta = json.loads((SRC / 'meta.json').read_text())
    G = meta['groups']; cid = [c[0] for c in meta['seeds']]; idx = {c: k for k, c in enumerate(cid)}
    P = np.load(SRC / 'proj_FRONT.npy'); nv = [i for i, g in enumerate(G) if g != 'vision']
    ex = examples(); base = m['base']
    Wt = np.array([tilt_weights(ex[b['uid']]['options']) for b in base]); zb = np.array([b['z'] for b in base]); zb -= zb.mean(1, keepdims=True)
    rows = []
    for rec in m['perturbations']:
        k = idx[rec['candidate_id']]; row = {'cid': rec['candidate_id'], 'pred_NOV': float(P[k, nv].sum()), 'pred_V': float(P[k, G.index('vision')]),
                                             'fidelity_ok': rec['vision_reset_mismatch'] == 0 and rec['restore_mismatch'] == 0, 'missing': rec['missing']}
        for cond in ('FULL', 'NOV'):
            z = np.array([x['z'] for x in rec[cond]]); z -= z.mean(1, keepdims=True); dz = z - zb; per_item = (Wt * dz).sum(1)
            row[f'T_{cond}'] = float(per_item.mean()); row[f'T_{cond}_pos'] = float((Wt * dz.mean(0)).sum(1).mean())
            if cond == 'FULL':
                row['T_FULL_odd'], row['T_FULL_even'] = float(per_item[1::2].mean()), float(per_item[0::2].mean())
        rows.append(row)
    n = len(rows); col = lambda k: np.array([x[k] for x in rows])
    TF, TN, pN, pV = col('T_FULL'), col('T_NOV'), col('pred_NOV'), col('pred_V')
    rh = r(col('T_FULL_odd'), col('T_FULL_even')); sb = 2 * rh / (1 + rh)
    miss = int(col('missing').sum()) + sum(x['missing'] for x in base); miss_frac = miss / (4 * len(base) * (2 * n + 1))
    pos_share = float(col('T_NOV_pos').var() / TN.var())
    tot = P[:, nv].sum(1); share = {G[i]: float(np.cov(P[:, i], tot)[0, 1] / tot.var(ddof=1)) for i in nv}
    R = {'n_perturbations': n, 'n_items': len(base), 'S1a_spearman_brown': sb, 'S1a_pass': sb >= 0.7,
         'S1b_fidelity_all_ok': bool(all(col('fidelity_ok'))), 'missing_letter_fraction': miss_frac, 'missing_flag': miss_frac > 0.01,
         'P1_position_var_share_NOV': pos_share, 'P1_content_ok': pos_share < 0.2,
         'R2_primary_r_predNOV_TNOV': r(pN, TN), 'R2_slope': float(np.polyfit(pN, TN, 1)[0]),
         'R2_share_r2_TNOV_TFULL': r(TN, TF) ** 2,
         'secondary': {'r_predNOV_TFULL': r(pN, TF), 'r_predNOV+predV_TFULL': r(pN + pV, TF),
                       'r_predNOV_content_part': r(pN, TN - col('T_NOV_pos'))},
         'L2_predicted_var_share_by_group': share, 'L2_mid_blocks_share': float(sum(share[g] for g in MID)),
         'L2_mid_supported': sum(share[g] for g in MID) >= 0.5,
         'sd': {'T_FULL': float(TF.std(ddof=1)), 'T_NOV': float(TN.std(ddof=1)), 'pred_NOV': float(pN.std(ddof=1))}}
    s = R['R2_primary_r_predNOV_TNOV']
    R['R2_verdict'] = 'INCONCLUSIVE (n<40)' if n < 40 else ('PASS' if s >= 0.6 else 'FALSIFIED' if s < 0.3 else 'INCONCLUSIVE')
    R['interpretable'] = bool(R['S1a_pass'] and R['S1b_fidelity_all_ok'] and R['P1_content_ok'])
    R['replication'] = ('REPLICATED' if R['interpretable'] and R['R2_verdict'] == 'PASS' else
                        'NOT REPLICATED' if R['interpretable'] and R['R2_verdict'] == 'FALSIFIED' else 'INCONCLUSIVE')
    (D / 'r2_results.json').write_text(json.dumps(R, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
    (D / 'r2_per_perturbation.json').write_text(json.dumps(rows, indent=1))
    return R


if __name__ == '__main__':
    print(json.dumps(main(), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))

"""Locked analysis for Stage 1+2 (results/paper-analysis/stage12/plan_lock.md). Applies the gates exactly as written."""
import json
from pathlib import Path

import numpy as np

D = Path('results/paper-analysis/stage12')
SRC = D / 'pod'


def r(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def main():
    m = json.loads((SRC / 'measurements.json').read_text()); pr = json.loads((SRC / 'predictions.json').read_text())
    base = m['base']; P = {p['candidate_id']: p for p in pr['predictions']}
    n_items = len(base)
    rows = []
    for rec in m['perturbations']:
        dF = np.array([x['B'] - b['B'] for x, b in zip(rec['FULL'], base)]); dN = np.array([x['B'] - b['B'] for x, b in zip(rec['NOV'], base)])
        rows.append({'cid': rec['candidate_id'], 'role': rec['role'], 'T_FULL': dF.mean(), 'T_NOV': dN.mean(),
                     'T_FULL_odd': dF[1::2].mean(), 'T_FULL_even': dF[0::2].mean(),
                     'pred_NOV': P[rec['candidate_id']]['pred_NOV'], 'pred_V': P[rec['candidate_id']]['pred_V'], 'missing': rec['missing'],
                     'fidelity_ok': rec['vision_reset_mismatch'] == 0 and rec['restore_mismatch'] == 0})
    n = len(rows); col = lambda k: np.array([x[k] for x in rows])
    TF, TN, pN, pV = col('T_FULL'), col('T_NOV'), col('pred_NOV'), col('pred_V')
    rh = r(col('T_FULL_odd'), col('T_FULL_even')); sb = 2 * rh / (1 + rh)
    miss = int(col('missing').sum()) + sum(x['missing'] for x in base); miss_frac = miss / (4 * n_items * (2 * n + 1))
    R = {'n_perturbations': n, 'n_items': n_items,
         'S1a_split_half_r': rh, 'S1a_spearman_brown': sb, 'S1a_pass': sb >= 0.7,
         'S1b_fidelity_all_ok': bool(all(col('fidelity_ok'))), 'missing_letter_fraction': miss_frac, 'missing_flag': miss_frac > 0.01,
         'sd': {'T_FULL': float(TF.std(ddof=1)), 'T_NOV': float(TN.std(ddof=1)), 'pred_NOV': float(pN.std(ddof=1)), 'pred_V': float(pV.std(ddof=1))},
         'S2_primary_r_predNOV_TNOV': r(pN, TN), 'S2_primary_slope': float(np.polyfit(pN, TN, 1)[0]),
         'S2_share_r_TNOV_TFULL': r(TN, TF), 'S2_share_r2': r(TN, TF) ** 2,
         'secondary': {'r_predNOV_TFULL': r(pN, TF), 'r_predNOV+predV_TFULL': r(pN + pV, TF), 'r_predV_TFULL-TNOV': r(pV, TF - TN),
                       'slope_predNOV_TFULL': float(np.polyfit(pN, TF, 1)[0])}}
    s2 = R['S2_primary_r_predNOV_TNOV']
    R['S2_primary_verdict'] = 'INCONCLUSIVE (n<40)' if n < 40 else ('PASS' if s2 >= 0.6 else 'FALSIFIED' if s2 < 0.3 else 'INCONCLUSIVE')
    R['interpretable'] = R['S1a_pass'] and R['S1b_fidelity_all_ok']
    R['stage3_GO'] = bool(R['interpretable'] and R['S2_primary_verdict'] == 'PASS' and R['S2_share_r2'] >= 0.5)
    w = [x for x in rows if x['role'] == 'winner']
    if w:
        R['winner'] = {k: float(w[0][k]) for k in ('T_FULL', 'T_NOV', 'pred_NOV', 'pred_V')}
        R['winner_T_FULL_percentile'] = float(100 * (TF < w[0]['T_FULL']).mean())
    (D / 'stage12_results.json').write_text(json.dumps(R, indent=1))
    (D / 'stage12_per_perturbation.json').write_text(json.dumps(rows, indent=1, default=float))
    return R


if __name__ == '__main__':
    print(json.dumps(main(), indent=1))

"""Locked analysis for R2b (results/paper-analysis/r2b/plan_lock.md): Qwen2.5-VL-7B, 363 probe items, NOV only."""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from r2_common import MID  # noqa: E402
from stage12_gradient import tilt_weights  # noqa: E402
import causal_diag_9504111 as cd  # noqa: E402

D = Path('results/paper-analysis/r2b')
SRC = D / 'pod'


def r(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def main():
    ex = {}
    for s in ('search', 'validation'):
        for l in (cd.DATA / f'{s}.jsonl').read_text(encoding='utf-8').splitlines():
            q = json.loads(l); ex[q['uid']] = q
    m = json.loads((SRC / 'measurements.json').read_text()); meta = json.loads((SRC / 'meta.json').read_text())
    G = meta['groups']; idx = {c[0]: k for k, c in enumerate(meta['seeds'])}
    P = np.load(SRC / 'proj_FRONT.npy'); nv = [i for i, g in enumerate(G) if g != 'vision']; base = m['base']
    Wt = np.array([tilt_weights(ex[b['uid']]['options']) for b in base]); zb = np.array([b['z'] for b in base]); zb -= zb.mean(1, keepdims=True)
    rows = []
    for rec in m['perturbations']:
        z = np.array([x['z'] for x in rec['NOV']]); z -= z.mean(1, keepdims=True); dz = z - zb; per = (Wt * dz).sum(1)
        rows.append({'cid': rec['candidate_id'], 'pred_NOV': float(P[idx[rec['candidate_id']], nv].sum()), 'pred_V': float(P[idx[rec['candidate_id']], G.index('vision')]),
                     'T_NOV': float(per.mean()), 'T_NOV_pos': float((Wt * dz.mean(0)).sum(1).mean()), 'odd': float(per[1::2].mean()), 'even': float(per[0::2].mean()),
                     'fidelity_ok': rec['vision_reset_mismatch'] == 0 and rec['restore_mismatch'] == 0, 'missing': rec['missing']})
    n = len(rows); col = lambda k: np.array([x[k] for x in rows])
    TN, pN = col('T_NOV'), col('pred_NOV'); rh = r(col('odd'), col('even')); sb = 2 * rh / (1 + rh)
    miss = int(col('missing').sum()) + sum(x['missing'] for x in base); miss_frac = miss / (4 * len(base) * (n + 1))
    pos_share = float(col('T_NOV_pos').var() / TN.var())
    tot = P[:, nv].sum(1); share = {G[i]: float(np.cov(P[:, i], tot)[0, 1] / tot.var(ddof=1)) for i in nv}
    R = {'n_perturbations': n, 'n_items': len(base), 'S1a_spearman_brown_TNOV': sb, 'S1a_pass': sb >= 0.7,
         'S1b_fidelity_all_ok': bool(all(col('fidelity_ok'))), 'missing_letter_fraction': miss_frac, 'missing_flag': miss_frac > 0.01,
         'P1_position_var_share_NOV': pos_share, 'P1_content_ok': pos_share < 0.2,
         'R2b_primary_r_predNOV_TNOV': r(pN, TN), 'R2b_slope': float(np.polyfit(pN, TN, 1)[0]),
         'reported': {'spearman': float(__import__('pandas').Series(pN).corr(__import__('pandas').Series(TN), method='spearman')),
                      'r_predNOV_content_part': r(pN, TN - col('T_NOV_pos')), 'sd_T_NOV': float(TN.std(ddof=1)), 'sd_pred_NOV': float(pN.std(ddof=1)),
                      'mid_blocks_share_80': float(sum(share[g] for g in MID)), 'group_share_80': share}}
    s = R['R2b_primary_r_predNOV_TNOV']
    R['R2b_verdict'] = 'INCONCLUSIVE (n<40)' if n < 40 else ('PASS' if s >= 0.6 else 'FALSIFIED' if s < 0.3 else 'INCONCLUSIVE')
    R['interpretable'] = bool(R['S1a_pass'] and R['S1b_fidelity_all_ok'] and R['P1_content_ok'] and not R['missing_flag'])
    R['replication'] = ('REPLICATED' if R['interpretable'] and R['R2b_verdict'] == 'PASS' else
                        'NOT REPLICATED' if R['interpretable'] and R['R2b_verdict'] == 'FALSIFIED' else 'INCONCLUSIVE')
    (D / 'r2b_results.json').write_text(json.dumps(R, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
    (D / 'r2b_per_perturbation.json').write_text(json.dumps(rows, indent=1))
    return R


if __name__ == '__main__':
    print(json.dumps(main(), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))

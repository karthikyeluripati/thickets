"""Locked analysis for the P0 pilot (results/paper-analysis/p0/plan_lock.md)."""
import gzip
import json
from pathlib import Path

import numpy as np

D = Path('results/paper-analysis/p0')
SRC = D / 'pod'


def r_(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def main():
    pred = np.load(SRC / 'pred.npy'); meas = np.load(SRC / 'meas.npy'); info = json.loads((SRC / 'p0_info.json').read_text())
    cands = json.loads((D / 'candidates.json').read_text())['candidates']; sig = np.array([c['sigma'] for c in cands])
    items = json.loads((D / 'items.json').read_text(encoding='utf-8'))['heldout']
    R = {'gate_mapping': info['mapping_gate'], 'gate_recompute_maxdiff': info['base_recompute_maxdiff'], 'gate_restore': info['restored_sample_equal']}
    m = sig <= 0.002; x, y = pred[:, m].ravel(), meas[:, m].ravel()
    R['primary_pooled_r'] = r_(x, y); R['primary_slope'] = float(np.polyfit(x, y, 1)[0]); R['n_pairs'] = int(len(x))
    R['decision'] = 'GO' if R['primary_pooled_r'] >= 0.5 else 'NO-GO' if R['primary_pooled_r'] < 0.3 else 'WEAK'
    R['r_sigma_0.005'] = r_(pred[:, sig == 0.005].ravel(), meas[:, sig == 0.005].ravel())
    R['r_by_sigma'] = {str(s): r_(pred[:, sig == s].ravel(), meas[:, sig == s].ravel()) for s in sorted(set(sig))}
    per = np.array([r_(pred[j, m], meas[j, m]) if meas[j, m].std() > 0 else np.nan for j in range(len(items))])
    R['per_item_r'] = {'median': float(np.nanmedian(per)), 'share_ge_0.5': float(np.nanmean(per >= 0.5))}
    R['mean_abs_dc_by_sigma'] = {str(s): float(np.abs(meas[:, sig == s]).mean()) for s in sorted(set(sig))}
    ev = D / 'pod/eval'
    if (ev / 'base_p0.json.gz').exists():
        base = json.loads(gzip.decompress((ev / 'base_p0.json.gz').read_bytes()))
        bc = np.array([base[r['id']]['correct'] for r in items], float); R['cot_base_acc'] = float(100 * bc.mean())
        ch = np.full_like(meas, np.nan); gains = {}
        for k, c in enumerate(cands):
            f = ev / 'cands' / f"cand_{c['index']}.json.gz"
            if f.exists():
                res = json.loads(gzip.decompress(f.read_bytes()))['res']; cc = np.array([res[r['id']]['correct'] for r in items], float)
                ch[:, k] = cc - bc; gains[k] = float(100 * (cc.mean() - bc.mean()))
        R['cot_gain_by_candidate'] = gains; R['cot_gain_sd'] = float(np.std(list(gains.values()))) if gains else None
        ok = ~np.isnan(ch)
        R['r_pred_dc_vs_cot_change'] = r_(pred[ok], ch[ok]) if ok.sum() > 2 else None
        R['r_meas_dc_vs_cot_change'] = r_(meas[ok], ch[ok]) if ok.sum() > 2 else None
    (D / 'p0_results.json').write_text(json.dumps(R, indent=1))
    return R


if __name__ == '__main__':
    print(json.dumps(main(), indent=1))

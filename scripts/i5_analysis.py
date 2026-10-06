"""Locked analysis for I5 (results/paper-analysis/i5/plan_lock.md)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

D = Path('results/paper-analysis/i5')
S12 = Path('results/paper-analysis/stage12')


def main():
    pred = np.load(D / 'pod/pred_items.npy'); meta = json.loads((D / 'pod/meta.json').read_text())
    m = json.loads((S12 / 'pod/measurements.json').read_text()); base = m['base']
    assert [b['uid'] for b in base] == meta['probe_uids']
    order = {c: k for k, c in enumerate(meta['candidate_ids'])}
    meas = np.zeros_like(pred)
    for rec in m['perturbations']:
        k = order[rec['candidate_id']]; meas[:, k] = [x['B'] - b['B'] for x, b in zip(rec['NOV'], base)]
    s12 = {x['cid']: x['pred_NOV'] for x in json.loads((S12 / 'stage12_per_perturbation.json').read_text())}
    agg = pred.mean(0); ref = np.array([s12[c] for c in meta['candidate_ids']])
    R = {'gate_r_mean_item_pred_vs_stage2_pred': float(np.corrcoef(agg, ref)[0, 1])}
    R['gate_pass'] = R['gate_r_mean_item_pred_vs_stage2_pred'] >= 0.99
    x, y = pred.ravel(), meas.ravel()
    R['I5_1_pooled_r'] = float(np.corrcoef(x, y)[0, 1]); R['I5_1_slope'] = float(np.polyfit(x, y, 1)[0]); R['n_pairs'] = int(len(x))
    R['I5_1_verdict'] = ('ITEM-LEVEL LAW' if R['I5_1_pooled_r'] >= 0.6 else 'AVERAGE-ONLY' if R['I5_1_pooled_r'] < 0.3 else 'INTERMEDIATE')
    per = np.array([np.corrcoef(pred[j], meas[j])[0, 1] if meas[j].std() > 0 else np.nan for j in range(pred.shape[0])])
    R['I5_2_per_item_r'] = {'median': float(np.nanmedian(per)), 'q25': float(np.nanquantile(per, .25)), 'q75': float(np.nanquantile(per, .75)),
                            'share_ge_0.6': float(np.nanmean(per >= 0.6)), 'items_constant_measure': int(np.isnan(per).sum())}
    margin = np.abs(np.array(meta['base_B_fp32']))
    R['I5_3_spearman_per_item_r_vs_abs_baseB'] = float(pd.Series(per).corr(pd.Series(margin), method='spearman'))
    R['measured_quantization_nats'] = 0.125
    (D / 'i5_results.json').write_text(json.dumps(R, indent=1))
    return R


if __name__ == '__main__':
    print(json.dumps(main(), indent=1))

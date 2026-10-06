"""Post-hoc check (after the locked S2 PASS): is the measured 'front' tilt just a letter-position (A-D) bias interacting
with where front options sit? For each perturbation: letter bias b_k = mean over probe items of the centered change
dz_k; position-implied tilt T_pos = mean_j sum_k w_jk b_k. Content tilt = T - T_pos."""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from stage12_gradient import tilt_weights  # noqa: E402

D = Path('results/paper-analysis/stage12')
m = json.loads((D / 'pod/measurements.json').read_text()); pr = {p['candidate_id']: p for p in json.loads((D / 'pod/predictions.json').read_text())['predictions']}
ex = {}
for s in ('search', 'validation'):
    for l in Path(f'examples/omnispatial-perspective-taking/{s}.jsonl').read_text(encoding='utf-8').splitlines():
        r = json.loads(l); ex[r['uid']] = r
base = m['base']; Wt = np.array([tilt_weights(ex[b['uid']]['options']) for b in base])  # items x 4
zb = np.array([b['z'] for b in base]); zb -= zb.mean(1, keepdims=True)
print('front-option share by letter position:', (Wt > 0).mean(0).round(3), ' mean weight by position:', Wt.mean(0).round(3))
out = {}
for cond in ('FULL', 'NOV'):
    T, Tpos, Tcont = [], [], []
    for rec in m['perturbations']:
        z = np.array([x['z'] for x in rec[cond]]); z -= z.mean(1, keepdims=True); dz = z - zb
        b = dz.mean(0); t = (Wt * dz).sum(1).mean(); tp = (Wt * b).sum(1).mean()
        T.append(t); Tpos.append(tp); Tcont.append(t - tp)
    T, Tpos, Tcont = map(np.array, (T, Tpos, Tcont)); pN = np.array([pr[r_['candidate_id']]['pred_NOV'] for r_ in m['perturbations']])
    out[cond] = {'sd_T': float(T.std()), 'sd_position_part': float(Tpos.std()), 'sd_content_part': float(Tcont.std()),
                 'r_T_position': float(np.corrcoef(T, Tpos)[0, 1]), 'var_share_content': float(Tcont.var() / T.var()),
                 'r_predNOV_content_part': float(np.corrcoef(pN, Tcont)[0, 1]), 'r_predNOV_position_part': float(np.corrcoef(pN, Tpos)[0, 1])}
(D / 'position_check.json').write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))

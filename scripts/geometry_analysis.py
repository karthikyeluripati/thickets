"""CPU analysis of GPU-A outputs: the locked falsifiers F1-F4 and S2d (results/paper-analysis/geometry-gpu-a/plan_lock.md)."""
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from causal_diag_9504111 import GROUPS  # noqa: E402

OUT = Path('results/paper-analysis/geometry-gpu-a')
GPU = OUT / 'gpu'
CD = Path('results/paper-analysis/causal-diagnostic')
RC = Path('results/paper-analysis/random-control-transfer')
LET = 'ABCD'
WIN = '25a60f0b59f103ebf782a31f94bb569d897a652a0bc1bd95c5cdc5521418b843'


def ld(f):
    return json.loads(gzip.decompress(Path(f).read_bytes()))


def vec(x):
    return np.array([x['letter_logprobs'][a] for a in LET], float)


def main():
    R = {'post_hoc': True}
    base = {x['uid']: vec(x) for x in ld(CD / 'session1/base_full.json.gz')}
    scored = {WIN: {x['uid']: vec(x) for x in ld(CD / 'session1/candidate_full.json.gz')}}
    ctrl = json.loads((RC / 'frozen_controls.json').read_text())['candidates']
    for c in ctrl:
        r = ld(RC / f"gpu/control_{c['seed']}.json.gz")
        scored[c['candidate_id']] = {x['uid']: vec(x) for ph in ('RERANK', 'TEST') for x in r[ph]}
    d = pd.read_parquet('results/paper-analysis/paper_master_predictions.parquet', columns=['candidate_id', 'phase', 'example_id', 'candidate_correct', 'base_correct'])
    d['candidate_id'] = d.candidate_id.astype(str); d['example_id'] = d.example_id.astype(str)
    hfb = json.loads((GPU / 'hf_base_letter_logprobs.json').read_text())
    F1x, F1y, per_cand = [], [], {}
    for ph in ('SEARCH', 'RERANK', 'TEST'):
        meta = json.loads((GPU / f'meta_{ph}.json').read_text()); P = np.load(GPU / f'pred_{ph}.npy')
        uids, cands, comp = meta['uids'], [c[0] for c in meta['candidates']], meta['comparator']
        Lb = {u: (np.array(hfb[u]) if ph == 'SEARCH' else base[u]) for u in uids}
        mB = np.array([Lb[u][comp[u][0]] - Lb[u][comp[u][1]] for u in uids])
        # F1 on scored candidates (RERANK/TEST only)
        if ph != 'SEARCH':
            for cid in scored:
                if cid not in cands: continue
                k = cands.index(cid)
                t = np.array([(scored[cid][u][comp[u][0]] - scored[cid][u][comp[u][1]]) - (base[u][comp[u][0]] - base[u][comp[u][1]]) for u in uids])
                F1x.append(P[:, k]); F1y.append(t)
                per_cand.setdefault(cid, {})[ph] = {'r': float(np.corrcoef(P[:, k], t)[0, 1]), 'slope': float(np.polyfit(P[:, k], t, 1)[0])}
        # F3 across candidates: predicted vs observed accuracy gains
        obs = d[(d.phase == ph) & d.candidate_id.isin(cands)].pivot(index='candidate_id', columns='example_id', values='candidate_correct')
        bc = d[(d.phase == ph) & (d.candidate_id == 'BASE')].set_index('example_id').base_correct
        if ph == 'TEST':  # controls' TEST outputs come from the random-control run
            gold = {}
            for s in ('test',):
                for l in Path(f'examples/omnispatial-perspective-taking/{s}.jsonl').read_text(encoding='utf-8').splitlines():
                    r = json.loads(l); gold[r['uid']] = LET[r['answer']]
            for c in ctrl:
                if c['candidate_id'] not in obs.index:
                    rr = ld(RC / f"gpu/control_{c['seed']}.json.gz")
                    obs.loc[c['candidate_id']] = pd.Series({x['uid']: x['parsed'] == gold[x['uid']] for x in rr['TEST']})
        obs = obs.loc[cands, uids].astype(float)
        g_obs = 100 * (obs.to_numpy() - bc.loc[uids].to_numpy(float)).mean(1)
        pred_corr = (mB[:, None] + P > 0).astype(float); g_pred = 100 * (pred_corr - (mB > 0)[:, None]).mean(0)
        R[f'F3_{ph}'] = {'n_candidates': len(cands), 'r_pred_vs_obs_gain': float(np.corrcoef(g_pred, g_obs)[0, 1]),
                         'spearman': float(pd.Series(g_pred).corr(pd.Series(g_obs), method='spearman'))}
        if WIN in cands:
            k = cands.index(WIN)
            R[f'winner_{ph}'] = {'pred_gain': float(g_pred[k]), 'obs_gain': float(g_obs[k]),
                                 'pred_rank_best_1': int(1 + (g_pred > g_pred[k]).sum()), 'n': len(cands)}
            if ph != 'SEARCH':
                bcv = bc.loc[uids].to_numpy(bool); cc = obs.loc[WIN].to_numpy().astype(bool)
                changed = bcv != cc
                pflip = np.sign(mB + P[:, k]) != np.sign(mB)
                R[f'F2_{ph}'] = {'changed': int(changed.sum()), 'predicted_flip_on_changed': int((pflip & changed).sum()),
                                 'predicted_flip_on_unchanged': int((pflip & ~changed).sum()), 'unchanged': int((~changed).sum())}
    x, y = np.concatenate(F1x), np.concatenate(F1y)
    R['F1'] = {'pooled_r': float(np.corrcoef(x, y)[0, 1]), 'slope': float(np.polyfit(x, y, 1)[0]), 'n_pairs': int(len(x)), 'per_candidate': per_cand}
    ch = sum(R[f'F2_{p}']['changed'] for p in ('RERANK', 'TEST')); hit = sum(R[f'F2_{p}']['predicted_flip_on_changed'] for p in ('RERANK', 'TEST'))
    R['F2'] = {'changed_answers': ch, 'predicted': hit, 'agreement': hit / ch}
    # F4 geometry
    lists = json.loads((OUT / 'candidate_lists.json').read_text())['ALL_candidates']
    proj = np.load(GPU / 'set_projection_all5000.npy'); k = [c[0] for c in lists].index(WIN)
    R['F4'] = {'winner_percentile': {ph: float(100 * (proj[:, i] < proj[k, i]).mean()) for i, ph in enumerate(('SEARCH', 'RERANK', 'TEST'))},
               'cosines': [s for s in json.loads((GPU / 'cost_log.json').read_text())['steps'] if s['step'] == 'S2c_sets'][0]['cosines']}
    # S2d: group partials vs measured single-group insertions on LOCALIZATION
    gp = {}
    for ph in ('RERANK', 'TEST'):
        f = GPU / f'group_pred_{ph}.json'
        if f.exists(): gp.update(json.loads(f.read_text()))
    xs, ys, pg = [], [], {}
    meta = {**json.loads((GPU / 'meta_RERANK.json').read_text())['comparator'], **json.loads((GPU / 'meta_TEST.json').read_text())['comparator']}
    for g in GROUPS:
        I = {x['uid']: vec(x) for x in ld(CD / f'session1/insertion_{g}_loc.json.gz')}
        gx, gy = [], []
        for u, parts in gp.items():
            y_, rb = meta[u]
            gx.append(parts[g]); gy.append((I[u][y_] - I[u][rb]) - (base[u][y_] - base[u][rb]))
        pg[g] = float(np.corrcoef(gx, gy)[0, 1]); xs += gx; ys += gy
    R['S2d'] = {'pooled_r': float(np.corrcoef(xs, ys)[0, 1]), 'per_group_r': pg, 'n_examples': len(gp)}
    R['decision'] = {'F1_pass': R['F1']['pooled_r'] >= 0.6, 'F1_falsified': R['F1']['pooled_r'] < 0.3,
                     'F2_falsified': R['F2']['agreement'] < 0.5, 'F3_RERANK_falsified': R['F3_RERANK']['r_pred_vs_obs_gain'] < 0.3,
                     'F3_RERANK_support': R['F3_RERANK']['r_pred_vs_obs_gain'] >= 0.5, 'S2d_falsified': R['S2d']['pooled_r'] < 0.5}
    (OUT / 'geometry_results.json').write_text(json.dumps(R, indent=1))
    return R


if __name__ == '__main__':
    print(json.dumps(main(), indent=1))

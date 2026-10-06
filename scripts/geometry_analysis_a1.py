"""CPU analysis of GPU-A run 2 under amendment A1 (plan_lock_A1.md + unchanged parts of plan_lock.md)."""
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from causal_diag_9504111 import GROUPS  # noqa: E402

OUT = Path('results/paper-analysis/geometry-gpu-a')
GPU = OUT / 'gpu2'
CD = Path('results/paper-analysis/causal-diagnostic')
RC = Path('results/paper-analysis/random-control-transfer')
LET = 'ABCD'
WIN = '25a60f0b59f103ebf782a31f94bb569d897a652a0bc1bd95c5cdc5521418b843'
WIN_SEED = 9504111


def ld(f):
    return json.loads(gzip.decompress(Path(f).read_bytes()))


def main():
    R = {'post_hoc': True, 'amendment': 'A1'}
    bc = json.loads((GPU / 'base_contrast_fp32.json').read_text())
    s_base, comp = bc['s'], bc['comparator']
    f1 = json.loads((GPU / 'f1_candidate_contrasts.json').read_text())
    ctrl = json.loads((RC / 'frozen_controls.json').read_text())['candidates']
    seed2cid = {WIN_SEED: WIN, **{c['seed']: c['candidate_id'] for c in ctrl}}
    # ---- F1 (HF-internal)
    xs, ys, per = [], [], {}
    for ph in ('RERANK', 'TEST'):
        meta = json.loads((GPU / f'meta_{ph}.json').read_text()); P = np.load(GPU / f'pred_{ph}.npy')
        cands = [c[0] for c in meta['candidates']]; uids = meta['uids']
        for seed, cid in seed2cid.items():
            if str(seed) not in f1 or cid not in cands: continue
            k = cands.index(cid)
            t = np.array([f1[str(seed)][u]['s'] - s_base[u] for u in uids]); p = P[:, k]
            xs.append(p); ys.append(t)
            per.setdefault(str(seed), {})[ph] = {'r': float(np.corrcoef(p, t)[0, 1]), 'slope': float(np.polyfit(p, t, 1)[0])}
    x, y = np.concatenate(xs), np.concatenate(ys)
    R['F1'] = {'pooled_r': float(np.corrcoef(x, y)[0, 1]), 'slope_obs_on_pred': float(np.polyfit(x, y, 1)[0]), 'n_pairs': int(len(x)),
               'residual_energy_ratio': float(((y - x) ** 2).sum() / (y ** 2).sum()), 'per_candidate': per}
    # ---- F2 / F3 against vLLM-observed answers
    d = pd.read_parquet('results/paper-analysis/paper_master_predictions.parquet', columns=['candidate_id', 'phase', 'example_id', 'candidate_correct', 'base_correct'])
    d['candidate_id'] = d.candidate_id.astype(str); d['example_id'] = d.example_id.astype(str)
    gold = {}
    for s in ('test',):
        for l in Path(f'examples/omnispatial-perspective-taking/{s}.jsonl').read_text(encoding='utf-8').splitlines():
            r = json.loads(l); gold[r['uid']] = LET[r['answer']]
    for ph in ('SEARCH', 'RERANK', 'TEST'):
        meta = json.loads((GPU / f'meta_{ph}.json').read_text()); P = np.load(GPU / f'pred_{ph}.npy')
        cands = [c[0] for c in meta['candidates']]; uids = meta['uids']
        mB = np.array([s_base[u] for u in uids])
        obs = d[(d.phase == ph) & d.candidate_id.isin(cands)].pivot(index='candidate_id', columns='example_id', values='candidate_correct')
        if ph == 'TEST':
            for c in ctrl:
                if c['candidate_id'] not in obs.index:
                    rr = ld(RC / f"gpu/control_{c['seed']}.json.gz")
                    obs.loc[c['candidate_id']] = pd.Series({q['uid']: q['parsed'] == gold[q['uid']] for q in rr['TEST']})
        obs = obs.loc[cands, uids].astype(float)
        bcv = d[(d.phase == ph) & (d.candidate_id == 'BASE')].set_index('example_id').base_correct.loc[uids].to_numpy(float)
        g_obs = 100 * (obs.to_numpy() - bcv).mean(1)
        g_pred = 100 * ((mB[:, None] + P > 0).astype(float) - (mB > 0)[:, None]).mean(0)
        R[f'F3_{ph}'] = {'n_candidates': len(cands), 'r_pred_vs_obs_gain': float(np.corrcoef(g_pred, g_obs)[0, 1]),
                         'spearman': float(pd.Series(g_pred).corr(pd.Series(g_obs), method='spearman')),
                         'base_margin_sign_agrees_vllm_base_correct': float(((mB > 0) == (bcv > 0)).mean())}
        np.save(OUT / f'a1_gain_pred_obs_{ph}.npy', np.vstack([g_pred, g_obs]))
        if WIN in cands:
            k = cands.index(WIN)
            R[f'winner_{ph}'] = {'pred_gain': float(g_pred[k]), 'obs_gain': float(g_obs[k]), 'pred_rank_best_1': int(1 + (g_pred > g_pred[k]).sum()), 'n': len(cands)}
            if ph != 'SEARCH':
                cc = obs.loc[WIN].to_numpy() > 0; changed = (bcv > 0) != cc
                pflip = np.sign(mB + P[:, k]) != np.sign(mB)
                R[f'F2_{ph}'] = {'changed': int(changed.sum()), 'predicted_flip_on_changed': int((pflip & changed).sum()),
                                 'predicted_flip_on_unchanged': int((pflip & ~changed).sum()), 'unchanged': int((~changed).sum())}
    ch = sum(R[f'F2_{p}']['changed'] for p in ('RERANK', 'TEST')); hit = sum(R[f'F2_{p}']['predicted_flip_on_changed'] for p in ('RERANK', 'TEST'))
    R['F2'] = {'changed_answers': ch, 'predicted': hit, 'agreement': hit / ch}
    # ---- F4
    lists = json.loads((OUT / 'candidate_lists.json').read_text())['ALL_candidates']
    proj = np.load(GPU / 'set_projection_all5000.npy'); k = [c[0] for c in lists].index(WIN)
    steps = json.loads((GPU / 'cost_log.json').read_text())['steps']
    R['F4'] = {'winner_percentile': {ph: float(100 * (proj[:, i] < proj[k, i]).mean()) for i, ph in enumerate(('SEARCH', 'RERANK', 'TEST'))},
               'sets': [s for s in steps if s['step'] == 'S2c_sets'][0]}
    # ---- S2d against session-1 measured insertion shifts (vLLM scores)
    base_v = {q['uid']: np.array([q['letter_logprobs'][a] for a in LET]) for q in ld(CD / 'session1/base_full.json.gz')}
    gp = {}
    for ph in ('RERANK', 'TEST'):
        f = GPU / f'group_pred_{ph}.json'
        if f.exists(): gp.update(json.loads(f.read_text()))
    xs, ys, pg = [], [], {}
    for g in GROUPS:
        I = {q['uid']: np.array([q['letter_logprobs'][a] for a in LET]) for q in ld(CD / f'session1/insertion_{g}_loc.json.gz')}
        gx, gy = [], []
        for u, parts in gp.items():
            yy, rb = comp[u]; gx.append(parts[g]); gy.append((I[u][yy] - I[u][rb]) - (base_v[u][yy] - base_v[u][rb]))
        pg[g] = float(np.corrcoef(gx, gy)[0, 1]); xs += gx; ys += gy
    R['S2d'] = {'pooled_r': float(np.corrcoef(xs, ys)[0, 1]), 'slope': float(np.polyfit(xs, ys, 1)[0]), 'per_group_r': pg, 'n_examples': len(gp)}
    R['decision'] = {'F1_support': R['F1']['pooled_r'] >= 0.6, 'F1_falsified': R['F1']['pooled_r'] < 0.3,
                     'F2_falsified': R['F2']['agreement'] < 0.5, 'F3_RERANK_falsified': R['F3_RERANK']['r_pred_vs_obs_gain'] < 0.3,
                     'F3_RERANK_support': R['F3_RERANK']['r_pred_vs_obs_gain'] >= 0.5, 'S2d_falsified': R['S2d']['pooled_r'] < 0.5}
    (OUT / 'geometry_results_A1.json').write_text(json.dumps(R, indent=1))
    return R


if __name__ == '__main__':
    print(json.dumps(main(), indent=1))

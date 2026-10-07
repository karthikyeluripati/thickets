"""Locked analysis for G1 (results/paper-analysis/g1/plan_lock.md)."""
from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np

D = Path('results/paper-analysis/g1')
SRC = D / 'pod'
B = 2000


def ld(f):
    return json.loads(gzip.decompress(Path(f).read_bytes()))


def r_(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and np.std(x) > 0 and np.std(y) > 0 else float('nan')


def main():
    items = json.loads((D / 'frozen_items.json').read_text(encoding='utf-8')); S, HO = items['selection'], items['heldout']
    bases = sorted(SRC.glob('base_gpu*.json.gz')); base = ld(bases[0])
    det = [np.mean([ld(f)[r['id']]['correct'] == base[r['id']]['correct'] for r in S + HO]) for f in bases]
    cands = {}
    for f in sorted((SRC / 'cands').glob('cand_*.json.gz')):
        x = ld(f); cands[x['candidate']['index']] = x
    R = {'n_candidates': len(cands), 'V_det_min_agreement': float(min(det)), 'V_det_pass': min(det) >= 0.99,
         'V_fid_all_ok': all(x['vision_changed'] == 0 and x['restore_mismatch'] == 0 for x in cands.values())}
    acc = lambda res, rows: 100 * np.mean([res[r['id']]['correct'] for r in rows])
    bS, bH = acc(base, S), acc(base, HO); R['base_acc'] = {'selection': bS, 'heldout': bH}
    idx = sorted(cands); sig = np.array([cands[i]['candidate']['sigma'] for i in idx])
    gS = np.array([acc(cands[i]['res'], S) - bS for i in idx]); gH = np.array([acc(cands[i]['res'], HO) - bH for i in idx])
    # ---- G1-A
    w = idx[int(np.argmax(gS))]; wres = cands[w]['res']
    img = np.array([r['imageId'] for r in HO]); groups = [np.flatnonzero(img == u) for u in np.unique(img)]; rng = np.random.default_rng(20261006)
    cw = np.array([wres[r['id']]['correct'] for r in HO], float); cb = np.array([base[r['id']]['correct'] for r in HO], float)
    bs = []
    for _ in range(B):
        ix = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))]); bs.append(100 * (cw[ix].mean() - cb[ix].mean()))
    ci = [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]; wS, wH = float(gS[idx.index(w)]), float(gH[idx.index(w)])
    verdict = 'INFLATED' if wH <= 0.5 * wS and ci[1] < wS else 'TRANSFERS' if wH >= 0.8 * wS and ci[0] > 0 else 'PARTIAL'
    top = np.argsort(-gS, kind='stable')[:10]
    R['G1_A'] = {'winner_index': w, 'winner_sigma': cands[w]['candidate']['sigma'], 'winner_selection_gain': wS, 'winner_heldout_gain': wH,
                 'winner_heldout_ci95': ci, 'verdict': verdict, 'top10_mean_selection': float(gS[top].mean()), 'top10_mean_heldout': float(gH[top].mean()),
                 'r_sel_ho_all': r_(gS, gH), 'r_sel_ho_by_sigma': {str(s): r_(gS[sig == s], gH[sig == s]) for s in sorted(set(sig))},
                 'gain_sd_by_sigma': {str(s): {'sel': float(gS[sig == s].std()), 'ho': float(gH[sig == s].std()), 'mean_ho': float(gH[sig == s].mean())} for s in sorted(set(sig))}}
    # ---- G1-B
    yn_ho = [r for r in HO if r['yesno']]

    def con(res, r):
        x = res[r['id']]; f = x['lp_floor']
        y = x['lp_yes'] if x['lp_yes'] is not None else f; n = x['lp_no'] if x['lp_no'] is not None else f
        return y - n
    T = np.array([np.mean([con(cands[i]['res'], r) - con(base, r) for r in yn_ho]) for i in idx])
    missing = sum((cands[i]['res'][r['id']]['lp_yes'] is None) + (cands[i]['res'][r['id']]['lp_no'] is None) for i in idx for r in yn_ho)
    proj = {}
    for X in ('HO', 'SEL'):
        f = SRC / f'proj_{X}.npy'
        if f.exists():
            meta = json.loads((SRC / 'meta.json').read_text()); pos = {c: k for k, c in enumerate(meta['candidates'])}
            P = np.load(f).sum(1); proj[X] = np.array([P[pos[i]] for i in idx])
    if 'HO' in proj:
        m = sig <= 0.002; rB = r_(proj['HO'][m], T[m])
        R['G1_B'] = {'n': int(m.sum()), 'r': rB, 'slope': float(np.polyfit(proj['HO'][m], T[m], 1)[0]) if m.sum() > 2 else None,
                     'verdict': 'INCONCLUSIVE (n<100)' if m.sum() < 100 else 'PASS' if rB >= 0.6 else 'FALSIFIED' if rB < 0.3 else 'INCONCLUSIVE',
                     'r_by_sigma': {str(s): r_(proj['HO'][sig == s], T[sig == s]) for s in sorted(set(sig))}, 'missing_letter_scores': int(missing)}
    else:
        R['G1_B'] = 'VOID (no gradient output / mapping gate failed)'
    # ---- G1-C
    def yes_share(res):
        return np.mean([str(res[r['id']]['pred']).strip().lower().startswith('yes') for r in yn_ho])
    ys = np.array([yes_share(cands[i]['res']) - yes_share(base) for i in idx]) * 100
    gy = [r for r in yn_ho if r['answer'] == 'yes']; gn = [r for r in yn_ho if r['answer'] == 'no']
    g_yes = np.array([acc(cands[i]['res'], gy) - acc(base, gy) for i in idx]); g_no = np.array([acc(cands[i]['res'], gn) - acc(base, gn) for i in idx])
    r2 = r_(ys, gH) ** 2 if len(idx) > 2 else float('nan')
    R['G1_C'] = {'n': len(idx), 'R2_HOgain_on_yesshift': r2, 'r_yesshift_gain_goldyes': r_(ys, g_yes), 'r_yesshift_gain_goldno': r_(ys, g_no),
                 'verdict': 'INCONCLUSIVE (n<200)' if len(idx) < 200 else 'PRIOR-DOMINATED' if r2 >= 0.30 else 'MINOR' if r2 < 0.10 else 'MODERATE',
                 'base_yes_rate_HO_yesno': float(yes_share(base)), 'gold_yes_rate_HO_yesno': len(gy) / len(yn_ho),
                 'top10_gain_goldyes_goldno': [[float(g_yes[k]), float(g_no[k])] for k in top]}
    # ---- G1-D / G1-E
    if 'SEL' in proj:
        R['G1_D'] = {'r_tauSEL_HOgain': r_(proj['SEL'], gH), 'r_tauSEL_yesshift': r_(proj['SEL'], ys)}
    votes = []
    for r in HO:
        c = Counter(str(cands[idx[k]]['res'][r['id']]['pred']).strip().lower() for k in top).most_common(1)[0][0]
        votes.append(c == r['answer'])
    R['G1_E'] = {'top10_vote_heldout_acc': float(100 * np.mean(votes)), 'base_heldout_acc': bH, 'note': 'approximate: exact-match on extracted answers'}
    (D / 'g1_results.json').write_text(json.dumps(R, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
    return R


if __name__ == '__main__':
    print(json.dumps(main(), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))

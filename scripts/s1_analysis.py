"""Locked analysis for Session 1 (results/paper-analysis/s1/plan_lock.md). CPU only.
Usage: python scripts/s1_analysis.py --a-dir <dir with A_pred.npy, A_meas.npy, A_info.json> --b-pred <B_pred.json>"""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

S1 = Path('results/paper-analysis/s1'); M1 = Path('results/paper-analysis/m1/pod'); LET = 'ABCD'


def r(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def boot_items(pred, meas, n=2000, seed=0):
    rng = np.random.default_rng(seed); J = pred.shape[0]; out = []
    for _ in range(n):
        i = rng.integers(0, J, J); out.append(r(pred[i].ravel(), meas[i].ravel()))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def rule(v, go=0.5, no=0.3):
    return 'GO' if v >= go else ('NO-GO' if v < no else 'INCONCLUSIVE')


def analyse_a(d):
    pred, meas = np.load(d / 'A_pred.npy'), np.load(d / 'A_meas.npy'); info = json.loads((d / 'A_info.json').read_text())
    sig = np.array([c['sigma'] for c in json.loads((S1 / 'frozen_A.json').read_text())['candidates']])[: pred.shape[1]]
    valid = bool(info['restored_exact']) and info['base_acc'] >= 0.30 and not info['smoke']
    m = sig <= 0.002; p, y = pred[:, m], meas[:, m]; r1 = r(p.ravel(), y.ravel())
    out = {'valid': valid, 'base_acc': info['base_acc'], 'restored_exact': info['restored_exact'],
           'A1_r_sigma_le_0.002': r1, 'A1_ci': boot_items(p, y), 'A1_outcome': rule(r1) if valid else 'UNINFORMATIVE',
           'slope_le_0.002': float(np.polyfit(p.ravel(), y.ravel(), 1)[0]),
           'r_by_sigma': {str(s): r(pred[:, sig == s].ravel(), meas[:, sig == s].ravel()) for s in sorted(set(sig))},
           'frac_perturbations_r_pos': float(np.mean([r(pred[:, k], meas[:, k]) > 0 for k in range(pred.shape[1])]))}
    return out


def load(p):
    x = json.loads(gzip.decompress(Path(p).read_bytes())); x = x['HOLDOUT'] if isinstance(x, dict) else x
    return {e['uid']: e for e in x}


def analyse_b(bp):
    pr = json.loads(Path(bp).read_text()); comp = json.loads((S1 / 'B_comparators.json').read_text())
    base, cand = load(M1 / 'base.json.gz'), load(M1 / 'cand_9504111.json.gz')
    rows = [json.loads(l) for l in (M1 / 'holdout.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    P, D, PN, PV, CH, CV, excl, chg = [], [], [], [], [], [], 0, []
    for row in rows:
        u = row['uid']; g, w = comp[u]; lb, lc = base[u]['letter_logprobs'], cand[u]['letter_logprobs']
        if any(x[LET[k]] is None for x in (lb, lc) for k in (g, w)):
            excl += 1; continue
        d = (lc[LET[g]] - lc[LET[w]]) - (lb[LET[g]] - lb[LET[w]]); q = pr[u]; pt = q['pred_NOV'] + q['pred_V']
        P.append(pt); D.append(d); PN.append(q['pred_NOV']); PV.append(q['pred_V']); CH.append(q['c_base_hf']); CV.append(lb[LET[g]] - lb[LET[w]])

        def arg(lp):
            return max(range(4), key=lambda k: (lp[LET[k]] if lp[LET[k]] is not None else -1e9))
        bc, cc = arg(lb) == g, arg(lc) == g
        if bc != cc:
            chg.append((1 if cc else -1, pt))
    P, D = np.array(P), np.array(D); b1 = r(P, D)
    rng = np.random.default_rng(0); bs = [r(P[i], D[i]) for i in (rng.integers(0, len(P), len(P)) for _ in range(2000))]
    agree = float(np.mean([np.sign(p) == s for s, p in chg])) if chg else float('nan')
    b2 = 'UNDERPOWERED' if len(chg) < 20 else ('SUPPORTED' if agree >= 0.70 else ('NOT SUPPORTED' if agree < 0.60 else 'INCONCLUSIVE'))
    return {'n_included': len(P), 'n_excluded_floor': excl, 'B1_r': b1, 'B1_ci': [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            'B1_outcome': rule(b1), 'B2_n_changed': len(chg), 'B2_n_repairs': sum(s > 0 for s, _ in chg), 'B2_sign_agreement': agree, 'B2_outcome': b2,
            'r_NOV_only': r(np.array(PN), D), 'r_V_only': r(np.array(PV), D), 'var_share_V': float(np.var(PV) / (np.var(PN) + np.var(PV))),
            'r_base_c_hf_vs_vllm': r(np.array(CH), np.array(CV)), 'slope': float(np.polyfit(P, D, 1)[0])}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--a-dir', type=Path); ap.add_argument('--b-pred', type=Path); ap.add_argument('--out', type=Path, default=S1 / 's1_results.json')
    a = ap.parse_args(); res = {}
    if a.a_dir: res['A'] = analyse_a(a.a_dir)
    if a.b_pred: res['B'] = analyse_b(a.b_pred)
    a.out.write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()

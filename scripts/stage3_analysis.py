"""Locked analysis for Stage 3 (results/paper-analysis/stage3/plan_lock.md). Applies the rules exactly as written."""
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

PA = Path('results/paper-analysis')
D = PA / 'stage3'
SRC = D / 'pod'
WIN = '25a60f0b59f103ebf782a31f94bb569d897a652a0bc1bd95c5cdc5521418b843'
LET = 'ABCD'


def r(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def stage3a(R):
    meta = json.loads((SRC / 'meta.json').read_text()); G = meta['groups']; cands = meta['candidates']; nv = [i for i, g in enumerate(G) if g != 'vision']
    tau = {X: np.load(SRC / f'proj_{X}.npy')[:, nv].sum(1) for X in ('P96', 'R', 'S')}
    cid = [c[0] for c in cands]; sig = np.array([c[2] for c in cands]); idx = {c: k for k, c in enumerate(cid)}
    # 3A-0
    s12 = json.loads((PA / 'stage12/stage12_per_perturbation.json').read_text())
    R['3A0_r_tauP96_measured_TNOV'] = r([tau['P96'][idx[x['cid']]] for x in s12], [x['T_NOV'] for x in s12])
    R['3A0_pass'] = R['3A0_r_tauP96_measured_TNOV'] >= 0.9
    d = pd.read_parquet(PA / 'paper_master_predictions.parquet', columns=['candidate_id', 'phase', 'example_id', 'candidate_correct', 'base_correct'])
    d['candidate_id'] = d.candidate_id.astype(str)
    gain = d[d.candidate_id != 'BASE'].groupby(['phase', 'candidate_id']).apply(lambda x: 100 * (x.candidate_correct.astype(float).mean() - x.base_correct.astype(float).mean()))
    gS = gain['SEARCH'].reindex(cid).to_numpy(); ok = ~np.isnan(gS)
    R['3A1_r_tauR_SEARCHgain'] = r(tau['R'][ok], gS[ok]); R['3A1_n'] = int(ok.sum())
    R['3A1_within_sigma0.002'] = r(tau['R'][ok & (sig == 0.002)], gS[ok & (sig == 0.002)])
    R['3A1_by_sigma'] = {str(s): r(tau['R'][ok & (sig == s)], gS[ok & (sig == s)]) for s in sorted(set(sig))}
    gR = gain['RERANK']; rr = [c for c in gR.index if c in idx]
    R['3A2_r_tauS_RERANKgain'] = r([tau['S'][idx[c]] for c in rr], gR.loc[rr].to_numpy()); R['3A2_n'] = len(rr)
    rng = np.random.default_rng(0); adv = np.isin(cid, rr); z = np.zeros(len(cid))
    for s in set(sig):
        m = sig == s; z[m] = (tau['R'][m] - tau['R'][m].mean()) / tau['R'][m].std()
    diff = z[adv].mean() - z[~adv].mean()
    bs = [z[rng.choice(np.flatnonzero(adv), adv.sum())].mean() - z[rng.choice(np.flatnonzero(~adv), (~adv).sum())].mean() for _ in range(2000)]
    R['3A3_enrichment_sd'] = float(diff); R['3A3_ci95'] = [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]; R['3A3_supports'] = diff >= 0.3
    w = idx[WIN]
    R['winner_percentile'] = {'tau_R_among_5000': float(100 * (tau['R'] < tau['R'][w]).mean()),
                              'tau_S_among_543': float(100 * (np.array([tau['S'][idx[c]] for c in rr]) < tau['S'][w]).mean())}
    gT = gain.get('TEST')
    tc = pd.read_csv(PA / 'selection-vs-specificity/test_measured_candidates.csv').set_index('candidate_id').g_T
    R['reported_r_tauRS_TESTgain'] = r([tau['R'][idx[c]] + tau['S'][idx[c]] for c in tc.index], tc.to_numpy()); R['reported_TEST_n'] = int(len(tc))
    def verdict(x):
        return 'PASS' if x >= 0.25 else 'FALSIFIED' if x < 0.10 else 'INCONCLUSIVE'
    R['3A1_verdict'] = verdict(R['3A1_r_tauR_SEARCHgain']); R['3A2_verdict'] = verdict(R['3A2_r_tauS_RERANKgain'])
    R['3A_conclusion'] = ('SUPPORTED: search selects along the tilt axis' if R['3A1_verdict'] == R['3A2_verdict'] == 'PASS'
                          else 'FALSIFIED' if R['3A1_verdict'] == R['3A2_verdict'] == 'FALSIFIED' else 'MIXED/INCONCLUSIVE')
    # group variance shares of tau_P96 across 5000 (for 3B-3)
    P = np.load(SRC / 'proj_P96.npy')[:, nv]; tot = P.sum(1)
    R['tauP96_group_var_share'] = {G[i]: float(np.cov(P[:, j], tot)[0, 1] / tot.var(ddof=1)) for j, i in enumerate(nv)}
    return idx


def stage3b(R, idx):
    f = SRC / 'insertions_3B.json'
    if not f.exists():
        R['3B'] = 'not run'; return
    m = json.loads(f.read_text()); meta = json.loads((SRC / 'meta.json').read_text()); G = meta['groups']; P = np.load(SRC / 'proj_P96.npy')
    s12 = {x['cid']: x for x in json.loads((PA / 'stage12/stage12_per_perturbation.json').read_text())}
    xs, ys, sums, tnov, meas = [], [], [], [], {}
    for rec in m['records']:
        k = idx[rec['candidate_id']]
        for g, t in rec['insertion_T'].items():
            xs.append(float(P[k, G.index(g)])); ys.append(t); meas.setdefault(g, []).append(t)
        sums.append(sum(rec['insertion_T'].values())); tnov.append(s12[rec['candidate_id']]['T_NOV'])
    n = len(m['records'])
    R['3B_n_perturbations'] = n
    R['3B1_pooled_r'] = r(xs, ys); R['3B1_slope'] = float(np.polyfit(xs, ys, 1)[0])
    R['3B1_verdict'] = 'INCONCLUSIVE (n<10)' if n < 10 else ('PASS' if R['3B1_pooled_r'] >= 0.6 else 'FALSIFIED' if R['3B1_pooled_r'] < 0.3 else 'INCONCLUSIVE')
    R['3B2_r_sum_vs_TNOV'] = r(sums, tnov)
    ssum = np.array(sums); R['3B3_measured_var_share'] = {g: float(np.cov(v, ssum)[0, 1] / ssum.var(ddof=1)) for g, v in meas.items()}
    dom = [g for g in meas if R['3B3_measured_var_share'][g] >= 0.5 and R['tauP96_group_var_share'].get(g, 0) >= 0.5]
    R['3B3_dominant'] = dom or 'distributed'
    R['per_group_r'] = {g: r([float(P[idx[rec['candidate_id']], G.index(g)]) for rec in m['records']], meas[g]) for g in meas}


def main():
    R = {}
    idx = stage3a(R); stage3b(R, idx)
    (D / 'stage3_results.json').write_text(json.dumps(R, indent=1))
    return R


if __name__ == '__main__':
    print(json.dumps(main(), indent=1))

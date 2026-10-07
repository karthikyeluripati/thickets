"""Locked analysis for the P1 pilot (results/paper-analysis/p1/plan_lock.md)."""
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

D = Path('results/paper-analysis/p1')
SRC = D / 'pod'
RANDOPT = Path('C:/Users/karth/AppData/Local/Temp/claude/c--Users-karth-OneDrive-Desktop-projects-thickets/f30c28bf-5da4-45f4-8bca-2f8edaf9c506/scratchpad/RandOpt')


def ld(f):
    return json.loads(gzip.decompress(Path(f).read_bytes()))


def main():
    sys.path.insert(0, str(RANDOPT)); from data_handlers.gqa import GQAHandler  # noqa
    h = GQAHandler(); norm = lambda s: h._normalize_answer(str(s))
    it = json.loads((D / 'items.json').read_text(encoding='utf-8')); S, H = it['selection'], it['heldout']; ALL = S + H
    gold = {r['id']: r['answer'] for r in ALL}
    base = ld(SRC / 'eval/base_p1.json.gz'); bans = {r['id']: norm(base[r['id']]['pred']) for r in ALL}
    cands = json.loads((D / 'candidates.json').read_text())['candidates']; P = {}
    for c in cands:
        f = SRC / f"eval/cands/cand_{c['index']}.json.gz"
        if f.exists(): P[c['index']] = ld(f)['res']
    Ts = sorted(float(p.name[len('samples_T'):-len('.json.gz')]) for p in SRC.glob('samples_T*.json.gz'))
    SM = {T: ld(SRC / f'samples_T{T}.json.gz') for T in Ts}
    R = {'n_perturbed': len(P), 'temps': Ts}
    fP = np.array([np.mean([norm(P[k][r['id']]['pred']) != bans[r['id']] for k in P]) for r in ALL])
    fT = {T: np.array([np.mean([norm(x['pred']) != bans[r['id']] for x in SM[T][r['id']]]) for r in ALL]) for T in Ts}
    Tstar = min(Ts, key=lambda T: abs(fT[T].mean() - fP.mean()))
    R['flip_rate_perturbations'] = float(fP.mean()); R['flip_rate_by_T'] = {str(T): float(fT[T].mean()) for T in Ts}; R['T_star'] = Tstar
    R['acc_by_T'] = {str(T): float(100 * np.mean([x['correct'] for r in ALL for x in SM[T][r['id']]])) for T in Ts}
    rho = float(pd.Series(fP).corr(pd.Series(fT[Tstar]), method='spearman'))
    R['P1_A'] = {'spearman': rho, 'verdict': 'SUPPORTS' if rho >= 0.6 else 'AGAINST' if rho < 0.3 else 'MIXED',
                 'spearman_by_T': {str(T): float(pd.Series(fP).corr(pd.Series(fT[T]), method='spearman')) for T in Ts}}
    acc = lambda res, rows: 100 * np.mean([res[r['id']]['correct'] for r in rows])
    bS, bH = acc(base, S), acc(base, H); R['base_acc'] = {'S': bS, 'H': bH}
    ks = sorted(P); gS = np.array([acc(P[k], S) - bS for k in ks]); gH = np.array([acc(P[k], H) - bH for k in ks])
    fis = lambda r, n: [float(np.tanh(np.arctanh(r) - 1.96 / np.sqrt(n - 3))), float(np.tanh(np.arctanh(r) + 1.96 / np.sqrt(n - 3)))]
    r1 = float(np.corrcoef(gS, gH)[0, 1]); ci = fis(r1, len(ks))
    ps = [{r['id']: SM[Tstar][r['id']][s] for r in ALL} for s in range(len(next(iter(SM[Tstar].values()))))]
    pS = np.array([acc(p, S) for p in ps]); pH = np.array([acc(p, H) for p in ps])
    R['P1_B'] = {'r_sel_ho': r1, 'ci95': ci, 'n': len(ks), 'pseudo_model_r_at_Tstar': float(np.corrcoef(pS, pH)[0, 1]),
                 'verdict': 'PERSISTENT' if r1 >= 0.3 and ci[0] > 0 else 'RESAMPLE-LIKE' if ci[0] <= 0 <= ci[1] and abs(r1) < 0.15 else 'INTERMEDIATE',
                 'gain_sd_sel': float(gS.std()), 'gain_sd_ho': float(gH.std()), 'mean_ho_gain': float(gH.mean())}

    def vote_correct(answer_lists):
        out = []
        for r, ans in zip(H, answer_lists):
            v = Counter(norm(a) for a in ans).most_common(1)[0][0]; out.append(bool(h._match_answer(v, norm(r['answer']))))
        return np.array(out, float)
    rng = np.random.default_rng(20261007); R['P1_C'] = {}
    order = sorted(ks, key=lambda k: (-gS[ks.index(k)], k))
    for K in (8, 16, 32):
        top = order[:K]
        vr = vote_correct([[P[k][r['id']]['pred'] for k in top] for r in H])
        vs = vote_correct([[x['pred'] for x in SM[Tstar][r['id']][:K]] for r in H])
        v7 = vote_correct([[x['pred'] for x in SM[0.7][r['id']][:K]] for r in H]) if 0.7 in SM else None
        rnd = np.mean([vote_correct([[P[k][r['id']]['pred'] for k in rng.choice(ks, K, replace=False)] for r in H]).mean() for _ in range(200)])
        bs = [(lambda ix: 100 * (vr[ix].mean() - vs[ix].mean()))(rng.integers(0, len(H), len(H))) for _ in range(2000)]
        cd = [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]
        R['P1_C'][f'K{K}'] = {'randopt_topK_vote': float(100 * vr.mean()), 'sc_vote_Tstar': float(100 * vs.mean()),
                              'sc_vote_T0.7': float(100 * v7.mean()) if v7 is not None else None, 'random_K_perturbation_vote': float(100 * rnd),
                              'diff_randopt_minus_sc_Tstar': float(100 * (vr.mean() - vs.mean())), 'diff_ci95': cd,
                              'verdict': 'ADVANTAGE' if cd[0] > 0 else 'NO ADVANTAGE'}
    a, b, c = R['P1_A']['verdict'], R['P1_B']['verdict'], R['P1_C']['K8']['verdict']
    R['overall'] = ('SUPPORTED: CoT thickets ~ self-consistency in weight space' if a == 'SUPPORTS' and b != 'PERSISTENT' and c == 'NO ADVANTAGE'
                    else 'NOT SUPPORTED: perturbations carry something beyond re-sampling' if b == 'PERSISTENT' or c == 'ADVANTAGE' else 'MIXED')
    (D / 'p1_results.json').write_text(json.dumps(R, indent=1))
    return R


if __name__ == '__main__':
    print(json.dumps(main(), indent=1))

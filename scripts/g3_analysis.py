"""Locked analysis for G3 (results/paper-analysis/g3-termination/plan_lock.md). CPU. Needs RandOpt (--upstream)."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--upstream', type=Path, required=True); ap.add_argument('--d', type=Path, required=True)
    ap.add_argument('--items', default='results/paper-analysis/g2-sameRun/pod/g2/items.json'); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--k', type=int, default=50); a = ap.parse_args()
    sys.path.insert(0, str(a.upstream))
    from data_handlers.gqa import GQAHandler
    h = GQAHandler()
    rows = json.loads(Path(a.items).read_text())['test']; n = len(rows); img = [r['imageId'] for r in rows]
    gt = [{'answer': r['answer'], 'full_answer': r['fullAnswer']} for r in rows]
    ld = lambda f: json.loads((a.d / f).read_text())

    def vote(keys, i):
        keys = [x for x in keys if x]
        return bool(keys) and bool(h.is_voted_answer_correct(Counter(keys).most_common(1)[0][0], gt[i]))
    nonterm = lambda g: g['fin'] == 'length' or not g['boxed']
    rng = np.random.default_rng(0)

    def boot(x, reps=10000):
        bs = np.array([x[rng.integers(0, n, n)].mean() for _ in range(reps)]) * 100
        return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    res = {'n': n}; D = {}; byb = {}
    for b in (256, 1024):
        base = ld(f'b{b}_base.json'); sc = ld(f'b{b}_sc.json'); mem = [ld(f'b{b}_rank{r}.json') for r in range(a.k)]
        R = {K: np.array([vote([mem[m][i]['key'] for m in range(K)], i) for i in range(n)], float) for K in (50, 10)}
        S = {K: np.array([vote([s['key'] for s in sc[i][:K]], i) for i in range(n)], float) for K in (50, 10)}
        nt_base = np.array([nonterm(g) for g in base], float); nt_mem = np.mean([[nonterm(g) for g in m] for m in mem], 0)
        nt_sc = np.mean([[nonterm(s) for s in sc[i]] for i in range(n)], 1)
        D[b] = R[50] - S[50]; byb[b] = dict(nt_base=nt_base)
        res[f'b{b}'] = {'base_acc': 100 * np.mean([g['c'] for g in base]), 'member_mean_acc': 100 * np.mean([[g['c'] for g in m] for m in mem]),
                        'randopt_K50': 100 * R[50].mean(), 'sc_K50': 100 * S[50].mean(), 'D_K50': 100 * D[b].mean(), 'D_K50_ci': boot(D[b]),
                        'randopt_K10': 100 * R[10].mean(), 'sc_K10': 100 * S[10].mean(), 'D_K10': 100 * (R[10] - S[10]).mean(), 'D_K10_ci': boot(R[10] - S[10]),
                        'nonterm_base': 100 * nt_base.mean(), 'nonterm_members': 100 * nt_mem.mean(), 'nonterm_sc': 100 * nt_sc.mean(),
                        'nonterm_base_minus_members': 100 * (nt_base - nt_mem).mean(), 'nonterm_base_minus_members_ci': boot(nt_base - nt_mem),
                        'mean_tokens': {'base': float(np.mean([g['ntok'] for g in base])), 'members': float(np.mean([[g['ntok'] for g in m] for m in mem])),
                                        'sc': float(np.mean([[s['ntok'] for s in sc[i]] for i in range(n)]))}}
        term = nt_base == 0
        res[f'b{b}']['D_K50_split_by_base256_terminated'] = None
    t256 = byb[256]['nt_base'] == 0
    for b in (256, 1024):
        res[f'b{b}']['D_K50_split_by_base256_terminated'] = {'terminated': 100 * D[b][t256].mean(), 'nonterminated': 100 * D[b][~t256].mean(),
                                                            'n_terminated': int(t256.sum())}
    delta = D[256] - D[1024]; dci = boot(delta); d1024 = res['b1024']['D_K50_ci']
    res['G3_1'] = {'delta_D256_minus_D1024': 100 * delta.mean(), 'delta_ci': dci, 'D1024_ci': d1024,
                   'outcome': ('TERMINATION ACCOUNT SUPPORTED' if dci[0] > 0 and d1024[0] <= 0 else
                               ('NOT SUPPORTED' if dci[0] <= 0 <= dci[1] else 'PARTIAL'))}
    g32 = res['b256']['nonterm_base_minus_members_ci']; res['G3_2'] = 'SUPPORTED' if g32[0] > 0 else 'NOT SUPPORTED'
    g2 = json.loads(Path('results/paper-analysis/g2-sameRun/g2_results.json').read_text())
    res['validity'] = {'base256_vs_g2': res['b256']['base_acc'] - g2['base_g2'], 'randopt256_vs_g2': res['b256']['randopt_K50'] - g2['K50']['randopt']}
    res['valid'] = abs(res['validity']['base256_vs_g2']) <= 0.5 and abs(res['validity']['randopt256_vs_g2']) <= 1.0
    if not res['valid']: res['G3_1']['outcome'] = 'INVALID-ENV (' + res['G3_1']['outcome'] + ')'
    txt = json.dumps(res, indent=1, default=lambda o: o.item()); a.out.write_text(txt); print(txt)


if __name__ == '__main__':
    main()

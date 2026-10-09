"""Locked analysis for GD (results/paper-analysis/gd-gqa-direct-search/plan_lock.md). CPU. Needs RandOpt (--upstream).
--d: GD run dir (out/test_rank*.json, out/test_base.json, out/topk.json; G2 runner format). --g2: G2 dir (CoT search).
--sc / --base: G4 direct-prompt SC@50 and greedy BASE files (g3_eval format: per item {'key', 'c', ...})."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ITEMS = Path('results/paper-analysis/g2-sameRun/pod/g2/items.json')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--upstream', type=Path, required=True); ap.add_argument('--d', type=Path, required=True)
    ap.add_argument('--g2', type=Path, required=True); ap.add_argument('--sc', type=Path, required=True); ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True); a = ap.parse_args()
    sys.path.insert(0, str(a.upstream))
    from data_handlers.gqa import GQAHandler
    h = GQAHandler(); items = json.loads(ITEMS.read_text())['test']; n = len(items)
    gt = [{'answer': r['answer'], 'full_answer': r['fullAnswer']} for r in items]

    def vote(keys, i):
        keys = [x for x in keys if x]
        return bool(keys) and bool(h.is_voted_answer_correct(Counter(keys).most_common(1)[0][0], gt[i]))
    boot = np.random.default_rng(0).integers(0, n, (10000, n))
    ci = lambda x: [float(v) for v in np.percentile(x[boot].mean(1) * 100, [2.5, 97.5])]
    out = lambda c: 'RANDOPT AHEAD' if c[0] > 0 else ('SC AHEAD' if c[1] < 0 else 'NO DIFFERENCE DETECTED')

    def run(d):
        T = [json.loads((d / f'out/test_rank{r}.json').read_text()) for r in range(50)]
        return ([t['ans'] for t in T], np.array([t['correct'] for t in T], float), np.array(json.loads((d / 'out/test_base.json').read_text())['correct'], float),
                json.loads((d / 'out/topk.json').read_text()))
    E, M, B, top = run(a.d); E2, M2, _, top2 = run(a.g2)
    sc = json.loads(a.sc.read_text()); base_g4 = np.array([x['c'] for x in json.loads(a.base.read_text())], float)
    R = {K: np.array([vote([E[m][i] for m in range(K)], i) for i in range(n)], float) for K in (50, 10)}
    S = {K: np.array([vote([x['key'] for x in sc[i][:K]], i) for i in range(n)], float) for K in (50, 10)}
    R2 = np.array([vote([E2[m][i] for m in range(50)], i) for i in range(n)], float)
    res = {'n': n, 'base_direct_this_run': 100 * B.mean(), 'base_direct_g4': 100 * base_g4.mean(),
           'valid_env': bool(abs(B.mean() - base_g4.mean()) * 100 <= 1.0), 'members_mean': 100 * M.mean(),
           'members_range': [100 * M.mean(1).min(), 100 * M.mean(1).max()], 'sigma_top': dict(Counter(str(t['sigma']) for t in top)),
           'top_reward': [top[0]['reward'], top[-1]['reward']], 'top50_overlap_with_cot_search': len({t['k'] for t in top} & {t['k'] for t in top2}),
           'vote_gain_members_to_vote': 100 * (R[50].mean() - M.mean()), 'sc_single_mean': 100 * float(np.mean([[x['c'] for x in row] for row in sc]))}
    for K in (50, 10):
        d = R[K] - S[K]; c = ci(d)
        res[f'K{K}'] = {'randopt_direct': 100 * R[K].mean(), 'sc_direct': 100 * S[K].mean(), 'D': 100 * d.mean(), 'ci': c, 'outcome': out(c),
                        'equivalent_2pp': bool(c[0] >= -2 and c[1] <= 2), 'n_randopt_only': int(((R[K] == 1) & (S[K] == 0)).sum()),
                        'n_sc_only': int(((R[K] == 0) & (S[K] == 1)).sum())}
    d = R[50] - R2; res['randopt_direct_minus_randopt_cot'] = {'randopt_cot': 100 * R2.mean(), 'D': 100 * d.mean(), 'ci': ci(d)}
    d = R[50] - base_g4; res['randopt_direct_minus_one_direct_generation'] = {'D': 100 * d.mean(), 'ci': ci(d)}
    d = M.mean(0) - base_g4; res['members_minus_direct_base'] = {'D': 100 * d.mean(), 'ci': ci(d)}
    c = res['K50']['ci']; res['GD_1'] = res['K50']['outcome'] + (' (EQUIVALENT within 2 pp)' if res['K50']['equivalent_2pp'] else '')
    if not res['valid_env']: res['GD_1'] = 'INVALID-ENV (' + res['GD_1'] + ')'
    txt = json.dumps(res, indent=1, default=lambda o: o.item()); a.out.write_text(txt); print(txt)


if __name__ == '__main__':
    main()

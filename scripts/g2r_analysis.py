"""Locked analysis for G2R (results/paper-analysis/g2r-seed/plan_lock.md). CPU. Needs RandOpt (--upstream) for GQAHandler.
--d1: G2 directory (population seed 42); --d2: G2R directory (second population seed). Both hold items.json and out/."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

import numpy as np

P2 = Path('results/paper-analysis/p2/pod')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--upstream', type=Path, required=True); ap.add_argument('--d1', type=Path, required=True)
    ap.add_argument('--d2', type=Path, required=True); ap.add_argument('--out', type=Path, required=True); a = ap.parse_args()
    sys.path.insert(0, str(a.upstream))
    from data_handlers.gqa import GQAHandler
    h = GQAHandler()

    def key(ans):  # == extract_answer_for_voting applied after extract_answer (as g2_analysis.py)
        n = h._normalize_answer(ans)
        return ' '.join(h._canonicalize(h._singularize(w)) for w in n.split()) if n else ''
    items = json.loads((a.d2 / 'items.json').read_text())['test']; ids = [r['id'] for r in items]; n = len(ids)
    assert ids == [r['id'] for r in json.loads((a.d1 / 'items.json').read_text())['test']], 'item mismatch'
    gt = [{'answer': r['answer'], 'full_answer': r['fullAnswer']} for r in items]
    sc = json.loads(gzip.decompress((P2 / 'gqa_vl3_T0.7.json.gz').read_bytes()))['res']

    def vote(keys, i):
        keys = [x for x in keys if x]
        return bool(keys) and bool(h.is_voted_answer_correct(Counter(keys).most_common(1)[0][0], gt[i]))

    def load(d):
        topk = json.loads((d / 'out/topk.json').read_text()); K = len(topk)
        T = [json.loads((d / f'out/test_rank{i}.json').read_text()) for i in range(K)]
        return topk, [t['ans'] for t in T], np.array([t['correct'] for t in T], float), np.array(json.loads((d / 'out/test_base.json').read_text())['correct'], float)
    run = {s: load(d) for s, d in (('s1', a.d1), ('s2', a.d2))}
    rng = np.random.default_rng(0); res = {'n_test': n}
    B = 10000; boot = rng.integers(0, n, (B, n))
    ci = lambda x: [float(v) for v in np.percentile(x[boot].mean(1) * 100, [2.5, 97.5])]
    for s, (topk, E, M, base) in run.items():
        res[s] = {'base': 100 * base.mean(), 'members_mean': 100 * M.mean(), 'members_range': [100 * M.mean(1).min(), 100 * M.mean(1).max()],
                  'sigma_top': dict(Counter(str(t['sigma']) for t in topk)), 'top_reward': [topk[0]['reward'], topk[-1]['reward']]}
        bs = (a.d1 if s == 's1' else a.d2) / 'out/base_select.json'
        if bs.exists(): res[s]['base_select_reward'] = json.loads(bs.read_text())['reward']
    D = {}
    for k in (50, 10):
        S = np.array([vote([key(x['a']) for x in sc[ids[i]][:k]], i) for i in range(n)], float)
        for s, (topk, E, M, base) in run.items():
            R = np.array([vote([E[m][i] for m in range(k)], i) for i in range(n)], float); d = R - S; lo, hi = ci(d); D[(s, k)] = d
            res[s][f'K{k}'] = {'randopt': 100 * R.mean(), 'sc': 100 * S.mean(), 'D': 100 * d.mean(), 'D_ci95': [lo, hi],
                               'outcome': 'RANDOPT AHEAD' if lo > 0 else ('SC AHEAD' if hi < 0 else 'NO DIFFERENCE DETECTED')}
        dd = D[('s2', k)] - D[('s1', k)]; mean2 = (D[('s1', k)] + D[('s2', k)]) / 2
        res[f'compare_K{k}'] = {'D_s2_minus_D_s1': 100 * dd.mean(), 'ci95': ci(dd), 'D_two_seed_mean': 100 * mean2.mean(), 'D_two_seed_mean_ci95': ci(mean2)}
    lo, hi = res['s2']['K50']['D_ci95']
    res['G2R_1'] = 'REPLICATED' if lo > 0 else ('REVERSED' if hi < 0 else 'NOT REPLICATED')
    res['validity_base_equal'] = bool(abs(res['s1']['base'] - res['s2']['base']) <= 1.0)
    res['seed_overlap_top50'] = len({t['seed'] for t in run['s1'][0]} & {t['seed'] for t in run['s2'][0]})
    txt = json.dumps(res, indent=1, default=lambda o: o.item()); a.out.write_text(txt); print(txt)


if __name__ == '__main__':
    main()

"""Locked analysis for Session C (results/paper-analysis/c-sameRun/plan_lock.md). Needs RandOpt on sys.path (--upstream)
for its GSM8K handler. Inputs: ensemble_answers.json (RandOpt arm, models in top-k order), randopt stdout log,
gsm8k_c15_{greedy,T0.7}.json.gz (SC arm), data/gsm8k/test.parquet."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import re
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--upstream', type=Path, required=True); ap.add_argument('--d', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True); a = ap.parse_args()
    sys.path.insert(0, str(a.upstream))
    from data_handlers.gsm8k import GSM8KHandler
    h = GSM8KHandler(); test = h.load_data(str(a.upstream / 'data/gsm8k/test.parquet'), split='test'); gt = [t['ground_truth'] for t in test]
    ok_r = lambda v, i: bool(v) and bool(h.is_answer_correct(h.format_answer_for_check(v), gt[i]))

    def vote(ans, i):
        ans = [x for x in ans if x]
        return ok_r(Counter(ans).most_common(1)[0][0], i) if ans else False
    E = json.loads((a.d / 'ensemble_answers.json').read_text()); n = len(gt)
    log = (a.d / 'randopt.log').read_text(errors='replace')
    printed = {int(k): int(c) for k, c in re.findall(r'K=(\d+): [\d.]+% \((\d+)/\d+\)', log)}
    sc = json.loads(gzip.decompress((a.d / 'gsm8k_c15_T0.7.json.gz').read_bytes()))['res']; ids = [str(i) for i in range(n)]
    gr = json.loads(gzip.decompress((a.d / 'gsm8k_c15_greedy.json.gz').read_bytes()))['res']
    base_printed = re.findall(r'Test accuracy: ([\d.]+)%', log)
    res = {'n_test': n, 'N_population': int(open(a.d / 'N.txt').read().split()[0]),
           'randopt_base_test_printed': float(base_printed[0]) if base_printed else None,
           'p2sc_greedy': 100 * float(np.mean([gr[i][0]['c'] for i in ids]))}
    rng = np.random.default_rng(0)
    for K in (50, 10):
        R = np.array([vote([E[m][i] for m in range(K)], i) for i in range(n)], float)
        S = np.array([vote([s['a'] for s in sc[ids[i]][:K]], i) for i in range(n)], float)
        S_p2 = np.array([Counter(s['a'] for s in sc[ids[i]][:K]).most_common(1)[0][0].replace(',', '') == gt[i] for i in range(n)], float)
        d = R - S; bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(10000)]) * 100; lo, hi = np.percentile(bs, [2.5, 97.5])
        out = 'RANDOPT AHEAD' if lo > 0 else ('SC AHEAD' if hi < 0 else 'NO DIFFERENCE DETECTED')
        res[f'K{K}'] = {'randopt': 100 * R.mean(), 'randopt_printed_correct': printed.get(K), 'reproduces_printed': printed.get(K) == int(R.sum()),
                        'sc': 100 * S.mean(), 'sc_p2rule': 100 * S_p2.mean(), 'D': 100 * d.mean(), 'D_ci95': [float(lo), float(hi)],
                        'outcome': out, 'equivalent_2pp': bool(lo >= -2 and hi <= 2),
                        'n_randopt_only': int(((R == 1) & (S == 0)).sum()), 'n_sc_only': int(((R == 0) & (S == 1)).sum())}
    res['C1_valid'] = res['K50']['reproduces_printed']
    a.out.write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()

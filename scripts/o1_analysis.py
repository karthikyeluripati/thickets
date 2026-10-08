"""Locked analysis for O1 (results/paper-analysis/o1-olmo-sameRun/plan_lock.md). Same vote rule, scorer and statistics
as Session C (scripts/c_analysis.py). Needs RandOpt (--upstream) with data/gsm8k/test.parquet.
--d1: population seed 42 run dir (ensemble_answers.json, randopt.log, N.txt); --d2: optional seed-43 run dir;
--sc-dir/--sc-tag: SC arm (gsm8k_<tag>_{greedy,T0.7}.json.gz)."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import re
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--upstream', type=Path, required=True); ap.add_argument('--d1', type=Path, required=True)
    ap.add_argument('--d2', type=Path, default=None); ap.add_argument('--sc-dir', type=Path, required=True); ap.add_argument('--sc-tag', default='o1')
    ap.add_argument('--out', type=Path, required=True); a = ap.parse_args()
    sys.path.insert(0, str(a.upstream))
    from data_handlers.gsm8k import GSM8KHandler
    h = GSM8KHandler(); test = h.load_data(str(a.upstream / 'data/gsm8k/test.parquet'), split='test'); gt = [t['ground_truth'] for t in test]; n = len(gt)
    ok = lambda v, i: bool(v) and bool(h.is_answer_correct(h.format_answer_for_check(v), gt[i]))

    def vote(ans, i):
        ans = [x for x in ans if x]
        return ok(Counter(ans).most_common(1)[0][0], i) if ans else False
    ids = [str(i) for i in range(n)]
    sc = json.loads(gzip.decompress((a.sc_dir / f'gsm8k_{a.sc_tag}_T0.7.json.gz').read_bytes()))['res']
    gr = json.loads(gzip.decompress((a.sc_dir / f'gsm8k_{a.sc_tag}_greedy.json.gz').read_bytes()))['res']
    S = {K: np.array([vote([s['a'] for s in sc[ids[i]][:K]], i) for i in range(n)], float) for K in (50, 10, 1)}
    boot = np.random.default_rng(0).integers(0, n, (10000, n))
    ci = lambda x: [float(v) for v in np.percentile(x[boot].mean(1) * 100, [2.5, 97.5])]
    res = {'n_test': n, 'sc_greedy': 100 * float(np.mean([gr[i][0]['c'] for i in ids])), 'sc_K50': 100 * S[50].mean(), 'sc_K10': 100 * S[10].mean(),
           'sc_single_sample_mean': 100 * float(np.mean([[ok(s['a'], i) for s in sc[ids[i]]] for i in range(n)]))}
    R = {}
    for tag, d in (('s42', a.d1), ('s43', a.d2)):
        if d is None: continue
        E = json.loads((d / 'ensemble_answers.json').read_text()); log = (d / 'randopt.log').read_text(errors='replace')
        printed = {int(k): int(c) for k, c in re.findall(r'K=(\d+): [\d.]+% \((\d+)/\d+\)', log)}
        base = re.findall(r'Test accuracy: ([\d.]+)%', log)
        mem = np.array([[ok(E[m][i], i) for i in range(n)] for m in range(len(E))], float)
        r = {'N_population': int(open(d / 'N.txt').read().split()[0]), 'base_test_printed': float(base[0]) if base else None,
             'members_mean': 100 * mem[:50].mean(), 'members_range': [100 * mem[:50].mean(1).min(), 100 * mem[:50].mean(1).max()],
             'vote_curve': {K: 100 * float(np.mean([vote([E[m][i] for m in range(K)], i) for i in range(n)])) for K in (1, 5, 10, 20, 50)}}
        for K in (50, 10):
            RK = np.array([vote([E[m][i] for m in range(K)], i) for i in range(n)], float); d_ = RK - S[K]; c = ci(d_)
            r[f'K{K}'] = {'randopt': 100 * RK.mean(), 'printed_correct': printed.get(K), 'reproduces_printed': printed.get(K) == int(RK.sum()),
                          'sc': 100 * S[K].mean(), 'D': 100 * d_.mean(), 'D_ci95': c,
                          'outcome': 'RANDOPT AHEAD' if c[0] > 0 else ('SC AHEAD' if c[1] < 0 else 'NO DIFFERENCE DETECTED'),
                          'equivalent_2pp': bool(c[0] >= -2 and c[1] <= 2), 'n_randopt_only': int(((RK == 1) & (S[K] == 0)).sum()),
                          'n_sc_only': int(((RK == 0) & (S[K] == 1)).sum())}
            R[(tag, K)] = RK
        res[tag] = r
    res['O1_1'] = res['s42']['K50']['outcome'] + (' (EQUIVALENT within 2 pp)' if res['s42']['K50']['equivalent_2pp'] else '')
    res['O1_valid'] = res['s42']['K50']['reproduces_printed']
    if 's43' in res:
        res['O1b_valid'] = res['s43']['K50']['reproduces_printed']
        for K in (50, 10):
            dd = R[('s43', K)] - R[('s42', K)]; m2 = (R[('s42', K)] + R[('s43', K)]) / 2 - S[K]
            res[f'compare_K{K}'] = {'D_s43_minus_D_s42': 100 * dd.mean(), 'ci95': ci(dd), 'D_two_seed_mean': 100 * m2.mean(), 'D_two_seed_mean_ci95': ci(m2)}
    if not res['O1_valid']: res['O1_1'] = 'INVALID (' + res['O1_1'] + ')'
    txt = json.dumps(res, indent=1, default=lambda o: o.item()); a.out.write_text(txt); print(txt)


if __name__ == '__main__':
    main()

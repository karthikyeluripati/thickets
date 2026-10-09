"""Locked analysis for Q2 (results/paper-analysis/q2-qwen-prompt/plan_lock.md). CPU. Needs RandOpt (--upstream) with
data/gsm8k/test.parquet. For each Qwen row: RandOpt K = 50 from the same-run dump (C or C3B), O2-style runs from --d
(<tag>/{randopt,plain,boxed}_base.json, <tag>/plain_sc.json), the same-run greedy file for the environment gate."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

import numpy as np

ROWS = {'q15': ('results/paper-analysis/c-sameRun/pod6/c/out/ensemble_answers.json', 'results/paper-analysis/c-sameRun/pod/c/out/gsm8k_c15_greedy.json.gz',
                'results/paper-analysis/c-sameRun/pod/c/out/gsm8k_c15_T0.7.json.gz'),
        'q3': ('results/paper-analysis/c3b-sameRun/pod/c3b/out/ensemble_answers.json', 'results/paper-analysis/c3b-sameRun/pod/c3b/out/gsm8k_c3b_greedy.json.gz',
               'results/paper-analysis/c3b-sameRun/pod/c3b/out/gsm8k_c3b_T0.7.json.gz')}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--upstream', type=Path, required=True); ap.add_argument('--d', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True); a = ap.parse_args()
    sys.path.insert(0, str(a.upstream))
    from data_handlers.gsm8k import GSM8KHandler
    h = GSM8KHandler(); gt = [t['ground_truth'] for t in h.load_data(str(a.upstream / 'data/gsm8k/test.parquet'), split='test')]; n = len(gt)
    ok = lambda v, i: bool(v) and bool(h.is_answer_correct(h.format_answer_for_check(v), gt[i]))

    def vote(ans, i):
        ans = [x for x in ans if x]
        return ok(Counter(ans).most_common(1)[0][0], i) if ans else False
    boot = np.random.default_rng(0).integers(0, n, (10000, n))
    ci = lambda x: [float(v) for v in np.percentile(x[boot].mean(1) * 100, [2.5, 97.5])]
    res = {'n': n}
    for tag, (ens, greedy, sct) in ROWS.items():
        d = a.d / tag; ld = lambda f: json.loads((d / f).read_text()); gz = lambda p: json.loads(gzip.decompress(Path(p).read_bytes()))['res']
        E = json.loads(Path(ens).read_text()); R50 = np.array([vote([E[m][i] for m in range(50)], i) for i in range(n)], float)
        g = gz(greedy); sc_ro = gz(sct)
        S_ro = np.array([vote([s['a'] for s in sc_ro[str(i)][:50]], i) for i in range(n)], float)
        acc = lambda recs: 100 * float(np.mean([ok(x['a'], i) for i, x in enumerate(recs)]))
        r = {'randopt_K50': 100 * R50.mean(), 'sc50_randopt_prompt': 100 * S_ro.mean(),
             'same_run_greedy': 100 * float(np.mean([ok(g[str(i)][0]['a'], i) for i in range(n)]))}
        rb = ld('randopt_base.json'); r['randopt_prompt_base'] = acc(rb)
        r['valid_env'] = bool(abs(r['randopt_prompt_base'] - r['same_run_greedy']) <= 1.0)
        pb = ld('plain_base.json'); B = np.array([ok(x['a'], i) for i, x in enumerate(pb)], float); r['plain_base'] = 100 * B.mean()
        r['prompt_damage_pp'] = r['plain_base'] - r['randopt_prompt_base']  # plain − RandOpt prompt, base greedy
        if (d / 'boxed_base.json').exists(): r['boxed_base'] = acc(ld('boxed_base.json'))
        sc = ld('plain_sc.json')
        for K in (50, 10):
            S = np.array([vote([x['a'] for x in sc[i][:K]], i) for i in range(n)], float); r[f'sc{K}_plain'] = 100 * S.mean()
            if K == 50: S50 = S
        dd = R50 - S50; c = ci(dd)
        r['D_randopt_minus_sc_plain'] = {'D': 100 * dd.mean(), 'ci': c, 'outcome': 'RANDOPT AHEAD' if c[0] > 0 else ('PROMPT-SC AHEAD' if c[1] < 0 else 'NO DIFFERENCE DETECTED'),
                                         'equivalent_2pp': bool(c[0] >= -2 and c[1] <= 2)}
        dd = R50 - B; r['D_randopt_minus_plain_base'] = {'D': 100 * dd.mean(), 'ci': ci(dd)}
        dd = S50 - S_ro; r['sc_plain_minus_sc_randopt_prompt'] = {'D': 100 * dd.mean(), 'ci': ci(dd)}
        dd = B - np.array([ok(x['a'], i) for i, x in enumerate(rb)], float); r['prompt_damage_ci'] = ci(dd)
        r['form'] = {p: {'mean_tokens': float(np.mean([x['ntok'] for x in recs])), 'has_####_pct': 100 * float(np.mean([x['hash4'] for x in recs])),
                         'boxed_pct': 100 * float(np.mean([x['boxed'] for x in recs])), 'hit_max_pct': 100 * float(np.mean([x['fin'] == 'length' for x in recs]))}
                     for p, recs in (('randopt', rb), ('plain', pb))}
        out = r['D_randopt_minus_sc_plain']['outcome'] + (' (EQUIVALENT within 2 pp)' if r['D_randopt_minus_sc_plain']['equivalent_2pp'] else '')
        r['outcome'] = out if r['valid_env'] else 'INVALID-ENV (' + out + ')'
        res[tag] = r
    txt = json.dumps(res, indent=1, default=lambda o: o.item()); a.out.write_text(txt); print(txt)


if __name__ == '__main__':
    main()

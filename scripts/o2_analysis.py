"""Locked analysis for O2 (results/paper-analysis/o2-olmo-prompt/plan_lock.md). CPU. Needs RandOpt (--upstream) for the
GSM8K handler. --o1: O1 directory (pod/o1: out/ensemble_answers.json, sc/gsm8k_o1_greedy.json.gz). --d: O2 outputs."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

import numpy as np

VALID_RANKS = [0, 1, 2, 3, 4, 45, 46, 47, 48, 49]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--upstream', type=Path, required=True); ap.add_argument('--o1', type=Path, required=True)
    ap.add_argument('--d', type=Path, required=True); ap.add_argument('--out', type=Path, required=True); a = ap.parse_args()
    sys.path.insert(0, str(a.upstream))
    from data_handlers.gsm8k import GSM8KHandler
    h = GSM8KHandler(); test = h.load_data(str(a.upstream / 'data/gsm8k/test.parquet'), split='test'); gt = [t['ground_truth'] for t in test]; n = len(gt)
    ok = lambda v, i: bool(v) and bool(h.is_answer_correct(h.format_answer_for_check(v), gt[i]))

    def vote(ans, i):
        ans = [x for x in ans if x]
        return ok(Counter(ans).most_common(1)[0][0], i) if ans else False
    ld = lambda name: json.loads((a.d / f'{name}.json').read_text())
    boot = np.random.default_rng(0).integers(0, n, (10000, n))
    ci = lambda x: [float(v) for v in np.percentile(x[boot].mean(1) * 100, [2.5, 97.5])]
    out = lambda c: 'RANDOPT AHEAD' if c[0] > 0 else ('PROMPT-SC AHEAD' if c[1] < 0 else 'NO DIFFERENCE DETECTED')
    E = json.loads((a.o1 / 'out/ensemble_answers.json').read_text())
    R50 = np.array([vote([E[m][i] for m in range(50)], i) for i in range(n)], float)
    o1_greedy = json.loads(gzip.decompress((a.o1 / 'sc/gsm8k_o1_greedy.json.gz').read_bytes()))['res']
    o1g = 100 * float(np.mean([ok(o1_greedy[str(i)][0]['a'], i) for i in range(n)]))  # rescored with RandOpt's scorer, as O2's arms
    form = lambda g: {'mean_tokens': float(np.mean([x['ntok'] for x in g])), 'hit_max_pct': 100 * float(np.mean([x['fin'] == 'length' for x in g])),
                      'has_####_pct': 100 * float(np.mean([x['hash4'] for x in g])), 'boxed_pct': 100 * float(np.mean([x['boxed'] for x in g]))}
    res = {'n': n, 'randopt_K50_o1': 100 * R50.mean(), 'o1_p2sc_greedy': o1g}
    rb = ld('randopt_base'); res['randopt_prompt_base'] = 100 * float(np.mean([ok(x['a'], i) for i, x in enumerate(rb)]))
    res['valid_env'] = bool(abs(res['randopt_prompt_base'] - o1g) <= 1.0)
    agree, dacc = [], []
    for r in VALID_RANKS:
        m = ld(f'randopt_rank{r}'); agree.append(np.mean([m[i]['a'] == E[r][i] for i in range(n)]))
        dacc.append(100 * (np.mean([ok(m[i]['a'], i) for i in range(n)]) - np.mean([ok(E[r][i], i) for i in range(n)])))
    res['fidelity'] = {'ranks': VALID_RANKS, 'mean_answer_agreement': float(np.mean(agree)), 'min_agreement': float(np.min(agree)),
                       'mean_abs_acc_diff_pp': float(np.mean(np.abs(dacc)))}
    res['valid_members'] = bool(np.mean(agree) >= 0.90 and np.mean(np.abs(dacc)) <= 2.0)
    res['form_randopt_base'] = form(rb)
    for p in ('plain', 'boxed'):
        if not (a.d / f'{p}_base.json').exists(): continue
        base = ld(f'{p}_base'); B = np.array([ok(x['a'], i) for i, x in enumerate(base)], float)
        r = {'base': 100 * B.mean(), 'form_base': form(base), 'base_minus_randopt_prompt_base': 100 * (B.mean() - res['randopt_prompt_base'] / 100)}
        d = R50 - B; r['randopt_minus_base_greedy'] = {'D': 100 * d.mean(), 'ci': ci(d)}
        if (a.d / f'{p}_sc.json').exists():
            sc = ld(f'{p}_sc'); r['form_sc'] = form([x for row in sc for x in row])
            for K in (50, 10):
                S = np.array([vote([x['a'] for x in sc[i][:K]], i) for i in range(n)], float); r[f'sc_K{K}'] = 100 * S.mean()
                if K == 50: S50 = S
            d = R50 - S50; c = ci(d)
            r['D_randopt_minus_sc'] = {'D': 100 * d.mean(), 'ci': c, 'outcome': out(c), 'equivalent_2pp': bool(c[0] >= -2 and c[1] <= 2),
                                       'n_randopt_only': int(((R50 == 1) & (S50 == 0)).sum()), 'n_sc_only': int(((R50 == 0) & (S50 == 1)).sum())}
            if all((a.d / f'{p}_rank{k}.json').exists() for k in range(50)):
                M = [ld(f'{p}_rank{k}') for k in range(50)]
                RM = np.array([vote([M[k][i]['a'] for k in range(50)], i) for i in range(n)], float); d = RM - S50; c = ci(d)
                r['members_mean'] = 100 * float(np.mean([[ok(M[k][i]['a'], i) for i in range(n)] for k in range(50)]))
                r['randopt_K50_this_prompt'] = 100 * RM.mean(); r['form_members'] = form([x for m in M for x in m])
                r['D_search_on_top_of_prompt'] = {'D': 100 * d.mean(), 'ci': c, 'outcome': out(c), 'valid': res['valid_members']}
        res[p] = r
    c = res['plain']['D_randopt_minus_sc']
    res['O2_1'] = c['outcome'] + (' (EQUIVALENT within 2 pp)' if c['equivalent_2pp'] else '')
    if not res['valid_env']: res['O2_1'] = 'INVALID-ENV (' + res['O2_1'] + ')'
    txt = json.dumps(res, indent=1, default=lambda o: o.item()); a.out.write_text(txt); print(txt)


if __name__ == '__main__':
    main()

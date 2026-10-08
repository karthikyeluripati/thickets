"""Locked analysis for G4 (results/paper-analysis/g4-direct-prompt/plan_lock.md). CPU. Needs RandOpt (--upstream).
--g3: extracted G3 outputs (out/b256_*.json: RandOpt members, SC and BASE under the CoT prompt; G3 reproduced G2 exactly).
--g2r: G2R out/ (seed-43 members' vote keys). --d: G4 outputs with subdirectories cot/, direct/, short/."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--upstream', type=Path, required=True); ap.add_argument('--g3', type=Path, required=True)
    ap.add_argument('--g2r', type=Path, required=True); ap.add_argument('--d', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--items', default='results/paper-analysis/g2-sameRun/pod/g2/items.json'); a = ap.parse_args()
    sys.path.insert(0, str(a.upstream))
    from data_handlers.gqa import GQAHandler
    h = GQAHandler()
    rows = json.loads(Path(a.items).read_text())['test']; n = len(rows)
    gt = [{'answer': r['answer'], 'full_answer': r['fullAnswer']} for r in rows]
    ld = lambda p: json.loads(p.read_text())

    def vote(keys, i):
        keys = [x for x in keys if x]
        return bool(keys) and bool(h.is_voted_answer_correct(Counter(keys).most_common(1)[0][0], gt[i]))
    rng = np.random.default_rng(0); boot_idx = rng.integers(0, n, (10000, n))

    def ci(x):
        return [float(v) for v in np.percentile(x[boot_idx].mean(1) * 100, [2.5, 97.5])]

    def outcome(c):
        return 'RANDOPT AHEAD' if c[0] > 0 else ('PROMPT-SC AHEAD' if c[1] < 0 else 'NO DIFFERENCE DETECTED')

    def arms(d):  # base greedy, SC votes (K 50/10), member votes (K 50/10) if present, generations for form stats
        base = ld(d / 'b256_base.json'); sc = ld(d / 'b256_sc.json')
        out = {'base': np.array([g['c'] for g in base], float), 'base_gen': base, 'sc_gen': [s for row in sc for s in row],
               'SC50': np.array([vote([s['key'] for s in sc[i][:50]], i) for i in range(n)], float),
               'SC10': np.array([vote([s['key'] for s in sc[i][:10]], i) for i in range(n)], float)}
        if (d / 'b256_rank0.json').exists():
            mem = [ld(d / f'b256_rank{r}.json') for r in range(50)]
            out['members'] = np.array([[g['c'] for g in m] for m in mem], float); out['mem_gen'] = [g for m in mem for g in m]
            for K in (50, 10): out[f'R{K}'] = np.array([vote([mem[m][i]['key'] for m in range(K)], i) for i in range(n)], float)
        return out
    form = lambda gens: {'mean_tokens': float(np.mean([g['ntok'] for g in gens])), 'direct_le32_pct': 100 * float(np.mean([g['ntok'] <= 32 for g in gens])),
                         'boxed_pct': 100 * float(np.mean([g['boxed'] for g in gens]))}
    cot = arms(a.g3 / 'out')
    s43 = [ld(a.g2r / f'test_rank{r}.json')['ans'] for r in range(50)]
    R43 = np.array([vote([s43[m][i] for m in range(50)], i) for i in range(n)], float)
    res = {'n': n, 'reference_cot': {'base': 100 * cot['base'].mean(), 'members_mean': 100 * cot['members'].mean(), 'randopt_K50_seed42': 100 * cot['R50'].mean(),
                                     'randopt_K50_seed43': 100 * R43.mean(), 'randopt_K10_seed42': 100 * cot['R10'].mean(), 'sc_K50': 100 * cot['SC50'].mean(),
                                     'sc_K10': 100 * cot['SC10'].mean(), 'form_base': form(cot['base_gen']), 'form_members': form(cot['mem_gen'])}}
    chk = ld(a.d / 'cot/b256_base.json'); res['validity_cot_base_g4_minus_g3'] = 100 * (np.mean([g['c'] for g in chk]) - cot['base'].mean())
    res['valid'] = bool(abs(res['validity_cot_base_g4_minus_g3']) <= 0.5)
    for p in ('direct', 'short'):
        x = arms(a.d / p); r = {'base': 100 * x['base'].mean(), 'sc_K50': 100 * x['SC50'].mean(), 'sc_K10': 100 * x['SC10'].mean(),
                                'form_base': form(x['base_gen']), 'form_sc': form(x['sc_gen'])}
        r['base_minus_cot_base'] = 100 * (x['base'] - cot['base']).mean(); r['base_minus_cot_base_ci'] = ci(x['base'] - cot['base'])
        r['base_minus_cot_members_mean'] = 100 * (x['base'].mean() - cot['members'].mean())
        r['sc_minus_cot_sc'] = 100 * (x['SC50'] - cot['SC50']).mean(); r['sc_minus_cot_sc_ci'] = ci(x['SC50'] - cot['SC50'])
        for K in (50, 10):  # does RandOpt (CoT prompt, the paper's setting) beat SC with this prompt?
            d = cot[f'R{K}'] - x[f'SC{K}']; c = ci(d)
            r[f'D_K{K}_randopt_cot_minus_sc_{p}'] = {'D': 100 * d.mean(), 'ci': c, 'outcome': outcome(c), 'equivalent_2pp': bool(c[0] >= -2 and c[1] <= 2)}
        d = R43 - x['SC50']; c = ci(d)
        r['D_K50_seed43'] = {'D': 100 * d.mean(), 'ci': c, 'outcome': outcome(c)}
        d = cot['R50'] - x['base']; r['randopt_cot_minus_base_greedy'] = {'D': 100 * d.mean(), 'ci': ci(d)}
        if 'R50' in x:  # selected perturbations evaluated with this prompt: does search add on top of the prompt?
            d = x['R50'] - x['SC50']; c = ci(d)
            r['members_mean'] = 100 * x['members'].mean(); r['randopt_K50'] = 100 * x['R50'].mean(); r['form_members'] = form(x['mem_gen'])
            r['D_K50_randopt_minus_sc_same_prompt'] = {'D': 100 * d.mean(), 'ci': c, 'outcome': outcome(c)}
        res[p] = r
    c = res['direct']['D_K50_randopt_cot_minus_sc_direct']
    res['G4_1'] = c['outcome'] + (' (EQUIVALENT within 2 pp)' if c['equivalent_2pp'] else '')
    if not res['valid']: res['G4_1'] = 'INVALID-ENV (' + res['G4_1'] + ')'
    txt = json.dumps(res, indent=1, default=lambda o: o.item()); a.out.write_text(txt); print(txt)


if __name__ == '__main__':
    main()

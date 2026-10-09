"""Locked analysis for GB (results/paper-analysis/gb-gsm8k-boxed-search/plan_lock.md). CPU. Needs RandOpt (--upstream)
with data/gsm8k/test.parquet. One row per call. --d: GB run dir (out/test_rank*.json, test_base.json, topk.json,
base_select.json). --sc/--base: the boxed-prompt SC@50 and greedy BASE of the same model (O2/Q2/PS, o2_eval format).
--ro-ens: the RandOpt-prompt search's ensemble answers (O1 or C); --ro-seeds: its top-50 seeds (for overlap)."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--upstream', type=Path, required=True); ap.add_argument('--d', type=Path, required=True)
    ap.add_argument('--sc', type=Path, required=True); ap.add_argument('--base', type=Path, required=True); ap.add_argument('--ro-ens', type=Path, required=True)
    ap.add_argument('--ro-seeds', type=Path, required=True); ap.add_argument('--sel-base-ref', type=float, required=True)  # PS selection accuracy, boxed (%)
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
    outc = lambda c: 'RANDOPT AHEAD' if c[0] > 0 else ('SC AHEAD' if c[1] < 0 else 'NO DIFFERENCE DETECTED')
    out = a.d / 'out'; E = [json.loads((out / f'test_rank{r}.json').read_text())['ans'] for r in range(50)]
    top = json.loads((out / 'topk.json').read_text()); B = np.array([ok(v, i) for i, v in enumerate(json.loads((out / 'test_base.json').read_text())['ans'])], float)
    M = np.array([[ok(E[m][i], i) for i in range(n)] for m in range(50)], float)
    sc = json.loads(a.sc.read_text()); Bref = np.array([ok(x['a'], i) for i, x in enumerate(json.loads(a.base.read_text()))], float)
    R = {K: np.array([vote([E[m][i] for m in range(K)], i) for i in range(n)], float) for K in (50, 10)}
    S = {K: np.array([vote([x['a'] for x in sc[i][:K]], i) for i in range(n)], float) for K in (50, 10)}
    Ero = json.loads(a.ro_ens.read_text()); Rro = np.array([vote([Ero[m][i] for m in range(50)], i) for i in range(n)], float)
    ro = json.loads(a.ro_seeds.read_text()); ro = ro['top_k_models'] if isinstance(ro, dict) else ro  # randopt.py top_k_seeds.json or a list
    ro_seeds = {int(t['seed']) for t in ro[:50]}
    sel_base = 100 * json.loads((out / 'base_select.json').read_text())['reward']
    res = {'n': n, 'base_boxed_this_run': 100 * B.mean(), 'base_boxed_ref': 100 * Bref.mean(), 'selection_base_this_run': sel_base,
           'selection_base_ref_PS': a.sel_base_ref,
           'valid_env': bool(abs(B.mean() - Bref.mean()) * 100 <= 1.0 and abs(sel_base - a.sel_base_ref) <= 1.0),
           'members_mean': 100 * M.mean(), 'members_range': [100 * M.mean(1).min(), 100 * M.mean(1).max()],
           'members_minus_boxed_base': {'D': 100 * (M.mean(0) - Bref).mean(), 'ci': ci(M.mean(0) - Bref)},
           'selection_top_reward': [100 * top[0]['reward'], 100 * top[-1]['reward']], 'sigma_top': dict(Counter(str(t['sigma']) for t in top)),
           'top50_overlap_with_randopt_prompt_search': len({int(t['seed']) for t in top} & ro_seeds),
           'vote_gain_over_members': 100 * (R[50].mean() - M.mean())}
    for K in (50, 10):
        d = R[K] - S[K]; c = ci(d)
        res[f'K{K}'] = {'randopt_boxed': 100 * R[K].mean(), 'sc_boxed': 100 * S[K].mean(), 'D': 100 * d.mean(), 'ci': c, 'outcome': outc(c),
                        'equivalent_2pp': bool(c[0] >= -2 and c[1] <= 2), 'n_randopt_only': int(((R[K] == 1) & (S[K] == 0)).sum()),
                        'n_sc_only': int(((R[K] == 0) & (S[K] == 1)).sum())}
    d = R[50] - Rro; res['randopt_boxed_minus_randopt_own_prompt'] = {'randopt_own_prompt': 100 * Rro.mean(), 'D': 100 * d.mean(), 'ci': ci(d)}
    d = R[50] - Bref; res['randopt_boxed_minus_one_boxed_generation'] = {'D': 100 * d.mean(), 'ci': ci(d)}
    res['GB_1'] = res['K50']['outcome'] + (' (EQUIVALENT within 2 pp)' if res['K50']['equivalent_2pp'] else '')
    if not res['valid_env']: res['GB_1'] = 'INVALID-ENV (' + res['GB_1'] + ')'
    txt = json.dumps(res, indent=1, default=lambda o: o.item()); a.out.write_text(txt); print(txt)


if __name__ == '__main__':
    main()

"""Parts B-K of the paper analysis. CPU only; reads the master table and committed locks.

Run: python paper/figures/scripts/analyses.py [part ...]   (default: all)
"""
from collections import Counter
import itertools
import json
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, 'paper/figures/scripts')
from common import (ACCENT, BASE_C, CAND, CANDIDATE, GOOD, GRID, INK, INK2, MUTED, ROOT, SIGMAS, correct_matrix, dump,
                    load, master, save, write_tex)

DF = master()
SL, VL = load(ROOT / 'locks/search.json'), load(ROOT / 'locks/validation.json')
PROTO = load('experiments/perspective_taking_n5000_protocol.json')
TOP50 = [r['candidate']['candidate_id'] for r in SL['top50']]
SEED_OF = {r['candidate']['candidate_id']: r['candidate']['seed'] for r in SL['records']}
SIGMA_OF = {r['candidate']['candidate_id']: r['candidate']['sigma'] for r in SL['records']}
AUDIT = set(PROTO['density_audit']['indices'])
AUDIT_IDS = [r['candidate']['candidate_id'] for r in SL['records'] if r['index'] in AUDIT]
RNG = np.random.default_rng(20261005)
BOOT = 10000


def acc_table():
    a = DF.groupby(['candidate_id', 'phase'], observed=True)['candidate_correct'].agg(['sum', 'size']).reset_index()
    return {(c, p): (s, n) for c, p, s, n in zip(a.candidate_id, a.phase, a['sum'], a['size'])}


ACC = acc_table()


def gain(cid, phase):
    s, n = ACC[(cid, phase)]; b, _ = ACC[('BASE', phase)]
    return 100 * (s - b) / n


def counts(g):
    g = np.asarray(g)
    return {'above_base': int((g > 1e-9).sum()), 'ge1': int((g >= 1 - 1e-9).sum()), 'ge3': int((g >= 3 - 1e-9).sum()), 'ge5': int((g >= 5 - 1e-9).sum())}


def majority(P, k):
    """Upstream randopt.py semantics: valid answers of the first k members in rank order, Counter.most_common."""
    out = []
    for j in range(P.shape[1]):
        ans = [P[i, j] for i in range(k) if P[i, j]]
        out.append(Counter(ans).most_common(1)[0][0] if ans else '')
    return np.array(out, dtype=object)


# ---------------------------------------------------------------- Part B
def part_b():
    sg = [gain(c, 'SEARCH') for c in SL['ranked_ids']]
    rg = [gain(c, 'RERANK') for c in TOP50]
    tg = [gain(c, 'TEST') for c in TOP50]
    ids, M, P, q = correct_matrix(DF, 'TEST', TOP50)
    gold = q.true_option.to_numpy()
    vote = majority(P, 50)
    vote_gain = 100 * ((vote == gold).mean() - q.base_correct.mean())
    top10 = [r['candidate']['candidate_id'] for r in VL['top10']]
    b = {p: ACC[('BASE', p)] for p in ('SEARCH', 'RERANK', 'TEST')}
    res = {'search': {'N': 5000, 'examples': 200, 'base_acc': 100 * b['SEARCH'][0] / 200, 'best_gain': max(sg), **counts(sg)},
           'rerank': {'K': 50, 'examples': 200, 'base_acc': 100 * b['RERANK'][0] / 200, 'best_gain': max(rg), **counts(rg)},
           'test': {'examples': 561, 'base_acc': 100 * b['TEST'][0] / 561, 'rerank_rank1_gain': gain(top10[0], 'TEST'),
                    'best_frozen_top10_gain': max(gain(c, 'TEST') for c in top10), 'best_top50_individual_gain': max(tg),
                    'top50_mean_gain': float(np.mean(tg)), 'top50_majority_vote_gain': vote_gain, **{f'top50_{k}': v for k, v in counts(tg).items()}}}
    dump('table_main_results', res)
    f = lambda x: f'{x:+.2f}'
    rows = [['\\textbf{SEARCH} (5{,}000 candidates)', '200', f"{res['search']['base_acc']:.2f}", f(res['search']['best_gain']),
             res['search']['above_base'], res['search']['ge1'], res['search']['ge3'], res['search']['ge5']],
            ['\\textbf{RERANK} (SEARCH top 50)', '200', f"{res['rerank']['base_acc']:.2f}", f(res['rerank']['best_gain']),
             res['rerank']['above_base'], res['rerank']['ge1'], res['rerank']['ge3'], res['rerank']['ge5']],
            ['\\textbf{TEST} (SEARCH top 50)', '561', f"{res['test']['base_acc']:.2f}", f(res['test']['best_top50_individual_gain']),
             res['test']['top50_above_base'], res['test']['top50_ge1'], res['test']['top50_ge3'], res['test']['top50_ge5']]]
    write_tex('table_main_results', ['Stage', 'Examples', 'Base acc.\\,(\\%)', 'Best gain (pp)', '$>$base', '$\\ge$+1', '$\\ge$+3', '$\\ge$+5'],
              rows, 'Three-stage results. RERANK is a second-stage selection set (historically named VALIDATION200), not an unbiased '
              f"validation estimate. On TEST, the RERANK rank-1 candidate scores {f(res['test']['rerank_rank1_gain'])} pp; the best of the "
              f"frozen RERANK top 10 {f(res['test']['best_frozen_top10_gain'])} pp; the SEARCH top-50 mean is {f(res['test']['top50_mean_gain'])} pp "
              f"and their majority vote {f(res['test']['top50_majority_vote_gain'])} pp.", 'tab:main', 'lrrrrrrr')
    print('B', json.dumps(res))


# ---------------------------------------------------------------- Part C
def part_c():
    stages = ['SEARCH', 'RERANK', 'TEST']
    base = [100 * ACC[('BASE', s)][0] / ACC[('BASE', s)][1] for s in stages]
    cand = [100 * ACC[(CANDIDATE, s)][0] / ACC[(CANDIDATE, s)][1] for s in stages]
    n = [ACC[('BASE', s)][1] for s in stages]
    dump('fig1_mirage_three_stage', pd.DataFrame({'stage': stages, 'n': n, 'base_correct': [ACC[('BASE', s)][0] for s in stages],
                                                  'candidate_correct': [ACC[(CANDIDATE, s)][0] for s in stages],
                                                  'base_acc': base, 'candidate_acc': cand, 'gain_pp': np.subtract(cand, base)}))
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    x = np.arange(3)
    ax.plot(x, base, '-o', color=BASE_C, ms=5, label='Base Qwen3-VL-8B', zorder=3)
    ax.plot(x, cand, '-o', color=CAND, ms=5, label='Candidate (seed 9504111, $\\sigma$=0.002)', zorder=4)
    for i, (b, c) in enumerate(zip(base, cand)):
        d = c - b
        ax.annotate(f'{d:+.1f} pp' if abs(d) >= 3 else f'{d:+.2f} pp' if i == 2 else f'{d:+.1f} pp', (i, max(b, c) + 1.1),
                    ha='center', color=CAND if d > 0 else ACCENT, fontsize=7.5, fontweight='bold')
    ax.set_xticks(x, ['SEARCH\n(n=200)', 'RERANK\n(n=200, selection)', 'TEST\n(official, n=561)'])
    ax.set_ylabel('Accuracy (%)'); ax.set_ylim(33, 50.5); ax.set_xlim(-.35, 2.35)
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, -.24), ncol=1, fontsize=6.8)
    save(fig, 'fig1_mirage_three_stage')
    print('C', [round(c - b, 2) for b, c in zip(base, cand)])


# ---------------------------------------------------------------- Part D
def part_d():
    sg = np.array([gain(c, 'SEARCH') for c in TOP50]); rg = np.array([gain(c, 'RERANK') for c in TOP50]); tg = np.array([gain(c, 'TEST') for c in TOP50])
    stat = {'search_vs_rerank': {'pearson': pearsonr(sg, rg).statistic, 'spearman': spearmanr(sg, rg).statistic},
            'rerank_vs_test': {'pearson': pearsonr(rg, tg).statistic, 'spearman': spearmanr(rg, tg).statistic},
            'search_vs_test': {'pearson': pearsonr(sg, tg).statistic, 'spearman': spearmanr(sg, tg).statistic}}
    best_test = TOP50[int(np.argmax(tg))]
    dump('fig2_transfer_scatter', pd.DataFrame({'candidate_id': TOP50, 'seed': [SEED_OF[c] for c in TOP50], 'sigma': [SIGMA_OF[c] for c in TOP50],
                                                'search_gain_pp': sg, 'rerank_gain_pp': rg, 'test_gain_pp': tg}))
    dump('fig2_transfer_stats', stat)
    jit = np.random.default_rng(0).uniform(-.12, .12, 50)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))
    i_c, i_b = TOP50.index(CANDIDATE), TOP50.index(best_test)
    for ax, (x, y, xl, yl, k, title) in zip(axes, ((sg + jit, rg, 'SEARCH gain (pp)', 'RERANK gain (pp)', 'search_vs_rerank', 'A  SEARCH $\\rightarrow$ RERANK'),
                                                    (rg, tg, 'RERANK gain (pp)', 'TEST gain (pp)', 'rerank_vs_test', 'B  RERANK $\\rightarrow$ TEST'))):
        ax.axhline(0, color=MUTED, lw=.8); ax.axvline(0, color=MUTED, lw=.8)
        ax.scatter(x, y, s=16, color=BASE_C, alpha=.55, edgecolor='white', linewidth=.5, zorder=3, label='SEARCH top 50')
        ax.scatter(x[i_c], y[i_c], s=42, color=CAND, edgecolor='white', linewidth=.8, zorder=5, label='Seed 9504111 (RERANK rank 1)')
        if k == 'rerank_vs_test':
            ax.scatter(x[i_b], y[i_b], s=42, marker='D', color=GOOD, edgecolor='white', linewidth=.8, zorder=5, label=f'Best TEST individual (seed {SEED_OF[best_test]})')
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_title(f"{title}   Pearson r = {stat[k]['pearson']:+.2f}, Spearman $\\rho$ = {stat[k]['spearman']:+.2f}", loc='left', fontsize=7.5)
    axes[0].text(.98, .03, 'x jittered $\\pm$0.12 pp\n(scores are 0.5-pp steps)', transform=axes[0].transAxes, ha='right', va='bottom', fontsize=6, color=MUTED)
    h1, l1 = axes[0].get_legend_handles_labels(); h2, l2 = axes[1].get_legend_handles_labels()
    fig.legend(h2, l2, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(.5, -.06))
    fig.tight_layout(rect=(0, .06, 1, 1))
    save(fig, 'fig2_transfer_scatter')
    print('D', json.dumps(stat), 'best test', SEED_OF[best_test], tg.max())


# ---------------------------------------------------------------- Part E
def part_e():
    subs = ['Allocentric', 'Egocentric', 'Hypothetical']
    def per_sub(phase):
        d = DF[(DF.phase == phase) & (DF.candidate_id == CANDIDATE)]
        out = {}
        for s in subs:
            x = d[d.subtask == s]
            out[s] = {'n': len(x), 'base_acc': 100 * x.base_correct.mean(), 'cand_acc': 100 * x.candidate_correct.mean(),
                      'gain': 100 * (x.candidate_correct.mean() - x.base_correct.mean()),
                      'wins': int((x.candidate_correct & ~x.base_correct).sum()), 'losses': int((~x.candidate_correct & x.base_correct).sum())}
        return d, out
    dR, R = per_sub('RERANK'); dT, T = per_sub('TEST')
    wR = {s: R[s]['n'] / 200 for s in subs}; wT = {s: T[s]['n'] / 561 for s in subs}
    obsR = sum(wR[s] * R[s]['gain'] for s in subs); obsT = sum(wT[s] * T[s]['gain'] for s in subs)
    cfA = sum(wT[s] * R[s]['gain'] for s in subs)   # TEST mixture, RERANK within-subtask behaviour
    cfB = sum(wR[s] * T[s]['gain'] for s in subs)   # RERANK mixture, TEST within-subtask behaviour
    # Stratified paired bootstrap over questions within each subtask (both splits independently).
    def strat(d):
        return {s: (d[d.subtask == s].candidate_correct.to_numpy().astype(int) - d[d.subtask == s].base_correct.to_numpy().astype(int)) for s in subs}
    DR, DT = strat(dR), strat(dT)
    bs = {k: [] for k in ('rerank', 'cfA', 'test', 'cfB')}
    for _ in range(BOOT):
        gR = {s: 100 * RNG.choice(DR[s], len(DR[s])).mean() for s in subs}
        gT = {s: 100 * RNG.choice(DT[s], len(DT[s])).mean() for s in subs}
        bs['rerank'].append(sum(wR[s] * gR[s] for s in subs)); bs['cfA'].append(sum(wT[s] * gR[s] for s in subs))
        bs['test'].append(sum(wT[s] * gT[s] for s in subs)); bs['cfB'].append(sum(wR[s] * gT[s] for s in subs))
    ci = {k: [float(np.quantile(v, .025)), float(np.quantile(v, .975))] for k, v in bs.items()}
    res = {'per_subtask_rerank': R, 'per_subtask_test': T, 'weights_rerank': wR, 'weights_test': wT,
           'rerank_observed': obsR, 'test_observed': obsT, 'cfA_test_mixture_rerank_behaviour': cfA,
           'cfB_rerank_mixture_test_behaviour': cfB, 'mixture_component': cfA - obsR, 'within_subtask_component': obsT - cfA,
           'bootstrap95': ci, 'P_cfA_le_0': float(np.mean(np.array(bs['cfA']) <= 0)),
           'note': 'Counterfactuals reweight per-subtask gains; they are descriptive accounting identities, not causal estimates.'}
    dump('fig3_shift_decomposition', res)
    rows = [[s, R[s]['n'], f"{R[s]['gain']:+.2f}", T[s]['n'], f"{T[s]['gain']:+.2f}"] for s in subs]
    rows += ['MIDRULE', ['Overall (own mixture)', 200, f'{obsR:+.2f}', 561, f'{obsT:+.2f}'],
             ['Counterfactual A: TEST mixture, RERANK within-subtask gains', '', '', '', f'{cfA:+.2f} [{ci["cfA"][0]:+.1f}, {ci["cfA"][1]:+.1f}]'],
             ['Counterfactual B: RERANK mixture, TEST within-subtask gains', '', f'{cfB:+.2f} [{ci["cfB"][0]:+.1f}, {ci["cfB"][1]:+.1f}]', '', '']]
    write_tex('table_shift_decomposition', ['Subtask', '$n_{\\mathrm{RERANK}}$', 'RERANK gain (pp)', '$n_{\\mathrm{TEST}}$', 'TEST gain (pp)'], rows,
              'Subtask-mixture decomposition for seed 9504111. Counterfactual A re-weights the RERANK within-subtask gains by the TEST subtask '
              'mixture; counterfactual B re-weights the TEST within-subtask gains by the RERANK mixture. Brackets: 95\\% stratified bootstrap '
              'intervals over questions. Accounting identities, not causal estimates.', 'tab:shift', 'lrrrr')
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    steps = [('RERANK\nobserved', obsR), ('Subtask mix\nchanged only\n(counterfactual A)', cfA), ('TEST\nobserved', obsT)]
    x = np.arange(3)
    vals = [v for _, v in steps]
    ax.bar(x, vals, color=[CAND, '#9cc2ee', ACCENT], width=.55, zorder=3)
    lo = [ci['rerank'][0], ci['cfA'][0], ci['test'][0]]; hi = [ci['rerank'][1], ci['cfA'][1], ci['test'][1]]
    ax.errorbar(x, vals, yerr=[np.subtract(vals, lo), np.subtract(hi, vals)], fmt='none', ecolor=INK2, elinewidth=.9, capsize=2.5, zorder=4)
    nl = '\n'
    for xm, txt in ((.5, f'mixture shift{nl}{cfA - obsR:+.2f} pp'), (1.5, f'within-subtask shift{nl}{obsT - cfA:+.2f} pp')):
        ax.text(xm, 11.6, txt, ha='center', va='center', fontsize=6.2, color=INK2)
    ax.axhline(0, color=MUTED, lw=.8)
    ax.set_xticks(x, [f'RERANK observed{nl}{obsR:+.2f} pp', f'TEST subtask mix,{nl}RERANK behaviour{nl}{cfA:+.2f} pp',
                      f'TEST observed{nl}{obsT:+.2f} pp'], fontsize=6.6)
    ax.set_ylabel('Gain over base (pp)'); ax.set_ylim(-6, 13.5); ax.set_xlim(-.5, 2.5)
    save(fig, 'fig3_shift_decomposition')
    print('E', json.dumps({k: res[k] for k in ('rerank_observed', 'cfA_test_mixture_rerank_behaviour', 'cfB_rerank_mixture_test_behaviour', 'test_observed', 'bootstrap95', 'P_cfA_le_0')}))
    print('  per-sub R', {s: round(R[s]['gain'], 2) for s in subs}, 'T', {s: round(T[s]['gain'], 2) for s in subs})


# ---------------------------------------------------------------- Part F
def part_f():
    search_q = DF[(DF.phase == 'SEARCH') & (DF.candidate_id == 'BASE')]
    texts, forms = set(search_q.question_text.astype(str)), set(search_q.question_form.astype(str))
    def cat(t, f):
        return 'exact text seen in SEARCH' if t in texts else 'form seen in SEARCH' if f in forms else 'unseen form'
    order = ['exact text seen in SEARCH', 'form seen in SEARCH', 'unseen form']
    out = {}
    for phase in ('RERANK', 'TEST'):
        d = DF[(DF.phase == phase) & (DF.candidate_id == CANDIDATE)].copy()
        d['cat'] = [cat(t, f) for t, f in zip(d.question_text.astype(str), d.question_form.astype(str))]
        ids, M, _, q = correct_matrix(DF, phase, [c for c in AUDIT_IDS if SIGMA_OF[c] == .002] if phase == 'RERANK' else None)
        qcat = [cat(t, f) for t, f in zip(q.question_text.astype(str), q.question_form.astype(str))]
        res = {}
        for c in order:
            x = d[d.cat == c]
            if not len(x): res[c] = {'n': 0}; continue
            wins, losses = int((x.candidate_correct & ~x.base_correct).sum()), int((~x.candidate_correct & x.base_correct).sum())
            r = {'n': len(x), 'base_acc': 100 * x.base_correct.mean(), 'cand_acc': 100 * x.candidate_correct.mean(),
                 'candidate_only': wins, 'base_only': losses, 'net': wins - losses, 'gain_pp': 100 * (wins - losses) / len(x)}
            if phase == 'RERANK':
                m = np.array([qc == c for qc in qcat]); b = q.base_correct.to_numpy()
                g = 100 * (M[:, m].mean(1) - b[m].mean())
                r['random_sigma002_mean_gain_pp'] = float(g.mean()); r['random_sigma002_n'] = len(ids)
            res[c] = r
        out[phase] = res
    dump('table_question_form_overlap', out)
    rows = []
    for phase in ('RERANK', 'TEST'):
        for c in order:
            r = out[phase][c]
            if not r['n']: rows.append([phase if c == order[0] else '', c, 0, '--', '--', '--', '--', '--', '--']); continue
            rows.append([phase if c == order[0] else '', c, r['n'], f"{r['base_acc']:.1f}", f"{r['cand_acc']:.1f}", r['candidate_only'], r['base_only'],
                         f"{r['gain_pp']:+.1f}", f"{r['random_sigma002_mean_gain_pp']:+.2f}" if 'random_sigma002_mean_gain_pp' in r else '--'])
        rows.append('MIDRULE')
    rows = rows[:-1]
    write_tex('table_question_form_overlap', ['Stage', 'Question form vs SEARCH', '$n$', 'Base', 'Cand.', 'Cand.-only', 'Base-only', 'Gain (pp)', 'Random $\\sigma{=}0.002$ gain'],
              rows, 'Seed 9504111 by recurrence of the question phrasing in SEARCH. ``Form\'\' is a normalized question-form family (first five '
              'lower-cased words); questions always differ in image. Last column: mean gain of the 125 precommitted random $\\sigma{=}0.002$ '
              'audit candidates on the same RERANK questions.', 'tab:forms', 'llrrrrrrr')
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    x = np.arange(3); w = .38
    g_c = [out['RERANK'][c].get('gain_pp', 0) for c in order]; g_r = [out['RERANK'][c].get('random_sigma002_mean_gain_pp', 0) for c in order]
    ax.bar(x - w / 2, g_c, w, color=CAND, label='Seed 9504111', zorder=3)
    ax.bar(x + w / 2, g_r, w, color='#9cc2ee', label='Random $\\sigma$=0.002 (mean of 125)', zorder=3)
    for i, c in enumerate(order):
        ax.text(i, max(g_c[i], g_r[i], 0) + .6, f"n={out['RERANK'][c]['n']}", ha='center', fontsize=6.5, color=INK2)
    ax.axhline(0, color=MUTED, lw=.8); ax.set_xticks(x, ['Exact text\nseen in SEARCH', 'Form seen\nin SEARCH', 'Unseen\nform'])
    ax.set_ylabel('RERANK gain (pp)'); ax.set_ylim(top=max(max(g_c), max(g_r)) + 2.2)
    ax.legend(frameon=False, fontsize=6.5, loc='upper center', bbox_to_anchor=(.5, -.2), ncol=2)
    save(fig, 'fig4_form_overlap_gain')
    print('F', json.dumps(out))


# ---------------------------------------------------------------- Part G
def part_g():
    out = {}
    for s in SIGMAS:
        ids, M, P, q = correct_matrix(DF, 'RERANK', [c for c in AUDIT_IDS if SIGMA_OF[c] == s])
        b = q.base_correct.to_numpy(); bp = q.base_prediction.astype(str).to_numpy()
        out[str(s)] = {'n_candidates': len(ids), 'change_answer': (P != bp).mean(0).tolist(),
                       'wrong_to_correct': np.where(~b, M.mean(0), 0).tolist(), 'correct_to_wrong': np.where(b, 1 - M.mean(0), 0).tolist()}
    ids, M, P, q = correct_matrix(DF, 'RERANK', [c for c in AUDIT_IDS if SIGMA_OF[c] == .002])
    b = q.base_correct.to_numpy()
    cand = DF[(DF.phase == 'RERANK') & (DF.candidate_id == CANDIDATE)].set_index('example_id').loc[q.example_id]
    cvec = cand.candidate_correct.to_numpy().astype(int) - b.astype(int)
    rate = M.mean(0) - b                        # mean signed gain of random sigma=0.002 candidates per question
    corr = pearsonr(cvec, rate).statistic
    w2c = np.where(~b, M.mean(0), np.nan); c2w = np.where(b, 1 - M.mean(0), np.nan)
    order = np.argsort(-np.nan_to_num(w2c, nan=-1) - np.nan_to_num(c2w, nan=0) * 0)
    base_wrong = np.where(~b)[0]; base_right = np.where(b)[0]
    bw = base_wrong[np.argsort(-w2c[base_wrong])]; br = base_right[np.argsort(-c2w[base_right])]
    top = [{'example_id': q.example_id[i], 'subtask': q.subtask[i], 'form': q.question_form[i], 'p_correction': float(w2c[i]),
            'candidate_9504111': int(cvec[i])} for i in bw[:10]]
    bins = [(0, 0), (1e-9, .1), (.1, .3), (.3, 1.01)]
    binned = []
    for lo, hi in bins:
        m = (~b) & (w2c >= lo) & (w2c <= hi) if lo == 0 else (~b) & (w2c > lo - 1e-12) & (w2c <= hi)
        binned.append({'bin': f'{lo:.0%}' if lo == hi else f'({lo:.0%},{min(hi, 1):.0%}]', 'n': int(m.sum()),
                       'candidate_gain_rate': float((cvec[m] > 0).mean()) if m.sum() else None})
    res = {'per_sigma': out, 'sigma002_top_corrections': top, 'candidate_vs_random_gain_corr': corr,
           'candidate_wins': int((cvec > 0).sum()), 'candidate_losses': int((cvec < 0).sum()),
           'candidate_win_questions_mean_random_correction': float(w2c[cvec > 0].mean()),
           'other_base_wrong_mean_random_correction': float(np.nanmean(w2c[(cvec <= 0) & ~b])),
           'binned_candidate_gain_rate_by_random_correction': binned}
    dump('fig5_fragile_items', res)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5), gridspec_kw={'width_ratios': [1.6, 1]})
    ax = axes[0]
    ax.bar(np.arange(len(bw)), w2c[bw], width=1, color=CAND, label=f'Base wrong $\\rightarrow$ correct (n={len(bw)} questions)', zorder=3)
    ax.bar(np.arange(len(br)) + len(bw) + 6, -c2w[br], width=1, color=ACCENT, label=f'Base correct $\\rightarrow$ wrong (n={len(br)})', zorder=3)
    wins = [k for k, i in enumerate(bw) if cvec[i] > 0]
    ax.scatter(wins, w2c[bw][wins] + .04, marker='v', s=10, color=INK, zorder=4, label='Seed 9504111 gains this question')
    ax.text(len(bw) * .55, .72, f'top items: {w2c[bw][0]:.0%}, {w2c[bw][1]:.0%}, {w2c[bw][2]:.0%}', fontsize=6.3, color=INK2)
    ax.axhline(0, color=MUTED, lw=.8)
    ax.set_xlabel('RERANK questions, sorted by flip frequency'); ax.set_ylabel('Fraction of 125 random\n$\\sigma$=0.002 candidates'); ax.set_xticks([])
    ax.set_title('A  Fragile items under random perturbation', loc='left'); ax.legend(frameon=False, fontsize=6.2, loc='lower right')
    ax = axes[1]
    labels = ['0', '$\leq$10%', '10-30%', '$>$30%']; vals = [x['candidate_gain_rate'] or 0 for x in binned]
    ax.bar(range(len(binned)), vals, color=CAND, width=.6, zorder=3)
    for i, x in enumerate(binned):
        ax.text(i, vals[i] + .02, f"{vals[i]:.0%}\n(n={x['n']})", ha='center', fontsize=6.3)
    ax.set_xticks(range(len(binned)), labels, fontsize=6.5); ax.set_ylim(0, max(vals) * 1.35 + .05)
    ax.set_xlabel('Random $\\sigma$=0.002 correction frequency'); ax.set_ylabel('Share corrected by 9504111')
    ax.set_title(f'B  9504111 follows the population (r = {corr:.2f})', loc='left')
    fig.tight_layout()
    save(fig, 'fig5_fragile_items')
    print('G', json.dumps({k: v for k, v in res.items() if k != 'per_sigma'}))


# ---------------------------------------------------------------- Part H
def part_h():
    obs = gain(CANDIDATE, 'RERANK')
    ids, M, P, q = correct_matrix(DF, 'RERANK', AUDIT_IDS)
    b = q.base_correct.to_numpy()
    g_audit = 100 * (M.mean(1) - b.mean())
    sig = np.array([SIGMA_OF[c] for c in ids])
    flip = {s: (M[sig == s] != b).mean(0) for s in SIGMAS}
    top50_mix = Counter(SIGMA_OF[c] for c in TOP50)
    Ms = (1, 5, 10, 25, 50, 125, 500)
    res = {'observed_rerank_max_top50': obs, 'top50_sigma_mix': {str(k): v for k, v in top50_mix.items()}, 'nulls': {}}
    # Null 1: empirical random candidates (subsample M of the 500 random audit candidates, without replacement).
    # Null 2: structured independent flips at per-question, per-sigma rates, uniform sigma mix.
    # Null 3: SEARCH-selected-composition null: 50 random audit candidates with the top-50 sigma mix (27/11/12/0).
    for name in ('empirical_random', 'structured_flip', 'search_selected_mix'):
        rows = []
        for m in Ms:
            if name == 'search_selected_mix' and m != 50: continue
            mx = []
            for _ in range(4000):
                if name == 'empirical_random':
                    mx.append(g_audit[RNG.choice(len(g_audit), m, replace=False)].max())
                elif name == 'structured_flip':
                    ss = RNG.choice(SIGMAS, m)
                    F = RNG.random((m, len(b))) < np.vstack([flip[s] for s in ss])
                    mx.append((100 * (np.where(F, ~b, b).mean(1) - b.mean())).max())
                else:
                    pick = np.concatenate([RNG.choice(np.where(sig == s)[0], k, replace=False) for s, k in top50_mix.items()])
                    mx.append(g_audit[pick].max())
            mx = np.array(mx)
            rows.append({'M': m, 'expected_max': float(mx.mean()), 'q025': float(np.quantile(mx, .025)), 'q975': float(np.quantile(mx, .975)),
                         'P_max_ge_observed': float((mx >= obs - 1e-9).mean())})
        res['nulls'][name] = rows
    dump('fig6_extreme_selection_null', res)
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for name, color, label, mk in (('empirical_random', BASE_C, 'Random candidates (500 audit, subsampled)', 'o'),
                                   ('structured_flip', MUTED, 'Independent flips at observed per-item rates', 's')):
        r = res['nulls'][name]
        xs = [x['M'] for x in r]
        ax.fill_between(xs, [x['q025'] for x in r], [x['q975'] for x in r], color=color, alpha=.15, linewidth=0)
        ax.plot(xs, [x['expected_max'] for x in r], '-' + mk, color=color, ms=3, label=label)
    s3 = res['nulls']['search_selected_mix'][0]
    ax.errorbar([50 * 1.12], [s3['expected_max']], yerr=[[s3['expected_max'] - s3['q025']], [s3['q975'] - s3['expected_max']]], fmt='D',
                color=GOOD, ms=4, capsize=2.5, label='50 random with SEARCH top-50 $\\sigma$ mix')
    ax.scatter([50], [obs], s=60, marker='*', color=CAND, zorder=5, label=f'Observed RERANK max of SEARCH top 50 ({obs:+.1f} pp)')
    ax.set_xscale('log'); ax.set_xticks(Ms, [str(m) for m in Ms]); ax.minorticks_off()
    ax.set_xlabel('Candidates inspected on RERANK (M)'); ax.set_ylabel('Maximum RERANK gain (pp)')
    ax.text(44, 7.5, 'capped at audit max (+7.5)', fontsize=5.8, color=INK2, ha='right', va='center')
    ax.set_ylim(-3.5, 9.2)
    ax.legend(frameon=False, fontsize=6, loc='upper center', bbox_to_anchor=(.5, -.2), ncol=1)
    save(fig, 'fig6_extreme_selection_null')
    print('H', json.dumps(res))


# ---------------------------------------------------------------- Part I
def part_i():
    ids, M, P, q = correct_matrix(DF, 'TEST', TOP50)
    b = q.base_correct.to_numpy(); bp = q.base_prediction.astype(str).to_numpy(); gold = q.true_option.astype(str).to_numpy()
    acc = 100 * M.mean(1)
    n = len(ids)
    dis = [float((P[i] != P[j]).mean()) for i, j in itertools.combinations(range(n), 2)]
    jac = []
    for i, j in itertools.combinations(range(n), 2):
        ei, ej = ~M[i], ~M[j]
        jac.append(float((ei & ej).sum() / max((ei | ej).sum(), 1)))
    votes = []
    for j in range(P.shape[1]):
        c = Counter(x for x in P[:, j] if x)
        tot = sum(c.values()); p = np.array([c.get(L, 0) / tot for L in 'ABCD'])
        ent = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
        top2 = sorted(c.values(), reverse=True) + [0]
        maj = Counter(x for x in P[:, j] if x).most_common(1)[0][0]
        votes.append({'entropy_bits': ent, 'margin': (top2[0] - top2[1]) / tot, 'base_is_majority': maj == bp[j], 'majority_correct': maj == gold[j],
                      'n_correct_members': int(M[:, j].sum())})
    V = pd.DataFrame(votes)
    oracle = (M.any(0)).mean()
    vote50 = majority(P, 50)
    k_curve = []
    for k in (1, 5, 10, 25, 50):
        v = majority(P, k)
        k_curve.append({'K': k, 'acc': 100 * (v == gold).mean(), 'gain': 100 * ((v == gold).mean() - b.mean()),
                        'majority_differs_from_base': int((v != bp).sum()), 'frac_differs': float((v != bp).mean()),
                        'ens_only_correct': int(((v == gold) & ~b).sum()), 'base_only_correct': int(((v != gold) & b).sum())})
    base_wrong = ~b
    res = {'individual': {'mean': float(acc.mean()), 'sd': float(acc.std(ddof=1)), 'min': float(acc.min()), 'max': float(acc.max()),
                          'above_base': int((acc > 100 * b.mean() + 1e-9).sum()), 'base_acc': 100 * b.mean()},
           'pairwise_prediction_disagreement': {'mean': float(np.mean(dis)), 'median': float(np.median(dis)), 'q05': float(np.quantile(dis, .05)), 'q95': float(np.quantile(dis, .95))},
           'pairwise_error_jaccard': {'mean': float(np.mean(jac)), 'median': float(np.median(jac)), 'min': float(np.min(jac))},
           'disagreement_with_base_per_member': {'mean': float((P != bp).mean()), 'max': float((P != bp).mean(1).max())},
           'per_question': {'mean_entropy_bits': float(V.entropy_bits.mean()), 'share_zero_entropy': float((V.entropy_bits == 0).mean()),
                            'mean_margin': float(V.margin.mean()), 'base_is_majority': float(V.base_is_majority.mean()),
                            'majority_correct': float(V.majority_correct.mean())},
           'oracle': {'base_correct': int(b.sum()), 'majority_correct': int((vote50 == gold).sum()), 'oracle_correct': int(M.any(0).sum()),
                      'oracle_acc': 100 * oracle, 'base_wrong_questions': int(base_wrong.sum()),
                      'base_wrong_fixed_by_any_member': int((M[:, base_wrong].any(0)).sum()),
                      'base_wrong_fixed_by_ge5_members': int((M[:, base_wrong].sum(0) >= 5).sum()),
                      'base_wrong_fixed_by_ge10_members': int((M[:, base_wrong].sum(0) >= 10).sum()),
                      'base_wrong_fixed_by_majority': int(((vote50 == gold) & base_wrong).sum()),
                      'members_correct_on_base_wrong_histogram': dict(Counter(M[:, base_wrong].sum(0).tolist())),
                      'base_right_broken_by_majority': int(((vote50 != gold) & b).sum())},
           'k_curve': k_curve}
    # Chance references for the oracle. (1) Symmetric churn: base-correct questions broken by >=1 / >=5 members.
    # (2) Guessing null: keep, per question, how many members changed the base answer; give each changed member a
    #     uniformly random non-base option independently; count base-wrong questions fixed by >=1 / >=5 / >=10 members.
    changed = (P != bp)
    res['oracle']['base_right_broken_by_any_member'] = int((~M[:, b]).any(0).sum())
    res['oracle']['base_right_broken_by_ge5_members'] = int(((~M[:, b]).sum(0) >= 5).sum())
    res['oracle']['base_right_broken_by_ge10_members'] = int(((~M[:, b]).sum(0) >= 10).sum())
    res['oracle']['base_right_questions'] = int(b.sum())
    nchg = changed[:, base_wrong].sum(0)
    sims = {1: [], 5: [], 10: []}
    for _ in range(2000):
        hits = RNG.binomial(nchg, 1 / 3)
        for t in sims: sims[t].append(int((hits >= t).sum()))
    res['oracle']['guessing_null_base_wrong_fixed'] = {f'ge{t}': {'mean': float(np.mean(v)), 'q95': float(np.quantile(v, .95)),
                                                              'observed': res['oracle'][f'base_wrong_fixed_by_{"any_member" if t == 1 else f"ge{t}_members"}']}
                                                       for t, v in sims.items()}
    # Do members agree on the same alternative when they leave the base answer? (coordination of changes)
    agree = []
    for j in np.where(base_wrong)[0]:
        alts = [x for x in P[:, j] if x and x != bp[j]]
        if len(alts) >= 2: agree.append(Counter(alts).most_common(1)[0][1] / len(alts))
    res['oracle']['mean_share_of_changed_members_choosing_same_alternative'] = float(np.mean(agree)) if agree else None
    dump('table_committee_diversity', res)
    o = res['oracle']
    rows = [['Individual TEST accuracy, mean (SD)', f"{res['individual']['mean']:.2f} ({res['individual']['sd']:.2f})"],
            ['Individual TEST accuracy, min / max', f"{res['individual']['min']:.2f} / {res['individual']['max']:.2f}"],
            ['Members above base (46.17\\%)', f"{res['individual']['above_base']} / 50"],
            ['Mean pairwise prediction disagreement', f"{100 * res['pairwise_prediction_disagreement']['mean']:.1f}\\%"],
            ['Mean pairwise error Jaccard', f"{res['pairwise_error_jaccard']['mean']:.3f}"],
            ['Mean member disagreement with base', f"{100 * res['disagreement_with_base_per_member']['mean']:.1f}\\%"],
            ['Questions with unanimous committee', f"{100 * res['per_question']['share_zero_entropy']:.1f}\\%"],
            ['Base answer is committee majority', f"{100 * res['per_question']['base_is_majority']:.1f}\\%"],
            'MIDRULE',
            ['Base correct / majority correct / oracle correct', f"{o['base_correct']} / {o['majority_correct']} / {o['oracle_correct']} (of 561)"],
            ['Base-wrong questions fixed by $\\ge$1 / $\\ge$5 / $\\ge$10 members', f"{o['base_wrong_fixed_by_any_member']} / {o['base_wrong_fixed_by_ge5_members']} / {o['base_wrong_fixed_by_ge10_members']} (of {o['base_wrong_questions']})"]]
    write_tex('table_committee_diversity', ['SEARCH top-50 committee on TEST', 'Value'], rows,
              'Diversity of the SEARCH-selected top-50 committee on the official TEST set. Oracle = at least one member correct.', 'tab:committee', 'lr')
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.3))
    ks = [x['K'] for x in k_curve]
    axes[0].plot(ks, [x['acc'] for x in k_curve], '-o', color=CAND, ms=4, label='Top-K majority vote')
    axes[0].axhline(100 * b.mean(), color=BASE_C, ls='--', lw=1, label=f'Base ({100 * b.mean():.2f}%)')
    axes[0].set_xscale('log'); axes[0].set_xticks(ks, [str(k) for k in ks]); axes[0].minorticks_off()
    axes[0].set_xlabel('K (prefix of SEARCH ranking)'); axes[0].set_ylabel('TEST accuracy (%)'); axes[0].legend(frameon=False, fontsize=6.5)
    axes[0].set_title('A  Ensemble accuracy vs K', loc='left')
    axes[1].bar(range(len(ks)), [x['majority_differs_from_base'] for x in k_curve], color=CAND, width=.6, zorder=3)
    for i, x in enumerate(k_curve):
        axes[1].text(i, x['majority_differs_from_base'] + .8, f"{x['ens_only_correct']}/{x['base_only_correct']}", ha='center', fontsize=6.2)
    axes[1].set_xticks(range(len(ks)), [str(k) for k in ks]); axes[1].set_xlabel('K')
    axes[1].set_ylim(0, 1.15 * max(x['majority_differs_from_base'] for x in k_curve))
    axes[1].set_ylabel('Questions where majority\n$\\neq$ base answer'); axes[1].set_title('B  Committee converges to base (labels: ens.-only / base-only correct)', loc='left', fontsize=7)
    fig.tight_layout()
    save(fig, 'fig7_committee_convergence')
    print('I', json.dumps({k: v for k, v in res.items() if k != 'k_curve'}), json.dumps(k_curve))


# ---------------------------------------------------------------- Part J
def part_j():
    rec = {r['candidate']['candidate_id']: r for r in SL['records']}
    scores = Counter(r['correct_count'] for r in SL['records'])
    cutoff = rec[TOP50[-1]]['correct_count']
    at_cut = [c for c in SL['ranked_ids'] if rec[c]['correct_count'] == cutoff]
    included = [c for c in at_cut if c in TOP50]; excluded = [c for c in at_cut if c not in TOP50]
    rr = {c: gain(c, 'RERANK') for c in at_cut if (c, 'RERANK') in ACC}
    res = {'top_score_counts': {str(s): scores[s] for s in sorted(scores, reverse=True)[:8]}, 'cutoff_score': cutoff,
           'tie_at_cutoff': len(at_cut), 'tie_included': len(included), 'tie_excluded': len(excluded),
           'top50_members_decided_by_tiebreak': len(included), 'share_top50_decided_by_tiebreak': len(included) / 50,
           'candidates_within_1_question_of_cutoff': sum(v for s, v in scores.items() if abs(s - cutoff) <= 1),
           'candidates_within_2_questions_of_cutoff': sum(v for s, v in scores.items() if abs(s - cutoff) <= 2),
           'rerank_gains_in_cutoff_tie': {str(SEED_OF[c]): v for c, v in rr.items()},
           'excluded_tie_members_with_rerank_outputs': [SEED_OF[c] for c in excluded if c in rr],
           'candidate_rank_among_tie_rerank': sorted(rr.values(), reverse=True).index(rr[CANDIDATE]) + 1 if CANDIDATE in rr else None,
           'tie_rerank_gain_mean': float(np.mean(list(rr.values()))), 'tie_rerank_gain_sd': float(np.std(list(rr.values()), ddof=1)),
           'search_binomial_se_pp_at_base': 100 * math.sqrt(.36 * .64 / 200)}
    dump('table_tie_instability', res)
    rows = [[s, v, 'yes' if int(s) > cutoff else ('tie-break' if int(s) == cutoff else 'no')] for s, v in res['top_score_counts'].items()]
    write_tex('table_tie_instability', ['SEARCH score (/200)', 'Candidates', 'In top 50?'], rows,
              f"SEARCH score counts near the top-50 cut-off. {len(included)} of the top 50 (including seed 9504111) were admitted from a "
              f"{len(at_cut)}-way tie at {cutoff}/200 by the hash tie-break. Among tie members evaluated on RERANK, gains ranged "
              f"{min(rr.values()):+.1f} to {max(rr.values()):+.1f} pp.", 'tab:tie', 'rrl')
    print('J', json.dumps(res))


# ---------------------------------------------------------------- Part K
def part_k():
    base_s = ACC[('BASE', 'SEARCH')][0]
    out = []
    ids_a, M, P, q = correct_matrix(DF, 'RERANK', AUDIT_IDS)
    ga = 100 * (M.mean(1) - q.base_correct.mean())
    for s in SIGMAS:
        sc = np.array([r['correct_count'] for r in SL['records'] if r['candidate']['sigma'] == s])
        sg = 100 * (sc - base_s) / 200
        ra = ga[[SIGMA_OF[c] == s for c in ids_a]]
        tg = [gain(c, 'TEST') for c in TOP50 if SIGMA_OF[c] == s]
        out.append({'sigma': s, 'search_mean_gain': float(sg.mean()), 'search_sd': float(sg.std(ddof=1)), 'search_max': float(sg.max()),
                    'rerank_random_mean_gain': float(ra.mean()), 'rerank_random_sd': float(ra.std(ddof=1)), **{f'rerank_random_{k}': v for k, v in counts(ra).items()},
                    'top50_selected': len(tg), 'top50_test_mean_gain': float(np.mean(tg)) if tg else None, 'top50_test_best_gain': float(max(tg)) if tg else None})
    dump('fig8_sigma_tradeoff', pd.DataFrame(out))
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.1))
    x = np.arange(4); lab = [f'{s:g}' for s in SIGMAS]
    axes[0].bar(x, [o['search_mean_gain'] for o in out], color=CAND, width=.55, zorder=3, label='Mean')
    axes[0].errorbar(x, [o['search_mean_gain'] for o in out], yerr=[o['search_sd'] for o in out], fmt='none', ecolor=INK2, capsize=2, lw=.8)
    axes[0].scatter(x, [o['search_max'] for o in out], marker='_', s=80, color=ACCENT, zorder=4, label='Max')
    axes[0].set_title('A  SEARCH (1,250 each)', loc='left'); axes[0].set_ylabel('Gain over base (pp)')
    axes[1].bar(x, [o['rerank_random_mean_gain'] for o in out], color=CAND, width=.55, zorder=3)
    axes[1].errorbar(x, [o['rerank_random_mean_gain'] for o in out], yerr=[o['rerank_random_sd'] for o in out], fmt='none', ecolor=INK2, capsize=2, lw=.8)
    for i, o in enumerate(out): axes[1].text(i, o['rerank_random_mean_gain'] + o['rerank_random_sd'] + .3, f"{o['rerank_random_ge3']}", ha='center', fontsize=6.3)
    axes[1].set_title('B  RERANK, random audit (125 each)\nlabels: count at least +3 pp', loc='left', fontsize=7)
    tv = [o['top50_test_mean_gain'] if o['top50_test_mean_gain'] is not None else 0 for o in out]
    axes[2].bar(x, tv, color=ACCENT, width=.55, zorder=3)
    axes[2].set_title('C  TEST mean gain, SEARCH top-50 members', loc='left', fontsize=7)
    for ax in axes:
        ax.axhline(0, color=MUTED, lw=.8); ax.set_xticks(x, lab); ax.set_xlabel('$\\sigma$')
    axes[2].set_xticks(x, [f"{l}\n(n={o['top50_selected']})" for l, o in zip(lab, out)])
    axes[0].legend(frameon=False, fontsize=6.2, loc='lower left')
    fig.tight_layout()
    save(fig, 'fig8_sigma_tradeoff')
    print('K', json.dumps(out))


PARTS = {'B': part_b, 'C': part_c, 'D': part_d, 'E': part_e, 'F': part_f, 'G': part_g, 'H': part_h, 'I': part_i, 'J': part_j, 'K': part_k}
if __name__ == '__main__':
    for p in (sys.argv[1:] or list(PARTS)):
        RNG = np.random.default_rng([20261005, ord(p)])  # per-part stream: results do not depend on which parts ran
        PARTS[p]()

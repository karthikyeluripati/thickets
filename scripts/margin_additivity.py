"""Margin and additivity follow-up for candidate 9504111 (POST HOC, CPU). Implements
results/paper-analysis/margin-additivity/margin_additivity_plan.md exactly (committed before analysis).

Usage: python scripts/margin_additivity.py [cpu|figures|reserved SESSION_DIR]
"""
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

D = Path('results/paper-analysis/causal-diagnostic')
OUT = Path('results/paper-analysis/margin-additivity')
FIGD = Path('paper/figures/margin-additivity')
LET = 'ABCD'
BINS = (0.0, 0.5, 1.0, 2.0, 4.0, np.inf)
BIN_LAB = ('[0,0.5)', '[0.5,1)', '[1,2)', '[2,4)', '[4,inf)')
GROUPS = ('vision', 'embed', 'lm_q1', 'lm_q2', 'lm_q3', 'lm_q4', 'final_norm_head')
MIN_SUPPORT = 5
BOOT, BOOT_SEED = 2000, 20261010
OFFSETS = {'A': 0.3023, 'B': -0.6404, 'C': 0.1201, 'D': 0.2180}  # frozen session-1 letter-offset model (in-sample on LOCALIZATION)


def ld(f):
    return {x['uid']: x for x in json.loads(gzip.decompress(Path(f).read_bytes()))}


def vec(x):
    return np.array([x['letter_logprobs'][a] for a in LET], float)


# ------------------------------------------------------------------------------------------ pure helpers
def margin_terms(LB, LC, y):
    """Exact score-movement accounting for one example. Returns m_B, m_C, t, s, comparator r_B, gold rank, top-2 gap."""
    wrong = [a for a in range(4) if a != y]
    rB = min(wrong, key=lambda a: (-LB[a], a))  # strongest wrong under BASE; tie -> lowest letter
    mB = LB[y] - LB[rB]
    mC = LC[y] - max(LC[a] for a in wrong)
    t = (LC[y] - LC[rB]) - (LB[y] - LB[rB])
    s = max(LC[a] for a in wrong) - LC[rB]
    rank = 1 + sum(LB[a] > LB[y] for a in wrong)
    srt = np.sort(LB)[::-1]
    return {'m_B': mB, 'm_C': mC, 't': t, 's': s, 'comparator': LET[rB], 'gold_rank': int(rank),
            'gold_tied': bool(any(LB[a] == LB[y] for a in wrong)), 'top2_gap': float(srt[0] - srt[1])}


def margin_bin(m):
    a = abs(m)
    for i in range(5):
        if BINS[i] <= a < BINS[i + 1]:
            return BIN_LAB[i]
    return BIN_LAB[-1]


def gain_identity(b, r, d):
    return 100 * ((1 - b) * r - b * d)


def decompose(wT, wR, eT, eR):
    """Symmetric accounting: g_T - g_R = composition + residual (same strata, weights sum to 1 in each phase)."""
    comp = 100 * np.sum((wT - wR) * (eT + eR) / 2)
    resid = 100 * np.sum((wT + wR) / 2 * (eT - eR))
    return comp, resid


def center(L):
    L = np.asarray(L, float)
    return L - L.mean(-1, keepdims=True)


def additive_prediction(zB, zIs):
    """zB: (n,4) centered BASE; zIs: list of (n,4) centered single-group insertions. Coefficient exactly 1 per group."""
    return zB + sum(zI - zB for zI in zIs)


# ------------------------------------------------------------------------------------------- analyses A-C
def table_all():
    man = json.loads((D / 'example_manifest.json').read_text())
    B, C = ld(D / 'session1/base_full.json.gz'), ld(D / 'session1/candidate_full.json.gz')
    rows = []
    for u, lab in man['labels'].items():
        LB, LC = vec(B[u]), vec(C[u]); y = LET.index(lab['gold'])
        mt = margin_terms(LB, LC, y)
        bc, cc = B[u]['parsed'] == lab['gold'], C[u]['parsed'] == lab['gold']
        rows.append({'uid': u, 'phase': lab['phase'], 'subtask': lab['subtask'], 'gold': lab['gold'], 'base_ans': B[u]['parsed'], 'cand_ans': C[u]['parsed'],
                     'base_correct': bc, 'cand_correct': cc, 'delta': int(cc) - int(bc), 'transition': lab['transition'],
                     'stratum': f"{'Bcorrect' if bc else 'Bwrong'}|{margin_bin(mt['m_B'])}", **mt,
                     'base_tied_top': bool(sum(LB == LB.max()) > 1), 'cand_tied_top': bool(sum(LC == LC.max()) > 1)})
    df = pd.DataFrame(rows)
    df['identity_err'] = (df.m_C - (df.m_B + df.t - df.s)).abs()
    df.to_csv(OUT / 'per_example_margins_all761.csv', index=False)
    return df


def opportunities(df):
    out = {}
    for ph, g in df.groupby('phase'):
        bw, bc = g[~g.base_correct], g[g.base_correct]
        out[ph] = {'n': len(g), 'base_correct': int(g.base_correct.sum()), 'base_wrong': int((~g.base_correct).sum()),
                   'gold_rank': {int(k): int(v) for k, v in g.gold_rank.value_counts().sort_index().items()},
                   'gold_tied_with_best_wrong': int(g.gold_tied.sum()),
                   'margin_bins': {s: int(v) for s, v in g.stratum.value_counts().sort_index().items()},
                   'repairs': int((bw.cand_correct).sum()), 'repair_rate': float(bw.cand_correct.mean()),
                   'regressions': int((~bc.cand_correct).sum()), 'regression_rate': float((~bc.cand_correct).mean()),
                   'wrong_to_wrong_changes': int(((~g.base_correct) & (~g.cand_correct) & (g.base_ans != g.cand_ans)).sum()),
                   'median_m_B_base_wrong': float(bw.m_B.median()), 'median_m_B_base_correct': float(bc.m_B.median()),
                   'share_base_wrong_close_lt1': float((bw.m_B.abs() < 1).mean()), 'share_base_correct_close_lt1': float((bc.m_B.abs() < 1).mean())}
        b, r, d = g.base_correct.mean(), bw.cand_correct.mean(), (~bc.cand_correct).mean()
        out[ph]['gain_pp_observed'] = float(100 * g.delta.mean()); out[ph]['gain_pp_identity'] = float(gain_identity(b, r, d))
        out[ph]['rate_by_bin'] = {s: {'n': len(x), 'base_correct': bool(s.startswith('Bcorrect')),
                                      'rate': float((~x.cand_correct).mean() if s.startswith('Bcorrect') else x.cand_correct.mean()),
                                      'events': int((~x.cand_correct).sum() if s.startswith('Bcorrect') else x.cand_correct.sum())}
                                  for s, x in g.groupby('stratum')}
    return out


def standardize(df, rng=None):
    """Common-support standardization and symmetric decomposition (Analysis B), plus standardized mean t (Analysis C)."""
    R, T = df[df.phase == 'RERANK'], df[df.phase == 'TEST']
    strata = sorted(set(df.stratum))
    nR, nT = R.stratum.value_counts(), T.stratum.value_counts()
    cs = [h for h in strata if nR.get(h, 0) >= MIN_SUPPORT and nT.get(h, 0) >= MIN_SUPPORT]
    Rc, Tc = R[R.stratum.isin(cs)], T[T.stratum.isin(cs)]
    wR = np.array([(Rc.stratum == h).mean() for h in cs]); wT = np.array([(Tc.stratum == h).mean() for h in cs])
    eR = np.array([Rc[Rc.stratum == h].delta.mean() for h in cs]); eT = np.array([Tc[Tc.stratum == h].delta.mean() for h in cs])
    tR = np.array([Rc[Rc.stratum == h].t.mean() for h in cs]); tT = np.array([Tc[Tc.stratum == h].t.mean() for h in cs])
    w = (wR + wT) / 2
    comp, resid = decompose(wT, wR, eT, eR)
    res = {'common_support_strata': cs, 'min_support': MIN_SUPPORT,
           'retained_RERANK': len(Rc) / len(R), 'retained_TEST': len(Tc) / len(T),
           'gap_full_pp': 100 * (T.delta.mean() - R.delta.mean()), 'gap_common_support_pp': 100 * (Tc.delta.mean() - Rc.delta.mean()),
           'gain_std_RERANK_pp': float(100 * (w * eR).sum()), 'gain_std_TEST_pp': float(100 * (w * eT).sum()),
           'gap_standardized_pp': float(100 * (w * (eT - eR)).sum()), 'composition_component_pp': float(comp), 'within_stratum_residual_pp': float(resid),
           'mean_t_full': {'RERANK': float(R.t.mean()), 'TEST': float(T.t.mean())},
           'mean_t_standardized': {'RERANK': float((w * tR).sum()), 'TEST': float((w * tT).sum())},
           'strata': {h: {'n_R': int(nR.get(h, 0)), 'n_T': int(nT.get(h, 0)), 'w_R': float(a), 'w_T': float(b_), 'e_R': float(c), 'e_T': float(d_),
                          't_R': float(x), 't_T': float(y_), 'p_t_pos_R': float((Rc[Rc.stratum == h].t > 0).mean()), 'p_t_pos_T': float((Tc[Tc.stratum == h].t > 0).mean())}
                      for h, a, b_, c, d_, x, y_ in zip(cs, wR, wT, eR, eT, tR, tT)},
           'excluded': {h: {'n_R': int(nR.get(h, 0)), 'n_T': int(nT.get(h, 0)),
                            'delta_sum_R': int(R[R.stratum == h].delta.sum()), 'delta_sum_T': int(T[T.stratum == h].delta.sum())}
                        for h in strata if h not in cs}}
    return res


def bootstrap(df):
    rng = np.random.default_rng(BOOT_SEED)
    R, T = df[df.phase == 'RERANK'].reset_index(drop=True), df[df.phase == 'TEST'].reset_index(drop=True)
    keys = ('gap_full_pp', 'gap_common_support_pp', 'gap_standardized_pp', 'composition_component_pp', 'within_stratum_residual_pp')
    vals = {k: [] for k in keys}; tdiff = []
    for _ in range(BOOT):
        b = pd.concat([R.iloc[rng.integers(0, len(R), len(R))], T.iloc[rng.integers(0, len(T), len(T))]])
        try:
            s = standardize(b)
        except Exception:
            continue
        for k in keys: vals[k].append(s[k])
        tdiff.append(s['mean_t_standardized']['TEST'] - s['mean_t_standardized']['RERANK'])
    q = lambda v: {'q025': float(np.quantile(v, .025)), 'q975': float(np.quantile(v, .975)), 'n': len(v)}
    return {**{k: q(v) for k, v in vals.items()}, 'mean_t_standardized_TEST_minus_RERANK': q(tdiff),
            'note': 'post hoc; resampling unit = example within phase; strata/common support recomputed per replicate'}


def movement(df):
    out = {}
    for (ph, bc), g in df.groupby(['phase', 'base_correct']):
        out[f"{ph}|{'Bcorrect' if bc else 'Bwrong'}"] = {
            'n': len(g), 't_mean': float(g.t.mean()), 't_median': float(g.t.median()), 'p_t_pos': float((g.t > 0).mean()),
            's_mean': float(g.s.mean()), 'p_s_pos': float((g.s > 0).mean()),
            'cross_up_mB_lt0_to_mC_gt0': int(((g.m_B < 0) & (g.m_C > 0)).sum()), 'cross_down_mB_gt0_to_mC_lt0': int(((g.m_B > 0) & (g.m_C < 0)).sum()),
            'ties_mC_eq0': int((g.m_C == 0).sum())}
    for ph, g in df.groupby('phase'):
        out[f'{ph}|all'] = {'n': len(g), 't_mean': float(g.t.mean()), 'p_t_pos': float((g.t > 0).mean()), 's_mean': float(g.s.mean())}
    out['identity_max_abs_error'] = float(df.identity_err.max())
    return out


# ---------------------------------------------------------------------------------------------- analysis D
def additivity(uids, labels, cond_dir, base_full, cand_full, suffix='loc'):
    B, C = ld(base_full), ld(cand_full)
    Ins = {g: ld(Path(cond_dir) / f'insertion_{g}_{suffix}.json.gz') for g in GROUPS}
    Rem = {g: ld(Path(cond_dir) / f'removal_{g}_{suffix}.json.gz') for g in GROUPS if (Path(cond_dir) / f'removal_{g}_{suffix}.json.gz').exists()}
    zB = center([vec(B[u]) for u in uids]); zC = center([vec(C[u]) for u in uids])
    zI = {g: center([vec(Ins[g][u]) for u in uids]) for g in GROUPS}
    zadd = additive_prediction(zB, [zI[g] for g in GROUPS])
    gold = np.array([LET.index(labels[u]['gold']) for u in uids])
    tr = np.array([labels[u]['transition'] for u in uids])
    cand_ans = np.array([LET.index(labels[u]['cand']) for u in uids]); base_ans = np.array([LET.index(labels[u]['base']) for u in uids])
    LBraw = np.array([vec(B[u]) for u in uids])
    off = np.array([OFFSETS[a] for a in LET])

    def m_of(z):
        return np.array([z[i, gold[i]] - max(z[i, a] for a in range(4) if a != gold[i]) for i in range(len(z))])

    def am(z):
        mx = z.max(1, keepdims=True); tie = (z == mx).sum(1) > 1
        return z.argmax(1), tie

    def metrics(mask):
        if not mask.any():
            return None
        e = zadd[mask] - zC[mask]; sh = zC[mask] - zB[mask]
        pa, ptie = am(zadd[mask]); po, _ = am(center(LBraw[mask] + off))
        mA, mC_ = m_of(zadd[mask]), m_of(zC[mask])
        pairs = [(a, b) for a in range(4) for b in range(a + 1, 4)]
        pc = np.concatenate([(zadd[mask][:, a] - zadd[mask][:, b]) - (zB[mask][:, a] - zB[mask][:, b]) for a, b in pairs])
        ac = np.concatenate([(zC[mask][:, a] - zC[mask][:, b]) - (zB[mask][:, a] - zB[mask][:, b]) for a, b in pairs])
        return {'n': int(mask.sum()), 'l2_error_median': float(np.median(np.linalg.norm(e, axis=1))), 'mae': float(np.abs(e).mean()),
                'shift_l2_median': float(np.median(np.linalg.norm(sh, axis=1))),
                'relative_residual_energy': float((e ** 2).sum() / (sh ** 2).sum()) if (sh ** 2).sum() > 0 else None,
                'argmax_agree_candidate': int((pa == cand_ans[mask]).sum()), 'argmax_ties': int(ptie.sum()),
                'base_answer_predictor_agree_candidate': int((base_ans[mask] == cand_ans[mask]).sum()),
                'offset_model_agree_candidate': int((po == cand_ans[mask]).sum()),
                'margin_mae': float(np.abs(mA - mC_).mean()), 'margin_corr': float(np.corrcoef(mA, mC_)[0, 1]) if mask.sum() > 2 else None,
                'margin_sign_agree': int((np.sign(mA) == np.sign(mC_)).sum()),
                'pairwise_contrast_change_corr': float(np.corrcoef(pc, ac)[0, 1]), 'pairwise_contrast_change_mae': float(np.abs(pc - ac).mean())}

    groups_ = {'repair': tr == 'repair', 'regression': tr == 'regression', 'wrong_to_wrong_diff': tr == 'both_wrong_diff',
               'unchanged_controls': np.isin(tr, ['both_correct', 'both_wrong_same']), 'changed_all': np.isin(tr, ['repair', 'regression', 'both_wrong_diff']),
               'all': np.ones(len(uids), bool)}
    phases = np.array([labels[u]['phase'] for u in uids])
    res = {k: metrics(m) for k, m in groups_.items()}
    for ph in ('RERANK', 'TEST'):
        for k in ('repair', 'regression'):
            res[f'{ph}|{k}'] = metrics(groups_[k] & (phases == ph))
    cons = {}
    if Rem:
        for g in GROUPS:
            if g in Rem:
                u_ = zI[g] - zB; v_ = zC - center([vec(Rem[g][u]) for u in uids])
                cons[g] = {'corr': float(np.corrcoef(u_.ravel(), v_.ravel())[0, 1]), 'mean_abs_u': float(np.abs(u_).mean()), 'mean_abs_v': float(np.abs(v_).mean())}
        U = np.concatenate([(zI[g] - zB).ravel() for g in Rem]); V = np.concatenate([(zC - center([vec(Rem[g][u]) for u in uids])).ravel() for g in Rem])
        cons['pooled_corr'] = float(np.corrcoef(U, V)[0, 1])
    per = pd.DataFrame({'uid': uids, 'transition': tr, 'phase': phases, 'gold': [labels[u]['gold'] for u in uids],
                        'base_ans': [labels[u]['base'] for u in uids], 'cand_ans': [labels[u]['cand'] for u in uids],
                        'add_ans': [LET[i] for i in am(zadd)[0]], 'add_tie': am(zadd)[1],
                        'm_B': m_of(zB), 'm_C': m_of(zC), 'm_add': m_of(zadd), 'err_l2': np.linalg.norm(zadd - zC, axis=1),
                        'shift_l2': np.linalg.norm(zC - zB, axis=1), **{f'zC_{a}': zC[:, i] for i, a in enumerate(LET)},
                        **{f'zadd_{a}': zadd[:, i] for i, a in enumerate(LET)}, **{f'zB_{a}': zB[:, i] for i, a in enumerate(LET)}})
    return res, cons, per


def cpu():
    OUT.mkdir(parents=True, exist_ok=True)
    df = table_all()
    R = {'post_hoc': True, 'opportunities': opportunities(df), 'standardization': standardize(df), 'bootstrap': bootstrap(df),
         'movement': movement(df)}
    man = json.loads((D / 'example_manifest.json').read_text())
    res, cons, per = additivity(man['localization'], man['labels'], D / 'session1', D / 'session1/base_full.json.gz', D / 'session1/candidate_full.json.gz')
    per.to_csv(OUT / 'additivity_localization_per_example.csv', index=False)
    R['additivity_localization'] = {'label': 'EXPLORATORY (LOCALIZATION, not reserved-example validation)', 'metrics': res, 'u_vs_v_consistency': cons}
    (OUT / 'margin_additivity_results.json').write_text(json.dumps(R, indent=1))
    return R


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
    if what == 'cpu':
        R = cpu()
        print(json.dumps({k: R[k] for k in ('opportunities',)}, indent=1)[:4000])


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sys.path.insert(0, 'paper/figures/scripts')
    from common import ACCENT, BASE_C, CAND, GOOD, MUTED  # noqa: F401
    FIGD.mkdir(parents=True, exist_ok=True)
    R = json.loads((OUT / 'margin_additivity_results.json').read_text())
    df = pd.read_csv(OUT / 'per_example_margins_all761.csv')

    def save(fig, n):
        fig.savefig(FIGD / f'{n}.pdf', metadata={'CreationDate': None}); fig.savefig(FIGD / f'{n}.png', dpi=300); plt.close(fig)

    # Fig 1: opportunities vs base correct-answer margin
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.7))
    x = np.arange(5)
    for ax, bc, key, title in ((axes[0], False, 'repair', 'A  BASE wrong: repair rate'), (axes[1], True, 'regression', 'B  BASE correct: regression rate')):
        for i, (ph, col) in enumerate((('RERANK', CAND), ('TEST', ACCENT))):
            ob = R['opportunities'][ph]['rate_by_bin']
            vals = [ob.get(f"{'Bcorrect' if bc else 'Bwrong'}|{b}", {}).get('rate', np.nan) for b in BIN_LAB]
            ns = [ob.get(f"{'Bcorrect' if bc else 'Bwrong'}|{b}", {}).get('n', 0) for b in BIN_LAB]
            ax.bar(x + (i - .5) * .38, vals, .38, color=col, label=ph, zorder=3)
            for xi, v, n in zip(x, vals, ns):
                ax.text(xi + (i - .5) * .38, (0 if np.isnan(v) else v) + .01, f'{n}', ha='center', fontsize=5.6, color=BASE_C)
        ax.set_xticks(x, BIN_LAB); ax.set_xlabel('|BASE correct-answer margin| (nats)'); ax.set_title(title, loc='left', fontsize=7.4)
        ax.set_ylim(0, .8)
    axes[0].set_ylabel('Rate (bar labels: n examples)'); axes[0].legend(frameon=False, fontsize=6.5)
    fig.tight_layout(); save(fig, 'fig_opportunity_by_margin')

    # Fig 2: observed vs standardized gap, and standardized score movement
    S, Bt = R['standardization'], R['bootstrap']
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.6), gridspec_kw={'width_ratios': [1.4, 1]})
    ax = axes[0]
    items = [('Observed\n(all 761)', S['gap_full_pp'], Bt['gap_full_pp']), ('Common\nsupport', S['gap_common_support_pp'], Bt['gap_common_support_pp']),
             ('Standard-\nized', S['gap_standardized_pp'], Bt['gap_standardized_pp']),
             ('Composition\ncomponent', S['composition_component_pp'], Bt['composition_component_pp'])]
    for i, (lab, v, ci) in enumerate(items):
        ax.bar(i, v, .6, color=MUTED if i == 3 else CAND, zorder=3)
        ax.errorbar(i, v, yerr=[[v - ci['q025']], [ci['q975'] - v]], color=BASE_C, capsize=3, lw=1)
    ax.axhline(0, color=MUTED, lw=.8); ax.set_xticks(range(4), [i[0] for i in items], fontsize=6.2)
    ax.set_ylabel('TEST − RERANK gain gap (pp)'); ax.set_title('A  Gap before and after opportunity matching', loc='left', fontsize=7.4)
    ax = axes[1]
    mt = S['mean_t_standardized']
    ax.bar([0, 1], [mt['RERANK'], mt['TEST']], .55, color=[CAND, ACCENT], zorder=3)
    ax.axhline(0, color=MUTED, lw=.8); ax.set_xticks([0, 1], ['RERANK', 'TEST'])
    ax.set_ylabel('Standardized mean t (nats)')
    ci = Bt['mean_t_standardized_TEST_minus_RERANK']
    ax.set_title(f"B  Score shift toward correct answer\n   (TEST − RERANK {mt['TEST'] - mt['RERANK']:+.2f} [{ci['q025']:+.2f}, {ci['q975']:+.2f}])", loc='left', fontsize=7.2)
    fig.tight_layout(); save(fig, 'fig_standardized_gap')

    # Fig 3: additive prediction vs actual (LOCALIZATION; MECHANISM-CHECK if present)
    sets = [('LOCALIZATION (exploratory)', OUT / 'additivity_localization_per_example.csv')]
    if (OUT / 'additivity_reserved_per_example.csv').exists():
        sets.append(('MECHANISM-CHECK (reserved)', OUT / 'additivity_reserved_per_example.csv'))
    fig, axes = plt.subplots(1, len(sets), figsize=(3.5 * len(sets), 3), squeeze=False)
    colors = {('repair',): (GOOD, 'repair'), ('regression',): (ACCENT, 'regression'), ('both_wrong_diff',): ('#9cc2ee', 'wrong to different wrong'),
              ('both_correct', 'both_wrong_same'): (MUTED, 'unchanged control')}
    for ax, (title, f) in zip(axes[0], sets):
        p = pd.read_csv(f)
        for ts, (col, lab) in colors.items():
            q = p[p.transition.isin(ts)]
            ax.scatter(q.m_C - q.m_B, q.m_add - q.m_B, s=14, color=col, alpha=.85, lw=0, label=lab)
        lim = np.nanmax(np.abs(np.r_[p.m_C - p.m_B, p.m_add - p.m_B])) * 1.05
        ax.plot([-lim, lim], [-lim, lim], color=BASE_C, lw=.8, ls='--'); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.axhline(0, color=MUTED, lw=.6); ax.axvline(0, color=MUTED, lw=.6)
        ax.set_xlabel('Actual change in correct-answer margin (nats)'); ax.set_ylabel('Sum of 7 group insertions (nats)')
        ax.set_title(title, loc='left', fontsize=7.4)
    axes[0][0].legend(frameon=False, fontsize=5.8, loc='upper left')
    fig.tight_layout(); save(fig, 'fig_additive_vs_actual')

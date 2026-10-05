"""Perturbation-radius GO / NO-GO audit (POST HOC, CPU only, deterministic).

Question: as sigma grows, do we get better TRANSFERABLE candidates, or only a wider, more damaging score
distribution whose right tail looks attractive under selection? Four radii, 1,250 candidates each (SEARCH200);
125 precommitted audit candidates each with SEARCH200 + RERANK200 (400-question train-side pool).
Replays use IDENTICAL question splits for all four sigmas in every trial, so radius is the only population-level
variable. Replay intervals describe one observed prediction matrix; they are not independent searches.

Usage: python paper/figures/scripts/radius_audit.py [compute|figures|all]
"""
import hashlib
import json
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from common import ACCENT, BASE_C, CAND, GOOD, INK2, MUTED, SIGMAS, write_tex  # noqa: E402
from behavioral_diversity import load_inputs, matrix, permute_within  # noqa: E402
from transfer_density import agg, split_replay, split_two, masks, trial_stats, pp_cut  # noqa: E402

OUT = Path('results/paper-analysis/radius-audit')
FIGD = Path('paper/figures/radius-audit')
SEED = 20261008
QUICK = bool(os.environ.get('RA_QUICK'))
T_REPLAY, T_BUDGET, T_DENS, NULL_REPS, NULL_TRIALS = (100, 50, 100, 5, 10) if QUICK else (5000, 2000, 5000, 200, 25)
BUDGET_M = (10, 25, 50, 100, 125)
SIG_LAB = [f'{s:g}' for s in SIGMAS]


def compute():
    OUT.mkdir(parents=True, exist_ok=True)
    df, lock, proto = load_inputs()
    recs = {r['candidate']['candidate_id']: r for r in lock['records']}
    order = [r['candidate']['candidate_id'] for r in sorted(lock['records'], key=lambda r: r['index'])]
    sigma = np.array([recs[c]['candidate']['sigma'] for c in order])
    audit = [c for c in order if recs[c]['index'] in set(proto['density_audit']['indices'])]
    asg = np.array([recs[c]['candidate']['sigma'] for c in audit])
    P, b_, g_, qS = matrix(df, 'SEARCH', order)
    CS, bS = P == g_, b_ == g_
    PaS, _, _, _ = matrix(df, 'SEARCH', audit)
    PaR, bRp, gR, qR = matrix(df, 'RERANK', audit)
    meta = pd.read_csv('results/paper-analysis/transfer-density/question_pool_400.csv')
    checks = {'candidates': len(set(order)), 'search_questions': P.shape[1], 'per_sigma': {str(s): int((sigma == s).sum()) for s in SIGMAS},
              'base_search_correct': int(bS.sum()), 'missing_predictions': int(((P < 0) | (P > 3)).sum()),
              'audit': len(audit), 'audit_per_sigma': {str(s): int((asg == s).sum()) for s in SIGMAS},
              'audit_complete': PaS.shape == (500, 200) and PaR.shape == (500, 200),
              'pool_unique_ids': int(meta.qid.nunique()), 'pool_duplicate_image_hashes': int(len(meta) - meta.image_sha256.nunique()),
              'pool_order_matches': meta.qid.tolist() == list(qS) + list(qR), 'sigma_known': bool(np.isin(sigma, SIGMAS).all())}
    print(json.dumps(checks, indent=1), flush=True)
    if checks['candidates'] != 5000 or checks['base_search_correct'] != 72 or checks['audit'] != 500 or not checks['audit_complete'] \
            or checks['pool_unique_ids'] != 400 or checks['pool_duplicate_image_hashes'] or not checks['pool_order_matches'] \
            or checks['per_sigma'] != {str(s): 1250 for s in SIGMAS} or checks['audit_per_sigma'] != {str(s): 125 for s in SIGMAS}:
        raise SystemExit('STOP: verification failed')
    R = {'post_hoc': True, 'seed': SEED, 'checks': checks, 'trials': {'replay': T_REPLAY, 'budget': T_BUDGET, 'density': T_DENS}}
    (OUT / 'candidate_ids_sigma.json').write_text(json.dumps({'search_order': order, 'sigma': sigma.tolist(), 'audit500': audit, 'audit_sigma': asg.tolist()}))

    # Part 1-2: population and tail on SEARCH200
    gain = 100 * (CS.sum(1) - bS.sum()) / 200
    changed = (P != b_).sum(1)
    pop = {}
    for s in SIGMAS:
        g = gain[sigma == s]; acc = 100 * CS[sigma == s].mean(1); gs = np.sort(g)[::-1]
        q = {p: float(np.percentile(g, p)) for p in (5, 25, 50, 75, 90, 95, 99)}
        pop[str(s)] = {'mean_acc': float(acc.mean()), 'median_acc': float(np.median(acc)), 'sd_acc': float(acc.std(ddof=1)),
                       'mean_gain': float(g.mean()), 'sd_gain': float(g.std(ddof=1)), **{f'q{p}_gain': v for p, v in q.items()},
                       'max_gain': float(g.max()), 'top10_mean_gain': float(gs[:10].mean()), 'top50_mean_gain': float(gs[:50].mean()),
                       'tail_spread_q99_minus_median': q[99] - q[50],
                       'frac_above_base': float((g > 0).mean()), 'frac_ge1pp': float((g >= 1 - 1e-9).mean()),
                       'frac_ge3pp': float((g >= 3 - 1e-9).mean()), 'frac_ge5pp': float((g >= 5 - 1e-9).mean()),
                       'median_answers_changed': float(np.median(changed[sigma == s])), 'mean_answers_changed': float(changed[sigma == s].mean())}
    R['population'] = pop
    pd.DataFrame({'candidate_id': order, 'sigma': sigma, 'search_gain_pp': gain, 'answers_changed': changed}).to_csv(OUT / 'search_population.csv', index=False)

    # Parts 3,5,6,8: per-sigma replay with identical splits across sigma
    C400 = np.hstack([PaS == g_, PaR == gR]); b400 = np.concatenate([bS, bRp == gR])
    D400 = C400.astype(np.int16) - b400.astype(np.int16)
    codes = pd.factorize(meta.subtask + meta.answer)[0]
    is_s = (meta.phase == 'SEARCH').to_numpy()
    hr = np.argsort(np.argsort([hashlib.sha256(('visual-rank-v1:' + c).encode()).hexdigest() for c in audit]))
    pools = {s: np.where(asg == s)[0] for s in SIGMAS}

    def select(D, pool, S, Rr, K):
        gS, gR_ = D[pool][:, S].sum(1), D[pool][:, Rr].sum(1)
        o = np.lexsort((hr[pool], -gS)); top = o[:K]
        w = top[np.lexsort((hr[pool][top], -gR_[top]))[0]]
        return pool[w], gS[w], gR_[w]

    def rd(C, b, idx, rows):
        bw, bc = idx[~b[idx]], idx[b[idx]]
        return C[np.ix_(rows, bw)].mean(), (~C[np.ix_(rows, bc)]).mean()

    rng = np.random.default_rng([SEED, 1]); rec = []; split_log = []
    for t in range(T_REPLAY):
        S, Rr, H = split_replay(codes, rng)
        split_log.append(np.concatenate([S, Rr, H]).astype(np.int16))
        hs, hrr = H[is_s[H]], H[~is_s[H]]
        for s in SIGMAS:
            w, gS, gR_ = select(D400, pools[s], S, Rr, 25)
            rep, dam = rd(C400, b400, H, [w])
            prep, pdam = rd(C400, b400, H, pools[s])
            rec.append({'trial': t, 'sigma': s, 'winner': audit[w], 'search_gain_pp': 100 * gS / 100, 'rerank_gain_pp': 100 * gR_ / 100,
                        'holdout_gain_pp': 100 * D400[w, H].sum() / 200, 'holdout_search_origin_pp': 100 * D400[w, hs].mean(),
                        'holdout_rerank_origin_pp': 100 * D400[w, hrr].mean(), 'pool_holdout_pp': 100 * D400[pools[s]][:, H].sum(1).mean() / 200,
                        'winner_repair_H': rep, 'winner_damage_H': dam, 'pool_repair_H': prep, 'pool_damage_H': pdam})
    tr = pd.DataFrame(rec)
    tr.to_csv(OUT / 'replay_per_trial.csv', index=False)
    np.savez_compressed(OUT / 'replay_splits.npz', splits=np.array(split_log), layout=np.array([100, 100, 200]))
    R['replay'] = {}
    for s in SIGMAS:
        d = tr[tr.sigma == s]
        R['replay'][str(s)] = {'search_gain': agg(d.search_gain_pp), 'rerank_gain': agg(d.rerank_gain_pp), 'holdout_gain': agg(d.holdout_gain_pp),
                               'optimism': agg(d.rerank_gain_pp - d.holdout_gain_pp), 'pool_holdout': agg(d.pool_holdout_pp),
                               'winner_minus_pool_holdout': agg(d.holdout_gain_pp - d.pool_holdout_pp),
                               'holdout_search_origin': agg(d.holdout_search_origin_pp), 'holdout_rerank_origin': agg(d.holdout_rerank_origin_pp),
                               'p_gt0': float((d.holdout_gain_pp > 0).mean()), 'p_ge1': float((d.holdout_gain_pp >= 1 - 1e-9).mean()),
                               'p_ge3': float((d.holdout_gain_pp >= 3 - 1e-9).mean()), 'p_ge5': float((d.holdout_gain_pp >= 5 - 1e-9).mean()),
                               'distinct_winners': int(d.winner.nunique()), 'top_winner_share': float(d.winner.value_counts().iloc[0] / len(d)),
                               'winner_repair_H': agg(d.winner_repair_H), 'winner_damage_H': agg(d.winner_damage_H),
                               'pool_repair_H': agg(d.pool_repair_H), 'pool_damage_H': agg(d.pool_damage_H)}
        rr = R['replay'][str(s)]
        rr['transfer_fraction'] = rr['holdout_gain']['mean'] / rr['rerank_gain']['mean'] if rr['rerank_gain']['mean'] > 0 else None
    # paired ordering of sigmas within trials (same splits)
    piv = tr.pivot(index='trial', columns='sigma', values='holdout_gain_pp')
    best = piv.idxmax(axis=1)
    R['replay_paired'] = {'share_best_holdout': {str(s): float((best == s).mean()) for s in SIGMAS},
                          'pairwise_P_row_gt_col': {f'{a:g}>{b:g}': float((piv[a] > piv[b]).mean()) for a in SIGMAS for b in SIGMAS if a != b},
                          'pairwise_mean_diff_ci': {f'{a:g}-{b:g}': agg(piv[a] - piv[b]) for i, a in enumerate(SIGMAS) for b in SIGMAS[i + 1:]}}
    # item-fragility null per sigma (selection noise only)
    nrng = np.random.default_rng([SEED, 2]); nrec = []
    for _ in range(NULL_REPS):
        Dn = permute_within(C400, asg, nrng).astype(np.int16) - b400.astype(np.int16)
        for _ in range(NULL_TRIALS):
            S, Rr, H = split_replay(codes, nrng)
            for s in SIGMAS:
                w, gS, gR_ = select(Dn, pools[s], S, Rr, 25)
                nrec.append({'sigma': s, 'rerank': gR_, 'holdout': Dn[w, H].sum() / 2})
    nr = pd.DataFrame(nrec)
    R['replay_null'] = {str(s): {'rerank_gain': agg(nr[nr.sigma == s].rerank), 'holdout_gain': agg(nr[nr.sigma == s].holdout),
                                 'p_gt0': float((nr[nr.sigma == s].holdout > 0).mean())} for s in SIGMAS}
    print('replay done', flush=True)

    # Part 10: search budget within each radius (same splits across sigma)
    brng = np.random.default_rng([SEED, 3]); brec = []
    for t in range(T_BUDGET):
        S, Rr, H = split_replay(codes, brng)
        subs = {M: brng.permutation(125)[:M] for M in BUDGET_M}  # same within-sigma positions for every sigma
        for s in SIGMAS:
            for M in BUDGET_M:
                pool = pools[s][np.sort(subs[M])]
                w, gS, gR_ = select(D400, pool, S, Rr, max(5, round(.2 * M)))
                brec.append({'sigma': s, 'M': M, 'rerank': gR_, 'holdout': D400[w, H].sum() / 2})
    br = pd.DataFrame(brec)
    R['budget'] = {str(s): {str(M): {'K': max(5, round(.2 * M)), 'rerank_gain': agg(br[(br.sigma == s) & (br.M == M)].rerank),
                                     'holdout_gain': agg(br[(br.sigma == s) & (br.M == M)].holdout),
                                     'optimism': agg(br[(br.sigma == s) & (br.M == M)].rerank - br[(br.sigma == s) & (br.M == M)].holdout)}
                            for M in BUDGET_M} for s in SIGMAS}
    print('budget done', flush=True)

    # Part 7: density vs transfer by radius, recomputed from source (all 5,000, SEARCH100 vs SEARCH100)
    codesS = codes[:200]
    drng = np.random.default_rng([SEED, 4])
    sp = [split_two(codesS, drng) for _ in range(T_DENS)]
    DS = CS.astype(np.int16) - bS.astype(np.int16)
    cuts = {'pp3': pp_cut('pp3', 3, 100), 'gt0': 1}
    R['density'] = {}
    for s in SIGMAS:
        ix = np.where(sigma == s)[0]; acc = {}
        for c0 in range(0, T_DENS, 1000):
            MA, MB = masks([x[0] for x in sp[c0:c0 + 1000]], 200), masks([x[1] for x in sp[c0:c0 + 1000]], 200)
            st = trial_stats((DS[ix] @ MA.T).astype(np.int32), (DS[ix] @ MB.T).astype(np.int32), cuts, spearman=False)
            for k, v in st.items():
                acc.setdefault(k, []).append(v)
        acc = {k: np.concatenate(v) for k, v in acc.items()}
        a, j = acc['pp3|rho_A'].mean(), acc['pp3|rho_joint'].mean()
        R['density'][str(s)] = {'rho_A_ge3pp': float(a), 'rho_B_ge3pp': float(acc['pp3|rho_B'].mean()), 'joint_ge3pp': float(j),
                                'tau_ge3pp_pooled': float(j / a) if a else None, 'joint_count_mean': float(j * len(ix)),
                                'gain_r': agg(acc['pearson']), 'rho_A_gt0': float(acc['gt0|rho_A'].mean()), 'joint_gt0': float(acc['gt0|rho_joint'].mean())}
    td = json.loads(Path('results/paper-analysis/transfer-density/transfer_density_results.json').read_text())['primary']
    R['density_crosscheck_vs_transfer_density_audit'] = {str(s): {'rho_A_ge3pp_previous': td[str(s)]['pp3|rho_A']['mean'],
                                                                  'gain_r_previous': td[str(s)]['pearson']['mean']} for s in SIGMAS}

    # Part 8 (SEARCH / RERANK) and Part 9
    rdS = {str(s): rd(C400, b400, np.arange(200), pools[s]) for s in SIGMAS}
    rdR = {str(s): rd(C400, b400, np.arange(200, 400), pools[s]) for s in SIGMAS}
    R['repair_damage'] = {str(s): {'SEARCH': {'repair': float(rdS[str(s)][0]), 'damage': float(rdS[str(s)][1])},
                                   'RERANK': {'repair': float(rdR[str(s)][0]), 'damage': float(rdR[str(s)][1])},
                                   'HOLDOUT_pool_mean': {'repair': R['replay'][str(s)]['pool_repair_H']['mean'], 'damage': R['replay'][str(s)]['pool_damage_H']['mean']},
                                   'HOLDOUT_selected_winner': {'repair': R['replay'][str(s)]['winner_repair_H']['mean'], 'damage': R['replay'][str(s)]['winner_damage_H']['mean']}}
                          for s in SIGMAS}
    bd = json.loads(Path('results/paper-analysis/behavioral-diversity/behavioral_diversity_results.json').read_text())
    R['behavior'] = {str(s): {'median_answers_changed': bd['distance'][str(s)]['answers_changed']['p50'],
                              'answer_r_entropy': bd['spectrum'][str(s)]['answer_raw']['r_entropy'], 'answer_ceiling': bd['spectrum'][str(s)]['answer_raw']['rank_ceiling'],
                              'answer_std_r_entropy': bd['spectrum'][str(s)]['answer_std']['r_entropy'],
                              'obs_over_null_raw': bd['null'][str(s)]['answer_raw']['r_entropy']['observed_over_null_mean'],
                              'gain_variance_pp2': pop[str(s)]['sd_gain'] ** 2} for s in SIGMAS}
    # monotonicity checks (Spearman over the four sigma points)
    def mono(v):
        return {'values': v, 'increasing': bool(all(b > a for a, b in zip(v, v[1:]))), 'decreasing': bool(all(b < a for a, b in zip(v, v[1:])))}
    R['monotonic'] = {'population_mean_gain': mono([pop[str(s)]['mean_gain'] for s in SIGMAS]),
                      'gain_sd': mono([pop[str(s)]['sd_gain'] for s in SIGMAS]),
                      'q99_gain': mono([pop[str(s)]['q99_gain'] for s in SIGMAS]), 'max_gain': mono([pop[str(s)]['max_gain'] for s in SIGMAS]),
                      'selected_rerank_gain': mono([R['replay'][str(s)]['rerank_gain']['mean'] for s in SIGMAS]),
                      'holdout_gain': mono([R['replay'][str(s)]['holdout_gain']['mean'] for s in SIGMAS]),
                      'optimism': mono([R['replay'][str(s)]['optimism']['mean'] for s in SIGMAS])}
    (OUT / 'radius_audit_results.json').write_text(json.dumps(R, indent=1, default=lambda x: x.item() if hasattr(x, 'item') else str(x)))
    print('COMPUTE_DONE', flush=True)


def save_ra(fig, name):
    FIGD.mkdir(parents=True, exist_ok=True)
    plt.rcParams['svg.hashsalt'] = name
    fig.savefig(FIGD / f'{name}.pdf', metadata={'CreationDate': None})
    fig.savefig(FIGD / f'{name}.png', dpi=300)
    plt.close(fig)


def figures_and_tables():
    R = json.loads((OUT / 'radius_audit_results.json').read_text())
    pop = pd.read_csv(OUT / 'search_population.csv')
    x = np.arange(4)

    # Figure 1: SEARCH distribution by radius
    fig, ax = plt.subplots(figsize=(4.6, 3))
    data = [pop.search_gain_pp[pop.sigma == s].values for s in SIGMAS]
    vp = ax.violinplot(data, positions=x, widths=.75, showextrema=False)
    for b in vp['bodies']:
        b.set_facecolor('#9cc2ee'); b.set_edgecolor('none'); b.set_alpha(.8)
    P = R['population']
    for key, mk, col, lab in (('mean_gain', 'o', CAND, 'Mean'), ('q50_gain', '_', BASE_C, 'Median'), ('q95_gain', '^', GOOD, '95th pct'),
                              ('q99_gain', 'v', GOOD, '99th pct'), ('max_gain', '*', ACCENT, 'Maximum')):
        ax.scatter(x, [P[str(s)][key] for s in SIGMAS], marker=mk, s=40 if mk != '*' else 70, color=col, zorder=4, label=lab)
    ax.plot(x, [P[str(s)]['mean_gain'] for s in SIGMAS], color=CAND, lw=1, zorder=3)
    ax.plot(x, [P[str(s)]['max_gain'] for s in SIGMAS], color=ACCENT, lw=1, zorder=3)
    ax.axhline(0, color=MUTED, lw=.8)
    ax.set_xticks(x, SIG_LAB); ax.set_xlabel('Perturbation radius σ (1,250 candidates each)'); ax.set_ylabel('SEARCH200 gain over base (pp)')
    ax.legend(frameon=False, fontsize=6.3, loc='lower left', ncol=2)
    fig.tight_layout(); save_ra(fig, 'fig_radius_search_distribution')

    # Figure 2: selection vs transfer
    rp, rn = R['replay'], R['replay_null']
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.9), gridspec_kw={'width_ratios': [1.25, 1]})
    ax = axes[0]
    for key, col, lab, a in (('search_gain', MUTED, 'Winner SEARCH100 gain', .5), ('rerank_gain', ACCENT, 'Winner RERANK100 gain (selected score)', 1),
                             ('holdout_gain', CAND, 'Same winner, fresh HOLDOUT200 gain', 1)):
        m = [rp[str(s)][key]['mean'] for s in SIGMAS]
        if key != 'search_gain':
            ax.fill_between(x, [rp[str(s)][key]['q025'] for s in SIGMAS], [rp[str(s)][key]['q975'] for s in SIGMAS], color=col, alpha=.13, lw=0)
        ax.plot(x, m, color=col, marker='o', ms=4.5, alpha=a, label=lab)
    ax.plot(x, [rn[str(s)]['holdout_gain']['mean'] for s in SIGMAS], color=CAND, ls='--', lw=1, alpha=.7, label='HOLDOUT under item-fragility null')
    ax.axhline(0, color=MUTED, lw=.8); ax.set_xticks(x, SIG_LAB); ax.set_xlabel('σ (125 audit candidates each)')
    ax.set_ylabel('Gain over base (pp)')
    ax.set_title('A  125 → top 25 → winner → fresh holdout (5,000 replays, 95% bands)', loc='left', fontsize=7.2)
    ax.legend(frameon=False, fontsize=5.9, loc='upper left')
    ax = axes[1]
    for key, col, lab in (('holdout_search_origin', '#5a9be0', 'HOLDOUT, SEARCH-origin items'), ('holdout_rerank_origin', '#123f7a', 'HOLDOUT, RERANK-origin items')):
        ax.plot(x, [rp[str(s)][key]['mean'] for s in SIGMAS], color=col, marker='o', ms=4.5, label=lab)
    ax.plot(x, [rp[str(s)]['pool_holdout']['mean'] for s in SIGMAS], color=MUTED, ls=':', marker='s', ms=3, label='Mean of all 125 candidates, HOLDOUT')
    ax.axhline(0, color=MUTED, lw=.8); ax.set_xticks(x, SIG_LAB); ax.set_xlabel('σ')
    ax.set_title('B  Winner HOLDOUT gain by question origin', loc='left', fontsize=7.2)
    ax.legend(frameon=False, fontsize=5.9, loc='upper left')
    fig.tight_layout(); save_ra(fig, 'fig_radius_selection_vs_transfer')

    # Figure 3: repair / damage
    rd = R['repair_damage']
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.7))
    for ax, key, title in ((axes[0], 'HOLDOUT_pool_mean', 'A  All 125 candidates per σ, fresh HOLDOUT'),
                           (axes[1], 'HOLDOUT_selected_winner', 'B  Selected winner, fresh HOLDOUT')):
        ax.plot(x, [100 * rd[str(s)][key]['repair'] for s in SIGMAS], color=GOOD, marker='o', ms=4.5, label='Repair rate: P(correct | base wrong)')
        ax.plot(x, [100 * rd[str(s)][key]['damage'] for s in SIGMAS], color=ACCENT, marker='o', ms=4.5, label='Damage rate: P(wrong | base correct)')
        ax.set_xticks(x, SIG_LAB); ax.set_xlabel('σ'); ax.set_ylim(0, None); ax.set_title(title, loc='left', fontsize=7.2)
    axes[0].set_ylabel('Rate (%)'); axes[0].legend(frameon=False, fontsize=6.2, loc='upper left')
    fig.tight_layout(); save_ra(fig, 'fig_radius_repair_damage')

    # Appendix figure: budget within radius
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.6), sharey=True)
    Ms = [int(m) for m in R['budget'][str(SIGMAS[0])]]
    cols = {0.00025: '#9cc2ee', 0.0005: '#5a9be0', 0.001: CAND, 0.002: '#123f7a'}
    for ax, key, title in ((axes[0], 'rerank_gain', 'A  Selected RERANK gain'), (axes[1], 'holdout_gain', 'B  Fresh HOLDOUT gain')):
        for s in SIGMAS:
            ax.plot(Ms, [R['budget'][str(s)][str(M)][key]['mean'] for M in Ms], color=cols[s], marker='o', ms=3.5, label=f'σ = {s:g}')
        ax.axhline(0, color=MUTED, lw=.8); ax.set_xticks(Ms); ax.set_xlabel('Candidates searched within σ (M)'); ax.set_title(title, loc='left', fontsize=7.2)
    axes[0].set_ylabel('Gain over base (pp)'); axes[0].legend(frameon=False, fontsize=6.2)
    fig.tight_layout(); save_ra(fig, 'fig_radius_budget')

    tables(R)


def tables(R):
    P = R['population']
    rows = []
    for s in SIGMAS:
        p = P[str(s)]
        rows.append([f'{s:g}', f"{p['mean_acc']:.2f}", f"{p['median_acc']:.1f}", f"{p['sd_acc']:.2f}",
                     ' / '.join(f"{p[f'q{q}_gain']:+.1f}" for q in (5, 25, 50, 75, 90, 95, 99)), f"{p['max_gain']:+.1f}",
                     f"{p['top10_mean_gain']:+.2f}", f"{100 * p['frac_above_base']:.1f}", f"{100 * p['frac_ge3pp']:.1f}", f"{100 * p['frac_ge5pp']:.2f}",
                     f"{p['median_answers_changed']:.0f}"])
    write_tex('table_radius_population', ['$\\sigma$', 'Mean acc.', 'Median', 'SD', 'Gain pct.\\ 5/25/50/75/90/95/99 (pp)', 'Max', 'Top-10 mean',
                                          '\\% $>$base', '\\% $\\ge$+3pp', '\\% $\\ge$+5pp', 'Median answers changed'], rows,
              'SEARCH200 population by perturbation radius (1,250 candidates each; base 72/200 = 36.0\\%). Gains in pp over base.',
              'tab:radius-population', colspec='lrrrlrrrrrr')
    rp, rn = R['replay'], R['replay_null']
    rows = []
    for s in SIGMAS:
        r, n = rp[str(s)], rn[str(s)]
        rows.append([f'{s:g}', f"{r['search_gain']['mean']:+.2f}", f"{r['rerank_gain']['mean']:+.2f}",
                     f"{r['holdout_gain']['mean']:+.2f} / {r['holdout_gain']['median']:+.1f}", f"[{r['holdout_gain']['q025']:+.1f}, {r['holdout_gain']['q975']:+.1f}]",
                     f"{r['p_gt0']:.2f}", f"{r['p_ge1']:.2f}", f"{r['p_ge3']:.2f}", f"{r['p_ge5']:.2f}", f"{r['optimism']['mean']:+.2f}",
                     f"{r['transfer_fraction']:.2f}" if r['transfer_fraction'] is not None else '--',
                     f"{r['holdout_search_origin']['mean']:+.2f} / {r['holdout_rerank_origin']['mean']:+.2f}", f"{n['holdout_gain']['mean']:+.2f}"])
    write_tex('table_radius_transfer', ['$\\sigma$', 'SEARCH', 'RERANK', 'HOLDOUT mean / median', 'HOLDOUT 95\\%', 'P($>$0)', 'P($\\ge$1)', 'P($\\ge$3)',
                                        'P($\\ge$5)', 'Optimism', 'Transfer fraction', 'HOLDOUT by origin S / R', 'Null HOLDOUT'], rows,
              'POST HOC replay within each radius: 125 precommitted audit candidates $\\to$ SEARCH100 top 25 $\\to$ RERANK100 winner $\\to$ fresh HOLDOUT200, '
              'with identical stratified splits for all four radii in each of 5,000 trials. Gains of the selected winner in pp; intervals are 2.5--97.5\\% of '
              'replays (one observed matrix, not independent searches). Optimism = RERANK $-$ HOLDOUT; transfer fraction = mean HOLDOUT / mean RERANK. '
              'Origin: HOLDOUT gain on SEARCH200- vs RERANK200-origin items. Null: item-fragility permutation within $\\sigma$.',
              'tab:radius-transfer', colspec='lrrrrrrrrrrrr')
    D = R['density']
    rows = [[f'{s:g}', f"{100 * D[str(s)]['rho_A_ge3pp']:.2f}", f"{100 * D[str(s)]['joint_ge3pp']:.3f}", f"{D[str(s)]['joint_count_mean']:.1f}",
             f"{D[str(s)]['tau_ge3pp_pooled']:.3f}", f"{D[str(s)]['gain_r']['mean']:.3f}"] for s in SIGMAS]
    write_tex('table_radius_density_transfer', ['$\\sigma$', 'Density $\\ge$+3pp on A (\\%)', 'Joint A$\\cap$B (\\%)', 'Joint count', 'Transfer rate', 'Gain $r$ A$\\leftrightarrow$B'], rows,
              'POST HOC. All 5,000 candidates (1,250 per $\\sigma$), SEARCH100 vs SEARCH100, 5,000 stratified splits; recomputed from source predictions. '
              'Transfer rate pooled (mean joint / mean density).', 'tab:radius-density', colspec='lrrrrr')
    rows = []
    for s in SIGMAS:
        r = R['repair_damage'][str(s)]
        rows.append([f'{s:g}'] + [f"{100 * r[k]['repair']:.1f} / {100 * r[k]['damage']:.1f}" for k in ('SEARCH', 'RERANK', 'HOLDOUT_pool_mean', 'HOLDOUT_selected_winner')])
    write_tex('table_radius_repair_damage', ['$\\sigma$', 'SEARCH', 'RERANK', 'HOLDOUT (all 125)', 'HOLDOUT (selected winner)'], rows,
              'POST HOC. Repair rate / damage rate (\\%) by radius, 125 audit candidates per $\\sigma$. Repair = P(correct $\\mid$ base wrong); damage = '
              'P(wrong $\\mid$ base correct). HOLDOUT values average the 5,000 replay holdouts.', 'tab:radius-repair', colspec='lrrrr')
    B = R['behavior']
    rows = [[f'{s:g}', f"{B[str(s)]['median_answers_changed']:g}", f"{B[str(s)]['answer_r_entropy']:.1f} / {B[str(s)]['answer_ceiling']}",
             f"{B[str(s)]['answer_std_r_entropy']:.1f}", f"{B[str(s)]['obs_over_null_raw']:.3f}", f"{B[str(s)]['gain_variance_pp2']:.2f}"] for s in SIGMAS]
    write_tex('table_radius_behavior', ['$\\sigma$', 'Median answers changed', 'Answer eff.\\ rank / ceiling', 'Std.\\ eff.\\ rank', 'Obs./null rank', 'Gain variance (pp$^2$)'],
              rows, 'POST HOC (appendix). Movement and behavioral structure by radius, from the behavioral-diversity audit (SEARCH200, 1,250 per $\\sigma$).',
              'tab:radius-behavior', colspec='lrrrrr')
    rows = []
    for s in SIGMAS:
        for M, v in R['budget'][str(s)].items():
            rows.append([f'{s:g}' if M == '10' else '', M, str(v['K']), f"{v['rerank_gain']['mean']:+.2f}", f"{v['holdout_gain']['mean']:+.2f}", f"{v['optimism']['mean']:+.2f}"])
        rows.append('MIDRULE')
    write_tex('table_radius_budget', ['$\\sigma$', '$M$', '$K$', 'RERANK gain', 'HOLDOUT gain', 'Optimism'], rows[:-1],
              'POST HOC (appendix). Search budget within each radius: random subsets of the 125 audit candidates, $K=\\max(5,\\mathrm{round}(0.2M))$, '
              'identical splits across $\\sigma$ (2,000 trials).', 'tab:radius-budget', colspec='lrrrrr')


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what in ('compute', 'all'):
        compute()
    if what in ('figures', 'all'):
        figures_and_tables()

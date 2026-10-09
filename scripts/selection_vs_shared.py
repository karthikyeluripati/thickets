"""Selection vs shared response for 9504111 (POST HOC, CPU only). Implements
results/paper-analysis/selection-vs-shared-response/analysis_plan.md (committed before analysis).

Usage: python scripts/selection_vs_shared.py [run|figures]
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

OUT = Path('results/paper-analysis/selection-vs-shared-response')
FIGD = Path('paper/figures/selection-vs-shared-response')
ROOT = Path('results/perspective-taking-n5000-20261004')
WINNER = '25a60f0b59f103ebf782a31f94bb569d897a652a0bc1bd95c5cdc5521418b843'
SEED, TRIALS, N_CONTROL = 20261011, 2000, 12
LET = 'ABCD'


# ------------------------------------------------------------------------------------------ pure helpers
def rank_hash(cid):
    return hashlib.sha256(('visual-rank-v1:' + cid).encode()).hexdigest()


def select_winner(ids, sel_correct, search_correct):
    """Original RERANK rule: selection correct desc, SEARCH200 correct desc, frozen hash asc."""
    return min(ids, key=lambda c: (-sel_correct[c], -search_correct[c], rank_hash(c)))


def accounting(gR_w, gT_w, muR, muT):
    """g_T* - g_R* = (mu_T - mu_R) + [(g_T* - mu_T) - (g_R* - mu_R)]."""
    group, winner = muT - muR, (gT_w - muT) - (gR_w - muR)
    return {'gap': gT_w - gR_w, 'group_mean_change': group, 'winner_advantage_change': winner}


def stratified_halves(codes, rng):
    """Sort by (random stratum order, random key); alternate assignment from a random offset."""
    k = codes.max() + 1
    o = np.lexsort((rng.random(len(codes)), rng.permutation(k)[codes]))
    lab = (np.arange(len(o)) + rng.integers(2)) % 2
    return np.sort(o[lab == 0]), np.sort(o[lab == 1])


def describe(g):
    g = np.asarray(g, float)
    return {'n': int(len(g)), 'mean': float(g.mean()), 'median': float(np.median(g)), 'sd': float(g.std(ddof=1)) if len(g) > 1 else 0.,
            'min': float(g.min()), 'max': float(g.max()), 'above_base': int((g > 0).sum())}


def position(x, g, member=True):
    """Rank of value x among g (1 = best). If x is itself one of g (member), it is not counted as its own tie."""
    g = np.asarray(g, float)
    return {'rank_best_1': int(1 + (g > x).sum()), 'tied_with': int((g == x).sum() - (1 if member else 0)),
            'share_strictly_below': float((g < x).mean()), 'n': int(len(g)), 'winner_is_member': member}


# --------------------------------------------------------------------------------------------- analysis
def run():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = json.loads((ROOT / 'locks/search.json').read_text())
    proto = json.loads(Path('experiments/perspective_taking_n5000_protocol.json').read_text())
    recs = {r['candidate']['candidate_id']: r for r in lock['records']}
    top50 = list(lock['ranked_ids'][:50])
    audit = [r['candidate']['candidate_id'] for r in sorted(lock['records'], key=lambda r: r['index']) if r['index'] in set(proto['density_audit']['indices'])]
    sig = {c: recs[c]['candidate']['sigma'] for c in recs}
    search_correct = {c: recs[c]['correct_count'] for c in recs}
    C = [c for c in top50 if sig[c] == 0.002]
    Dset = [c for c in audit if sig[c] == 0.002]
    sets = {'A_winner': [WINNER], 'B_search_top50': top50, 'C_top50_sigma0.002': C, 'D_audit_sigma0.002': Dset, 'E_audit_all500': audit}
    (OUT / 'candidate_sets.json').write_text(json.dumps({**sets, 'sigma': {c: sig[c] for s in sets.values() for c in s},
                                                        'overlap_top50_audit': sorted(set(top50) & set(audit)),
                                                        'overlap_C_D': sorted(set(C) & set(Dset))}, indent=1))
    d = pd.read_parquet('results/paper-analysis/paper_master_predictions.parquet',
                        columns=['candidate_id', 'phase', 'example_id', 'subtask', 'true_option', 'candidate_correct', 'base_correct'])
    d = d[d.phase.isin(['RERANK', 'TEST'])]
    d['candidate_id'] = d.candidate_id.astype(str)
    base = {ph: d[(d.candidate_id == 'BASE') & (d.phase == ph)] for ph in ('RERANK', 'TEST')}
    n = {ph: len(base[ph]) for ph in base}
    checks = {'rerank_n': n['RERANK'], 'test_n': n['TEST'], 'base_rerank': int(base['RERANK'].base_correct.sum()), 'base_test': int(base['TEST'].base_correct.sum()),
              'top50_rerank_complete': int(d[(d.phase == 'RERANK') & d.candidate_id.isin(top50)].groupby('candidate_id').size().eq(200).sum()),
              'top50_test_complete': int(d[(d.phase == 'TEST') & d.candidate_id.isin(top50)].groupby('candidate_id').size().eq(561).sum()),
              'audit_rerank_complete': int(d[(d.phase == 'RERANK') & d.candidate_id.isin(audit)].groupby('candidate_id').size().eq(200).sum()),
              'audit_with_test': int(d[(d.phase == 'TEST') & d.candidate_id.isin(audit)].candidate_id.nunique()),
              'C_size': len(C), 'D_size': len(Dset)}
    print(json.dumps(checks), flush=True)
    if checks['top50_rerank_complete'] != 50 or checks['top50_test_complete'] != 50 or checks['audit_rerank_complete'] != 500 or checks['base_rerank'] != 79 or checks['base_test'] != 259:
        raise SystemExit('STOP: coverage check failed')

    # per-candidate gains, repairs, regressions
    stats = {}
    for ph in ('RERANK', 'TEST'):
        x = d[(d.phase == ph) & (d.candidate_id != 'BASE')]
        g = x.groupby('candidate_id')
        stats[ph] = pd.DataFrame({'correct': g.candidate_correct.sum(), 'repairs': g.apply(lambda q: int((q.candidate_correct & ~q.base_correct).sum())),
                                  'regressions': g.apply(lambda q: int((~q.candidate_correct & q.base_correct).sum()))})
        stats[ph]['gain_pp'] = 100 * (stats[ph].correct - base[ph].base_correct.sum()) / n[ph]
    A = {'checks': checks, 'sets': {}}
    for name, ids in sets.items():
        A['sets'][name] = {}
        for ph in ('RERANK', 'TEST'):
            have = [c for c in ids if c in stats[ph].index]
            if len(have) < len(ids):
                A['sets'][name][ph] = {'status': 'not measured' if not have else f'partially measured ({len(have)}/{len(ids)}; not reported as the set)', 'measured_ids': len(have)}
                continue
            s = stats[ph].loc[have]
            A['sets'][name][ph] = {**describe(s.gain_pp), 'repairs_total': int(s.repairs.sum()), 'regressions_total': int(s.regressions.sum()),
                                   'sigma_mix': {str(k): int(v) for k, v in pd.Series([sig[c] for c in have]).value_counts().items()}}
            if name != 'A_winner':
                A['sets'][name][ph]['winner_position'] = position(stats[ph].loc[WINNER, 'gain_pp'], s.gain_pp, member=WINNER in have)
    A['winner'] = {ph: {k: float(stats[ph].loc[WINNER, k]) for k in ('gain_pp', 'repairs', 'regressions')} for ph in stats}

    # Analysis B accounting
    B = {}
    for name, ids in (('B_search_top50', top50), ('C_top50_sigma0.002', C)):
        for incl in (True, False):
            ref = ids if incl else [c for c in ids if c != WINNER]
            r = accounting(stats['RERANK'].loc[WINNER, 'gain_pp'], stats['TEST'].loc[WINNER, 'gain_pp'],
                           stats['RERANK'].loc[ref, 'gain_pp'].mean(), stats['TEST'].loc[ref, 'gain_pp'].mean())
            r.update({'reference_n': len(ref), 'winner_in_mean': incl, 'mu_R': float(stats['RERANK'].loc[ref, 'gain_pp'].mean()),
                      'mu_T': float(stats['TEST'].loc[ref, 'gain_pp'].mean()),
                      'identity_error': abs(r['gap'] - r['group_mean_change'] - r['winner_advantage_change'])})
            B[f"{name}|{'winner_included' if incl else 'leave_winner_out'}"] = {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in r.items()}
        # per-candidate paired phase difference within the selected set (descriptive)
        diff = stats['TEST'].loc[ids, 'gain_pp'] - stats['RERANK'].loc[ids, 'gain_pp']
        B[f'{name}|paired_TEST_minus_RERANK'] = {**describe(diff), 'winner': float(diff.loc[WINNER]), 'winner_position_most_negative_1': int(1 + (diff < diff.loc[WINNER]).sum()),
                                                  'corr_RERANK_TEST': float(np.corrcoef(stats['RERANK'].loc[ids, 'gain_pp'], stats['TEST'].loc[ids, 'gain_pp'])[0, 1])}

    # Analysis C: RERANK-only reselection
    xr = d[(d.phase == 'RERANK')]
    qorder = base['RERANK'].example_id.astype(str).tolist()
    codes = pd.factorize(base['RERANK'].subtask.astype(str) + base['RERANK'].true_option.astype(str))[0]
    bvec = base['RERANK'].base_correct.to_numpy(bool)
    piv = xr[xr.candidate_id.isin(top50)].assign(eid=lambda q: q.example_id.astype(str)).pivot(index='candidate_id', columns='eid', values='candidate_correct')[qorder]
    M = piv.to_numpy(bool); ids = list(piv.index)
    rng = np.random.default_rng(SEED)
    splits = [stratified_halves(codes, rng) for _ in range(TRIALS)]
    np.savez_compressed(OUT / 'rerank_split_manifest.npz', selection=np.array([s[0] for s in splits], np.int16), heldout=np.array([s[1] for s in splits], np.int16),
                        question_ids=np.array(qorder))
    cellcounts = pd.Series(codes).value_counts().sort_index()
    out = {}
    for setname, cand in (('B_search_top50', top50), ('C_top50_sigma0.002', C)):
        ix = [ids.index(c) for c in cand]
        rows = []
        for t, (S, H) in enumerate(splits):
            selc = {c: int(M[i, S].sum()) for c, i in zip(cand, ix)}
            w = select_winner(cand, selc, search_correct)
            wi = ids.index(w)
            sel_gain = 100 * (M[wi, S].sum() - bvec[S].sum()) / len(S)
            hold = 100 * (M[wi, H].sum() - bvec[H].sum()) / len(H)
            mean_hold = 100 * (M[ix][:, H].sum(1).mean() - bvec[H].sum()) / len(H)
            mean_sel = 100 * (M[ix][:, S].sum(1).mean() - bvec[S].sum()) / len(S)
            rows.append({'trial': t, 'winner': w, 'winner_sigma': sig[w], 'is_9504111': w == WINNER, 'selection_gain_pp': sel_gain, 'heldout_gain_pp': hold,
                         'uniform_member_heldout_pp': mean_hold, 'uniform_member_selection_pp': mean_sel, 'increment_over_uniform_pp': hold - mean_hold,
                         'selection_minus_heldout_pp': sel_gain - hold})
        tr = pd.DataFrame(rows); tr.to_csv(OUT / f'rerank_reselection_trials_{setname}.csv', index=False)
        q = lambda s: {'mean': float(s.mean()), 'median': float(s.median()), 'q025': float(s.quantile(.025)), 'q975': float(s.quantile(.975))}
        wf = tr.winner.value_counts()
        out[setname] = {'n_candidates': len(cand), 'trials': TRIALS, 'selection_gain': q(tr.selection_gain_pp), 'heldout_gain': q(tr.heldout_gain_pp),
                        'uniform_member_heldout': q(tr.uniform_member_heldout_pp), 'increment_over_uniform': q(tr.increment_over_uniform_pp),
                        'selection_minus_heldout': q(tr.selection_minus_heldout_pp), 'share_increment_positive': float((tr.increment_over_uniform_pp > 0).mean()),
                        'share_heldout_positive': float((tr.heldout_gain_pp > 0).mean()), 'share_winner_is_9504111': float(tr.is_9504111.mean()),
                        'distinct_winners': int(wf.size), 'top_winners': [{'candidate_id': c, 'sigma': sig[c], 'wins': int(k)} for c, k in wf.iloc[:5].items()]}
    Cres = {'cell_counts_subtask_x_letter': {str(k): int(v) for k, v in cellcounts.items()}, 'results': out}

    # Analysis D proposal: outcome-independent control sample
    ctrl = sorted(Dset, key=lambda c: hashlib.sha256(('selection-vs-shared-control-v1:' + c).encode()).hexdigest())[:N_CONTROL]
    Dp = {'rule': "12 smallest SHA256('selection-vs-shared-control-v1:'+candidate_id) among the 125 sigma=0.002 audit candidates; no outcomes read",
          'candidates': [{'candidate_id': c, 'seed': recs[c]['candidate']['seed'], 'sigma': sig[c], 'manifest_index': recs[c]['index'], 'in_search_top50': c in set(top50),
                          'test_outputs_exist': c in stats['TEST'].index} for c in ctrl]}
    R = {'post_hoc': True, 'analysis_A': A, 'analysis_B': B, 'analysis_C': Cres, 'control_proposal': Dp}
    (OUT / 'selection_vs_shared_results.json').write_text(json.dumps(R, indent=1, default=lambda x: x.item() if hasattr(x, 'item') else str(x)))
    return R


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sys.path.insert(0, 'paper/figures/scripts')
    from common import ACCENT, BASE_C, CAND, GOOD, MUTED  # noqa: F401
    FIGD.mkdir(parents=True, exist_ok=True)
    R = json.loads((OUT / 'selection_vs_shared_results.json').read_text())
    sets = json.loads((OUT / 'candidate_sets.json').read_text())
    d = pd.read_parquet('results/paper-analysis/paper_master_predictions.parquet', columns=['candidate_id', 'phase', 'candidate_correct', 'base_correct'])
    d = d[d.phase.isin(['RERANK', 'TEST'])]; d['candidate_id'] = d.candidate_id.astype(str)
    gain = {}
    for ph, n, b in (('RERANK', 200, 79), ('TEST', 561, 259)):
        x = d[(d.phase == ph) & (d.candidate_id != 'BASE')].groupby('candidate_id').candidate_correct.sum()
        gain[ph] = 100 * (x - b) / n

    def save(fig, nm):
        fig.savefig(FIGD / f'{nm}.pdf', metadata={'CreationDate': None}); fig.savefig(FIGD / f'{nm}.png', dpi=300); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={'width_ratios': [1.1, 1]})
    ax = axes[0]
    top, C = sets['B_search_top50'], sets['C_top50_sigma0.002']
    sig = sets['sigma']
    for c in top:
        col = CAND if sig[c] == 0.002 else '#9cc2ee'
        ax.plot([0, 1], [gain['RERANK'][c], gain['TEST'][c]], color=col, lw=.7, alpha=.6)
    ax.plot([0, 1], [gain['RERANK'][WINNER], gain['TEST'][WINNER]], color=ACCENT, lw=2.4, marker='o', label='9504111')
    ax.plot([0, 1], [gain['RERANK'][top].mean(), gain['TEST'][top].mean()], color=BASE_C, lw=2, ls='--', marker='s', label='mean of SEARCH top 50')
    ax.plot([], [], color=CAND, lw=1, label='top-50 member, σ = 0.002'); ax.plot([], [], color='#9cc2ee', lw=1, label='top-50 member, σ < 0.002')
    ax.axhline(0, color=MUTED, lw=.8); ax.set_xticks([0, 1], ['RERANK200', 'TEST561']); ax.set_xlim(-.25, 1.25)
    ax.set_ylabel('Gain over base (pp)'); ax.set_title('A  Same 50 SEARCH-selected candidates', loc='left', fontsize=7.4)
    ax.legend(frameon=False, fontsize=5.9, loc='lower left')
    ax = axes[1]
    D = sets['D_audit_sigma0.002']
    ax.hist(gain['RERANK'][D], bins=np.arange(-6.25, 9, .5), color='#9cc2ee', edgecolor='white', lw=.3, label='125 random audit, σ = 0.002 (RERANK)')
    ax.hist(gain['RERANK'][C], bins=np.arange(-6.25, 9, .5), color=CAND, alpha=.7, edgecolor='white', lw=.3, label='27 top-50 members, σ = 0.002 (RERANK)')
    ax.axvline(gain['RERANK'][WINNER], color=ACCENT, lw=2, label='9504111 RERANK (+8.0)')
    ax.axvline(gain['TEST'][WINNER], color=ACCENT, lw=1.5, ls=':', label='9504111 TEST (−2.67)')
    ax.set_xlabel('Gain over base (pp)'); ax.set_yticks([]); ax.set_title('B  RERANK distributions (TEST not measured for audit)', loc='left', fontsize=7.4)
    ax.legend(frameon=False, fontsize=5.6, loc='upper left')
    fig.tight_layout(); save(fig, 'fig_winner_vs_group')

    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    res = R['analysis_C']['results']
    labs = [('selection_gain', 'Winner, selection half', ACCENT), ('heldout_gain', 'Same winner, held-out half', CAND), ('uniform_member_heldout', 'Uniform member, held-out half', MUTED)]
    for j, (setname, tag) in enumerate((('B_search_top50', 'SEARCH top 50'), ('C_top50_sigma0.002', 'σ = 0.002 members (27)'))):
        for i, (k, lab, col) in enumerate(labs):
            v = res[setname][k]; xpos = j * 4 + i
            ax.bar(xpos, v['mean'], .8, color=col, zorder=3, label=lab if j == 0 else None)
            ax.errorbar(xpos, v['mean'], yerr=[[v['mean'] - v['q025']], [v['q975'] - v['mean']]], color=BASE_C, capsize=2.5, lw=.9)
    ax.set_xticks([1, 5], ['SEARCH top 50', 'σ = 0.002 members (27)']); ax.axhline(0, color=MUTED, lw=.8)
    ax.set_ylabel('Gain over base (pp)'); ax.set_title('RERANK-only reselection: 100 / 100 splits (2,000 partitions)', loc='left', fontsize=7.2)
    ax.set_ylim(-3, 18); ax.legend(frameon=False, fontsize=6, loc='upper center', ncol=3)
    fig.tight_layout(); save(fig, 'fig_rerank_reselection')


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'run'
    if what == 'run':
        R = run()
        print(json.dumps({k: R[k] for k in ('analysis_B',)}, indent=1))
    else:
        figures()

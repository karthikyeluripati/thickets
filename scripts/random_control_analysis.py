"""CPU analysis of the random-control transfer run (implements analysis_plan.md, committed before outputs were read)."""
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from margin_additivity import margin_terms, margin_bin  # noqa: E402
from selection_vs_shared import accounting  # noqa: E402

OUT = Path('results/paper-analysis/random-control-transfer')
GPU = OUT / 'gpu'
CD = Path('results/paper-analysis/causal-diagnostic')
LET = 'ABCD'
WINNER_SEED = 9504111
BOOT, BOOT_SEED = 10000, 20261012


def vec(x):
    return np.array([x['letter_logprobs'][a] for a in LET], float)


def load_rows():
    gold, phase = {}, {}
    for s, ph in (('validation', 'RERANK'), ('test', 'TEST')):
        for l in Path(f'examples/omnispatial-perspective-taking/{s}.jsonl').read_text(encoding='utf-8').splitlines():
            r = json.loads(l); gold[r['uid']] = LET[r['answer']]; phase[r['uid']] = ph
    return gold, phase


def candidate_table(outs, base, gold):
    """outs/base: {phase: [records]} -> per-phase correct, gain, repairs, regressions."""
    res = {}
    for ph, n, b0 in (('RERANK', 200, 79), ('TEST', 561, 259)):
        bmap = {x['uid']: x['parsed'] == gold[x['uid']] for x in base[ph]}
        cc = {x['uid']: x['parsed'] == gold[x['uid']] for x in outs[ph]}
        res[ph] = {'correct': int(sum(cc.values())), 'gain_pp': 100 * (sum(cc.values()) - b0) / n,
                   'repairs': int(sum(cc[u] and not bmap[u] for u in cc)), 'regressions': int(sum(bmap[u] and not cc[u] for u in cc)),
                   'tied_top_generated': int(sum(sum(vec(x) == vec(x).max()) > 1 for x in outs[ph]))}
    return res


def run():
    gold, phase = load_rows()
    ctrl = json.loads((OUT / 'frozen_controls.json').read_text())['candidates']
    base = json.loads(gzip.decompress((GPU / 'base.json.gz').read_bytes()))
    cand9 = {x['uid']: x for x in json.loads(gzip.decompress((CD / 'session1/candidate_full.json.gz').read_bytes()))}
    target = {ph: [cand9[x['uid']] for x in base[ph]] for ph in base}
    recs = {}
    for c in ctrl:
        f = GPU / f"control_{c['seed']}.json.gz"
        if f.exists():
            recs[c['seed']] = json.loads(gzip.decompress(f.read_bytes()))
    rows = []
    fid = {}
    for c in ctrl:
        if c['seed'] not in recs:
            rows.append({'seed': c['seed'], 'in_search_top50': c['in_search_top50'], 'status': 'not run (budget/stop)'}); continue
        r = recs[c['seed']]
        t = candidate_table(r, base, gold)
        fid[c['seed']] = {k: r[k] for k in ('fingerprint_ok', 'RERANK_text_identical_to_stored', 'TEST_text_identical_to_stored', 'TEST_stored_available',
                                            'RERANK_missing_scores', 'TEST_missing_scores', 'base_restored_mismatches')}
        rows.append({'seed': c['seed'], 'in_search_top50': c['in_search_top50'], 'status': 'measured',
                     'R_correct': t['RERANK']['correct'], 'T_correct': t['TEST']['correct'], 'g_R': t['RERANK']['gain_pp'], 'g_T': t['TEST']['gain_pp'],
                     'gap': t['TEST']['gain_pp'] - t['RERANK']['gain_pp'], 'R_repairs': t['RERANK']['repairs'], 'R_regressions': t['RERANK']['regressions'],
                     'T_repairs': t['TEST']['repairs'], 'T_regressions': t['TEST']['regressions'],
                     'tied_top_R': t['RERANK']['tied_top_generated'], 'tied_top_T': t['TEST']['tied_top_generated']})
    tw = candidate_table(target, base, gold)
    win = {'seed': WINNER_SEED, 'g_R': tw['RERANK']['gain_pp'], 'g_T': tw['TEST']['gain_pp'], 'gap': tw['TEST']['gain_pp'] - tw['RERANK']['gain_pp'],
           'R_correct': tw['RERANK']['correct'], 'T_correct': tw['TEST']['correct'], 'R_repairs': tw['RERANK']['repairs'], 'R_regressions': tw['RERANK']['regressions'],
           'T_repairs': tw['TEST']['repairs'], 'T_regressions': tw['TEST']['regressions']}
    df = pd.DataFrame(rows); df.to_csv(OUT / 'per_control_metrics.csv', index=False)
    m = df[df.status == 'measured']
    rng = np.random.default_rng(BOOT_SEED)
    gaps = m.gap.to_numpy()
    boot = [rng.choice(gaps, len(gaps)).mean() for _ in range(BOOT)]
    summ = {'n_measured': int(len(m)), 'mean_gap': float(gaps.mean()), 'median_gap': float(np.median(gaps)), 'min_gap': float(gaps.min()), 'max_gap': float(gaps.max()),
            'sd_gap': float(gaps.std(ddof=1)), 'n_RERANK_gain_gt_TEST_gain': int((m.g_R > m.g_T).sum()),
            'n_above_base_RERANK': int((m.g_R > 0).sum()), 'n_above_base_TEST': int((m.g_T > 0).sum()),
            'mean_g_R': float(m.g_R.mean()), 'mean_g_T': float(m.g_T.mean()),
            'bootstrap_mean_gap_95': [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
            'bootstrap_note': 'resampling unit = control candidates (12), conditional on the fixed RERANK/TEST questions; post hoc',
            'winner_gap_below_all_controls': bool(win['gap'] < gaps.min()), 'winner_gap_rank_most_negative_1': int(1 + (gaps < win['gap']).sum()),
            'winner_g_R_above_all_controls': bool(win['g_R'] > m.g_R.max()), 'winner_g_T_below_all_controls': bool(win['g_T'] < m.g_T.min())}
    acct = accounting(win['g_R'], win['g_T'], m.g_R.mean(), m.g_T.mean())
    acct = {'gap': acct['gap'], 'random_control_mean_change': acct['group_mean_change'], 'winner_advantage_change': acct['winner_advantage_change'],
            'mu_R': float(m.g_R.mean()), 'mu_T': float(m.g_T.mean())}

    # ---- score comparison
    M = json.loads((Path('results/paper-analysis/margin-additivity/margin_additivity_results.json')).read_text())['standardization']
    w = {h: (v['w_R'] + v['w_T']) / 2 for h, v in M['strata'].items()}
    bvec = {x['uid']: vec(x) for ph in base for x in base[ph]}
    bcorr = {x['uid']: x['parsed'] == gold[x['uid']] for ph in base for x in base[ph]}

    def tstats(outs):
        res = {}
        for ph in ('RERANK', 'TEST'):
            rr = []
            for x in outs[ph]:
                u = x['uid']; y = LET.index(gold[u]); mt = margin_terms(bvec[u], vec(x), y)
                rr.append({'t': mt['t'], 'bc': bcorr[u], 'stratum': f"{'Bcorrect' if bcorr[u] else 'Bwrong'}|{margin_bin(mt['m_B'])}"})
            q = pd.DataFrame(rr)
            std = sum(w[h] * q[q.stratum == h].t.mean() for h in w if (q.stratum == h).any()) / sum(w[h] for h in w if (q.stratum == h).any())
            res[ph] = {'mean_t': float(q.t.mean()), 'mean_t_Bcorrect': float(q[q.bc].t.mean()), 'mean_t_Bwrong': float(q[~q.bc].t.mean()),
                       'std_mean_t': float(std), 'p_t_pos': float((q.t > 0).mean())}
        return res
    T = {'9504111': tstats(target)}
    for s, r in recs.items():
        T[str(s)] = tstats(r)

    def dz(outs, ph):
        return np.array([(vec(x) - vec(x).mean()) - (bvec[x['uid']] - bvec[x['uid']].mean()) for x in outs[ph]])
    cmp = {}
    for ph in ('RERANK', 'TEST'):
        D = {s: dz(r, ph) for s, r in recs.items()}
        ref = np.mean(list(D.values()), 0)
        tgt = dz(target, ph)

        def metrics(a, ref_):
            cos = float((a * ref_).sum() / (np.linalg.norm(a) * np.linalg.norm(ref_)))
            pq = [float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y))) for x, y in zip(a, ref_) if np.linalg.norm(x) > 0 and np.linalg.norm(y) > 0]
            return {'pooled_cosine': cos, 'median_per_question_cosine': float(np.median(pq)), 'residual_ratio': float(np.linalg.norm(a - ref_) / np.linalg.norm(a)),
                    'shift_norm_per_question': float(np.linalg.norm(a, axis=1).mean())}
        loo = {}
        for s in D:
            others = np.mean([D[k] for k in D if k != s], 0)
            loo[str(s)] = metrics(D[s], others)
        cmp[ph] = {'target_vs_control_mean': metrics(tgt, ref), 'controls_vs_leave_one_out_mean': loo,
                   'controls_loo_median': {k: float(np.median([v[k] for v in loo.values()])) for k in ('pooled_cosine', 'residual_ratio', 'median_per_question_cosine')},
                   'reference_mean_shift_norm_per_question': float(np.linalg.norm(ref, axis=1).mean())}
    R = {'post_hoc': True, 'fidelity': fid, 'winner': win, 'summary': summ, 'accounting': acct, 'score_t': T, 'score_vector_comparison': cmp}
    (OUT / 'random_control_results.json').write_text(json.dumps(R, indent=1))
    return R


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sys.path.insert(0, 'paper/figures/scripts')
    from common import ACCENT, BASE_C, CAND, MUTED  # noqa: F401
    figd = Path('paper/figures/random-control-transfer'); figd.mkdir(parents=True, exist_ok=True)
    R = json.loads((OUT / 'random_control_results.json').read_text())
    df = pd.read_csv(OUT / 'per_control_metrics.csv'); m = df[df.status == 'measured']

    def save(fig, nm):
        fig.savefig(figd / f'{nm}.pdf', metadata={'CreationDate': None}); fig.savefig(figd / f'{nm}.png', dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.6, 3.1))
    for _, r in m.iterrows():
        ax.plot([0, 1], [r.g_R, r.g_T], color=CAND if not r.in_search_top50 else '#5a9be0', lw=1, marker='o', ms=3, alpha=.8)
    w = R['winner']
    ax.plot([0, 1], [w['g_R'], w['g_T']], color=ACCENT, lw=2.4, marker='o', ms=5, label='9504111 (selected winner)')
    ax.plot([0, 1], [m.g_R.mean(), m.g_T.mean()], color=BASE_C, lw=2, ls='--', marker='s', label=f'mean of {len(m)} random controls')
    ax.plot([], [], color=CAND, lw=1, label='random σ = 0.002 control')
    ax.axhline(0, color=MUTED, lw=.8); ax.set_xticks([0, 1], ['RERANK200', 'TEST561']); ax.set_xlim(-.25, 1.25)
    ax.set_ylabel('Gain over base (pp)'); ax.legend(frameon=False, fontsize=6, loc='upper right')
    fig.tight_layout(); save(fig, 'fig_paired_gains')

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8), sharey=True)
    for ax, key, title in ((axes[0], 'mean_t_Bwrong', 'A  BASE-wrong questions'), (axes[1], 'mean_t_Bcorrect', 'B  BASE-correct questions')):
        for s, v in R['score_t'].items():
            if s == '9504111': continue
            ax.plot([0, 1], [v['RERANK'][key], v['TEST'][key]], color=CAND, lw=1, marker='o', ms=3, alpha=.75)
        v = R['score_t']['9504111']
        ax.plot([0, 1], [v['RERANK'][key], v['TEST'][key]], color=ACCENT, lw=2.4, marker='o', ms=5, label='9504111')
        ax.axhline(0, color=MUTED, lw=.8); ax.set_xticks([0, 1], ['RERANK', 'TEST']); ax.set_xlim(-.25, 1.25)
        ax.set_title(title, loc='left', fontsize=7.4)
    axes[0].set_ylabel('Mean shift toward correct answer, t (nats)'); axes[0].legend(frameon=False, fontsize=6)
    fig.tight_layout(); save(fig, 'fig_paired_score_shift')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'figures':
        figures()
    else:
        R = run(); print(json.dumps({k: R[k] for k in ('summary', 'accounting', 'winner')}, indent=1))

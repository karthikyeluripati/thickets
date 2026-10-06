"""Selection vs candidate-specificity for 9504111 (POST HOC, CPU only). Implements
results/paper-analysis/selection-vs-specificity/analysis_plan.md (committed before analysis)."""
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from selection_vs_shared import select_winner  # noqa: E402

OUT = Path('results/paper-analysis/selection-vs-specificity')
SVS = Path('results/paper-analysis/selection-vs-shared-response')
RC = Path('results/paper-analysis/random-control-transfer')
ROOT = Path('results/perspective-taking-n5000-20261004')
WIN = '25a60f0b59f103ebf782a31f94bb569d897a652a0bc1bd95c5cdc5521418b843'
HIGH_R = 4.0
BOOT, SEED = 10000, 20261013


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = json.loads((ROOT / 'locks/search.json').read_text())
    recs = {r['candidate']['candidate_id']: r for r in lock['records']}
    top50 = list(lock['ranked_ids'][:50]); search_correct = {c: recs[c]['correct_count'] for c in recs}
    proto = json.loads(Path('experiments/perspective_taking_n5000_protocol.json').read_text())
    audit = [r['candidate']['candidate_id'] for r in sorted(lock['records'], key=lambda r: r['index']) if r['index'] in set(proto['density_audit']['indices'])]
    d = pd.read_parquet('results/paper-analysis/paper_master_predictions.parquet', columns=['candidate_id', 'phase', 'example_id', 'candidate_correct', 'base_correct', 'true_option'])
    d = d[d.phase.isin(['RERANK', 'TEST'])]; d['candidate_id'] = d.candidate_id.astype(str); d['example_id'] = d.example_id.astype(str)
    qR = d[(d.phase == 'RERANK') & (d.candidate_id == 'BASE')].example_id.tolist()
    qT = d[(d.phase == 'TEST') & (d.candidate_id == 'BASE')].example_id.tolist()
    bR = d[(d.phase == 'RERANK') & (d.candidate_id == 'BASE')].base_correct.to_numpy(bool)
    bT = d[(d.phase == 'TEST') & (d.candidate_id == 'BASE')].base_correct.to_numpy(bool)
    goldT = dict(zip(qT, d[(d.phase == 'TEST') & (d.candidate_id == 'BASE')].true_option.astype(str)))

    def mat(ph, ids, q):
        x = d[(d.phase == ph) & d.candidate_id.isin(ids)].pivot(index='candidate_id', columns='example_id', values='candidate_correct')
        return x.loc[ids, q].to_numpy(bool)
    # TEST matrix: top 50 from master; 11 non-overlapping controls from the random-control GPU run
    ctrl = json.loads((RC / 'frozen_controls.json').read_text())['candidates']
    ctrl_ids = [c['candidate_id'] for c in ctrl]
    T50 = mat('TEST', top50, qT)
    extra = [c for c in ctrl if c['candidate_id'] not in set(top50)]
    Tc = []
    for c in extra:
        r = json.loads(gzip.decompress((RC / f"gpu/control_{c['seed']}.json.gz").read_bytes()))
        mp = {x['uid']: x['parsed'] == goldT[x['uid']] for x in r['TEST']}
        Tc.append([mp[u] for u in qT])
    test_ids = top50 + [c['candidate_id'] for c in extra]
    TM = np.vstack([T50, np.array(Tc, bool)])
    gT = 100 * (TM.sum(1) - bT.sum()) / len(qT)
    rer_ids = list(dict.fromkeys(top50 + audit))
    RM = mat('RERANK', rer_ids, qR)
    gR_all = dict(zip(rer_ids, 100 * (RM.sum(1) - bR.sum()) / len(qR)))
    tbl = pd.DataFrame({'candidate_id': test_ids, 'g_R': [gR_all[c] for c in test_ids], 'g_T': gT,
                        'group': ['top50' if c in set(top50) else 'random_control' for c in test_ids],
                        'is_random_control': [c in set(ctrl_ids) for c in test_ids], 'sigma': [recs[c]['candidate']['sigma'] for c in test_ids]})
    tbl['gap'] = tbl.g_T - tbl.g_R
    tbl.to_csv(OUT / 'test_measured_candidates.csv', index=False)
    w = tbl[tbl.candidate_id == WIN].iloc[0]
    R = {'post_hoc': True, 'n_test_measured': len(tbl)}

    # refined accounting relative to the 12 controls
    c12 = tbl[tbl.is_random_control]
    R['accounting_refined'] = {'gap_winner': float(w.gap), 'control_mean_gap': float(c12.gap.mean()),
                               'winner_RERANK_excess_over_controls': float(w.g_R - c12.g_R.mean()),
                               'winner_TEST_deficit_vs_controls': float(w.g_T - c12.g_T.mean()),
                               'identity_check': float(w.gap - c12.gap.mean() - ((w.g_T - c12.g_T.mean()) - (w.g_R - c12.g_R.mean())))}

    # A/B cross-fitting with the frozen partitions
    man = np.load(SVS / 'rerank_split_manifest.npz')
    assert list(man['question_ids']) == qR
    S, H = man['selection'], man['heldout']
    def crossfit(pool, with_test):
        ix = [rer_ids.index(c) for c in pool]; rows = []
        for s, h in zip(S, H):
            selc = {c: int(RM[i, s].sum()) for c, i in zip(pool, ix)}
            wn = select_winner(pool, selc, search_correct); wi = rer_ids.index(wn)
            row = {'winner': wn, 'is_9504111': wn == WIN, 'sel_gain': 100 * (RM[wi, s].sum() - bR[s].sum()) / len(s),
                   'held_gain': 100 * (RM[wi, h].sum() - bR[h].sum()) / len(h),
                   'uniform_held': 100 * (RM[ix][:, h].sum(1).mean() - bR[h].sum()) / len(h)}
            if with_test:
                row['test_gain'] = float(tbl.set_index('candidate_id').loc[wn, 'g_T'])
            rows.append(row)
        return pd.DataFrame(rows)
    cf = crossfit(top50, True); cf.to_csv(OUT / 'crossfit_top50_trials.csv', index=False)
    q = lambda s: {'mean': float(s.mean()), 'q025': float(s.quantile(.025)), 'q975': float(s.quantile(.975))}
    ws = cf[cf.is_9504111]
    top_T = tbl[tbl.group == 'top50']
    R['crossfit_top50'] = {
        'selection_rule': 'selection-half correct desc, SEARCH200 correct desc, SHA256(visual-rank-v1:id) asc; among the fixed SEARCH top 50',
        'partitions': int(len(cf)), 'effective_information': '200 RERANK questions; partitions are not independent experiments',
        'p_9504111_selected': float(cf.is_9504111.mean()), 'distinct_winners': int(cf.winner.nunique()),
        'winner9504111_full_RERANK_gain': float(w.g_R), 'winner9504111_held_gain_when_selected': q(ws.held_gain),
        'winner9504111_selection_gain_when_selected': q(ws.sel_gain),
        'winner9504111_RERANK_optimism': float(w.g_R - ws.held_gain.mean()),
        'selected_held_gain_all_trials': q(cf.held_gain), 'uniform_member_held_gain': q(cf.uniform_held),
        'selected_TEST_gain_all_trials': q(cf.test_gain), 'selected_TEST_gain_when_not_9504111': q(cf[~cf.is_9504111].test_gain) if (~cf.is_9504111).any() else None,
        'uniform_top50_TEST_mean': float(top_T.g_T.mean()), 'random_control_TEST_mean': float(c12.g_T.mean()),
        'selected_TEST_minus_heldout_RERANK': q(cf.test_gain - cf.held_gain),
        'uniform_TEST_minus_RERANK_top50': float((top_T.g_T - top_T.g_R).mean())}
    cfa = crossfit(audit, False)
    R['crossfit_audit500_RERANK_only'] = {'selected_held_gain': q(cfa.held_gain), 'uniform_held_gain': q(cfa.uniform_held),
                                         'selection_minus_held': q(cfa.sel_gain - cfa.held_gain), 'distinct_winners': int(cfa.winner.nunique()),
                                         'top_winners': [{'seed': recs[c]['candidate']['seed'], 'wins': int(k), 'full_RERANK_gain': float(gR_all[c]),
                                                          'TEST_measured': c in set(test_ids)} for c, k in cfa.winner.value_counts().iloc[:5].items()]}

    # C. conditional on RERANK strength
    others = tbl[tbl.candidate_id != WIN]
    hi = others[others.g_R >= HIGH_R].sort_values('g_R', ascending=False)
    b, a = np.polyfit(others.g_R, others.g_T, 1)
    resid_sd = float(np.std(others.g_T - (a + b * others.g_R), ddof=2))
    R['conditional'] = {'high_RERANK_threshold_pp': HIGH_R,
                        'high_RERANK_group': [{'seed': recs[c]['candidate']['seed'], 'group': g, 'g_R': float(r_), 'g_T': float(t_)}
                                              for c, g, r_, t_ in zip(hi.candidate_id, hi.group, hi.g_R, hi.g_T)],
                        'winner_TEST_rank_among_all_61_worst_1': int(1 + (tbl.g_T < w.g_T).sum()),
                        'winner_TEST_below_all_high_RERANK': bool((w.g_T < hi.g_T).all()) if len(hi) else None,
                        'ols_gT_on_gR_others': {'slope': float(b), 'intercept': float(a), 'residual_sd': resid_sd, 'r': float(np.corrcoef(others.g_R, others.g_T)[0, 1]),
                                                'max_other_g_R': float(others.g_R.max()), 'prediction_at_winner_g_R': float(a + b * w.g_R),
                                                'winner_residual': float(w.g_T - (a + b * w.g_R)), 'winner_residual_in_resid_sd': float((w.g_T - (a + b * w.g_R)) / resid_sd),
                                                'note': 'descriptive; prediction at g_R=8 extrapolates beyond the other candidates'},
                        'tie_note': 'none'}
    # paired question bootstrap on TEST
    rng = np.random.default_rng(SEED)
    wi = test_ids.index(WIN); ci = [test_ids.index(c) for c in ctrl_ids]; hi_ix = [test_ids.index(c) for c in hi.candidate_id]
    D = TM.astype(float) - bT.astype(float)
    def boot(ref_ix):
        diff = D[wi] - D[ref_ix].mean(0)
        v = [100 * diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(BOOT)]
        return {'observed': float(100 * diff.mean()), 'q025': float(np.quantile(v, .025)), 'q975': float(np.quantile(v, .975)),
                'note': 'TEST questions resampled with replacement; candidates fixed'}
    R['test_question_bootstrap'] = {'winner_minus_12_controls': boot(ci), 'winner_minus_high_RERANK_group': boot(hi_ix) if hi_ix else None}
    # same on RERANK for the excess
    wiR = rer_ids.index(WIN); ciR = [rer_ids.index(c) for c in ctrl_ids]
    DR = RM.astype(float) - bR.astype(float); diffR = DR[wiR] - DR[ciR].mean(0)
    vR = [100 * diffR[rng.integers(0, 200, 200)].mean() for _ in range(BOOT)]
    R['rerank_question_bootstrap_winner_minus_controls'] = {'observed': float(100 * diffR.mean()), 'q025': float(np.quantile(vR, .025)), 'q975': float(np.quantile(vR, .975))}
    # GPU proposal set: audit candidates with RERANK >= +5, not TEST-measured
    prop = [c for c in audit if gR_all[c] >= 5.0]
    R['gpu_proposal_candidates'] = [{'candidate_id': c, 'seed': recs[c]['candidate']['seed'], 'sigma': recs[c]['candidate']['sigma'], 'g_R': float(gR_all[c]),
                                     'TEST_measured': c in set(test_ids), 'in_top50': c in set(top50), 'expected_state_id': recs[c]['candidate_state_id']} for c in prop]
    (OUT / 'selection_vs_specificity_results.json').write_text(json.dumps(R, indent=1))
    return R


if __name__ == '__main__':
    print(json.dumps(main(), indent=1)[:9000])

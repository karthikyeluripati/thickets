"""Post-hoc, CPU-only: does the nonlinear vision response explain 9504111's RERANK-positive / TEST-negative behavior?

Uses only existing artifacts (causal-diagnostic session 1 insertions/removals, GPU-A run-2 first-order predictions,
HF-internal candidate contrasts, master predictions). H* is treated as falsified; everything here is hypothesis-generating.
Score: the A1 contrast s_j = z[gold] - z[r_B] (r_B = BASE strongest wrong, from base_contrast_fp32.json); Delta = s(X) - s(BASE).
"""
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from causal_diag_9504111 import GROUPS  # noqa: E402

PA = Path('results/paper-analysis')
CD = PA / 'causal-diagnostic'
GA = PA / 'geometry-gpu-a'
GPU = GA / 'gpu2'
RC = PA / 'random-control-transfer'
OUT = PA / 'vision-nonlinearity'
LET = 'ABCD'
WIN = '25a60f0b59f103ebf782a31f94bb569d897a652a0bc1bd95c5cdc5521418b843'
WIN_SEED = 9504111
LM = ('lm_q1', 'lm_q2', 'lm_q3', 'lm_q4')
OTHER = ('embed', 'final_norm_head')
CELLS = ('repair', 'regression', 'both_wrong_diff', 'both_correct', 'both_wrong_same')
B = 5000  # bootstrap resamples
rng = np.random.default_rng(20261006)


def ld(f):
    return json.loads(gzip.decompress(Path(f).read_bytes()))


def vecs(f):
    return {q['uid']: np.array([q['letter_logprobs'][a] for a in LET], float) for q in ld(f)}


def fit_stats(pred, obs):
    pred, obs = np.asarray(pred, float), np.asarray(obs, float); res = obs - pred
    return {'n': int(len(obs)), 'pearson_r': float(np.corrcoef(pred, obs)[0, 1]),
            'spearman_r': float(pd.Series(pred).corr(pd.Series(obs), method='spearman')),
            'slope_obs_on_pred': float(np.polyfit(pred, obs, 1)[0]), 'residual_sd': float(res.std(ddof=1)),
            'residual_energy_ratio': float((res ** 2).sum() / (obs ** 2).sum()),
            'explained_energy': float(1 - (res ** 2).sum() / (obs ** 2).sum()),
            'mae': float(np.abs(res).mean()), 'rmse': float(np.sqrt((res ** 2).mean())), 'sd_pred': float(pred.std(ddof=1)), 'sd_obs': float(obs.std(ddof=1))}


def cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / s) if s > 0 else float('nan')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    R = {'post_hoc': True, 'H_star': 'falsified under A1 (F1 r=0.137, S2d r=0.239); not reinterpreted here',
         'score': 's = z[gold] - z[r_B] (A1 contrast; vLLM letter log-probs for measured hybrids); Delta = s(X) - s(BASE)'}
    man = json.loads((CD / 'example_manifest.json').read_text()); labels = man['labels']; LOC = man['localization']
    comp = json.loads((GPU / 'base_contrast_fp32.json').read_text())['comparator']
    zB, zC = vecs(CD / 'session1/base_full.json.gz'), vecs(CD / 'session1/candidate_full.json.gz')
    zI = {g: vecs(CD / f'session1/insertion_{g}_loc.json.gz') for g in GROUPS}
    zR = {g: vecs(CD / f'session1/removal_{g}_loc.json.gz') for g in GROUPS}
    gp = {**json.loads((GPU / 'group_pred_RERANK.json').read_text()), **json.loads((GPU / 'group_pred_TEST.json').read_text())}
    s = lambda z, u: float(z[comp[u][0]] - z[comp[u][1]])

    # ---------------- 1. reconstruction / verification
    df = pd.DataFrame({'uid': LOC})
    df['phase'] = [labels[u]['phase'] for u in LOC]; df['cell'] = [labels[u]['transition'] for u in LOC]
    df['base_correct'] = df.cell.isin(['regression', 'both_correct'])
    df['mB'] = [s(zB[u], u) for u in LOC]; df['d_total'] = [s(zC[u], u) - s(zB[u], u) for u in LOC]
    for g in GROUPS:
        df[f'ins_{g}'] = [s(zI[g][u], u) - s(zB[u], u) for u in LOC]
        df[f'rem_{g}'] = [s(zC[u], u) - s(zR[g][u], u) for u in LOC]  # effect of g in the candidate context
        df[f'lin_{g}'] = [gp[u][g] for u in LOC]
    df['d_vision'] = df.ins_vision; df['d_LM'] = df[[f'ins_{g}' for g in LM]].sum(1)
    df['d_embed'] = df.ins_embed; df['d_head'] = df.ins_final_norm_head; df['d_other'] = df.d_embed + df.d_head
    df['d_sum'] = df.d_vision + df.d_LM + df.d_other; df['interaction'] = df.d_total - df.d_sum
    df['lin_LM'] = df[[f'lin_{g}' for g in LM]].sum(1); df['lin_other'] = df.lin_embed + df.lin_final_norm_head
    df['lin_vision'] = df.lin_vision; df['nl_vision'] = df.d_vision - df.lin_vision
    df['nl_LM'] = df.d_LM - df.lin_LM
    df['is_rerank'] = (df.phase == 'RERANK').astype(float)
    df['changed'] = df.cell.isin(['repair', 'regression', 'both_wrong_diff'])
    # population weights: each LOC example stands for N_cell(population)/n_cell(LOC) examples of its phase x cell
    pop = pd.Series({(v['phase'], v['transition']): 0 for v in labels.values()})
    for v in labels.values(): pop[(v['phase'], v['transition'])] += 1
    nloc = df.groupby(['phase', 'cell']).size()
    df['w'] = [pop[(p, c)] / nloc[(p, c)] for p, c in zip(df.phase, df.cell)]
    # check: weighted LOC mean of d_total vs the true population mean over all 761
    allu = list(labels); pop_dt = {ph: float(np.mean([s(zC[u], u) - s(zB[u], u) for u in allu if labels[u]['phase'] == ph])) for ph in ('RERANK', 'TEST')}
    wm = lambda x, m: float(np.average(df.loc[m, x], weights=df.loc[m, 'w']))
    R['1_reconstruction'] = {
        'groups': list(GROUPS), 'n_localization': len(LOC), 'by_phase_cell_n_loc': {f'{p}|{c}': int(n) for (p, c), n in nloc.items()},
        'population_phase_cell_n': {f'{p}|{c}': int(n) for (p, c), n in pop.items()},
        'construction': 'INSERTION_g = BASE weights with only group g replaced by the candidate bytes (base + sigma*eps restricted to g); '
                        'REMOVAL_g = candidate weights with group g reset to BASE. vLLM 0.11, same engine/prompts as the candidate; '
                        'letter log-probs A-D of the first answer token.',
        'same_examples': 'yes: all 7 insertions, 7 removals, BASE and candidate are scored on the same 83 LOCALIZATION examples',
        'additivity': 'groups are disjoint, so insertions are exactly additive in PARAMETER space (sum of group deltas = full delta); '
                      'in SCORE space additivity is only approximate (tested below and in margin-additivity)',
        'linear_predictions': 'GPU-A run 2 group partials sigma*<fold_g(grad s), eps> (HF fp32 contrast gradient at BASE), same comparator',
        'engine_noise_floor_nats': 'HF fp32 vs vLLM base contrast |diff|: median 0.136, p90 0.362, p99 0.647 (V0c); vLLM scores quantized at 0.125',
        'check_S2d_per_group_r': {g: float(np.corrcoef(df[f'lin_{g}'], df[f'ins_{g}'])[0, 1]) for g in GROUPS},
        'check_weighted_mean_d_total_vs_population': {ph: {'weighted_loc': wm('d_total', df.phase == ph), 'population_761': pop_dt[ph]} for ph in pop_dt},
    }

    # ---------------- 2. decomposition accuracy (score space)
    def flips(pred_delta, m):
        sel = df[m]; act = (sel.mB + sel.d_total) > 0; prd = (sel.mB + pred_delta[m]) > 0; base = sel.mB > 0
        ch = act != base
        return {'contrast_sign_agreement': float((act == prd).mean()), 'changed_contrast_sign_n': int(ch.sum()),
                'changed_predicted': int(((prd != base) & ch).sum()), 'false_flips_on_unchanged': int(((prd != base) & ~ch).sum())}
    allm = np.ones(len(df), bool)
    preds = {'sum_all_groups': df.d_sum, 'vision_only': df.d_vision, 'LM_only': df.d_LM, 'vision_plus_LM': df.d_vision + df.d_LM,
             'LM_plus_other': df.d_LM + df.d_other,
             'linear_LM_other_plus_measured_vision': df.lin_LM + df.lin_other + df.d_vision,
             'all_linear_first_order': df.lin_LM + df.lin_other + df.lin_vision}
    R['2_decomposition'] = {k: {**fit_stats(v, df.d_total), **flips(v, allm)} for k, v in preds.items()}
    # full A-D vector check (argmax answers) for the sum, reproduces margin-additivity
    ans_ok = 0; chg = 0
    for u in LOC:
        za = zB[u] + sum(zI[g][u] - zB[u] for g in GROUPS)
        if labels[u]['cand'] != labels[u]['base']:
            chg += 1; ans_ok += LET[int(np.argmax(za))] == labels[u]['cand']
    R['2_decomposition']['sum_all_groups']['full_vector_candidate_answer_on_changed'] = f'{ans_ok}/{chg}'

    # ---------------- 3. RERANK vs TEST asymmetry by component
    comps = ['d_total', 'd_vision', 'd_LM', 'd_other', 'interaction', 'lin_vision', 'nl_vision', 'lin_LM', 'nl_LM']

    # stratified bootstrap (resample within phase x cell), vectorized: index draws shared across components
    strata = [np.flatnonzero(((df.phase == p) & (df.cell == c)).to_numpy()) for (p, c) in nloc.index]
    draws = np.concatenate([st[rng.integers(0, len(st), (B, len(st)))] for st in strata], axis=1)  # B x 83
    wv = df.w.to_numpy(); isR = df.is_rerank.to_numpy()[draws] > 0; ww = wv[draws]

    def boot_gap(col):
        x = df[col].to_numpy()[draws]
        r_ = (x * ww * isR).sum(1) / (ww * isR).sum(1); t_ = (x * ww * ~isR).sum(1) / (ww * ~isR).sum(1)
        return [float(np.quantile(r_ - t_, .025)), float(np.quantile(r_ - t_, .975))]
    A3 = {}
    for col in comps:
        e = {}
        for ph in ('RERANK', 'TEST'):
            m = df.phase == ph; x = df.loc[m, col]
            e[ph] = {'weighted_mean': wm(col, m), 'unweighted_mean': float(x.mean()), 'median': float(x.median()),
                     'standardized_weighted_mean': wm(col, m) / float(x.std(ddof=1)),
                     'base_correct_wmean': wm(col, m & df.base_correct), 'base_wrong_wmean': wm(col, m & ~df.base_correct),
                     **{f'{c}_mean': float(df.loc[m & (df.cell == c), col].mean()) for c in CELLS}}
        e['gap_RERANK_minus_TEST_weighted'] = e['RERANK']['weighted_mean'] - e['TEST']['weighted_mean']
        e['gap_95ci_stratified_bootstrap'] = boot_gap(col)
        A3[col] = e
    gt = A3['d_total']['gap_RERANK_minus_TEST_weighted']
    A3['share_of_total_gap'] = {c: A3[c]['gap_RERANK_minus_TEST_weighted'] / gt for c in ('d_vision', 'd_LM', 'd_other', 'interaction', 'lin_vision', 'nl_vision')}
    # matched comparisons within cells (removes cell-mix): mean RERANK - TEST per cell for vision vs total
    A3['within_cell_gap'] = {c: {col: float(df.loc[(df.phase == 'RERANK') & (df.cell == c), col].mean() - df.loc[(df.phase == 'TEST') & (df.cell == c), col].mean())
                                 for col in ('d_total', 'd_vision', 'd_LM', 'nl_vision', 'interaction')} for c in CELLS}
    R['3_asymmetry'] = A3

    # ---------------- 4. nonlinear vision residual
    A4 = {'nl_vision_vs_d_total': fit_stats(df.nl_vision, df.d_total),
          'nl_vision_sd': float(df.nl_vision.std(ddof=1)), 'lin_vision_sd': float(df.lin_vision.std(ddof=1)), 'd_vision_sd': float(df.d_vision.std(ddof=1)),
          'nl_vision_vs_d_vision_r': float(np.corrcoef(df.nl_vision, df.d_vision)[0, 1]),
          'lin_vision_vs_d_vision_r': float(np.corrcoef(df.lin_vision, df.d_vision)[0, 1]),
          'cohen_d_changed_vs_unchanged_nl_vision': cohen_d(df.nl_vision[df.changed], df.nl_vision[~df.changed]),
          'cohen_d_RERANK_vs_TEST_nl_vision': cohen_d(df.nl_vision[df.phase == 'RERANK'], df.nl_vision[df.phase == 'TEST']),
          'by_phase_cell_mean_nl_vision': {f'{p}|{c}': float(g.nl_vision.mean()) for (p, c), g in df.groupby(['phase', 'cell'])},
          'repair_vs_regression': {ph: {'repair_mean': float(df.nl_vision[(df.phase == ph) & (df.cell == 'repair')].mean()),
                                        'regression_mean': float(df.nl_vision[(df.phase == ph) & (df.cell == 'regression')].mean())} for ph in ('RERANK', 'TEST')}}
    # LOO-CV regression (weighted by population weights would overweight a few; use unweighted OLS, report both in-sample and LOO R^2)
    def loo(cols):
        X = np.column_stack([np.ones(len(df))] + [df[c].to_numpy() for c in cols]); y = df.d_total.to_numpy(); pr = np.zeros(len(y))
        for i in range(len(y)):
            m = np.arange(len(y)) != i; b = np.linalg.lstsq(X[m], y[m], rcond=None)[0]; pr[i] = X[i] @ b
        bfull = np.linalg.lstsq(X, y, rcond=None)[0]
        return {'coef': dict(zip(['intercept'] + list(cols), map(float, bfull))), 'loo_r2': float(1 - ((y - pr) ** 2).sum() / ((y - y.mean()) ** 2).sum())}
    A4['regression_loo'] = {'M1_LM_lin+other_lin': loo(['lin_LM', 'lin_other']),
                            'M2_+vision_lin': loo(['lin_LM', 'lin_other', 'lin_vision']),
                            'M3_+vision_nonlinear_residual': loo(['lin_LM', 'lin_other', 'lin_vision', 'nl_vision'])}
    A4['regression_loo']['M0_phase_only'] = loo(['is_rerank'])
    A4['regression_loo']['M3b_+phase'] = loo(['lin_LM', 'lin_other', 'lin_vision', 'nl_vision', 'is_rerank'])
    A4['caveat'] = ('nl_vision + lin_vision = measured vision insertion, and d_total ~ sum of insertions (section 2), so M3 > M2 is '
                    'partly mechanical: it shows the measured vision term matters for d_total, not that its nonlinearity causes the phase asymmetry.')
    R['4_nonlinear_vision'] = A4

    # ---------------- 5. interaction residual + context dependence (insertion vs removal)
    I_ = df.interaction
    A5 = {'sd': float(I_.std(ddof=1)), 'energy_share_of_d_total': float((I_ ** 2).sum() / (df.d_total ** 2).sum()),
          'mean': float(I_.mean()), 'r_with_d_total': float(np.corrcoef(I_, df.d_total)[0, 1]),
          'spearman_abs_with_abs_mB': float(pd.Series(I_.abs()).corr(df.mB.abs(), method='spearman')),
          'by_phase_weighted_mean': {ph: wm('interaction', df.phase == ph) for ph in ('RERANK', 'TEST')},
          'cohen_d_RERANK_vs_TEST': cohen_d(I_[df.phase == 'RERANK'], I_[df.phase == 'TEST']),
          'by_cell_mean': {c: float(I_[df.cell == c].mean()) for c in CELLS},
          'close_margin_abs_mB_lt_1_mean_abs': float(I_[df.mB.abs() < 1].abs().mean()), 'far_margin_mean_abs': float(I_[df.mB.abs() >= 1].abs().mean())}
    ctx = {}
    for g in GROUPS:
        d = df[f'rem_{g}'] - df[f'ins_{g}']
        ctx[g] = {'r_ins_rem': float(np.corrcoef(df[f'ins_{g}'], df[f'rem_{g}'])[0, 1]), 'sd_ins': float(df[f'ins_{g}'].std(ddof=1)),
                  'sd_context_diff': float(d.std(ddof=1)), 'gap_RERANK_minus_TEST_context_diff_weighted':
                  float(np.average(d[df.phase == 'RERANK'], weights=df.w[df.phase == 'RERANK']) - np.average(d[df.phase == 'TEST'], weights=df.w[df.phase == 'TEST']))}
    A5['context_dependence_removal_minus_insertion'] = ctx
    R['5_interaction'] = A5

    # ---------------- 6/7. controls: group-level quantities
    R['6_7_controls_group_level'] = {
        'available': False,
        'reason': 'No group insertions/removals or first-order group partials exist for any of the 12 random controls '
                  '(session 1 measured only 9504111; GPU-A group partials were computed only for 9504111 on LOCALIZATION). '
                  'Control vision effects, vision nonlinear residuals, and their directions therefore cannot be computed without new GPU work.'}

    # whole-model linearization departure for 9504111 vs the 12 controls (available: HF-internal measured t and first-order pred, all 761)
    bc = json.loads((GPU / 'base_contrast_fp32.json').read_text())['s']
    f1 = json.loads((GPU / 'f1_candidate_contrasts.json').read_text())
    ctrl = json.loads((RC / 'frozen_controls.json').read_text())['candidates']; seed2cid = {WIN_SEED: WIN, **{c['seed']: c['candidate_id'] for c in ctrl}}
    W = []
    for seed, cid in seed2cid.items():
        row = {'seed': seed, 'is_winner': seed == WIN_SEED}
        for ph in ('RERANK', 'TEST'):
            meta = json.loads((GPU / f'meta_{ph}.json').read_text()); P = np.load(GPU / f'pred_{ph}.npy'); k = [c[0] for c in meta['candidates']].index(cid)
            t = np.array([f1[str(seed)][u]['s'] - bc[u] for u in meta['uids']]); p = P[:, k]; res = t - p
            row.update({f'{ph}_mean_t': t.mean(), f'{ph}_mean_lin': p.mean(), f'{ph}_mean_nonlin': res.mean(),
                        f'{ph}_sd_t': t.std(), f'{ph}_sd_lin': p.std(), f'{ph}_sd_nonlin': res.std(), f'{ph}_r': np.corrcoef(p, t)[0, 1]})
        for q in ('t', 'lin', 'nonlin'):
            row[f'gap_{q}'] = row[f'RERANK_mean_{q}'] - row[f'TEST_mean_{q}']
        W.append(row)
    W = pd.DataFrame(W); W.to_csv(OUT / 'whole_model_linearization_winner_vs_controls.csv', index=False)
    win = W[W.is_winner].iloc[0]; cs = W[~W.is_winner]
    R['6_whole_model_winner_vs_controls'] = {
        'note': 'Whole-model only (all groups together; HF-internal fp32 contrast, all 761 examples). Not a vision decomposition.',
        **{k: {'winner': float(win[k]), 'controls_min': float(cs[k].min()), 'controls_median': float(cs[k].median()), 'controls_max': float(cs[k].max()),
               'winner_rank_desc_of_13': int(1 + (cs[k] > win[k]).sum())}
           for k in ('RERANK_sd_nonlin', 'TEST_sd_nonlin', 'RERANK_r', 'TEST_r', 'gap_t', 'gap_lin', 'gap_nonlin', 'RERANK_mean_nonlin', 'TEST_mean_nonlin')}}

    # ---------------- 8. selection + nonlinear response across measured candidates (accuracy gains)
    d = pd.read_parquet(PA / 'paper_master_predictions.parquet', columns=['candidate_id', 'phase', 'example_id', 'candidate_correct', 'base_correct'])
    d['candidate_id'] = d.candidate_id.astype(str); d['example_id'] = d.example_id.astype(str)
    gold = {}
    for l in Path('examples/omnispatial-perspective-taking/test.jsonl').read_text(encoding='utf-8').splitlines():
        r = json.loads(l); gold[r['uid']] = LET[r['answer']]
    gains = {}
    for ph in ('RERANK', 'TEST'):
        meta = json.loads((GPU / f'meta_{ph}.json').read_text()); P = np.load(GPU / f'pred_{ph}.npy')
        cands = [c[0] for c in meta['candidates']]; uids = meta['uids']; mB = np.array([bc[u] for u in uids])
        obs = d[(d.phase == ph) & d.candidate_id.isin(cands)].pivot(index='candidate_id', columns='example_id', values='candidate_correct')
        if ph == 'TEST':
            for c in ctrl:
                if c['candidate_id'] not in obs.index:
                    rr = ld(RC / f"gpu/control_{c['seed']}.json.gz"); obs.loc[c['candidate_id']] = pd.Series({q['uid']: q['parsed'] == gold[q['uid']] for q in rr['TEST']})
        obs = obs.loc[cands, uids].astype(float)
        bcv = d[(d.phase == ph) & (d.candidate_id == 'BASE')].set_index('example_id').base_correct.loc[uids].to_numpy(float)
        g_obs = 100 * (obs.to_numpy() - bcv).mean(1); g_lin = 100 * ((mB[:, None] + P > 0).astype(float) - (mB > 0)[:, None]).mean(0)
        gains[ph] = pd.DataFrame({'obs': g_obs, 'lin': g_lin, 'nonlin': g_obs - g_lin}, index=cands)
    tc = pd.read_csv(PA / 'selection-vs-specificity/test_measured_candidates.csv').set_index('candidate_id')
    J = gains['RERANK'].join(gains['TEST'], lsuffix='_R', rsuffix='_T', how='inner'); J['group'] = tc.group.reindex(J.index)
    J.to_csv(OUT / 'gain_decomposition_test_measured_candidates.csv')
    top = J[J.group == 'top50']; wr = J.loc[WIN]
    pct = lambda col, frame: float(100 * (frame[col] < wr[col]).mean())
    R['8_selection_vs_signature'] = {
        'n_RERANK_candidates': int(len(gains['RERANK'])), 'n_TEST_measured': int(len(J)), 'n_top50': int(len(top)),
        'winner': {k: float(wr[k]) for k in ('obs_R', 'lin_R', 'nonlin_R', 'obs_T', 'lin_T', 'nonlin_T')},
        'winner_percentile_among_543_RERANK': {k: float(100 * (gains['RERANK'][k] < gains['RERANK'].loc[WIN, k]).mean()) for k in ('obs', 'lin', 'nonlin')},
        'winner_percentile_among_top50': {k: pct(k, top) for k in ('obs_R', 'lin_R', 'nonlin_R', 'obs_T', 'lin_T', 'nonlin_T')},
        'top50_mean': {k: float(top[k].mean()) for k in ('obs_R', 'lin_R', 'nonlin_R', 'obs_T', 'lin_T', 'nonlin_T')},
        'controls_mean': {k: float(J[J.group == 'random_control'][k].mean()) for k in ('obs_R', 'lin_R', 'nonlin_R', 'obs_T', 'lin_T', 'nonlin_T')},
        'across_61_corr': {'nonlin_R_vs_nonlin_T': float(np.corrcoef(J.nonlin_R, J.nonlin_T)[0, 1]),
                           'lin_R_vs_lin_T': float(np.corrcoef(J.lin_R, J.lin_T)[0, 1]),
                           'obs_R_vs_obs_T': float(np.corrcoef(J.obs_R, J.obs_T)[0, 1])},
        'across_top50_corr': {'nonlin_R_vs_nonlin_T': float(np.corrcoef(top.nonlin_R, top.nonlin_T)[0, 1]),
                              'obs_R_vs_obs_T': float(np.corrcoef(top.obs_R, top.obs_T)[0, 1])},
        'across_543_RERANK_share_of_obs_variance': {'var_obs': float(gains['RERANK'].obs.var()), 'var_lin': float(gains['RERANK'].lin.var()),
                                                    'var_nonlin': float(gains['RERANK'].nonlin.var()),
                                                    'r_obs_nonlin': float(np.corrcoef(gains['RERANK'].obs, gains['RERANK'].nonlin)[0, 1])}}

    # ---------------- 9. analytical counterfactuals (contrast-sign correctness on LOC, population-weighted accuracy change per phase)
    def acc_change(delta):
        out = {}
        corr_c = (df.mB + delta) > 0; corr_b = df.mB > 0
        for ph in ('RERANK', 'TEST'):
            m = df.phase == ph; out[ph] = float(100 * np.average((corr_c[m].astype(float) - corr_b[m].astype(float)), weights=df.w[m]))
        return out
    true_acc = {ph: float(100 * np.average(((df.cell == 'repair').astype(float) - (df.cell == 'regression').astype(float))[df.phase == ph], weights=df.w[df.phase == ph])) for ph in ('RERANK', 'TEST')}
    cf = {'actual_candidate_contrast_sign': df.d_total, 'sum_of_groups': df.d_sum,
          'remove_vision_entirely': df.d_total - df.d_vision, 'linearize_vision (remove nl_vision)': df.d_total - df.nl_vision,
          'remove_interaction': df.d_sum, 'remove_LM': df.d_total - df.d_LM, 'only_linear_LM_plus_measured_vision': df.lin_LM + df.d_vision,
          'only_vision': df.d_vision}
    R['9_counterfactual'] = {'label': 'ANALYTICAL counterfactual on measured score components (not a causal intervention)',
                             'true_accuracy_change_pp_population': true_acc,
                             'contrast_sign_accuracy_change_pp': {k: acc_change(v) for k, v in cf.items()}}
    for k in R['9_counterfactual']['contrast_sign_accuracy_change_pp']:
        v = R['9_counterfactual']['contrast_sign_accuracy_change_pp'][k]; v['gap_R_minus_T'] = v['RERANK'] - v['TEST']

    # ---------------- 10. simpler explanations on all 761 (winner vs controls), vLLM/HF
    m_all = {u: bc[u] for u in allu}
    simp = {}
    for seed, cid in seed2cid.items():
        t = np.array([f1[str(seed)][u]['s'] - bc[u] for u in allu]); mb = np.array([m_all[u] for u in allu]); ph = np.array([labels[u]['phase'] for u in allu])
        X = np.column_stack([np.ones(len(t)), mb, (mb > 0).astype(float)]); b = np.linalg.lstsq(X, t, rcond=None)[0]; res = t - X @ b
        simp[str(seed)] = {'slope_on_base_margin': float(b[1]), 'r2_margin_correctness': float(1 - res.var() / t.var()),
                           'raw_gap_R_minus_T': float(t[ph == 'RERANK'].mean() - t[ph == 'TEST'].mean()),
                           'margin_adjusted_gap_R_minus_T': float(res[ph == 'RERANK'].mean() - res[ph == 'TEST'].mean())}
    S = pd.DataFrame(simp).T; w_ = S.loc[str(WIN_SEED)]; c_ = S.drop(str(WIN_SEED))
    R['10_simpler'] = {'per_candidate': simp, 'winner_rank_desc_of_13': {k: int(1 + (c_[k] > w_[k]).sum()) for k in S.columns},
                       'controls_range': {k: [float(c_[k].min()), float(c_[k].max())] for k in S.columns},
                       'loc_vision_vs_mB_r': float(np.corrcoef(df.d_vision, df.mB)[0, 1]), 'loc_total_vs_mB_r': float(np.corrcoef(df.d_total, df.mB)[0, 1]),
                       'loc_LM_vs_mB_r': float(np.corrcoef(df.d_LM, df.mB)[0, 1])}
    df.to_csv(OUT / 'loc_decomposition_per_example.csv', index=False)
    (OUT / 'vision_nonlinearity_results.json').write_text(json.dumps(R, indent=1, default=float))
    return R


if __name__ == '__main__':
    main(); print('ok')

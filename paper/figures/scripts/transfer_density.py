"""Empirical improvement density vs cross-sample transfer (POST HOC, CPU only, deterministic).

Question: when a random nearby perturbation looks task-improving on one finite question sample, how often does
that improvement stay attached to the same perturbation on an independent sample?

Repeated split trials quantify the stability of ONE observed prediction matrix; they are not independent
repetitions of search or training. All intervals are descriptive (split/replay percentiles), not tests.
TEST predictions are used only in Part 10 (seed 9504111 and the frozen top 50), never to design the analysis.

Usage: python paper/figures/scripts/transfer_density.py [compute|figures|all]
"""
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import ACCENT, BASE_C, CAND, CANDIDATE, GOOD, INK2, MUTED, SIGMAS, write_tex  # noqa: E402
from behavioral_diversity import load_inputs, matrix, permute_within  # noqa: E402

OUT = Path('results/paper-analysis/transfer-density')
FIGD = Path('paper/figures/transfer-density')
EX = Path('examples/omnispatial-perspective-taking')
L = 'ABCD'
SEED = 20261007
import os
QUICK = bool(os.environ.get('TD_QUICK'))
T_PRIMARY, T_SIZE, T_REPLAY, T_REPLAY_M = (200, 100, 200, 100) if QUICK else (5000, 2000, 5000, 2000)
NULL_REPS_PRIMARY, NULL_SPLITS_PRIMARY = (10, 20) if QUICK else (500, 100)
NULL_REPS_SIZE, NULL_SPLITS_SIZE = (10, 10) if QUICK else (500, 50)
NULL_REPS_REPLAY, NULL_TRIALS_REPLAY = (10, 10) if QUICK else (200, 50)
SIZES = (25, 50, 75, 100, 150, 200)
POOL_M = (10, 25, 50, 100, 250, 500)
PP = (('gt0', 0.0), ('pp1', 1.0), ('pp3', 3.0), ('pp5', 5.0))
CNT = (('n1', 1), ('n2', 2), ('n3', 3), ('n5', 5))
TOPQ = (.01, .02, .05, .10)
TOPK = (10, 25, 50, 100, 250)
SIG_COL = {0.00025: '#9cc2ee', 0.0005: '#5a9be0', 0.001: CAND, 0.002: '#123f7a'}


def pp_cut(name, m, n):
    """Realized integer net-question cutoff for a pp margin on n questions ('gt0' means > 0)."""
    return 1 if name == 'gt0' else max(1, int(np.ceil(m * n / 100 - 1e-9)))


# -------------------------------------------------------------------------------------------- splits
def strat_order(codes, rng):
    k = codes.max() + 1
    return np.lexsort((rng.random(len(codes)), rng.permutation(k)[codes]))


def split_two(codes, rng):
    """Stratified disjoint halves: stratum-sorted random order, alternate assignment from a random offset."""
    o = strat_order(codes, rng)
    lab = (np.arange(len(o)) + rng.integers(2)) % 2
    return o[lab == 0], o[lab == 1]


def subsample(idx, codes, n, rng):
    """Proportionally stratified systematic sample of n from idx."""
    if n >= len(idx):
        return idx
    o = idx[strat_order(codes[idx], rng)]
    step = len(o) / n
    return o[np.floor(rng.random() * step + np.arange(n) * step).astype(int)]


def split_replay(codes, rng):
    o = strat_order(codes, rng)
    lab = (np.arange(len(o)) + rng.integers(4)) % 4
    return o[lab == 0], o[lab == 1], o[(lab == 2) | (lab == 3)]


def masks(splits, Q):
    M = np.zeros((len(splits), Q), np.float32)
    for t, s in enumerate(splits):
        M[t, s] = 1
    return M


# ------------------------------------------------------------------------------------- per-trial stats
def colcorr(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    d = np.sqrt((X ** 2).sum(0) * (Y ** 2).sum(0))
    return np.where(d > 0, (X * Y).sum(0) / np.where(d > 0, d, 1), np.nan)


def colrank(X):
    """Average ranks per column (ties averaged), vectorized."""
    n, t = X.shape
    o = np.argsort(X, axis=0, kind='stable')
    xs = np.take_along_axis(X, o, 0)
    r = np.empty_like(X, dtype=np.float64)
    ranks = np.arange(1, n + 1, dtype=np.float64)[:, None].repeat(t, 1)
    # average ties: for each column, groups of equal values get mean rank
    new = np.vstack([np.ones((1, t), bool), xs[1:] != xs[:-1]])
    gid = np.cumsum(new, 0) - 1
    for j in range(t):
        g = gid[:, j]
        mean = np.bincount(g, ranks[:, j]) / np.bincount(g)
        r[o[:, j], j] = mean[g]
    return r


def trial_stats(GA, GB, cuts, spearman=True):
    """GA, GB: (N, T) net-gain counts. cuts: {name: count cutoff}. Returns per-trial arrays."""
    out = {'pearson': colcorr(GA.astype(np.float64), GB.astype(np.float64))}
    if spearman:
        out['spearman'] = colcorr(colrank(GA), colrank(GB))
    N = GA.shape[0]
    for name, c in cuts.items():
        IA, IB = GA >= c, GB >= c
        a, b, j = IA.sum(0), IB.sum(0), (IA & IB).sum(0)
        u = a + b - j
        with np.errstate(invalid='ignore', divide='ignore'):
            out[f'{name}|rho_A'] = a / N; out[f'{name}|rho_B'] = b / N; out[f'{name}|rho_joint'] = j / N
            out[f'{name}|tau_A_to_B'] = np.where(a > 0, j / np.maximum(a, 1), np.nan)
            out[f'{name}|tau_B_to_A'] = np.where(b > 0, j / np.maximum(b, 1), np.nan)
            out[f'{name}|lift'] = np.where((a > 0) & (b > 0), j * N / np.maximum(a * b, 1), np.nan)
            out[f'{name}|jaccard'] = np.where(u > 0, j / np.maximum(u, 1), np.nan)
    return out


def topq_stats(GA, GB, rng, qs=TOPQ, ks=TOPK):
    """Rank-based overlap with random tie-breaking (ties are pervasive for integer gains)."""
    N, T = GA.shape
    sA = GA + rng.random(GA.shape) * .999; sB = GB + rng.random(GB.shape) * .999
    rA = np.argsort(np.argsort(-sA, 0), 0); rB = np.argsort(np.argsort(-sB, 0), 0)  # 0 = best
    out = {}
    for q in qs:
        k = max(1, int(round(q * N)))
        out[f'top{q:g}|overlap'] = ((rA < k) & (rB < k)).sum(0) / k
        out[f'top{q:g}|random_expectation'] = np.full(T, k / N)
    for k in ks:
        if k > N: continue
        selA = rA < k
        out[f'topK{k}|mean_gain_B'] = (GB * selA).sum(0) / k
        out[f'topK{k}|pool_mean_gain_B'] = GB.mean(0)
        out[f'topK{k}|mean_rank_pct_B'] = 100 * ((rB + 1) * selA).sum(0) / k / N
        out[f'topK{k}|frac_in_topK_B'] = (selA & (rB < k)).sum(0) / k
        out[f'topK{k}|enrichment'] = out[f'topK{k}|frac_in_topK_B'] / (k / N)
    return out


def agg(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if not len(v):
        return None
    return {'mean': float(v.mean()), 'median': float(np.median(v)), 'sd': float(v.std(ddof=1)) if len(v) > 1 else 0.,
            'q025': float(np.quantile(v, .025)), 'q975': float(np.quantile(v, .975)), 'n': int(len(v))}


def null_cmp(obs_mean, null_means):
    v = np.asarray([x for x in null_means if x is not None and not np.isnan(x)], float)
    if not len(v) or obs_mean is None:
        return None
    return {'observed': obs_mean, 'null_mean': float(v.mean()), 'null_q025': float(np.quantile(v, .025)),
            'null_q975': float(np.quantile(v, .975)), 'percentile_of_observed': float(100 * (v <= obs_mean).mean())}


# -------------------------------------------------------------------------------------- repair selectivity
def selectivity(C, b):
    """Per candidate: repair rate R = P(correct | base wrong), damage rate B = P(wrong | base correct), S = R - B."""
    bw, bc = ~b, b
    R = C[:, bw].mean(1); D = (~C[:, bc]).mean(1)
    return R, D, R - D


def corr2(x, y):
    from scipy import stats
    return {'n': int(len(x)), 'pearson': float(np.corrcoef(x, y)[0, 1]), 'spearman': float(stats.spearmanr(x, y)[0])}


def perm_corr(x, y, strata, rng, n=1000):
    """Correlation of x and y with candidate identity of y permuted within strata (sigma)."""
    obs = float(np.corrcoef(x, y)[0, 1]); v = []
    for _ in range(n):
        p = np.arange(len(y))
        for s in np.unique(strata):
            m = np.where(strata == s)[0]; p[m] = rng.permutation(m)
        v.append(np.corrcoef(x, y[p])[0, 1])
    return null_cmp(obs, v)


# --------------------------------------------------------------------------------------------- compute
def compute():
    OUT.mkdir(parents=True, exist_ok=True)
    df, lock, proto = load_inputs()
    ids = json.loads(Path('results/paper-analysis/behavioral-diversity/candidate_ids.json').read_text())
    recs = {r['candidate']['candidate_id']: r for r in lock['records']}
    order = [r['candidate']['candidate_id'] for r in sorted(lock['records'], key=lambda r: r['index'])]
    if order != ids['search_order']:
        raise SystemExit('STOP: candidate order differs from behavioral-diversity audit')
    sigma = np.array([recs[c]['candidate']['sigma'] for c in order])
    audit_idx = set(proto['density_audit']['indices'])
    audit = [c for c in order if recs[c]['index'] in audit_idx]
    top50 = list(lock['ranked_ids'][:50])
    rows = {s: {json.loads(l)['uid']: json.loads(l) for l in (EX / f'{s}.jsonl').read_text(encoding='utf-8').splitlines()} for s in ('search', 'validation', 'test')}

    P, b_, g_, qS = matrix(df, 'SEARCH', order)
    CS, bS = (P == g_), (b_ == g_)
    PaS, _, _, _ = matrix(df, 'SEARCH', audit)
    PaR, bRp, gR, qR = matrix(df, 'RERANK', audit)
    CaS, CaR, bR = (PaS == g_), (PaR == gR), (bRp == gR)
    asg = sigma[[order.index(c) for c in audit]]
    meta = [dict(qid=q, phase='SEARCH', subtask=rows['search'][q]['sub_task_type'], answer=L[rows['search'][q]['answer']],
                 image_sha256=rows['search'][q]['image_sha256']) for q in qS] + \
           [dict(qid=q, phase='RERANK', subtask=rows['validation'][q]['sub_task_type'], answer=L[rows['validation'][q]['answer']],
                 image_sha256=rows['validation'][q]['image_sha256']) for q in qR]
    meta = pd.DataFrame(meta)
    checks = {'candidates': len(order), 'unique': len(set(order)), 'search_questions': CS.shape[1], 'per_sigma': {str(s): int((sigma == s).sum()) for s in SIGMAS},
              'base_search_complete': len(bS) == 200, 'audit': len(audit), 'audit_per_sigma': {str(s): int((asg == s).sum()) for s in SIGMAS},
              'audit_search_rerank_complete': CaS.shape == (500, 200) and CaR.shape == (500, 200), 'pool_questions': len(meta),
              'pool_unique_qids': int(meta.qid.nunique()), 'pool_duplicate_image_hashes': int(len(meta) - meta.image_sha256.nunique()),
              'gold_letters_match_jsonl': bool(all(L[g] == L[rows['search'][q]['answer']] for g, q in zip(g_, qS))
                                                  and all(L[g] == L[rows['validation'][q]['answer']] for g, q in zip(gR, qR)))}
    print(json.dumps(checks, indent=1), flush=True)
    if checks['unique'] != 5000 or checks['audit'] != 500 or checks['pool_duplicate_image_hashes'] or not checks['audit_search_rerank_complete'] \
            or checks['per_sigma'] != {str(s): 1250 for s in SIGMAS} or checks['audit_per_sigma'] != {str(s): 125 for s in SIGMAS}:
        raise SystemExit('STOP: verification failed')
    meta.to_csv(OUT / 'question_pool_400.csv', index=False)
    (OUT / 'candidate_ids.json').write_text(json.dumps({'search_order': order, 'sigma': sigma.tolist(), 'audit500': audit, 'top50': top50}))
    R = {'post_hoc': True, 'seed': SEED, 'checks': checks,
         'trials': {'primary': T_PRIMARY, 'size': T_SIZE, 'replay': T_REPLAY, 'replay_M': T_REPLAY_M},
         'nulls': {'primary': [NULL_REPS_PRIMARY, NULL_SPLITS_PRIMARY], 'size': [NULL_REPS_SIZE, NULL_SPLITS_SIZE], 'replay': [NULL_REPS_REPLAY, NULL_TRIALS_REPLAY]}}

    codesS = pd.factorize(meta.subtask[:200] + meta.answer[:200])[0]
    codes400 = pd.factorize(meta.subtask + meta.answer)[0]
    groups = {'all': np.arange(5000), **{str(s): np.where(sigma == s)[0] for s in SIGMAS}}
    DS = CS.astype(np.int16) - bS.astype(np.int16)

    # ---------------- Part 2 / 5 / 6 / 12: all 5,000 on SEARCH100 vs SEARCH100
    rng = np.random.default_rng([SEED, 2])
    splits = [split_two(codesS, rng) for _ in range(T_PRIMARY)]
    np.savez_compressed(OUT / 'splits_primary_search100x100.npz', A=np.array([s[0] for s in splits], np.int16), B=np.array([s[1] for s in splits], np.int16))
    cuts = {**{n: pp_cut(n, m, 100) for n, m in PP}, **{n: c for n, c in CNT}}
    R['primary_cuts'] = cuts
    per = {g: {} for g in groups}
    trng = np.random.default_rng([SEED, 21])
    for c0 in range(0, T_PRIMARY, 500):
        sl = splits[c0:c0 + 500]
        MA, MB = masks([s[0] for s in sl], 200), masks([s[1] for s in sl], 200)
        GA, GB = (DS @ MA.T).astype(np.int32), (DS @ MB.T).astype(np.int32)
        for g, ix in groups.items():
            st = trial_stats(GA[ix], GB[ix], cuts)
            st.update(topq_stats(GA[ix], GB[ix], trng))
            for k, v in st.items():
                per[g].setdefault(k, []).append(v)
        print('primary', c0, flush=True)
    per = {g: {k: np.concatenate(v) for k, v in d.items()} for g, d in per.items()}
    R['primary'] = {g: {k: agg(v) for k, v in d.items()} for g, d in per.items()}
    # null: within-sigma per-question permutation of candidate identity, fixed subset of splits
    nsl = splits[:NULL_SPLITS_PRIMARY]
    MA, MB = masks([s[0] for s in nsl], 200), masks([s[1] for s in nsl], 200)
    nrng = np.random.default_rng([SEED, 3]); nper = {g: {} for g in groups}
    for r in range(NULL_REPS_PRIMARY):
        Cn = permute_within(CS, sigma, nrng)
        Dn = Cn.astype(np.int16) - bS.astype(np.int16)
        GA, GB = (Dn @ MA.T).astype(np.int32), (Dn @ MB.T).astype(np.int32)
        for g, ix in groups.items():
            st = trial_stats(GA[ix], GB[ix], cuts, spearman=False)
            st.update(topq_stats(GA[ix], GB[ix], nrng))
            for k, v in st.items():
                nper[g].setdefault(k, []).append(np.nanmean(v) if np.isfinite(v).any() else np.nan)
        if r % 100 == 0: print('primary null', r, flush=True)
    R['primary_null'] = {g: {k: null_cmp(R['primary'][g][k]['mean'] if R['primary'][g].get(k) else None, v) for k, v in d.items()} for g, d in nper.items()}

    # ---------------- Part 3: sample-size curve, audit 500 on the 400-question pool
    D400 = np.hstack([CaS, CaR]).astype(np.int16) - np.concatenate([bS, bR]).astype(np.int16)
    C400 = np.hstack([CaS, CaR])
    b400 = np.concatenate([bS, bR])
    R['size'] = {}; R['size_null'] = {}; R['size_cuts'] = {}
    size_splits = {}
    for n in SIZES:
        rng = np.random.default_rng([SEED, 4, n])
        sp = []
        for _ in range(T_SIZE):
            A, B = split_two(codes400, rng)
            sp.append((subsample(A, codes400, n, rng), subsample(B, codes400, n, rng)))
        size_splits[n] = sp
        cuts_n = {**{nm: pp_cut(nm, m, n) for nm, m in PP}, **{nm: c for nm, c in CNT}}
        R['size_cuts'][n] = cuts_n
        MA, MB = masks([s[0] for s in sp], 400), masks([s[1] for s in sp], 400)
        GA, GB = (D400 @ MA.T).astype(np.int32), (D400 @ MB.T).astype(np.int32)
        st = trial_stats(GA, GB, cuts_n); st.update(topq_stats(GA, GB, np.random.default_rng([SEED, 41, n])))
        R['size'][n] = {k: agg(v) for k, v in st.items()}
        MA, MB = MA[:NULL_SPLITS_SIZE], MB[:NULL_SPLITS_SIZE]
        nrng = np.random.default_rng([SEED, 5, n]); nv = {}
        for r in range(NULL_REPS_SIZE):
            Dn = permute_within(C400, asg, nrng).astype(np.int16) - b400.astype(np.int16)
            GA, GB = (Dn @ MA.T).astype(np.int32), (Dn @ MB.T).astype(np.int32)
            st = trial_stats(GA, GB, cuts_n, spearman=False); st.update(topq_stats(GA, GB, nrng))
            for k, v in st.items():
                nv.setdefault(k, []).append(np.nanmean(v) if np.isfinite(v).any() else np.nan)
        R['size_null'][n] = {k: null_cmp(R['size'][n][k]['mean'] if R['size'][n].get(k) else None, v) for k, v in nv.items()}
        print('size', n, flush=True)
    np.savez_compressed(OUT / 'splits_size_curve.npz', **{f'A_{n}': np.array([s[0] for s in sp], np.int16) for n, sp in size_splits.items()},
                        **{f'B_{n}': np.array([s[1] for s in sp], np.int16) for n, sp in size_splits.items()})

    # ---------------- Parts 7-9: two-stage selection replay
    hrank = np.argsort(np.argsort([hashlib.sha256(('visual-rank-v1:' + c).encode()).hexdigest() for c in audit]))  # 0 = first in tie-break

    def replay(Cmat, pool_rng, trials, M, K, split_rng, record=False):
        D = Cmat.astype(np.int16) - b400.astype(np.int16)
        out = []
        for t in range(trials):
            S, Rr, H = split_replay(codes400, split_rng)
            pool = np.arange(500) if M == 500 else np.sort(pool_rng.choice(500, M, replace=False))
            gS, gR, gH = D[pool][:, S].sum(1), D[pool][:, Rr].sum(1), D[pool][:, H].sum(1)
            o = np.lexsort((hrank[pool], -gS))  # SEARCH gain desc, hash tie-break
            top = o[:K]
            w = top[np.lexsort((hrank[pool][top], -gR[top]))[0]]
            rec = {'M': M, 'K': K, 'winner': int(pool[w]), 'sigma': float(asg[pool[w]]), 'search_gain_pp': 100 * gS[w] / len(S),
                   'rerank_gain_pp': 100 * gR[w] / len(Rr), 'holdout_gain_pp': 100 * gH[w] / len(H),
                   'winner_search_rank': int(np.where(o == w)[0][0] + 1), 'pool_mean_holdout_pp': float(100 * gH.mean() / len(H)),
                   'topK_mean_holdout_pp': float(100 * gH[top].mean() / len(H)), 'pool_best_holdout_pp': float(100 * gH.max() / len(H))}
            if record:
                rec.update(S=S.tolist(), R=Rr.tolist())
            out.append(rec)
        return out

    def summarize(recs):
        d = pd.DataFrame(recs)
        gap = d.rerank_gain_pp - d.holdout_gain_pp
        return {'trials': len(d), 'search_gain': agg(d.search_gain_pp), 'rerank_gain': agg(d.rerank_gain_pp), 'holdout_gain': agg(d.holdout_gain_pp),
                'optimism_gap': agg(gap), 'p_holdout_gt0': float((d.holdout_gain_pp > 0).mean()), 'p_holdout_ge1pp': float((d.holdout_gain_pp >= 1 - 1e-9).mean()),
                'p_holdout_ge3pp': float((d.holdout_gain_pp >= 3 - 1e-9).mean()), 'p_holdout_ge5pp': float((d.holdout_gain_pp >= 5 - 1e-9).mean()),
                'pool_mean_holdout': agg(d.pool_mean_holdout_pp), 'topK_mean_holdout': agg(d.topK_mean_holdout_pp),
                'winner_minus_pool_mean_holdout': agg(d.holdout_gain_pp - d.pool_mean_holdout_pp),
                'winner_sigma_share': {str(s): float((d.sigma == s).mean()) for s in SIGMAS},
                'distinct_winners': int(d.winner.nunique()), 'top_winner_share': float(d.winner.value_counts().iloc[0] / len(d)),
                'top5_winner_share': float(d.winner.value_counts().iloc[:5].sum() / len(d))}

    prim = replay(C400, None, T_REPLAY, 500, 50, np.random.default_rng([SEED, 7]))
    pdf = pd.DataFrame(prim); pdf['winner_id'] = [audit[i] for i in pdf.winner]
    pdf.to_csv(OUT / 'replay_primary_trials.csv', index=False)
    R['replay_primary'] = summarize(prim)
    R['replay_primary']['winner_frequency_top10'] = [{'candidate_id': audit[i], 'sigma': float(asg[i]), 'wins': int(c)}
                                                    for i, c in pdf.winner.value_counts().iloc[:10].items()]
    R['replay_M'] = {}; R['replay_M_fraction'] = {}
    for M in POOL_M:
        K = min(50, max(5, round(.1 * M)))
        R['replay_M'][M] = summarize(replay(C400, np.random.default_rng([SEED, 8, M]), T_REPLAY_M, M, K, np.random.default_rng([SEED, 9, M])))
        Kf = max(1, round(.1 * M))
        R['replay_M_fraction'][M] = summarize(replay(C400, np.random.default_rng([SEED, 10, M]), T_REPLAY_M, M, Kf, np.random.default_rng([SEED, 11, M])))
        print('replay M', M, flush=True)
    # Part 9: replay under the item-fragility null (fresh permuted matrix per replicate)
    nrng = np.random.default_rng([SEED, 12]); R['replay_null'] = {}
    for M in POOL_M:
        K = min(50, max(5, round(.1 * M)))
        recs = []
        for r in range(NULL_REPS_REPLAY if M == 500 else max(20, NULL_REPS_REPLAY // 4)):
            Cn = permute_within(C400, asg, nrng)
            recs += replay(Cn, nrng, NULL_TRIALS_REPLAY, M, K, nrng)
        R['replay_null'][M] = summarize(recs)
        print('replay null M', M, flush=True)

    # ---------------- Part 10: repair selectivity
    Pt, bT_, gT, _ = matrix(df, 'TEST', top50)
    CT, bT = (Pt == gT), (bT_ == gT)
    PtR, _, _, _ = matrix(df, 'RERANK', top50)
    PtS, _, _, _ = matrix(df, 'SEARCH', top50)
    ci = top50.index(CANDIDATE)
    sel = {}
    for name, C_, b in (('SEARCH', PtS == g_, bS), ('RERANK', PtR == gR, bR), ('TEST', CT, bT)):
        Rr, Dd, Ss = selectivity(C_, b)
        sel[name] = {'repair': Rr, 'damage': Dd, 'S': Ss}
    R['selectivity_9504111'] = {ph: {'repair_rate': float(v['repair'][ci]), 'damage_rate': float(v['damage'][ci]), 'selectivity': float(v['S'][ci]),
                                     'repairs': int(((PtS if ph == 'SEARCH' else PtR if ph == 'RERANK' else Pt)[ci] == (g_ if ph == 'SEARCH' else gR if ph == 'RERANK' else gT))[~(bS if ph == 'SEARCH' else bR if ph == 'RERANK' else bT)].sum()),
                                     'base_wrong': int((~(bS if ph == 'SEARCH' else bR if ph == 'RERANK' else bT)).sum())}
                                for ph, v in sel.items()}
    aS, aR = selectivity(CaS, bS), selectivity(CaR, bR)
    srng = np.random.default_rng([SEED, 13])
    R['selectivity_audit'] = {'all': {**corr2(aS[2], aR[2]), 'perm_within_sigma': perm_corr(aS[2], aR[2], asg, srng)}}
    for s in SIGMAS:
        m = asg == s
        R['selectivity_audit'][str(s)] = {**corr2(aS[2][m], aR[2][m]), 'perm': perm_corr(aS[2][m], aR[2][m], np.zeros(m.sum()), srng)}
    R['selectivity_top50_rerank_test'] = corr2(sel['RERANK']['S'], sel['TEST']['S'])
    R['selectivity_top50_rerank_test']['perm'] = perm_corr(sel['RERANK']['S'], sel['TEST']['S'], sigma[[order.index(c) for c in top50]], srng)
    R['selectivity_top50_summary'] = {ph: agg(v['S']) for ph, v in sel.items()}
    R['selectivity_audit_summary'] = {'SEARCH': agg(aS[2]), 'RERANK': agg(aR[2])}
    pd.DataFrame({'candidate_id': audit, 'sigma': asg, 'S_search': aS[2], 'S_rerank': aR[2], 'repair_search': aS[0], 'damage_search': aS[1],
                  'repair_rerank': aR[0], 'damage_rerank': aR[1]}).to_csv(OUT / 'audit_repair_selectivity.csv', index=False)
    pd.DataFrame({'candidate_id': top50, 'S_rerank': sel['RERANK']['S'], 'S_test': sel['TEST']['S'], 'S_search': sel['SEARCH']['S']}).to_csv(OUT / 'top50_repair_selectivity.csv', index=False)

    # ---------------- Part 11: behavioral-direction persistence vs task-advantage persistence (audit 500, within sigma)
    BD = json.loads(Path('results/paper-analysis/behavioral-diversity/behavioral_diversity_results.json').read_text())['cross_split']
    gS_a, gR_a = (CaS.astype(int) - bS).sum(1), (CaR.astype(int) - bR).sum(1)
    R['behavior_vs_task'] = {}
    for s in SIGMAS:
        m = asg == s; z = np.zeros(m.sum())
        R['behavior_vs_task'][str(s)] = {
            'answer_cka': {'observed': BD['cka']['answer'][str(s)]['observed'], 'null_mean': BD['cka']['answer'][str(s)]['global_perm']['null_mean'],
                           'percentile': BD['cka']['answer'][str(s)]['global_perm']['percentile_of_observed']},
            'direction_only_cka': {'observed': BD['cka_direction_only']['answer'][str(s)]['observed'],
                                   'null_mean': BD['cka_direction_only']['answer'][str(s)]['global_perm']['null_mean'],
                                   'percentile': BD['cka_direction_only']['answer'][str(s)]['global_perm']['percentile_of_observed']},
            'distance_to_base_r': BD['distance_to_base'][str(s)]['pearson'],
            'gain_correlation': perm_corr(gS_a[m].astype(float), gR_a[m].astype(float), z, srng),
            'selectivity_correlation': R['selectivity_audit'][str(s)]['perm']}
    (OUT / 'transfer_density_results.json').write_text(json.dumps(R, indent=1, default=lambda x: x.item() if hasattr(x, 'item') else str(x)))
    print('COMPUTE_DONE', flush=True)


# ---------------------------------------------------------------------------------------------- figures
def save_td(fig, name):
    FIGD.mkdir(parents=True, exist_ok=True)
    plt.rcParams['svg.hashsalt'] = name
    fig.savefig(FIGD / f'{name}.pdf', metadata={'CreationDate': None})
    fig.savefig(FIGD / f'{name}.png', dpi=300)
    plt.close(fig)


def pooled(st, key, N=None):
    """Pooled estimates from trial-mean densities: tau = E[joint]/E[rho_A], lift = E[joint]/(E[rho_A]E[rho_B])."""
    a, b, j = (st[f'{key}|{k}']['mean'] for k in ('rho_A', 'rho_B', 'rho_joint'))
    N = N or 500
    return {'rho_A': a, 'rho_B': b, 'joint': j, 'tau': j / a if a else np.nan, 'lift': j / (a * b) if a * b else np.nan, 'joint_count': j * N}


def pooled_null(sn, key, N=None):
    if not sn.get(f'{key}|rho_joint'):
        return None
    a, b, j = (sn[f'{key}|{k}']['null_mean'] for k in ('rho_A', 'rho_B', 'rho_joint'))
    N = N or 500
    return {'tau': j / a if a else np.nan, 'lift': j / (a * b) if a * b else np.nan, 'joint_count': j * N}


def figures_and_tables():
    R = json.loads((OUT / 'transfer_density_results.json').read_text())
    sizes = [int(n) for n in R['size']]
    thr = (('gt0', '> 0', CAND), ('pp3', '≥ +3 pp', GOOD), ('pp5', '≥ +5 pp', ACCENT))

    # Figure A: pooled transfer rate and lift vs set size (pooled = ratio of trial-mean densities; points whose
    # mean joint improver count is < 2 candidates are not plotted)
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
    for key, lab, col in thr:
        tau, rb, lift, nl = [], [], [], []
        for n in sizes:
            pt, pn = pooled(R['size'][str(n)], key), pooled_null(R['size_null'][str(n)], key)
            ok = pt['joint_count'] >= 2
            tau.append(pt['tau'] if ok else np.nan); rb.append(pt['rho_B']); lift.append(pt['lift'] if ok else np.nan)
            nl.append(pn['lift'] if pn and pn['joint_count'] >= 2 else np.nan)
        axes[0].plot(sizes, tau, color=col, marker='o', ms=4, label=f'τ(A→B), gain {lab}')
        axes[0].plot(sizes, rb, color=col, ls=':', lw=1.3)
        axes[1].plot(sizes, lift, color=col, marker='o', ms=4, label=f'observed, {lab}')
        axes[1].plot(sizes, nl, color=col, ls='--', lw=1.1, marker='s', ms=3, alpha=.7)
    axes[0].set_xlabel('Questions per evaluation set (n)'); axes[0].set_ylabel('Empirical transfer rate (pooled)')
    axes[0].set_title('A  P(improver on B | improver on A), 500 audit cand.\n   dotted: ρ_B (no-persistence reference)', loc='left', fontsize=7.3)
    axes[0].legend(frameon=False, fontsize=6.2, loc='center right', bbox_to_anchor=(1, .55)); axes[0].set_ylim(0, None)
    axes[1].axhline(1, color=MUTED, lw=.8)
    axes[1].set_xlabel('Questions per evaluation set (n)'); axes[1].set_ylabel('Transfer lift τ / ρ_B (pooled)')
    axes[1].set_title('B  Lift: observed (solid) vs item-fragility null (dashed)', loc='left', fontsize=7.3)
    axes[1].legend(frameon=False, fontsize=6.2, loc='upper left')
    for ax in axes: ax.set_xticks(sizes)
    fig.tight_layout(); save_td(fig, 'fig_transfer_density_vs_samples')

    # Figure B: replay vs pool size
    Ms = [int(m) for m in R['replay_M']]
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
    ax = axes[0]
    for key, lab, col, ls in (('rerank_gain', 'Selected winner, RERANK100 gain', ACCENT, '-'), ('holdout_gain', 'Same winner, fresh HOLDOUT200 gain', CAND, '-')):
        m = [R['replay_M'][str(M)][key]['mean'] for M in Ms]
        lo = [R['replay_M'][str(M)][key]['q025'] for M in Ms]; hi = [R['replay_M'][str(M)][key]['q975'] for M in Ms]
        ax.fill_between(Ms, lo, hi, color=col, alpha=.13, lw=0)
        ax.plot(Ms, m, color=col, marker='o', ms=4, ls=ls, label=lab)
    nh = [R['replay_null'][str(M)]['holdout_gain']['mean'] for M in Ms]
    nr = [R['replay_null'][str(M)]['rerank_gain']['mean'] for M in Ms]
    ax.plot(Ms, nr, color=ACCENT, ls='--', lw=1, alpha=.7, label='Item-fragility null: RERANK')
    ax.plot(Ms, nh, color=CAND, ls='--', lw=1, alpha=.7, label='Item-fragility null: HOLDOUT')
    ax.axhline(0, color=MUTED, lw=.8); ax.set_xscale('log'); ax.set_xticks(Ms, [str(M) for M in Ms])
    ax.set_xlabel('Candidate pool size M (audit candidates)'); ax.set_ylabel('Gain over base (pp)')
    ax.set_title('A  Two-stage selection replay (mean, 95% of trials)', loc='left', fontsize=7.3)
    ax.legend(frameon=False, fontsize=5.8, loc='upper left')
    ax = axes[1]
    for key, lab, col in (('p_holdout_gt0', 'P(HOLDOUT gain > 0)', CAND), ('p_holdout_ge3pp', 'P(HOLDOUT ≥ +3 pp)', GOOD), ('p_holdout_ge5pp', 'P(HOLDOUT ≥ +5 pp)', ACCENT)):
        ax.plot(Ms, [R['replay_M'][str(M)][key] for M in Ms], color=col, marker='o', ms=4, label=lab)
        ax.plot(Ms, [R['replay_null'][str(M)][key] for M in Ms], color=col, ls='--', lw=1, alpha=.7)
    ax.set_xscale('log'); ax.set_xticks(Ms, [str(M) for M in Ms]); ax.set_ylim(0, 1)
    ax.set_xlabel('Candidate pool size M'); ax.set_ylabel('Probability over replay trials')
    ax.set_title('B  Does the selected winner improve on HOLDOUT?\n   solid observed, dashed null', loc='left', fontsize=7.3)
    ax.legend(frameon=False, fontsize=6, loc='upper left')
    fig.tight_layout(); save_td(fig, 'fig_selection_replay_vs_pool_size')

    # Figure C: repair selectivity transfer
    a = pd.read_csv(OUT / 'audit_repair_selectivity.csv'); t = pd.read_csv(OUT / 'top50_repair_selectivity.csv')
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    ax = axes[0]
    for s in SIGMAS:
        m = a.sigma == s
        ax.scatter(a.S_search[m], a.S_rerank[m], s=8, color=SIG_COL[s], alpha=.75, lw=0, label=f'σ = {s:g}')
    ax.axhline(0, color=MUTED, lw=.7); ax.axvline(0, color=MUTED, lw=.7)
    sa = R['selectivity_audit']
    ax.set_xlabel('Repair selectivity S on SEARCH200'); ax.set_ylabel('S on RERANK200')
    ax.set_title(f"A  500 random audit candidates (r = {sa['all']['pearson']:.2f};\n   within σ: " +
                 ', '.join(f"{sa[str(s)]['pearson']:.2f}" for s in SIGMAS) + ')', loc='left', fontsize=7.2)
    ax.legend(frameon=False, fontsize=6, loc='upper left', markerscale=1.5)
    ax = axes[1]
    w = t.candidate_id == CANDIDATE
    ax.scatter(t.S_rerank[~w], t.S_test[~w], s=14, color=CAND, alpha=.8, lw=0, label='SEARCH top-50 members')
    ax.scatter(t.S_rerank[w], t.S_test[w], s=60, marker='*', color=ACCENT, zorder=4, label='Seed 9504111')
    ax.axhline(0, color=MUTED, lw=.7); ax.axvline(0, color=MUTED, lw=.7)
    ax.set_xlabel('S on RERANK200 (selection set)'); ax.set_ylabel('S on TEST561')
    st = R['selectivity_top50_rerank_test']
    ax.set_title(f"B  Frozen top 50: RERANK → TEST (r = {st['pearson']:.2f})", loc='left', fontsize=7.2)
    ax.legend(frameon=False, fontsize=6, loc='lower left')
    fig.tight_layout(); save_td(fig, 'fig_repair_selectivity_transfer')

    # Figure D: improver-set stability
    pr, pn = R['primary']['all'], R['primary_null']['all']
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.7))
    ax = axes[0]
    keys = [('n1', '≥1'), ('n2', '≥2'), ('n3', '≥3'), ('n5', '≥5')]
    x = np.arange(len(keys))
    jo = [pr[f'{k}|jaccard']['mean'] for k, _ in keys]
    jn = [pn[f'{k}|jaccard']['null_mean'] for k, _ in keys]
    ax.bar(x - .17, jo, .34, color=CAND, label='Observed', zorder=3)
    ax.bar(x + .17, jn, .34, color=MUTED, label='Item-fragility null', zorder=3)
    ax.set_xticks(x, [f'net {l}' for _, l in keys]); ax.set_ylabel('Jaccard(E_A, E_B)')
    ax.set_title('A  Improver sets, all 5,000, SEARCH100 vs SEARCH100', loc='left', fontsize=7.3)
    ax.legend(frameon=False, fontsize=6)
    ax = axes[1]
    qs = [f'top{q:g}' for q in TOPQ]
    x = np.arange(len(qs))
    ax.bar(x - .25, [pr[f'{q}|overlap']['mean'] for q in qs], .25, color=CAND, label='Observed', zorder=3)
    ax.bar(x, [pn[f'{q}|overlap']['null_mean'] for q in qs], .25, color=MUTED, label='Item-fragility null', zorder=3)
    ax.bar(x + .25, [pr[f'{q}|random_expectation']['mean'] for q in qs], .25, color='#d8d6d0', label='Random', zorder=3)
    ax.set_xticks(x, ['top 1%', 'top 2%', 'top 5%', 'top 10%']); ax.set_ylabel('Overlap of top-q sets')
    ax.set_title('B  Rank-based overlap (random tie-breaking)', loc='left', fontsize=7.3)
    ax.legend(frameon=False, fontsize=6)
    fig.tight_layout(); save_td(fig, 'fig_improver_set_stability')

    tables(R)


def lift_fmt(p):
    """Pooled lift, or '--' when the mean joint improver count is below 2 candidates (ratio not interpretable)."""
    if not p or np.isnan(p['lift']) or p['joint_count'] < 2:
        return '--'
    return f"{p['lift']:.2f}"


def f3(x):
    return '--' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.3f}'


def tables(R):
    rows = []
    for n in [int(k) for k in R['size']]:
        for key, lab in (('gt0', '$>0$'), ('pp1', '$\\ge$+1 pp'), ('pp3', '$\\ge$+3 pp'), ('pp5', '$\\ge$+5 pp')):
            s, nn = R['size'][str(n)], R['size_null'][str(n)]
            g = lambda k: s.get(f'{key}|{k}', {}) or {}
            nl = nn.get(f'{key}|lift') or {}
            pt, pq = pooled(s, key), pooled_null(nn, key)
            rows.append([str(n) if key == 'gt0' else '', f"{lab} ($\\ge${R['size_cuts'][str(n)][key]})", f3(pt['rho_A']), f3(pt['rho_B']),
                         f3(pt['joint']), f3(pt['tau']), lift_fmt(pt), lift_fmt(pq), f3(g('jaccard').get('mean'))])
        rows.append('MIDRULE')
    p, pn = R['primary']['all'], R['primary_null']['all']
    for key, lab in (('gt0', '$>0$'), ('pp3', '$\\ge$+3 pp'), ('pp5', '$\\ge$+5 pp')):
        pt, pq = pooled(p, key, 5000), pooled_null(pn, key, 5000)
        rows.append(['100 (5,000)' if key == 'gt0' else '', f"{lab} ($\\ge${R['primary_cuts'][key]})", f3(pt['rho_A']), f3(pt['rho_B']),
                     f3(pt['joint']), f3(pt['tau']), lift_fmt(pt), lift_fmt(pq), f3(p[f'{key}|jaccard']['mean'])])
    write_tex('table_transfer_density', ['$n$', 'Threshold (net questions)', '$\\rho_A$', '$\\rho_B$', 'Joint $\\rho$', '$\\tau_{A\\to B}$', 'Lift', 'Null lift', 'Jaccard'], rows,
              'POST HOC. Empirical improvement density on one question sample ($\\rho_A$), on a disjoint sample ($\\rho_B$), jointly, the empirical transfer '
              'rate $\\tau_{A\\to B}=P(g_B\\ge m\\mid g_A\\ge m)$ and lift $\\tau/\\rho_B$ (1 = no persistence), pooled over split trials ($\\tau$ = mean joint density / mean $\\rho_A$; '
              'lift omitted where the mean joint improver count is below 2 candidates). Upper block: 500 precommitted audit candidates on '
              'stratified disjoint splits of the 400 SEARCH+RERANK questions (2,000 splits per $n$). Last block: all 5,000 candidates, SEARCH100 vs SEARCH100 '
              '(5,000 splits). Null: $\\sigma$-stratified per-question permutation of candidate identity. Means over split trials; these describe one observed '
              'prediction matrix, not repeated searches.', 'tab:transfer-density', colspec='llrrrrrrr')

    rows = []
    for M in [int(k) for k in R['replay_M']]:
        d, nl = R['replay_M'][str(M)], R['replay_null'][str(M)]
        rows.append([str(M), str(min(50, max(5, round(.1 * M)))), f"{d['search_gain']['mean']:+.2f}", f"{d['rerank_gain']['mean']:+.2f}",
                     f"{d['holdout_gain']['mean']:+.2f} [{d['holdout_gain']['q025']:+.1f}, {d['holdout_gain']['q975']:+.1f}]",
                     f"{d['optimism_gap']['mean']:+.2f}", f"{d['p_holdout_gt0']:.2f}", f"{d['p_holdout_ge3pp']:.2f}",
                     f"{nl['holdout_gain']['mean']:+.2f}", f"{nl['p_holdout_gt0']:.2f}"])
    write_tex('table_selection_replay', ['$M$', '$K$', 'SEARCH gain', 'RERANK gain', 'HOLDOUT gain [95\\% of trials]', 'Optimism', 'P(H$>$0)', 'P(H$\\ge$3pp)',
                                         'Null H gain', 'Null P(H$>$0)'], rows,
              'POST HOC replay of the two-stage procedure on the 500 precommitted audit candidates: SEARCH100 $\\to$ top $K$ $\\to$ RERANK100 winner $\\to$ '
              'fresh HOLDOUT200 (disjoint, stratified, 400 train-side questions; hash tie-break as in the study; 2,000 trials per $M$, random pools for $M<500$). '
              'Gains in pp over base. Optimism = RERANK $-$ HOLDOUT gain of the winner. Null: item-fragility permutation.', 'tab:selection-replay', colspec='rrrrrrrrrr')

    if 'replay_holdout_by_origin' in R:
        o = R['replay_holdout_by_origin']
        rows = [['SEARCH200-origin', f"{o['holdout_search_origin_n']['mean']:.0f}", f"{o['win_sel_search_origin_pp']['mean']:+.2f}",
                 f"{o['win_hold_search_pp']['mean']:+.2f} [{o['win_hold_search_pp']['q025']:+.1f}, {o['win_hold_search_pp']['q975']:+.1f}]",
                 f"{o['pool_hold_search_pp']['mean']:+.2f}", f"{o['p_win_hold_search_gt0']:.2f}"],
                ['RERANK200-origin', f"{200 - o['holdout_search_origin_n']['mean']:.0f}", f"{o['win_sel_rerank_origin_pp']['mean']:+.2f}",
                 f"{o['win_hold_rerank_pp']['mean']:+.2f} [{o['win_hold_rerank_pp']['q025']:+.1f}, {o['win_hold_rerank_pp']['q975']:+.1f}]",
                 f"{o['pool_hold_rerank_pp']['mean']:+.2f}", f"{o['p_win_hold_rerank_gt0']:.2f}"]]
        write_tex('table_replay_holdout_origin', ['Question origin', 'HOLDOUT items', 'Winner gain on selection items', 'Winner HOLDOUT gain [95\\%]',
                                                  'Pool mean HOLDOUT', 'P(winner $>$ 0)'], rows,
                  'POST HOC. Primary replay ($M$=500, $K$=50, 5,000 trials; winners identical to Table~\\ref{tab:selection-replay}) with the winner\'s gain split '
                  'by the origin of the questions. SEARCH200 and RERANK200 are both drawn from the official train split with identical subtask$\\times$answer '
                  'strata. Gains in pp over base.', 'tab:replay-origin', colspec='lrrrrr')

    rows = []
    s9 = R['selectivity_9504111']
    for ph in ('SEARCH', 'RERANK', 'TEST'):
        v = s9[ph]
        rows.append([f'9504111, {ph}', f"{v['repair_rate']:.3f} ({v['repairs']}/{v['base_wrong']})", f"{v['damage_rate']:.3f}", f"{v['selectivity']:+.3f}", '--', '--'])
    rows.append('MIDRULE')
    sa = R['selectivity_audit']
    rows.append(['Audit 500: SEARCH vs RERANK', '--', '--', '--', f"{sa['all']['pearson']:.2f} / {sa['all']['spearman']:.2f}",
                 f"{sa['all']['perm_within_sigma']['null_mean']:.2f} ({sa['all']['perm_within_sigma']['percentile_of_observed']:.0f})"])
    for s in SIGMAS:
        rows.append([f'\\quad within $\\sigma$={s:g}', '--', '--', '--', f"{sa[str(s)]['pearson']:.2f} / {sa[str(s)]['spearman']:.2f}",
                     f"{sa[str(s)]['perm']['null_mean']:.2f} ({sa[str(s)]['perm']['percentile_of_observed']:.0f})"])
    st = R['selectivity_top50_rerank_test']
    rows.append(['Top 50: RERANK vs TEST', '--', '--', '--', f"{st['pearson']:.2f} / {st['spearman']:.2f}",
                 f"{st['perm']['null_mean']:.2f} ({st['perm']['percentile_of_observed']:.0f})"])
    write_tex('table_repair_selectivity', ['', 'Repair rate', 'Damage rate', 'Selectivity $S$', 'Transfer $r$ (Pearson / Spearman)', 'Perm.\\ null mean (pct)'], rows,
              'POST HOC. Repair rate $=P(\\mathrm{correct}\\mid\\mathrm{base\\ wrong})$, damage rate $=P(\\mathrm{wrong}\\mid\\mathrm{base\\ correct})$, $S$ = repair $-$ damage, '
              'recomputed from raw predictions. Transfer: correlation of per-candidate $S$ across disjoint question sets; null permutes candidate identity '
              '(within $\\sigma$ for pooled rows).', 'tab:repair-selectivity', colspec='lrrrrr')

    rows = []
    for g in [str(s) for s in SIGMAS] + ['all']:
        p, pn = R['primary'][g], R['primary_null'][g]
        for key, lab in (('gt0', '$>0$'), ('pp3', '$\\ge$+3 pp')):
            Ng = 5000 if g == 'all' else 1250
            pt, pq = pooled(p, key, Ng), pooled_null(pn, key, Ng)
            rows.append([g if key == 'gt0' else '', lab, f3(pt['rho_A']), f3(pt['rho_B']), f3(pt['joint']), f3(pt['tau']), lift_fmt(pt), lift_fmt(pq),
                         f"{p['pearson']['mean']:.3f}" if key == 'gt0' else '', f"{pn['pearson']['null_mean']:.3f}" if key == 'gt0' else ''])
    write_tex('table_sigma_transfer', ['$\\sigma$', 'Threshold', '$\\rho_A$', '$\\rho_B$', 'Joint', '$\\tau_{A\\to B}$', 'Lift', 'Null lift', 'Gain $r$', 'Null $r$'], rows,
              'POST HOC. Transfer density by $\\sigma$: 1,250 candidates per $\\sigma$, SEARCH100 vs SEARCH100 (5,000 stratified splits); null as before.',
              'tab:sigma-transfer', colspec='llrrrrrrrr')

    rows = []
    for k in TOPK:
        p, pn = R['primary']['all'], R['primary_null']['all']
        rows.append([str(k), f"{p[f'topK{k}|mean_gain_B']['mean']:+.2f}", f"{p[f'topK{k}|pool_mean_gain_B']['mean']:+.2f}", f"{p[f'topK{k}|mean_rank_pct_B']['mean']:.1f}",
                     f"{p[f'topK{k}|frac_in_topK_B']['mean']:.3f}", f"{p[f'topK{k}|enrichment']['mean']:.2f}", f"{pn[f'topK{k}|enrichment']['null_mean']:.2f}"])
    write_tex('table_rank_persistence', ['Top $K$ on A', 'Mean net gain on B (questions)', 'Pool mean on B', 'Mean rank on B (pct, 50 = random)', 'Share in top $K$ on B',
                                         'Enrichment', 'Null enrichment'], rows,
              'POST HOC. Rank persistence for all 5,000 candidates, SEARCH100 (A) vs SEARCH100 (B), random tie-breaking, 5,000 splits. Enrichment = share / ($K/N$).',
              'tab:rank-persistence', colspec='rrrrrrr')

    rows = []
    for s in SIGMAS:
        v = R['behavior_vs_task'][str(s)]
        rows.append([f'{s:g}', f"{v['answer_cka']['observed']:.3f} / {v['answer_cka']['null_mean']:.3f}", f"{v['direction_only_cka']['observed']:.3f} / {v['direction_only_cka']['null_mean']:.3f}",
                     f"{v['distance_to_base_r']:.2f}", f"{v['gain_correlation']['observed']:.2f} ({v['gain_correlation']['percentile_of_observed']:.0f})",
                     f"{v['selectivity_correlation']['observed']:.2f} ({v['selectivity_correlation']['percentile_of_observed']:.0f})"])
    write_tex('table_behavior_vs_task', ['$\\sigma$', 'Answer CKA obs./null', 'Direction-only CKA obs./null', 'Distance-to-base $r$', 'Gain $r$ (null pct)',
                                         'Selectivity $r$ (null pct)'], rows,
              'POST HOC. 125 audit candidates per $\\sigma$, SEARCH200 vs RERANK200: persistence of behavioral geometry (from the behavioral-diversity audit) '
              'versus persistence of task advantage. Null percentiles from candidate-identity permutation.', 'tab:behavior-vs-task', colspec='lrrrrr')


def replay_origin_stage():
    """Supplementary: re-run the primary replay (same seeds, verified identical winners) and split each winner's
    HOLDOUT gain by question origin (SEARCH200-origin vs RERANK200-origin), against the pool mean on the same items."""
    df, lock, proto = load_inputs()
    ids = json.loads((OUT / 'candidate_ids.json').read_text())
    audit = ids['audit500']
    meta = pd.read_csv(OUT / 'question_pool_400.csv')
    PaS, bS_, g_, _ = matrix(df, 'SEARCH', audit)
    PaR, bR_, gR, _ = matrix(df, 'RERANK', audit)
    C400 = np.hstack([PaS == g_, PaR == gR]); b400 = np.concatenate([bS_ == g_, bR_ == gR])
    D = C400.astype(np.int16) - b400.astype(np.int16)
    codes400 = pd.factorize(meta.subtask + meta.answer)[0]
    is_search = (meta.phase == 'SEARCH').to_numpy()
    hrank = np.argsort(np.argsort([hashlib.sha256(('visual-rank-v1:' + c).encode()).hexdigest() for c in audit]))
    saved = pd.read_csv(OUT / 'replay_primary_trials.csv')
    rng = np.random.default_rng([SEED, 7]); rec = []
    for t in range(len(saved)):
        S, Rr, H = split_replay(codes400, rng)
        gS, gR_ = D[:, S].sum(1), D[:, Rr].sum(1)
        o = np.lexsort((hrank, -gS)); top = o[:50]
        w = top[np.lexsort((hrank[top], -gR_[top]))[0]]
        hs, hr = H[is_search[H]], H[~is_search[H]]
        rec.append({'winner': int(w), 'n_hold_search': len(hs), 'n_hold_rerank': len(hr),
                    'win_hold_search_pp': 100 * D[w, hs].sum() / len(hs), 'win_hold_rerank_pp': 100 * D[w, hr].sum() / len(hr),
                    'pool_hold_search_pp': 100 * D[:, hs].sum(1).mean() / len(hs), 'pool_hold_rerank_pp': 100 * D[:, hr].sum(1).mean() / len(hr),
                    'win_sel_search_origin_pp': 100 * D[w, np.concatenate([S, Rr])[is_search[np.concatenate([S, Rr])]]].mean(),
                    'win_sel_rerank_origin_pp': 100 * D[w, np.concatenate([S, Rr])[~is_search[np.concatenate([S, Rr])]]].mean()})
    r = pd.DataFrame(rec)
    same = bool((r.winner.values == saved.winner.values).all())
    out = {'winners_identical_to_primary_replay': same, 'trials': len(r), 'holdout_search_origin_n': agg(r.n_hold_search),
           **{k: agg(r[k]) for k in ('win_hold_search_pp', 'win_hold_rerank_pp', 'pool_hold_search_pp', 'pool_hold_rerank_pp',
                                    'win_sel_search_origin_pp', 'win_sel_rerank_origin_pp')},
           'p_win_hold_search_gt0': float((r.win_hold_search_pp > 0).mean()), 'p_win_hold_rerank_gt0': float((r.win_hold_rerank_pp > 0).mean())}
    R = json.loads((OUT / 'transfer_density_results.json').read_text())
    R['replay_holdout_by_origin'] = out
    (OUT / 'transfer_density_results.json').write_text(json.dumps(R, indent=1, default=lambda x: x.item() if hasattr(x, 'item') else str(x)))
    print(json.dumps({k: (round(v['mean'], 3) if isinstance(v, dict) else v) for k, v in out.items()}, indent=1))


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what == 'origin':
        replay_origin_stage()
    if what in ('compute', 'all'):
        compute()
    if what in ('figures', 'all'):
        figures_and_tables()

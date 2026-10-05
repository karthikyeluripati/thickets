"""Behavioral diversity of the N = 5,000 weight-space neighbourhood (CPU only, deterministic).

Spectral "effective rank" here is an EFFECTIVE BEHAVIORAL DIMENSION of the observed prediction variation,
not a count of independent models. Inputs: the paper master table (raw outputs re-parsed), the search lock,
the protocol's precommitted density-audit indices, and raw candidate files for spot checks.
Outputs: results/paper-analysis/behavioral-diversity/, paper/figures/behavioral-diversity/, paper/tables/.

Usage: python paper/figures/scripts/behavioral_diversity.py [compute|figures|all]
"""
import gzip
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from common import ACCENT, BASE_C, CAND, GOOD, INK2, MASTER, MUTED, ROOT, SIGMAS, load, write_tex  # noqa: E402

OUT = Path('results/paper-analysis/behavioral-diversity')
FIGD = Path('paper/figures/behavioral-diversity')
L = 'ABCD'
SEED = 20261006
import os
N_NULL, N_COMM, N_PERM = (5, 50, 20) if os.environ.get('BD_QUICK') else (1000, 10000, 1000)
SIG_COL = {0.00025: '#9cc2ee', 0.0005: '#5a9be0', 0.001: CAND, 0.002: '#123f7a'}


# ----------------------------------------------------------------------------------------------- data
def load_inputs():
    df = pd.read_parquet(MASTER, columns=['candidate_id', 'seed', 'sigma', 'manifest_index', 'in_density_audit', 'phase', 'example_id',
                                          'true_option', 'base_prediction', 'candidate_prediction'])
    lock = load(ROOT / 'locks/search.json')
    proto = json.loads(Path('experiments/perspective_taking_n5000_protocol.json').read_text())
    return df, lock, proto


def matrix(df, phase, ids):
    """(len(ids) x Q) answer codes 0-3 in the base row order, plus base and gold codes."""
    x = df[df.phase == phase]
    b = x[x.candidate_id == 'BASE']
    qs = b.example_id.astype(str).tolist()
    base = np.array([L.index(v) for v in b.base_prediction.astype(str)], np.int8)
    gold = np.array([L.index(v) for v in b.true_option.astype(str)], np.int8)
    y = x[x.candidate_id.isin(ids)]
    piv = pd.DataFrame({'cid': y.candidate_id.astype(str).values, 'eid': y.example_id.astype(str).values,
                        'p': y.candidate_prediction.astype(str).map(L.index).values}).pivot(index='cid', columns='eid', values='p')
    if set(piv.index) != set(ids) or piv.isna().any().any() or piv.shape[1] != len(qs):
        raise SystemExit(f'INCOMPLETE {phase} matrix: {piv.shape}, missing {len(set(ids) - set(piv.index))}')
    return piv.loc[ids, qs].to_numpy(np.int8), base, gold, qs


def onehot(P):
    n, q = P.shape
    o = np.zeros((n, q, 4), np.float32)
    o[np.arange(n)[:, None], np.arange(q)[None, :], P] = 1
    return o.reshape(n, q * 4)


def d_corr(P, base, gold):
    return ((P == gold).astype(np.float32) - (base == gold).astype(np.float32))


def d_ans(P, base):
    ob = np.zeros((len(base), 4), np.float32); ob[np.arange(len(base)), base] = 1
    return onehot(P) - ob.reshape(1, -1)


# ------------------------------------------------------------------------------------------ spectra
def eig(D, standardize=False):
    Z = D.astype(np.float64)
    Z = Z - Z.mean(0)
    sd = Z.std(0)
    keep = sd > 1e-9
    Z = Z[:, keep]
    if standardize:
        Z = Z / sd[keep]
    n, p = Z.shape
    G = Z.T @ Z if p <= n else Z @ Z.T
    lam = np.clip(np.linalg.eigvalsh(G)[::-1], 0, None)
    return lam, int(keep.sum())


def spec_metrics(lam, n, nonconst, ceiling, curve=False):
    tot = lam.sum()
    if tot <= 0:
        return {'r_entropy': 0., 'r_PR': 0.}
    p = lam / tot
    pz = p[p > 0]
    cum = np.cumsum(p)
    k = lambda t: int(np.searchsorted(cum, t - 1e-12) + 1)
    m = {'r_entropy': float(np.exp(-(pz * np.log(pz)).sum())), 'r_PR': float(1 / (p ** 2).sum()),
         'numerical_rank': int((lam > lam[0] * 1e-10).sum()), 'rank_ceiling': int(ceiling), 'nonconstant_features': nonconst, 'n': n,
         'k50': k(.5), 'k80': k(.8), 'k90': k(.9), 'k95': k(.95), 'k99': k(.99),
         'pc1': float(p[0]), 'top5': float(p[:5].sum()), 'top10': float(p[:10].sum()), 'top20': float(p[:20].sum())}
    if curve:
        m['cum'] = cum[:600].tolist()
    return m


def ceilings(P, base, gold):
    n = len(P)
    mj = np.array([len(np.unique(P[:, j])) for j in range(P.shape[1])])
    corr_nc = int((d_corr(P, base, gold).std(0) > 0).sum())
    return min(n - 1, int((mj - 1).sum())), min(n - 1, corr_nc)


def full_spectrum(P, base, gold, curve=False):
    """Both representations x raw/standardized."""
    ca, cc = ceilings(P, base, gold)
    out = {}
    for rep, D, c in (('answer', d_ans(P, base), ca), ('correctness', d_corr(P, base, gold), cc)):
        for st in (False, True):
            lam, nc = eig(D, st)
            out[f"{rep}_{'std' if st else 'raw'}"] = spec_metrics(lam, len(P), nc, c, curve)
    return out


def permute_within(P, strata, rng):
    """Sigma-stratified per-question permutation: an independent shuffle of candidate identity for every question."""
    Q = P.copy()
    for s in np.unique(strata):
        rows = np.where(strata == s)[0]
        perm = np.argsort(rng.random((len(rows), P.shape[1])), axis=0)
        Q[rows] = np.take_along_axis(P[rows], perm, axis=0)
    return Q


def null_summary(obs, vals):
    v = np.asarray(vals, float)
    return {'observed': obs, 'null_mean': float(v.mean()), 'null_sd': float(v.std(ddof=1)), 'null_median': float(np.median(v)),
            'null_q025': float(np.quantile(v, .025)), 'null_q975': float(np.quantile(v, .975)),
            'percentile_of_observed': float(100 * (v <= obs).mean()), 'observed_over_null_mean': float(obs / v.mean()) if v.mean() else None}


# --------------------------------------------------------------------------------------- committees
def committee_metrics(P, base, gold, with_rank=True):
    n, q = P.shape
    oh = onehot(P)
    iu = np.triu_indices(n, 1)
    dis = 1 - (oh @ oh.T)[iu] / q
    W = (P != gold).astype(np.float32)
    inter = W @ W.T; c = W.sum(1); union = c[:, None] + c[None, :] - inter
    jac = np.where(union > 0, inter / np.maximum(union, 1), 1.)[iu]
    m = {'pairwise_disagreement': float(dis.mean()), 'error_jaccard': float(jac.mean()), 'hamming_from_base': float((P != base).mean())}
    if with_rank:
        for rep, D in (('answer', d_ans(P, base)), ('correctness', d_corr(P, base, gold))):
            for st in (False, True):
                lam, nc = eig(D, st)
                mm = spec_metrics(lam, n, nc, 0)
                m[f"{rep}_{'std' if st else 'raw'}_r_entropy"] = mm['r_entropy']
                m[f"{rep}_{'std' if st else 'raw'}_r_PR"] = mm['r_PR']
    return m, dis


# ------------------------------------------------------------------------------------------- compute
def compute():
    OUT.mkdir(parents=True, exist_ok=True)
    df, lock, proto = load_inputs()
    R = {'seed': SEED, 'n_null': N_NULL, 'n_committees': N_COMM, 'n_perm': N_PERM}

    # Part 0: verification
    recs = {r['candidate']['candidate_id']: r for r in lock['records']}
    ids = [r['candidate']['candidate_id'] for r in sorted(lock['records'], key=lambda r: r['index'])]
    sigma = np.array([recs[c]['candidate']['sigma'] for c in ids])
    top50 = list(lock['ranked_ids'][:50])
    audit_idx = set(proto['density_audit']['indices'])
    audit = [c for c in ids if recs[c]['index'] in audit_idx]
    s_master = df[(df.phase == 'SEARCH') & (df.candidate_id != 'BASE')].drop_duplicates('candidate_id')
    test_ids = set(df[(df.phase == 'TEST') & (df.candidate_id != 'BASE')].candidate_id.astype(str))
    checks = {'search_unique_candidates': len(set(ids)), 'records': len(lock['records']),
              'master_search_candidates_equal_lock': set(s_master.candidate_id.astype(str)) == set(ids),
              'sigma_counts': {str(s): int((sigma == s).sum()) for s in SIGMAS},
              'top50_equals_lock_top50_field': top50 == [c['candidate']['candidate_id'] for c in lock['top50']],
              'top50_sigma_mix': {str(s): int(sum(recs[c]['candidate']['sigma'] == s for c in top50)) for s in SIGMAS},
              'audit_count': len(audit), 'audit_sigma': {str(s): int(sum(recs[c]['candidate']['sigma'] == s for c in audit)) for s in SIGMAS},
              'audit_equals_master_flag': set(audit) == set(s_master[s_master.in_density_audit].candidate_id.astype(str)),
              'test_candidates_equal_top50': test_ids == set(top50), 'top50_in_audit': len(set(top50) & set(audit))}
    P, base, gold, qs = matrix(df, 'SEARCH', ids)
    checks['search_matrix_shape'] = list(P.shape)
    checks['search_scores_match_lock'] = bool(all(int((P[i] == gold).sum()) == recs[c]['correct_count'] for i, c in enumerate(ids)))
    checks['base_search_correct'] = int((base == gold).sum())
    # Independent spot check against raw candidate files (25 candidates, fixed seed) and raw base outputs.
    raw_base = [o['text'].strip().upper()[:1] for o in load(ROOT / 'baseline/base.json.gz')['search']['outputs']]
    checks['base_matches_raw'] = [L[b] for b in base] == raw_base
    rng = np.random.default_rng(SEED)
    spot = set(rng.choice(len(ids), 25, replace=False).tolist())
    want = {ids[i]: i for i in spot}
    ok = 0
    for d in sorted(ROOT.glob('search-shard-*')):
        for f in (d / 'candidates').glob('*.json.gz'):
            if f.name[:-8] in want:
                raw = load(f)
                pr = [o['text'].strip().upper()[:1] for o in raw['splits']['search']['outputs']]
                ok += pr == [L[v] for v in P[want[f.name[:-8]]]]
    checks['raw_spot_check_25_identical'] = ok == 25
    R['checks'] = checks
    print(json.dumps(checks, indent=1), flush=True)
    bad = [k for k, v in checks.items() if v is False]
    if bad or checks['sigma_counts'] != {str(s): 1250 for s in SIGMAS} or checks['audit_count'] != 500:
        raise SystemExit(f'STOP: verification failed {bad}')
    (OUT / 'candidate_ids.json').write_text(json.dumps({'search_order': ids, 'sigma': sigma.tolist(), 'top50': top50, 'audit500': audit,
                                                        'search_question_ids': qs}, indent=0))

    # Part 2: behavioral distance
    Dc = d_corr(P, base, gold)
    bw = base != gold
    per = {'answers_changed': (P != base).sum(1), 'correctness_changes': (Dc != 0).sum(1), 'repairs': (Dc > 0).sum(1),
           'regressions': (Dc < 0).sum(1), 'wrong_to_wrong_changes': ((P != base) & (P != gold) & bw).sum(1)}
    def dist(v):
        q = np.percentile(v, [5, 25, 50, 75, 95, 99])
        return {'mean': float(v.mean()), 'sd': float(v.std(ddof=1)), 'min': int(v.min()), 'max': int(v.max()),
                **{f'p{p}': float(x) for p, x in zip((5, 25, 50, 75, 95, 99), q)}}
    R['distance'] = {'all': {k: dist(v) for k, v in per.items()}}
    for s in SIGMAS:
        R['distance'][str(s)] = {k: dist(v[sigma == s]) for k, v in per.items()}
    R['distance']['identical_to_base'] = {'all': int((per['answers_changed'] == 0).sum()), **{str(s): int((per['answers_changed'][sigma == s] == 0).sum()) for s in SIGMAS}}
    pd.DataFrame({'candidate_id': ids, 'sigma': sigma, **per}).to_csv(OUT / 'per_candidate_distance.csv', index=False)

    # Parts 3-6: spectra + sigma-stratified per-question permutation null
    groups = {'all': np.arange(len(ids)), **{str(s): np.where(sigma == s)[0] for s in SIGMAS}}
    R['spectrum'] = {g: full_spectrum(P[ix], base, gold, curve=(g == 'all')) for g, ix in groups.items()}
    keys = ('r_entropy', 'r_PR', 'k90', 'k95')
    nulls = {g: {r: {k: [] for k in keys} for r in R['spectrum'][g]} for g in groups}
    null_curves = {'answer_raw': [], 'correctness_raw': []}
    rng = np.random.default_rng([SEED, 1])
    for t in range(N_NULL):
        Q = permute_within(P, sigma, rng)
        for g, ix in groups.items():
            sp = full_spectrum(Q[ix], base, gold, curve=(g == 'all'))
            for r, m in sp.items():
                for k in keys:
                    nulls[g][r][k].append(m[k])
                if g == 'all' and r in null_curves:
                    null_curves[r].append(m['cum'])
        if t % 100 == 0:
            print('null', t, flush=True)
    R['null'] = {g: {r: {k: null_summary(R['spectrum'][g][r][k], v) for k, v in d.items()} for r, d in gd.items()} for g, gd in nulls.items()}
    curves = {}
    for r, cs in null_curves.items():
        Lm = min(len(c) for c in cs)
        A = np.array([c[:Lm] for c in cs])
        curves[r] = {'median': np.median(A, 0).tolist(), 'q025': np.quantile(A, .025, 0).tolist(), 'q975': np.quantile(A, .975, 0).tolist()}
    R['null_curves'] = curves

    # Part 7: SEARCH top 50 vs sigma-matched random committees
    pos = {c: i for i, c in enumerate(ids)}
    t_ix = np.array([pos[c] for c in top50])
    obs, _ = committee_metrics(P[t_ix], base, gold)
    mix = {s: int((sigma[t_ix] == s).sum()) for s in SIGMAS}
    rng = np.random.default_rng([SEED, 2])
    pools = {s: np.where(sigma == s)[0] for s in SIGMAS}
    vals = {k: [] for k in obs}
    for t in range(N_COMM):
        ix = np.concatenate([rng.choice(pools[s], k, replace=False) for s, k in mix.items() if k])
        m, _ = committee_metrics(P[ix], base, gold)
        for k in vals:
            vals[k].append(m[k])
    R['selection'] = {'top50_sigma_mix': {str(s): k for s, k in mix.items()}, 'metrics': {k: null_summary(obs[k], v) for k, v in vals.items()}}
    np.savez_compressed(OUT / 'selection_null_samples.npz', **{k: np.array(v) for k, v in vals.items()})
    # within-committee permutation null for the SEARCH top 50 (structure inside the selected group)
    R['top50_search_spectrum'] = full_spectrum(P[t_ix], base, gold)
    R['top50_search_null'] = committee_perm_null(P[t_ix], base, gold, sigma[t_ix], [SEED, 3])

    # Part 8: SELECTED_COMMITTEE_TEST_GEOMETRY
    PT, bT, gT, _ = matrix(df, 'TEST', top50)
    mT, disT = committee_metrics(PT, bT, gT)
    spT = full_spectrum(PT, bT, gT, curve=True)
    oh = onehot(PT); cnt = oh.reshape(50, -1, 4).sum(0)
    R['test_top50'] = {'label': 'SELECTED_COMMITTEE_TEST_GEOMETRY', 'metrics': mT, 'spectrum': spT,
                       'null': committee_perm_null(PT, bT, gT, sigma[t_ix], [SEED, 4]),
                       'pairwise_disagreement_pct': {f'p{p}': float(np.percentile(disT, p)) for p in (0, 5, 25, 50, 75, 95, 100)},
                       'distance_to_base_answers': {f'p{p}': float(np.percentile((PT != bT).sum(1), p)) for p in (0, 25, 50, 75, 100)},
                       'base_is_majority_share': float(np.mean(cnt.argmax(1) == bT)), 'majority_correct': int((cnt.argmax(1) == gT).sum()),
                       'base_correct': int((bT == gT).sum())}

    # Part 9: cross-split geometry for the 500 precommitted audit candidates
    a_ix = np.array([pos[c] for c in audit])
    PS = P[a_ix]
    PR, bR, gR, _ = matrix(df, 'RERANK', audit)
    asg = sigma[a_ix]
    hS, hR = (PS != base).mean(1), (PR != bR).mean(1)
    cross = {'distance_to_base': {'all': corr_pair(hS, hR)}}
    for s in SIGMAS:
        cross['distance_to_base'][str(s)] = corr_pair(hS[asg == s], hR[asg == s])
    dS = 1 - (onehot(PS) @ onehot(PS).T) / PS.shape[1]
    dR = 1 - (onehot(PR) @ onehot(PR).T) / PR.shape[1]
    cross['pairwise'] = mantel(dS, dR, asg, [SEED, 5])
    cross['pairwise_within_sigma'] = {}
    for s in SIGMAS:
        m = asg == s
        cross['pairwise_within_sigma'][str(s)] = mantel(dS[np.ix_(m, m)], dR[np.ix_(m, m)], asg[m], [SEED, 6, int(s * 1e5)], n=N_PERM)
    cross['cka'] = {}
    for rep, fS, fR in (('answer', d_ans(PS, base), d_ans(PR, bR)), ('correctness', d_corr(PS, base, gold), d_corr(PR, bR, gR))):
        cross['cka'][rep] = {'all': cka_perm(fS, fR, asg, [SEED, 7, len(rep)])}
        for s in SIGMAS:
            m = asg == s
            cross['cka'][rep][str(s)] = cka_perm(fS[m], fR[m], asg[m], [SEED, 8, len(rep), int(s * 1e5)])
    R['cross_split'] = cross
    pd.DataFrame({'candidate_id': audit, 'sigma': asg, 'search_hamming': hS, 'rerank_hamming': hR}).to_csv(OUT / 'audit_cross_split_distance.csv', index=False)

    # Part 10: unique patterns
    R['unique'] = {'answer_vectors': uniq(P, lambda u: (u != base).sum(1)), 'changed_answer_sets': uniq(P != base, lambda u: u.sum(1)),
                   'correctness_signatures': uniq(Dc.astype(np.int8), lambda u: (u != 0).sum(1))}
    G = onehot(P) @ onehot(P).T
    np.fill_diagonal(G, -1)
    nn = P.shape[1] - G.max(1)
    R['unique']['nearest_neighbour_answer_distance'] = {'mean': float(nn.mean()), **{f'p{p}': float(np.percentile(nn, p)) for p in (5, 25, 50, 75, 95)},
                                                        'share_with_neighbour_within_1': float((nn <= 1).mean()), 'share_with_neighbour_within_2': float((nn <= 2).mean())}
    (OUT / 'behavioral_diversity_results.json').write_text(json.dumps(R, indent=1, default=float))
    print('COMPUTE_DONE', flush=True)


def committee_perm_null(P, base, gold, strata, seed, n=N_NULL):
    rng = np.random.default_rng(seed)
    obs = full_spectrum(P, base, gold)
    v = {r: {'r_entropy': [], 'r_PR': [], 'k90': []} for r in obs}
    for _ in range(n):
        sp = full_spectrum(permute_within(P, strata, rng), base, gold)
        for r in v:
            for k in v[r]:
                v[r][k].append(sp[r][k])
    return {r: {k: null_summary(obs[r][k], x) for k, x in d.items()} for r, d in v.items()}


def corr_pair(a, b):
    return {'n': int(len(a)), 'pearson': float(stats.pearsonr(a, b)[0]), 'pearson_p': float(stats.pearsonr(a, b)[1]),
            'spearman': float(stats.spearmanr(a, b)[0]), 'spearman_p': float(stats.spearmanr(a, b)[1])}


def mantel(dS, dR, strata, seed, n=N_PERM):
    iu = np.triu_indices(len(dS), 1)
    x, y = dS[iu], dR[iu]
    rS = stats.rankdata(x); rR_full = np.zeros_like(dR); rR_full[iu] = stats.rankdata(y); rR_full = rR_full + rR_full.T
    out = {'pairs': int(len(x)), 'pearson': float(np.corrcoef(x, y)[0, 1]), 'spearman': float(np.corrcoef(rS, rR_full[iu])[0, 1])}
    rng = np.random.default_rng(seed)
    for name, strat in (('global_perm', np.zeros(len(dS))), ('sigma_stratified_perm', strata)):
        pv, sv = [], []
        for _ in range(n):
            perm = np.arange(len(dS))
            for s in np.unique(strat):
                m = np.where(strat == s)[0]
                perm[m] = rng.permutation(m)
            pv.append(np.corrcoef(x, dR[np.ix_(perm, perm)][iu])[0, 1])
            sv.append(np.corrcoef(rS, rR_full[np.ix_(perm, perm)][iu])[0, 1])
        out[name] = {'pearson': null_summary(out['pearson'], pv), 'spearman': null_summary(out['spearman'], sv)}
    return out


def cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    return float(np.linalg.norm(X.T @ Y) ** 2 / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)))


def cka_perm(X, Y, strata, seed, n=N_PERM):
    X = (X - X.mean(0)).astype(np.float64); Y = (Y - Y.mean(0)).astype(np.float64)
    obs = cka(X, Y)
    rng = np.random.default_rng(seed)
    out = {'n': int(len(X)), 'observed': obs}
    for name, strat in (('global_perm', np.zeros(len(X))), ('sigma_stratified_perm', strata)):
        if name == 'sigma_stratified_perm' and len(np.unique(strata)) == 1:
            continue
        v = []
        for _ in range(n):
            perm = np.arange(len(X))
            for s in np.unique(strat):
                m = np.where(strat == s)[0]
                perm[m] = rng.permutation(m)
            v.append(cka(X, Y[perm]))
        out[name] = null_summary(obs, v)
    return out


def uniq(M, size):
    """Unique rows; `size(u)` gives each pattern's number of changed answers relative to base."""
    u, c = np.unique(M, axis=0, return_counts=True)
    order = np.argsort(-c, kind='stable')
    sz = size(u)
    return {'unique': int(len(u)), 'largest_group': int(c.max()), 'singletons': int((c == 1).sum()),
            'top10': [{'count': int(c[i]), 'changed_vs_base': int(sz[i])} for i in order[:10]]}


# ------------------------------------------------------------------------------------------- figures
def save_bd(fig, name):
    FIGD.mkdir(parents=True, exist_ok=True)
    plt.rcParams['svg.hashsalt'] = name
    fig.savefig(FIGD / f'{name}.pdf', metadata={'CreationDate': None})
    fig.savefig(FIGD / f'{name}.png', dpi=300)
    plt.close(fig)


def figures_and_tables():
    R = json.loads((OUT / 'behavioral_diversity_results.json').read_text())
    sp, nl = R['spectrum'], R['null']

    # Figure A: cumulative explained variance, all 5,000
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.7))
    for ax, rep, title in ((axes[0], 'correctness_raw', 'A  Correctness deviation (200 features)'),
                           (axes[1], 'answer_raw', 'B  Answer deviation (800 one-hot features)')):
        obs = np.array(sp['all'][rep]['cum']); cv = R['null_curves'][rep]
        k = np.arange(1, len(obs) + 1); kn = np.arange(1, len(cv['median']) + 1)
        ax.fill_between(kn, cv['q025'], cv['q975'], color=MUTED, alpha=.25, lw=0, label='Null 95% band')
        ax.plot(kn, cv['median'], color=BASE_C, lw=1.4, ls='--', label='Null median (σ-stratified per-question permutation)')
        ax.plot(k, obs, color=CAND, lw=1.8, label='Observed, N = 5,000')
        ax.set_xscale('log'); ax.set_ylim(0, 1.02); ax.set_xlabel('Principal components'); ax.set_title(title, loc='left')
        m, n = sp['all'][rep], nl['all'][rep]
        ax.text(.98, .05, f"r_entropy {m['r_entropy']:.1f} (null {n['r_entropy']['null_median']:.1f})\n"
                          f"r_PR {m['r_PR']:.1f} (null {n['r_PR']['null_median']:.1f})\nk90 {m['k90']} (null {n['k90']['null_median']:.0f})",
                transform=ax.transAxes, ha='right', va='bottom', fontsize=6.3, color=INK2)
    axes[0].set_ylabel('Cumulative explained variance')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(.5, -.08), fontsize=6.5)
    fig.tight_layout(rect=(0, .06, 1, 1))
    save_bd(fig, 'fig_behavioral_spectrum')

    # Figure B: effective dimension by sigma (rows: representation; cols: raw / standardized)
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.4))
    x = np.arange(4)
    for i, rep in enumerate(('answer', 'correctness')):
        for j, st in enumerate(('raw', 'std')):
            ax = axes[i, j]; r = f'{rep}_{st}'
            o = [sp[str(s)][r]['r_entropy'] for s in SIGMAS]
            nm = [nl[str(s)][r]['r_entropy']['null_median'] for s in SIGMAS]
            lo = [nl[str(s)][r]['r_entropy']['null_q025'] for s in SIGMAS]; hi = [nl[str(s)][r]['r_entropy']['null_q975'] for s in SIGMAS]
            ax.fill_between(x, lo, hi, color=MUTED, alpha=.25, lw=0)
            ax.plot(x, nm, color=BASE_C, ls='--', marker='s', ms=4, lw=1.2, label='Null median (95% band)')
            ax.plot(x, o, color=CAND, marker='o', ms=5, label='Observed')
            ax.set_xticks(x, [f'{s:g}' for s in SIGMAS]); ax.set_ylim(bottom=0)
            ax.set_title(f"{'ABCD'[2 * i + j]}  {rep}, {'raw' if st == 'raw' else 'standardized'}", loc='left', fontsize=7.5)
            if j == 0: ax.set_ylabel('Entropy effective rank')
            if i == 1: ax.set_xlabel('σ (1,250 candidates each)')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(.5, -.03), fontsize=6.8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    save_bd(fig, 'fig_effective_dimension_sigma')

    # Figure C: selection vs sigma-matched random committees
    S = np.load(OUT / 'selection_null_samples.npz'); sel = R['selection']['metrics']
    SM = np.load(OUT / 'selection_score_matched_samples.npz'); ssel = R['selection_score_matched']['metrics']
    mets = (('answer_raw_r_entropy', 'Answer effective rank (raw)'), ('pairwise_disagreement', 'Mean pairwise disagreement'),
            ('correctness_raw_r_entropy', 'Correctness effective rank (raw)'))
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5))
    for ax, (k, lab) in zip(axes, mets):
        bins = np.histogram_bin_edges(np.concatenate([S[k], SM[k]]), 50)
        ax.hist(S[k], bins=bins, color='#9cc2ee', edgecolor='white', lw=.3, density=True, label='σ-matched random committees')
        ax.hist(SM[k], bins=bins, color=MUTED, alpha=.55, edgecolor='white', lw=.3, density=True, label='σ + score-matched random committees')
        ax.axvline(sel[k]['observed'], color=ACCENT, lw=1.8, label='Selected SEARCH top 50')
        ax.set_xlabel(lab); ax.set_yticks([])
        ax.text(.97, .95, f"pct vs σ-matched {sel[k]['percentile_of_observed']:.1f}\nvs σ+score {ssel[k]['percentile_of_observed']:.1f}",
                transform=ax.transAxes, ha='right', va='top', fontsize=6.1, color=INK2)
    axes[0].set_ylabel('Density')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(.5, -.02), fontsize=6.5)
    fig.tight_layout(rect=(0, .08, 1, 1))
    save_bd(fig, 'fig_selection_diversity')

    # Figure D: cross-split geometry
    cs = R['cross_split']; dd = pd.read_csv(OUT / 'audit_cross_split_distance.csv')
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), gridspec_kw={'width_ratios': [1, 1.15]})
    ax = axes[0]
    rng = np.random.default_rng(0)
    for s in SIGMAS:
        m = dd.sigma == s
        ax.scatter(200 * dd.search_hamming[m] + rng.uniform(-.25, .25, m.sum()), 200 * dd.rerank_hamming[m] + rng.uniform(-.25, .25, m.sum()),
                   s=8, color=SIG_COL[s], alpha=.75, lw=0, label=f'σ = {s:g}')
    ax.set_xlabel('Answers changed vs base, SEARCH200'); ax.set_ylabel('Answers changed vs base, RERANK200')
    db = cs['distance_to_base']
    ax.set_title(f"A  Distance to base (Spearman {db['all']['spearman']:.2f} overall)", loc='left', fontsize=7.5)
    ax.legend(frameon=False, fontsize=6, loc='upper left', markerscale=1.5)
    ax = axes[1]
    rows = [('Pairwise distance r', cs['pairwise']['spearman']['observed'] if isinstance(cs['pairwise']['spearman'], dict) else cs['pairwise']['spearman'],
             cs['pairwise']['sigma_stratified_perm']['spearman']),
            ('CKA, answer', cs['cka']['answer']['all']['observed'], cs['cka']['answer']['all']['sigma_stratified_perm']),
            ('CKA, correctness', cs['cka']['correctness']['all']['observed'], cs['cka']['correctness']['all']['sigma_stratified_perm'])]
    for s in SIGMAS:
        c = cs['cka']['answer'][str(s)]
        rows.append((f'CKA answer, σ={s:g}', c['observed'], c['global_perm']))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, o, n) in zip(y, rows):
        ax.plot([n['null_q025'], n['null_q975']], [yi, yi], color=MUTED, lw=4, solid_capstyle='butt', alpha=.6)
        ax.plot(n['null_median'], yi, 's', color=BASE_C, ms=4)
        ax.plot(o, yi, 'o', color=ACCENT, ms=6)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=6.6); ax.set_xlim(left=min(0, ax.get_xlim()[0]))
    ax.set_xlabel('Statistic (orange observed; grey null 95%, square median)')
    ax.set_title('B  SEARCH vs RERANK geometry, 500 audit candidates', loc='left', fontsize=7.5)
    fig.tight_layout()
    save_bd(fig, 'fig_cross_split_geometry')

    tables(R)


def tables(R):
    dist = R['distance']
    rows = []
    for g, lab in [('all', 'All 5,000')] + [(str(s), f'$\\sigma$={s:g}') for s in SIGMAS]:
        for k, kl in (('answers_changed', 'answers changed'), ('correctness_changes', 'correctness changes'), ('repairs', 'repairs'),
                      ('regressions', 'regressions'), ('wrong_to_wrong_changes', 'wrong$\\to$wrong')):
            d = dist[g][k]
            rows.append([lab if k == 'answers_changed' else '', kl, f"{d['mean']:.2f}", f"{d['sd']:.2f}", f"{d['p5']:g}", f"{d['p25']:g}",
                         f"{d['p50']:g}", f"{d['p75']:g}", f"{d['p95']:g}", f"{d['p99']:g}", f"{d['min']}--{d['max']}"])
        rows.append('MIDRULE')
    write_tex('table_behavioral_distance', ['Population', 'Per candidate (of 200)', 'Mean', 'SD', 'P5', 'P25', 'Median', 'P75', 'P95', 'P99', 'Range'],
              rows[:-1], 'Behavioral distance of each SEARCH candidate from the base model on the 200 SEARCH questions. '
              'Repairs: base wrong $\\to$ candidate right; regressions: the reverse; wrong$\\to$wrong: a different wrong answer.',
              'tab:behavioral-distance', colspec='llrrrrrrrrr')

    sp, nl = R['spectrum'], R['null']
    rows = []
    for rep, rl, dim in (('answer', 'Answer dev.', 800), ('correctness', 'Correctness dev.', 200)):
        for g, lab in [('all', 'All 5,000')] + [(str(s), f'$\\sigma$={s:g} (1,250)') for s in SIGMAS]:
            r, s_, n = sp[g][f'{rep}_raw'], sp[g][f'{rep}_std'], nl[g][f'{rep}_raw']['r_entropy']
            rows.append([lab, f'{rl} ({dim})', f"{r['r_entropy']:.1f}", f"{r['r_PR']:.1f}", f"{s_['r_entropy']:.1f}",
                         f"{nl[g][f'{rep}_std']['r_entropy']['null_median']:.1f}", f"{r['k90']}",
                         f"{n['null_median']:.1f}", f"[{n['null_q025']:.1f}, {n['null_q975']:.1f}]", f"{r['rank_ceiling']}"])
        for key, lab in (('top50_search', 'SEARCH top 50'), ('test_top50', 'TEST top 50$^\\dagger$')):
            spc = R['top50_search_spectrum'] if key == 'top50_search' else R['test_top50']['spectrum']
            nn = R['top50_search_null'] if key == 'top50_search' else R['test_top50']['null']
            r, s_ = spc[f'{rep}_raw'], spc[f'{rep}_std']
            n = nn[f'{rep}_raw']['r_entropy']
            rows.append([lab, f"{rl} ({4 * 561 if key == 'test_top50' and rep == 'answer' else 561 if key == 'test_top50' else dim})",
                         f"{r['r_entropy']:.1f}", f"{r['r_PR']:.1f}", f"{s_['r_entropy']:.1f}", f"{nn[f'{rep}_std']['r_entropy']['null_median']:.1f}",
                         f"{r['k90']}", f"{n['null_median']:.1f}", f"[{n['null_q025']:.1f}, {n['null_q975']:.1f}]", f"{r['rank_ceiling']}"])
        rows.append('MIDRULE')
    write_tex('table_behavioral_effective_dimension',
              ['Population', 'Representation (features)', 'Raw $r_{\\mathrm{ent}}$', 'Raw $r_{\\mathrm{PR}}$', 'Std.\\ $r_{\\mathrm{ent}}$',
               'Std.\\ null med.', 'k90', 'Raw null med.', 'Raw null 95\\%', 'Ceiling'], rows[:-1],
              'Effective behavioral dimension (entropy effective rank $r_{\\mathrm{ent}}$, participation ratio $r_{\\mathrm{PR}}$) of candidate '
              'prediction deviations from the base, with a $\\sigma$-stratified per-question permutation null (1,000 replicates) that preserves every '
              'question\'s answer distribution per $\\sigma$ but destroys persistent candidate identity. These are dimensions of observed behavioral '
              'variation, not counts of independent models. Ceilings differ between representations and populations; compare each row with its own null. '
              '$^\\dagger$SELECTED\\_COMMITTEE\\_TEST\\_GEOMETRY: 561 TEST questions; the null permutes within the committee.',
              'tab:effective-dimension', colspec='llrrrrrrrr')

    sel = R['selection']['metrics']
    rows = []
    for k, lab in (('answer_raw_r_entropy', 'Answer eff.\\ rank (raw)'), ('answer_std_r_entropy', 'Answer eff.\\ rank (std.)'),
                   ('correctness_raw_r_entropy', 'Correctness eff.\\ rank (raw)'), ('correctness_std_r_entropy', 'Correctness eff.\\ rank (std.)'),
                   ('pairwise_disagreement', 'Mean pairwise disagreement'), ('error_jaccard', 'Mean error Jaccard'),
                   ('hamming_from_base', 'Mean distance from base')):
        m = sel[k]; sm = R['selection_score_matched']['metrics'][k]
        rows.append([lab, f"{m['observed']:.3f}", f"{m['null_median']:.3f} [{m['null_q025']:.3f}, {m['null_q975']:.3f}]", f"{m['percentile_of_observed']:.1f}",
                     f"{sm['null_median']:.3f} [{sm['null_q025']:.3f}, {sm['null_q975']:.3f}]", f"{sm['percentile_of_observed']:.1f}"])
    ssm = R['selection_score_matched']
    write_tex('table_selection_diversity', ['Metric (SEARCH200)', 'Selected top 50', '$\\sigma$-matched median [95\\%]', 'Pct.',
                                            '$\\sigma$+score-matched median [95\\%]', 'Pct.'], rows,
              f"Diversity of the frozen SEARCH top 50 versus {R['n_committees']:,} random committees of 50 drawn from the 5,000 candidates with "
              f"exactly the same $\\sigma$ composition ({', '.join(f'{v}$\\times${k}' for k, v in R['selection']['top50_sigma_mix'].items() if v)}), "
              f"and versus {ssm['n_committees']:,} committees additionally matched member by member on SEARCH correct count ($\\pm$1, widened only "
              f"when exhausted; mean gap {ssm['mean_score_gap_matched_minus_selected']:+.2f} questions). Score matching removes the agreement that high "
              "accuracy on the selection questions forces mechanically.",
              'tab:selection-diversity', colspec='lrrrrr')

    cs = R['cross_split']; rows = []
    db = cs['distance_to_base']
    for g, lab in [('all', 'All 500')] + [(str(s), f'$\\sigma$={s:g}') for s in SIGMAS]:
        rows.append(['Distance to base', lab, f"{db[g]['pearson']:.2f}", f"{db[g]['spearman']:.2f}", '--', '--'])
    rows.append('MIDRULE')
    pw = cs['pairwise']
    rows.append(['Pairwise distance', 'All 500', f"{pw['pearson']:.2f}", f"{pw['spearman']:.2f}",
                 f"{pw['sigma_stratified_perm']['spearman']['null_median']:.2f} [{pw['sigma_stratified_perm']['spearman']['null_q025']:.2f}, {pw['sigma_stratified_perm']['spearman']['null_q975']:.2f}]",
                 f"{pw['sigma_stratified_perm']['spearman']['percentile_of_observed']:.1f}"])
    for s in SIGMAS:
        w = cs['pairwise_within_sigma'][str(s)]
        rows.append(['', f'within $\\sigma$={s:g}', f"{w['pearson']:.2f}", f"{w['spearman']:.2f}",
                     f"{w['global_perm']['spearman']['null_median']:.2f} [{w['global_perm']['spearman']['null_q025']:.2f}, {w['global_perm']['spearman']['null_q975']:.2f}]",
                     f"{w['global_perm']['spearman']['percentile_of_observed']:.1f}"])
    rows.append('MIDRULE')
    for rep in ('answer', 'correctness'):
        c = cs['cka'][rep]
        rows.append([f'Linear CKA, {rep}', 'All 500', f"{c['all']['observed']:.3f}", '--',
                     f"{c['all']['sigma_stratified_perm']['null_median']:.3f} [{c['all']['sigma_stratified_perm']['null_q025']:.3f}, {c['all']['sigma_stratified_perm']['null_q975']:.3f}]",
                     f"{c['all']['sigma_stratified_perm']['percentile_of_observed']:.1f}"])
        for s in SIGMAS:
            cc = c[str(s)]
            rows.append(['', f'within $\\sigma$={s:g}', f"{cc['observed']:.3f}", '--',
                         f"{cc['global_perm']['null_median']:.3f} [{cc['global_perm']['null_q025']:.3f}, {cc['global_perm']['null_q975']:.3f}]",
                         f"{cc['global_perm']['percentile_of_observed']:.1f}"])
    if 'cka_direction_only' in cs:
        rows.append('MIDRULE')
        for rep in ('answer', 'correctness'):
            c = cs['cka_direction_only'][rep]
            for s in SIGMAS:
                cc = c[str(s)]
                rows.append([f'Direction-only CKA, {rep}' if s == SIGMAS[0] else '', f'within $\\sigma$={s:g} (n={cc["n"]})', f"{cc['observed']:.3f}", '--',
                             f"{cc['global_perm']['null_median']:.3f} [{cc['global_perm']['null_q025']:.3f}, {cc['global_perm']['null_q975']:.3f}]",
                             f"{cc['global_perm']['percentile_of_observed']:.1f}"])
    write_tex('table_cross_split_geometry', ['Statistic', 'Candidates', 'Pearson / CKA', 'Spearman', 'Permutation null median [95\\%]', 'Pct.'], rows,
              'Does candidate behavioral geometry on SEARCH200 persist on the disjoint RERANK200 questions? Same 500 precommitted random-audit '
              'candidates. Pairwise distance: normalized answer Hamming distance over all candidate pairs (Mantel-style permutation of candidate '
              'identity, 1,000 replicates; within $\\sigma$ for the pooled rows so that $\\sigma$ alone cannot create agreement). Direction-only: deviation vectors scaled to unit length per '
              'candidate, removing movement magnitude. Pct.: percentile of '
              'the observed value in its null.', 'tab:cross-split', colspec='llrrrr')


def direction_stage():
    """Supplementary: does cross-split agreement survive removing each candidate's movement magnitude?
    Deviation vectors are scaled to unit length per candidate (candidates identical to base on either split are dropped),
    then linear CKA within each sigma against a candidate-identity permutation null."""
    df, lock, proto = load_inputs()
    ids = json.loads((OUT / 'candidate_ids.json').read_text())
    audit = ids['audit500']; sig = dict(zip(ids['search_order'], ids['sigma']))
    PS, bS, gS, _ = matrix(df, 'SEARCH', audit)
    PR, bR, gR, _ = matrix(df, 'RERANK', audit)
    asg = np.array([sig[c] for c in audit])
    R = json.loads((OUT / 'behavioral_diversity_results.json').read_text())
    out = {}
    for rep, fS, fR in (('answer', d_ans(PS, bS), d_ans(PR, bR)), ('correctness', d_corr(PS, bS, gS), d_corr(PR, bR, gR))):
        nS, nR = np.linalg.norm(fS, axis=1), np.linalg.norm(fR, axis=1)
        keep = (nS > 0) & (nR > 0)
        uS, uR = fS[keep] / nS[keep, None], fR[keep] / nR[keep, None]
        out[rep] = {'dropped_zero_movement': int((~keep).sum()), 'all': cka_perm(uS, uR, asg[keep], [SEED, 9, len(rep)])}
        for s in SIGMAS:
            m = asg[keep] == s
            out[rep][str(s)] = cka_perm(uS[m], uR[m], asg[keep][m], [SEED, 10, len(rep), int(s * 1e5)])
    R['cross_split']['cka_direction_only'] = out
    (OUT / 'behavioral_diversity_results.json').write_text(json.dumps(R, indent=1, default=float))
    print(json.dumps({r: {g: (round(v['observed'], 3), round(v.get('global_perm', {}).get('null_median', float('nan')), 3),
                              v.get('global_perm', {}).get('percentile_of_observed')) for g, v in d.items() if isinstance(v, dict)} for r, d in out.items()}, indent=1))


def score_matched_stage(n=2000, tol=1):
    """Supplementary: is the selected committee's agreement explained by its members' SEARCH accuracy?
    Each top-50 member is replaced by a random non-top-50 candidate with the same sigma and SEARCH correct count within
    +/- tol (pools widened by 1 until non-empty). Agreement on questions both answer correctly is mechanical, so this
    separates selection-by-score from any further behavioral concentration."""
    df, lock, proto = load_inputs()
    ids = json.loads((OUT / 'candidate_ids.json').read_text())
    order, sigma, top50 = ids['search_order'], np.array(ids['sigma']), ids['top50']
    P, base, gold, _ = matrix(df, 'SEARCH', order)
    score = (P == gold).sum(1)
    pos = {c: i for i, c in enumerate(order)}
    t_ix = np.array([pos[c] for c in top50]); is_top = np.zeros(len(order), bool); is_top[t_ix] = True
    # Sequential draws without replacement: members are visited in a random order and each takes a random unused
    # candidate with the same sigma and |score difference| <= w, widening w only if that pool is exhausted.
    obs, _ = committee_metrics(P[t_ix], base, gold)
    rng = np.random.default_rng([SEED, 11])
    vals = {k: [] for k in obs}
    widths_all, gaps = [], []
    for t in range(n):
        used = np.zeros(len(order), bool); ix = []
        for i in rng.permutation(t_ix):
            w = tol
            while True:
                pool = np.where((sigma == sigma[i]) & (np.abs(score - score[i]) <= w) & ~is_top & ~used)[0]
                if len(pool): break
                w += 1
            j = rng.choice(pool); used[j] = True; ix.append(j); widths_all.append(w); gaps.append(int(score[j] - score[i]))
        ix = np.array(ix)
        assert len(np.unique(ix)) == len(ix)
        m, _ = committee_metrics(P[ix], base, gold)
        for k in vals: vals[k].append(m[k])
        if t % 500 == 0: print('score-matched', t, flush=True)
    widths = {'mean': float(np.mean(widths_all)), 'max': int(np.max(widths_all)), 'share_tol': float(np.mean(np.array(widths_all) == tol))}
    R = json.loads((OUT / 'behavioral_diversity_results.json').read_text())
    R['selection_score_matched'] = {'n_committees': n, 'tolerance': tol, 'pool_widths': widths,
                                    'mean_score_gap_matched_minus_selected': float(np.mean(gaps)), 'top50_mean_score': float(score[t_ix].mean()),
                                    'metrics': {k: null_summary(obs[k], v) for k, v in vals.items()}}
    np.savez_compressed(OUT / 'selection_score_matched_samples.npz', **{k: np.array(v) for k, v in vals.items()})
    (OUT / 'behavioral_diversity_results.json').write_text(json.dumps(R, indent=1, default=float))
    print(json.dumps({k: (round(v['observed'], 3), round(v['null_median'], 3), round(v['null_q025'], 3), round(v['null_q975'], 3), v['percentile_of_observed'])
                      for k, v in R['selection_score_matched']['metrics'].items()}, indent=1), widths)


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what == 'direction':
        direction_stage()
    if what == 'score_matched':
        score_matched_stage()
    if what in ('compute', 'all'):
        compute()
    if what in ('figures', 'all'):
        figures_and_tables()

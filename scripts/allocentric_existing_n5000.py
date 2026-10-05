"""Post-hoc exploratory follow-up: Allocentric specialists inside the EXISTING N=5000 population.

Step 1-2 (CPU only): rescore all 5,000 existing candidates on the Allocentric subset of the
original SEARCH200 from their stored raw outputs; rank; compare with the mixed ranking.
No new perturbation, sigma, mask or search inference. Writes only new files.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics

ROOT = Path('results/perspective-taking-n5000-20261004')
DATA = Path('examples/omnispatial-perspective-taking')
OUT = ROOT / 'allocentric-followup'
LETTERS = 'ABCD'


def load(path):
    raw = Path(path).read_bytes()
    return json.loads(gzip.decompress(raw) if str(path).endswith('.gz') else raw)


def parse(text):
    """Frozen official direct rule: first non-space character, upper-cased, valid if A-D."""
    m = re.match(r'\s*(\S)', text)
    ch = m.group(1).upper() if m else ''
    return ch if ch in LETTERS else ''


def rank_hash(cid):
    return hashlib.sha256(('visual-rank-v1:' + cid).encode()).hexdigest()


def step12():
    from collections import Counter
    import numpy as np
    from scipy.stats import spearmanr
    OUT.mkdir(parents=True, exist_ok=True)
    search = [json.loads(l) for l in (DATA / 'search.jsonl').read_text(encoding='utf-8').splitlines()]
    allo = [i for i, r in enumerate(search) if r['sub_task_type'] == 'Allocentric']
    gold = [LETTERS[search[i]['answer']] for i in allo]
    base_out = load(ROOT / 'baseline/base.json.gz')['search']['outputs']
    if [o['uid'] for o in base_out] != [r['uid'] for r in search]: raise SystemExit('base order mismatch')
    base_pred = [parse(base_out[i]['text']) for i in allo]
    base = sum(p == g for p, g in zip(base_pred, gold))
    lock = load(ROOT / 'locks/search.json')
    mixed_rank = {cid: i + 1 for i, cid in enumerate(lock['ranked_ids'])}
    recs = []
    for k in range(50):
        d = ROOT / f'search-shard-{k:03d}' / 'candidates'
        for f in sorted(d.glob('*.json.gz')):
            raw = load(f)
            outs = raw['splits']['search']['outputs']
            if [o['uid'] for o in outs] != [r['uid'] for r in search]: raise SystemExit('candidate order mismatch')
            pred = [parse(outs[i]['text']) for i in allo]
            c = sum(p == g for p, g in zip(pred, gold))
            cid = raw['candidate']['candidate_id']
            recs.append({'candidate_id': cid, 'seed': raw['candidate']['seed'], 'sigma': raw['candidate']['sigma'],
                         'candidate_state_id': raw['candidate_state_id'], 'allocentric_correct': c,
                         'allocentric_accuracy': c / len(allo), 'gain_pp': 100 * (c - base) / len(allo),
                         'mixed_search_correct': sum(parse(o['text']) == LETTERS[r['answer']] for o, r in zip(outs, search)),
                         'original_search_rank': mixed_rank[cid], 'prediction_distribution': dict(Counter(pred)),
                         'allocentric_correct_vector': [p == g for p, g in zip(pred, gold)]})
    if len(recs) != 5000 or len({r['candidate_id'] for r in recs}) != 5000: raise SystemExit('population incomplete')
    ranked = sorted(recs, key=lambda r: (-r['allocentric_correct'], rank_hash(r['candidate_id'])))
    for i, r in enumerate(ranked): r['allocentric_rank'] = i + 1
    c = np.array([r['allocentric_correct'] for r in ranked])
    g = 100 * (c - base) / len(allo)
    per_q = np.array([r['allocentric_correct_vector'] for r in ranked])
    summary = {'allocentric_search_examples': len(allo), 'base_correct': base, 'base_accuracy': base / len(allo),
               'base_prediction_distribution': dict(Counter(base_pred)), 'gold_distribution': dict(Counter(gold)),
               'mean_correct': float(c.mean()), 'median_correct': float(np.median(c)), 'sd_correct': float(c.std(ddof=1)),
               'best': {k: ranked[0][k] for k in ('seed', 'sigma', 'allocentric_correct', 'gain_pp', 'original_search_rank')},
               'histogram': dict(sorted(Counter(c.tolist()).items())),
               'above_base': int((g > 1e-9).sum()), 'at_least_3pp': int((g >= 3 - 1e-9).sum()),
               'at_least_5pp': int((g >= 5 - 1e-9).sum()), 'at_least_10pp': int((g >= 10 - 1e-9).sum()),
               'one_question_pp': 100 / len(allo),
               'by_sigma': {str(s): {'n': int(m.sum()), 'mean_gain_pp': float(g[m].mean()), 'sd_gain_pp': float(g[m].std(ddof=1)),
                                     'max_gain_pp': float(g[m].max()), 'at_least_5pp': int((g[m] >= 5 - 1e-9).sum()),
                                     'at_least_10pp': int((g[m] >= 10 - 1e-9).sum())}
                            for s in (.00025, .0005, .001, .002) for m in [np.array([r['sigma'] == s for r in ranked])]},
               'top200_cutoff_correct': int(c[199]), 'top200_ties_at_cutoff': int((c == c[199]).sum()),
               'top50_cutoff_correct': int(c[49]),
               'per_question_fraction_correct': per_q.mean(axis=0).tolist(),
               'per_question_base_correct': [p == gl for p, gl in zip(base_pred, gold)]}
    mixed = np.array([r['mixed_search_correct'] for r in ranked])
    summary['spearman_allocentric_vs_mixed'] = float(spearmanr(c, mixed).statistic)
    top50_allo = {r['candidate_id'] for r in ranked[:50]}
    top50_mixed = set(lock['ranked_ids'][:50])
    summary['top50_overlap_allocentric_vs_mixed'] = len(top50_allo & top50_mixed)
    summary['top200_allocentric_vs_mixed_top200_overlap'] = len({r['candidate_id'] for r in ranked[:200]} & set(lock['ranked_ids'][:200]))
    # Noise reference: expected maximum gain over 5,000 draws if every candidate were the base model with
    # independent per-question flips at the observed discordance rate (sign-flip null).
    rng = np.random.default_rng(20261005)
    base_vec = np.array([p == gl for p, gl in zip(base_pred, gold)])
    disc = np.array([int((v != base_vec).sum()) for v in per_q])
    null_max = []
    for _ in range(2000):
        d = rng.choice(disc, 5000)
        null_max.append(((2 * rng.binomial(d, .5) - d) / len(allo) * 100).max())
    obs_tail = {t: int((g >= t - 1e-9).sum()) for t in (3, 5, 7)}
    null_tail = {t: [] for t in (3, 5, 7)}
    for _ in range(500):
        d = rng.choice(disc, 5000); ng = (2 * rng.binomial(d, .5) - d) / len(allo) * 100
        for t in null_tail: null_tail[t].append(int((ng >= t - 1e-9).sum()))
    summary['observed_vs_null_tail_counts'] = {str(t): {'observed': obs_tail[t], 'null_mean': float(np.mean(v)),
                                                        'null_p95': float(np.quantile(v, .95))} for t, v in null_tail.items()}
    summary['null_max_gain_over_5000_pp'] = {'mean': float(np.mean(null_max)), 'p95': float(np.quantile(null_max, .95)),
                                              'median_discordant_questions': float(np.median(disc)),
                                              'model': 'each candidate keeps its observed number of disagreements with base; each disagreement is a fair coin (no true effect)'}
    out = OUT / 'existing_n5000_allocentric_search_ranking.json'
    if out.exists(): out.unlink()
    out.write_text(json.dumps({'summary': summary, 'ranking': [
        {k: v for k, v in r.items() if k != 'allocentric_correct_vector'} for r in ranked]}, indent=1))
    show = {k: v for k, v in summary.items() if k not in ('per_question_fraction_correct', 'per_question_base_correct', 'histogram')}
    print(json.dumps(show, indent=1)); print('histogram', summary['histogram'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('step', choices=['cpu'])
    a = p.parse_args()
    step12()

"""Frozen Perspective-Taking N=5000 existence test: manifest, rankings and decisions.

Search ranking reads only SEARCH records; validation ranking reads only the
VALIDATION outputs of the frozen top 50; the density audit never selects; no
test candidate output is read before the top-10 lock is committed.
"""
from collections import Counter
import math
import random

from .omnispatial import EPS, LETTERS, compare, counts, ident, label_audit, paired, summarize, verify_dataset  # noqa: F401
from .omnispatial_data import load
from .visual_final import rank_hash

SPLITS = ('search', 'validation', 'test')
PHASE_SPLITS = {'baseline': SPLITS, 'search': ('search',), 'validation': ('validation',), 'test': ('test',)}


def splits(protocol):
    sets = {s: load(protocol['dataset'], s) for s in SPLITS}
    for s, rows in sets.items():
        if len(rows) != protocol['populations'][s]['n'] or ident(rows) != protocol['populations'][s]['sha256']:
            raise ValueError('evaluation population changed: ' + s)
    h = {s: {r['image_sha256'] for r in rows} for s, rows in sets.items()}
    if h['search'] & h['validation'] or (h['search'] | h['validation']) & h['test']:
        raise ValueError('image shared across splits')
    return sets


def manifest(protocol):
    """Candidate i = 0..4999: seed seed_start+i, sigma sigma_mixture[i % 4]; shard i // shard_size."""
    c = protocol['candidates']
    sig = c['sigma_mixture']
    rows = [{'index': i, 'seed': c['seed_start'] + i, 'sigma': sig[i % 4], 'shard': i // c['shard_size']} for i in range(c['count'])]
    if c['count'] != 5000 or sig != [.00025, .0005, .001, .002] or any(sum(r['sigma'] == s for r in rows) != 1250 for s in sig):
        raise ValueError('candidate population differs from the frozen 4 x 1250 sigma mixture')
    return rows


def shard_plan(protocol, shard):
    rows = [r for r in manifest(protocol) if r['shard'] == shard]
    if not rows:
        raise ValueError('unknown search shard')
    return [(r['seed'], r['sigma']) for r in rows]


def density_audit(protocol):
    """500 manifest indices, 125 per sigma, drawn with a committed seed before any output exists."""
    rng = random.Random(protocol['density_audit']['seed'])
    rows = manifest(protocol)
    chosen = []
    for s in protocol['candidates']['sigma_mixture']:
        pool = [r['index'] for r in rows if r['sigma'] == s]
        chosen += sorted(rng.sample(pool, protocol['density_audit']['per_sigma']))
    return sorted(chosen)


def verify_inputs(protocol):
    proof = verify_dataset(protocol)
    splits(protocol)
    rows = manifest(protocol)
    seeds = [r['seed'] for r in rows]
    if len(set(seeds)) != len(seeds) or set(seeds) & set(protocol['previous_candidate_seeds']):
        raise ValueError('reused candidate seed')
    if density_audit(protocol) != protocol['density_audit']['indices']:
        raise ValueError('density audit selection changed')
    return proof


def rank_search(records, protocol):
    """SEARCH correct count descending, then SHA256('visual-rank-v1:'+candidate_id)."""
    if len(records) != 5000 or len({r['candidate']['candidate_id'] for r in records}) != 5000:
        raise ValueError('ranking requires all 5000 audited candidates exactly once')
    if any(r['split'] != 'search' or r['examples'] != protocol['populations']['search']['n'] for r in records):
        raise ValueError('ranking accepts complete SEARCH records only')
    return sorted(records, key=lambda r: (-r['correct_count'], rank_hash(r['candidate']['candidate_id'])))


def rank_validation(top, validation_correct):
    """VALIDATION correct desc, SEARCH correct desc, then the frozen hash."""
    return sorted(top, key=lambda r: (-validation_correct[r['candidate']['candidate_id']], -r['correct_count'],
                                      rank_hash(r['candidate']['candidate_id'])))


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def density(gains_by_sigma):
    out = {}
    for key, gains in list(gains_by_sigma.items()) + [('all', [g for v in gains_by_sigma.values() for g in v])]:
        n = len(gains)
        out[str(key)] = {'n': n, **{f'at_least_{pp}pp': {'count': c, 'fraction': c / n, 'wilson95': wilson(c, n)}
                                    for pp, c in ((1, sum(g >= .01 - EPS for g in gains)), (3, sum(g >= .03 - EPS for g in gains)),
                                                  (5, sum(g >= .05 - EPS for g in gains)))}}
    return out


def majority(outputs, rows, k):
    """Majority vote of the frozen top k: invalid answers ignored, ties to the first
    occurrence in frozen validation order, all-invalid counts as incorrect."""
    result = []
    for i, r in enumerate(rows):
        preds = [v[i]['prediction'] for v in outputs[:k] if v[i]['valid_letter']]
        tally = Counter(preds)
        best = max(tally.values()) if tally else 0
        winner = next((p for p in preds if tally[p] == best), '')
        result.append({'uid': r['uid'], 'prediction': winner, 'valid_letter': winner in LETTERS,
                       'correct': winner == chr(65 + r['answer'])})
    return result


def expert(result, protocol, controls_pass):
    gate = protocol['gate']
    checks = {'gain_at_least_5pp': result['gain'] >= gate['expert_gain'] - EPS,
              'bootstrap_lower_positive': result['paired']['paired_bootstrap95'][0] > 0,
              'positive_in_two_sub_tasks': sum(g > EPS for g in result['sub_task_gain'].values()) >= gate['positive_sub_tasks'],
              'answer_bias_spread': result['label_audit']['bias_spread_pass'],
              'runtime_controls': controls_pass}
    return {'checks': checks, 'transferable_expert': all(checks.values())}


def decision(final, protocol, controls_pass):
    if not controls_pass:
        raise ValueError('operational failure: controls must pass before a scientific decision')
    if len(final) != 10:
        raise ValueError('operational failure: exactly the ten frozen candidates are required')
    g = protocol['gate']
    verdicts = [expert(r, protocol, controls_pass) for r in final]
    gains = [r['gain'] for r in final]
    primary = verdicts[0]['transferable_expert']
    five = [i for i, x in enumerate(gains) if x >= g['expert_gain'] - EPS]
    replicated = {'at_least_3_with_3pp': sum(x >= .03 - EPS for x in gains) >= 3, 'at_least_1_with_5pp': bool(five),
                  'mean_top10_at_least_3pp': sum(gains) / 10 >= .03 - EPS,
                  'a_5pp_candidate_passes_expert_checks': any(verdicts[i]['transferable_expert'] for i in five),
                  'controls': controls_pass}
    result = ('GO_VISUAL_NEURAL_THICKET_PERSPECTIVE_TAKING' if primary else
              'GO_VISUAL_NEURAL_THICKET_PERSPECTIVE_TAKING_REPLICATED' if all(replicated.values()) else
              'NO_GO_VISUAL_NEURAL_THICKET_PERSPECTIVE_N5000')
    return {'decision': result, 'primary_pass': primary, 'replicated': replicated, 'replicated_pass': all(replicated.values()),
            'experts': verdicts, 'transferable_expert_ranks': [i + 1 for i, v in enumerate(verdicts) if v['transferable_expert']],
            'rank1_gain': gains[0], 'best_top10_gain': max(gains), 'mean_top10_gain': sum(gains) / 10,
            'top10_counts': counts(gains)}

"""Frozen v1 tail-search decisions. No baseline or population-mean gates."""
from collections import Counter
import hashlib
import math
from pathlib import Path

import numpy as np

from .line_tracing import load_split, sha
from .visual_decisions import rank_selection as original_rank, summarize
from .visual_runtime import read
from .visual_v21 import population_id, verify_dataset


def verify_inputs(protocol):
    proof = verify_dataset(protocol)
    reference = protocol['reference']
    index_path = Path(reference['index'])
    if sha(index_path) != reference['index_sha256']:
        raise ValueError('historical reference index changed')
    for name, expected in read(index_path).items():
        if sha(index_path.parent / name) != expected:
            raise ValueError('historical artifact changed: ' + name)
    sets = {s: load_split(protocol['dataset'], s) for s in ('selection', 'heldout')}
    for s, rows in sets.items():
        if len(rows) != protocol['populations'][s]['n'] or population_id(rows) != protocol['populations'][s]['sha256']:
            raise ValueError('evaluation population changed')
    for field in ('id', 'seed', 'image_sha256'):
        if {r[field] for r in sets['selection']} & {r[field] for r in sets['heldout']}:
            raise ValueError('selection/held-out overlap')
    old = read(reference['calibration_records'])
    best = sorted(old, key=lambda r: (-r['correct_count'], rank_hash(r['candidate']['candidate_id'])))[0]
    if best != protocol['diagnostic']['selected'] or best['correct_count'] != 50:
        raise ValueError('existing calibration diagnostic identity changed')
    fresh = set(range(protocol['search']['seed_start'], protocol['search']['seed_start'] + 300))
    if fresh & set(protocol['previous_candidate_seeds']):
        raise ValueError('reused candidate seed')
    return proof


def evaluation_sets(protocol, phase):
    if phase not in ('diagnostic', 'search', 'heldout'):
        raise ValueError('only diagnostic, search and heldout phases are permitted')
    split = 'selection' if phase == 'search' else 'heldout'
    rows = load_split(protocol['dataset'], split)
    if len(rows) != protocol['populations'][split]['n'] or population_id(rows) != protocol['populations'][split]['sha256']:
        raise ValueError('wrong evaluation population')
    return {split: rows}


def rank_hash(cid):
    return hashlib.sha256(('visual-rank-v1:' + cid).encode()).hexdigest()


def rank_selection(records, protocol):
    for r in records:
        if r['candidate']['sigma'] != protocol['sigma'] or r['population_sha256'] != protocol['populations']['selection']['sha256']:
            raise ValueError('ranking requires frozen sigma and selection population')
    return original_rank(records, protocol['search'])


def paired(base, candidate, indices):
    b = np.asarray([o['correct'] for o in base], dtype=bool)
    c = np.asarray([o['correct'] for o in candidate], dtype=bool)
    if len(b) != len(c):
        raise ValueError('paired cardinality mismatch')
    delta = c.astype(np.int8) - b.astype(np.int8)
    wins, losses = int(np.sum(c & ~b)), int(np.sum(b & ~c))
    discordant = wins + losses
    p = min(1., 2 * sum(math.comb(discordant, i) for i in range(min(wins, losses) + 1)) / 2**discordant) if discordant else 1.
    return {'gain': float(delta.mean()), 'base_only_correct': losses, 'candidate_only_correct': wins,
            'both_correct': int(np.sum(b & c)), 'both_wrong': int(np.sum(~b & ~c)),
            'paired_bootstrap95': np.quantile(delta[indices].mean(axis=1), [.025, .975], method='linear').tolist(),
            'mcnemar_exact_two_sided_p': p, 'discordant_pairs': discordant}


def label_audit(base, candidate, rows, limit):
    tables = {}
    positive_net = []
    for label in ('1', '2', '3', '4'):
        ids = [i for i, r in enumerate(rows) if r['answer'] == label]
        b = sum(base[i]['correct'] for i in ids)
        c = sum(candidate[i]['correct'] for i in ids)
        positive_net.append(max(0, c - b))
        tables[label] = {'n': len(ids), 'base_correct': b, 'candidate_correct': c,
                         'base_accuracy': b/len(ids), 'candidate_accuracy': c/len(ids),
                         'net_correct_gain': c-b,
                         'base_confusion': dict(Counter(base[i]['answer'] or 'invalid' for i in ids)),
                         'candidate_confusion': dict(Counter(candidate[i]['answer'] or 'invalid' for i in ids))}
    total = sum(positive_net)
    concentration = max(positive_net)/total if total else None
    return {'by_true_label': tables, 'base_prediction_counts': dict(Counter(o['answer'] or 'invalid' for o in base)),
            'candidate_prediction_counts': dict(Counter(o['answer'] or 'invalid' for o in candidate)),
            'positive_net_gain_count': total, 'largest_positive_net_gain_share': concentration,
            'single_label_concentration_limit': limit,
            'label_spread_pass': concentration is not None and concentration < limit}


def decision(individual, base, protocol, controls_pass):
    if not controls_pass or len(individual) != 10:
        raise ValueError('operational failure: ten audited selected experts and controls are required')
    config = protocol['gate']
    first = individual[0]
    gains = [r['comparison']['gain'] for r in individual]
    primary = {'effect_size': gains[0] >= config['single_expert_gain'] - 1e-12,
               'bootstrap_lower_positive': first['comparison']['paired_bootstrap95'][0] > 0,
               'two_difficulties_improve': sum(d > 1e-12 for d in first['difficulty_gain'].values()) >= 2,
               'label_spread': first['label_audit']['label_spread_pass'], 'controls': controls_pass}
    density = {'strictly_above_base': sum(g > 1e-12 for g in gains),
               'at_least_1pp': sum(g >= .01-1e-12 for g in gains),
               'at_least_3pp': sum(g >= .03-1e-12 for g in gains),
               'at_least_5pp': sum(g >= .05-1e-12 for g in gains),
               'mean_individual_gain': sum(gains)/10,
               'mean_individual_accuracy': base['accuracy'] + sum(gains)/10}
    replication = {'three_at_least_3pp': density['at_least_3pp'] >= 3,
                   'mean_at_least_3pp': density['mean_individual_gain'] >= .03-1e-12,
                   'one_at_least_5pp': density['at_least_5pp'] >= 1, 'controls': controls_pass}
    strong, replicated = all(primary.values()), all(replication.values())
    result = 'GO_VISUAL_THICKET_SINGLE_EXPERT' if strong else 'GO_VISUAL_THICKET_REPLICATED_EXPERTS' if replicated else 'NO_GO_VISUAL_THICKET_LINE_TRACING'
    return {'decision': result, 'pass': strong or replicated, 'primary': primary,
            'primary_pass': strong, 'replication': replication, 'replication_pass': replicated,
            'top10_density': density, 'controls_pass': controls_pass,
            'effect_size_without_all_primary_checks': primary['effect_size'] and not strong,
            'ensemble_is_not_a_go_route': True}

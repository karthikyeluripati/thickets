"""Frozen OmniSpatial Complex Spatial Logic existence test: ranking and decisions.

Search ranking reads only search records; validation ordering reads only
validation outputs of the frozen top 30; nothing reads test outputs before the
final candidates are committed.
"""
from collections import Counter
import hashlib
import math
from pathlib import Path
import subprocess

import numpy as np

from .omnispatial_data import load, sha256
from .visual_final import rank_hash
from .visual_v21 import population_id

EPS = 1e-12
LETTERS = ('A', 'B', 'C', 'D')
SPLITS = ('search', 'validation', 'test')
PHASE_SPLITS = {'baseline': SPLITS, 'search': ('search',), 'validation': ('validation',), 'test': ('test',)}


def verify_dataset(protocol):
    """Every committed dataset file equals its blob in the frozen dataset commit."""
    root, pin = Path(protocol['dataset']), protocol['dataset_pin']
    entries = subprocess.check_output(['git', 'ls-tree', '-rz', pin['commit'], '--', root.as_posix()]).split(b'\0')
    paths = set()
    for entry in filter(None, entries):
        meta, name = entry.decode().split('\t', 1)
        mode, kind, oid = meta.split()
        data = Path(name).read_bytes()
        if kind != 'blob' or hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest() != oid:
            raise ValueError('dataset differs from frozen commit: ' + name)
        paths.add(Path(name).resolve())
    if paths != {p.resolve() for p in root.rglob('*') if p.is_file()} or len(paths) != pin['files']:
        raise ValueError('frozen dataset inventory changed')
    if sha256((root / 'manifest.json').read_bytes()) != pin['manifest_sha256']:
        raise ValueError('frozen dataset manifest changed')
    return {'commit': pin['commit'], 'files_verified': len(paths), 'manifest_sha256': pin['manifest_sha256']}


def ident(rows):
    return population_id([{'id': r['uid'], 'image_sha256': r['image_sha256']} for r in rows])


def splits(protocol):
    sets = {s: load(protocol['dataset'], s) for s in SPLITS}
    for s, rows in sets.items():
        if len(rows) != protocol['populations'][s]['n'] or ident(rows) != protocol['populations'][s]['sha256']:
            raise ValueError('evaluation population changed: ' + s)
    uids = [r['uid'] for rows in sets.values() for r in rows]
    if len(set(uids)) != len(uids):
        raise ValueError('split overlap')
    h = {s: {r['image_sha256'] for r in rows} for s, rows in sets.items()}
    if h['search'] & h['validation'] or (h['search'] | h['validation']) & h['test']:
        raise ValueError('identical image shared across splits')
    return sets


def candidate_plan(protocol):
    """Seed seed_start+i gets sigma_mixture[i % 4]; exactly 100 per sigma."""
    c = protocol['candidates']
    sigmas = c['sigma_mixture']
    plan = [(c['seed_start'] + i, sigmas[i % len(sigmas)]) for i in range(c['count'])]
    if c['count'] != 400 or sigmas != [.00025, .0005, .001, .002] or any(sum(s == x for _, s in plan) != 100 for x in sigmas):
        raise ValueError('candidate population differs from the frozen 4x100 sigma mixture')
    return plan


def shard_plan(protocol, shard):
    size = protocol['candidates']['shard_size']
    if not 0 <= shard < protocol['candidates']['count'] // size:
        raise ValueError('unknown search shard')
    return candidate_plan(protocol)[shard * size:(shard + 1) * size]


def verify_inputs(protocol):
    proof = verify_dataset(protocol)
    splits(protocol)
    seeds = [s for s, _ in candidate_plan(protocol)]
    if len(set(seeds)) != len(seeds) or set(seeds) & set(protocol['previous_candidate_seeds']):
        raise ValueError('reused candidate seed')
    return proof


def summarize(outputs, rows):
    if [o['uid'] for o in outputs] != [r['uid'] for r in rows]:
        raise ValueError('output/example identity mismatch')
    def group(idx):
        n = len(idx)
        c = sum(outputs[i]['correct'] for i in idx)
        return {'n': n, 'correct_count': c, 'accuracy': c / n if n else None,
                'invalid_count': sum(not outputs[i]['valid_letter'] for i in idx)}
    subs = sorted({r['sub_task_type'] for r in rows})
    return {**group(list(range(len(rows)))),
            'sub_task': {s: group([i for i, r in enumerate(rows) if r['sub_task_type'] == s]) for s in subs},
            'prediction_marginal': dict(sorted(Counter(o['prediction'] if o['valid_letter'] else 'invalid' for o in outputs).items())),
            'confusion_true_by_pred': {t: dict(sorted(Counter(o['prediction'] if o['valid_letter'] else 'invalid'
                                                              for o, r in zip(outputs, rows) if chr(65 + r['answer']) == t).items()))
                                       for t in LETTERS}}


def rank_search(records, protocol):
    """SEARCH accuracy (correct count on the fixed split) descending, then SHA256('visual-rank-v1:'+id)."""
    if len(records) != protocol['candidates']['count'] or len({r['candidate']['candidate_id'] for r in records}) != len(records):
        raise ValueError('ranking requires every audited search candidate exactly once')
    if any(r['split'] != 'search' or r['examples'] != protocol['populations']['search']['n'] for r in records):
        raise ValueError('ranking accepts complete search records only')
    return sorted(records, key=lambda r: (-r['correct_count'], rank_hash(r['candidate']['candidate_id'])))


def rank_validation(top, correct):
    """VALIDATION correct count descending, then frozen search rank (position in top)."""
    return sorted(top, key=lambda r: (-correct[r['candidate']['candidate_id']], top.index(r)))


def label_audit(base, cand, rows, limit):
    """Positive net correctness gains by TRUE answer letter; full confusion matrices."""
    by, positive = {}, {}
    for t in LETTERS:
        idx = [i for i, r in enumerate(rows) if chr(65 + r['answer']) == t]
        b, c = sum(base[i]['correct'] for i in idx), sum(cand[i]['correct'] for i in idx)
        by[t] = {'n': len(idx), 'base_correct': b, 'candidate_correct': c, 'net_correct_gain': c - b,
                 'base_accuracy': b / len(idx) if idx else None, 'candidate_accuracy': c / len(idx) if idx else None}
        positive[t] = max(0, c - b)
    total = sum(positive.values())
    share = {t: positive[t] / total for t in LETTERS} if total else None
    worst = max(share.values()) if share else None
    pred = lambda outs, t: dict(sorted(Counter(outs[i]['prediction'] if outs[i]['valid_letter'] else 'invalid'
                                               for i, r in enumerate(rows) if chr(65 + r['answer']) == t).items()))
    return {'by_true_letter': by, 'positive_gain_share': share, 'largest_positive_gain_share': worst,
            'positive_net_gain_total': total, 'limit': limit, 'bias_spread_pass': worst is not None and worst < limit,
            'base_confusion': {t: pred(base, t) for t in LETTERS}, 'candidate_confusion': {t: pred(cand, t) for t in LETTERS},
            'base_prediction_marginal': dict(sorted(Counter(o['prediction'] if o['valid_letter'] else 'invalid' for o in base).items())),
            'candidate_prediction_marginal': dict(sorted(Counter(o['prediction'] if o['valid_letter'] else 'invalid' for o in cand).items()))}


def paired(base, cand, indices):
    b = np.asarray([o['correct'] for o in base], dtype=bool)
    c = np.asarray([o['correct'] for o in cand], dtype=bool)
    if len(b) != len(c):
        raise ValueError('paired cardinality mismatch')
    delta = c.astype(np.int8) - b.astype(np.int8)
    wins, losses = int(np.sum(c & ~b)), int(np.sum(b & ~c))
    d = wins + losses
    p = min(1., 2 * sum(math.comb(d, i) for i in range(min(wins, losses) + 1)) / 2 ** d) if d else 1.
    return {'gain': float(delta.mean()), 'candidate_only_correct': wins, 'base_only_correct': losses,
            'both_correct': int(np.sum(b & c)), 'both_wrong': int(np.sum(~b & ~c)),
            'paired_bootstrap95': np.quantile(delta[indices].mean(axis=1), [.025, .975], method='linear').tolist(),
            'mcnemar_exact_two_sided_p': p, 'discordant_pairs': d}


def compare(base_outputs, outputs, rows, indices, limit):
    base, metric = summarize(base_outputs, rows), summarize(outputs, rows)
    return {'summary': metric, 'gain': metric['accuracy'] - base['accuracy'],
            'sub_task_gain': {s: metric['sub_task'][s]['accuracy'] - base['sub_task'][s]['accuracy'] for s in metric['sub_task']},
            'paired': paired(base_outputs, outputs, indices), 'label_audit': label_audit(base_outputs, outputs, rows, limit)}


def counts(gains):
    return {'strictly_above_base': sum(g > EPS for g in gains), 'at_least_1pp': sum(g >= .01 - EPS for g in gains),
            'at_least_3pp': sum(g >= .03 - EPS for g in gains), 'at_least_5pp': sum(g >= .05 - EPS for g in gains)}


def expert(result, protocol, controls_pass):
    gate = protocol['gate']
    subs = result['sub_task_gain']
    checks = {'gain_at_least_5pp': result['gain'] >= gate['expert_gain'] - EPS,
              'bootstrap_lower_positive': result['paired']['paired_bootstrap95'][0] > 0,
              'positive_in_two_sub_tasks': sum(g > EPS for g in subs.values()) >= gate['positive_sub_tasks'],
              'answer_bias_spread': result['label_audit']['bias_spread_pass'],
              'runtime_controls': controls_pass}
    return {'checks': checks, 'transferable_expert': all(checks.values())}


def decision(final, protocol, controls_pass):
    if not controls_pass:
        raise ValueError('operational failure: controls must pass before a scientific decision')
    if len(final) != 5:
        raise ValueError('operational failure: exactly the five frozen candidates are required')
    verdicts = [expert(r, protocol, controls_pass) for r in final]
    go = any(v['transferable_expert'] for v in verdicts)
    return {'decision': 'GO_OMNISPATIAL_VISUAL_EXPERT' if go else 'NO_GO_OMNISPATIAL_VISUAL_EXPERT',
            'experts': verdicts, 'transferable_expert_ranks': [i + 1 for i, v in enumerate(verdicts) if v['transferable_expert']],
            'rank1_gain': final[0]['gain'], 'best_top5_gain': max(r['gain'] for r in final)}

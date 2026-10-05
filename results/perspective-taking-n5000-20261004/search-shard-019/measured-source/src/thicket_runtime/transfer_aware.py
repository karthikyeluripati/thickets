"""Frozen VANILLA vs TRANSFER_MIN ranking, validation gate and final decision.

Both rules rank the exact same 400 audited search records. Nothing here reads
validation or test outcomes when ranking, and nothing reads test outcomes when
choosing the final candidate set.
"""
import hashlib
from pathlib import Path
import subprocess

import numpy as np

from .line_tracing import sha
from .transfer_aware_data import FOLDS, load, load_folds
from .visual_final import label_audit, paired, rank_hash
from .visual_decisions import summarize
from .visual_v21 import population_id

EPS = 1e-12
METHODS = ('VANILLA', 'TRANSFER_MIN')
PHASE_SPLITS = {'baseline': ('search', 'validation', 'test'), 'search': ('search',),
                'validation': ('validation',), 'test': ('test',)}


def verify_dataset(protocol):
    """Every dataset file must equal its blob in the frozen dataset commit."""
    root, pin = Path(protocol['dataset']), protocol['dataset_pin']
    entries = subprocess.check_output(['git', 'ls-tree', '-rz', pin['commit'], '--', root.as_posix()]).split(b'\0')
    paths = set()
    for entry in filter(None, entries):
        meta, name = entry.decode().split('\t', 1)
        mode, kind, oid = meta.split()
        data = Path(name).read_bytes()
        if kind != 'blob' or mode != '100644' or hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() != oid:
            raise ValueError('dataset differs from frozen commit: ' + name)
        paths.add(Path(name).resolve())
    if paths != {p.resolve() for p in root.rglob('*') if p.is_file()} or len(paths) != pin['files']:
        raise ValueError('frozen dataset file inventory changed')
    if sha(root / 'manifest.json') != pin['manifest_sha256']:
        raise ValueError('frozen dataset manifest changed')
    return {'commit': pin['commit'], 'files_verified': len(paths), 'manifest_sha256': pin['manifest_sha256']}


def splits(protocol):
    sets = {s: load(protocol['dataset'], s) for s in ('search', 'validation', 'test')}
    for s, rows in sets.items():
        if len(rows) != protocol['populations'][s]['n'] or population_id(rows) != protocol['populations'][s]['sha256']:
            raise ValueError('evaluation population changed: ' + s)
    for field in ('id', 'seed', 'image_sha256'):
        values = [r[field] for rows in sets.values() for r in rows]
        if len(set(values)) != len(values):
            raise ValueError('split overlap in ' + field)
    return sets


def folds(protocol, search_rows):
    value = load_folds(protocol['dataset'])
    ids = [r['id'] for r in search_rows]
    if list(value) != list(FOLDS) or sorted(i for f in FOLDS for i in value[f]) != sorted(ids):
        raise ValueError('folds must partition search300')
    if {f: len(value[f]) for f in FOLDS} != {f: protocol['folds'][f]['n'] for f in FOLDS}:
        raise ValueError('fold sizes changed')
    index = {rid: i for i, rid in enumerate(ids)}
    return {f: [index[rid] for rid in value[f]] for f in FOLDS}


def verify_inputs(protocol):
    proof = verify_dataset(protocol)
    sets = splits(protocol)
    folds(protocol, sets['search'])
    plan = candidate_plan(protocol)
    seeds = [s for s, _ in plan]
    if len(set(seeds)) != len(seeds) or set(seeds) & set(protocol['previous_candidate_seeds']):
        raise ValueError('reused candidate seed')
    return proof


def candidate_plan(protocol):
    """Seed seed_start+i gets sigma[i % 4]; exactly 100 per sigma, never changed."""
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


def fold_record(correct, fold_index, base):
    """Search metrics for one candidate from its per-example correctness vector."""
    c = np.asarray(correct, dtype=bool)
    fold = {f: float(c[i].mean()) for f, i in fold_index.items()}
    gain = {f: fold[f] - base['fold_accuracy'][f] for f in FOLDS}
    return {'pooled_correct': int(c.sum()), 'pooled_accuracy': float(c.mean()),
            'pooled_gain': float(c.mean()) - base['pooled_accuracy'],
            'fold_accuracy': fold, 'fold_gain': gain,
            'transfer_score': min(gain.values()), 'mean_fold_gain': sum(gain.values()) / 3}


def rank(records, method):
    """VANILLA: pooled correct desc. TRANSFER_MIN: min fold gain, mean fold gain desc. Hash tie-break."""
    if len(records) != 400 or len({r['candidate']['candidate_id'] for r in records}) != 400:
        raise ValueError('ranking requires all 400 distinct audited search candidates')
    if method == 'VANILLA':
        key = lambda r: (-r['search']['pooled_correct'], rank_hash(r['candidate']['candidate_id']))
    elif method == 'TRANSFER_MIN':
        # Fold gains are multiples of 1/100 after subtracting fixed base values;
        # rounding to 1e-9 removes float noise only, never a real difference.
        key = lambda r: (-round(r['search']['transfer_score'], 9), -round(r['search']['mean_fold_gain'], 9),
                         rank_hash(r['candidate']['candidate_id']))
    else:
        raise ValueError('only the two frozen ranking rules exist')
    return sorted(records, key=key)


def counts(gains):
    return {'strictly_above_base': sum(g > EPS for g in gains), 'at_least_1pp': sum(g >= .01 - EPS for g in gains),
            'at_least_3pp': sum(g >= .03 - EPS for g in gains), 'at_least_5pp': sum(g >= .05 - EPS for g in gains)}


def correlation(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    def pearson(a, b):
        return None if a.std() == 0 or b.std() == 0 else float(np.corrcoef(a, b)[0, 1])
    def ranks(a):
        order = a.argsort(kind='stable'); r = np.empty(len(a)); r[order] = np.arange(len(a))
        for v in np.unique(a): r[a == v] = r[a == v].mean()
        return r
    return {'n': len(x), 'pearson': pearson(x, y), 'spearman': pearson(ranks(x), ranks(y))}


def validation_order(method_top, gains):
    """Validation gain desc, then the method's frozen search rank (list position)."""
    return sorted(method_top, key=lambda r: (-round(gains[r['candidate']['candidate_id']], 9), method_top.index(r)))


def validation_gate(lock, val, protocol):
    """val: candidate_id -> validation gain. Conditions A or B, exactly as frozen."""
    g = protocol['validation_gate']
    k = protocol['validation']['top_k']
    summary = {}
    for m in METHODS:
        top = lock['top'][m]
        if len(top) != k or any(r['candidate']['candidate_id'] not in val for r in top):
            raise ValueError('operational failure: every frozen top-k candidate needs validation')
        gains = [val[r['candidate']['candidate_id']] for r in top]
        summary[m] = {'best_validation_gain': max(gains), 'top5_mean_validation_gain': sum(gains[:5]) / 5,
                      'top10_mean_validation_gain': sum(gains[:10]) / 10, 'top20_counts': counts(gains),
                      'top10_counts': counts(gains[:10])}
    T, V = summary['TRANSFER_MIN'], summary['VANILLA']
    cond_a = {'transfer_best_at_least_3pp': T['best_validation_gain'] >= g['a_best_gain'] - EPS,
              'transfer_best_beats_vanilla_best_by_2pp': T['best_validation_gain'] - V['best_validation_gain'] >= g['a_margin'] - EPS}
    tc, vc = T['top10_counts']['at_least_3pp'], V['top10_counts']['at_least_3pp']
    cond_b = {'transfer_top10_at_least_3_with_3pp': tc >= g['b_count'],
              'transfer_top10_count_exceeds_vanilla_by_2': tc - vc >= g['b_margin']}
    a, b = all(cond_a.values()), all(cond_b.values())
    return {'methods': summary, 'condition_a': cond_a, 'condition_a_pass': a, 'condition_b': cond_b,
            'condition_b_pass': b, 'pass': a or b,
            'decision': 'PASS_TO_FINAL_TEST' if a or b else 'NO_GO_TRANSFER_AWARE_VISUAL_SEARCH'}


def final_selection(lock, val):
    """Best and top-5 validation candidates of each method; deduplicated union."""
    chosen, union = {}, []
    for m in METHODS:
        order = validation_order(lock['top'][m], val)
        chosen[m] = {'best': order[0], 'top5': order[:5]}
        for r in order[:5]:
            if r['candidate']['candidate_id'] not in [u['candidate']['candidate_id'] for u in union]:
                union.append(r)
    return chosen, union


def expert(metric, comparison, base, audit, protocol):
    """Genuine visual expert checks 1-4 (runtime controls are checked globally)."""
    gate = protocol['final_gate']
    levels = {l: metric['difficulty'][l]['accuracy'] - base['difficulty'][l]['accuracy'] for l in ('easy', 'medium', 'hard')}
    checks = {'gain_at_least_5pp': comparison['gain'] >= gate['expert_gain'] - EPS,
              'bootstrap_lower_positive': comparison['paired_bootstrap95'][0] > 0,
              'two_difficulties_improve': sum(d > EPS for d in levels.values()) >= 2,
              'label_concentration_below_80pct': audit['label_spread_pass']}
    return {'difficulty_gain': levels, 'checks': checks, 'genuine_expert': all(checks.values())}


def evaluate_final(selection, outputs, base_outputs, rows, protocol, indices):
    base = summarize(base_outputs, rows)
    limit = protocol['final_gate']['label_concentration_limit']
    per = {}
    for cid, outs in outputs.items():
        metric = summarize(outs, rows)
        comparison = paired(base_outputs, outs, indices)
        audit = label_audit(base_outputs, outs, rows, limit)
        per[cid] = {'test': metric, 'comparison': comparison, 'label_audit': audit,
                    **expert(metric, comparison, base, audit, protocol)}
    return base, per


def final_decision(selection, per, protocol, controls_pass):
    if not controls_pass:
        raise ValueError('operational failure: runtime controls must pass before a scientific decision')
    gate = protocol['final_gate']
    cid = lambda r: r['candidate']['candidate_id']
    methods = {}
    for m in METHODS:
        five = [per[cid(r)]['comparison']['gain'] for r in selection[m]['top5']]
        primary = per[cid(selection[m]['best'])]
        methods[m] = {'primary_candidate_id': cid(selection[m]['best']), 'primary_gain': primary['comparison']['gain'],
                      'primary_genuine_expert': primary['genuine_expert'], 'best_final_gain': max(five),
                      'top5_mean_final_gain': sum(five) / 5, 'top5_at_least_3pp': sum(g >= .03 - EPS for g in five),
                      'any_genuine_expert': any(per[cid(r)]['genuine_expert'] for r in selection[m]['top5'])}
    T, V = methods['TRANSFER_MIN'], methods['VANILLA']
    strong = {'transfer_primary_genuine_expert': T['primary_genuine_expert'], 'controls': controls_pass,
              'vanilla_primary_not_expert_or_transfer_beats_by_2pp':
                  (not V['primary_genuine_expert']) or T['primary_gain'] - V['primary_gain'] >= gate['beat_margin'] - EPS}
    secondary = {'transfer_at_least_3_reach_3pp': T['top5_at_least_3pp'] >= gate['secondary_count'],
                 'transfer_top5_mean_at_least_3pp': T['top5_mean_final_gain'] >= gate['secondary_mean'] - EPS,
                 'vanilla_not_equivalent': not (V['top5_at_least_3pp'] >= gate['secondary_count'] and
                                                V['top5_mean_final_gain'] >= gate['secondary_mean'] - EPS)}
    result = ('GO_TRANSFER_AWARE_VISUAL_SEARCH' if all(strong.values()) else
              'PROMISING_BUT_NOT_EXPERT' if all(secondary.values()) else 'NO_GO_TRANSFER_AWARE_VISUAL_SEARCH')
    return {'decision': result, 'methods': methods, 'strong': strong, 'strong_pass': all(strong.values()),
            'secondary': secondary, 'secondary_pass': all(secondary.values()) and not all(strong.values())}

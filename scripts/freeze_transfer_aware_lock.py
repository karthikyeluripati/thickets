"""Write one transfer-aware lock from audited phases; the caller commits it next.

baseline:   base identities, base summaries and base fold accuracies.
ranking:    VANILLA and TRANSFER_MIN over the same 400 search records; top-20s.
validation: frozen first-bridge gate and, only if it passes, the final union.
"""
import argparse
from pathlib import Path
import sys

import numpy as np

from thicket_runtime.cli import revision_info
from thicket_runtime.committee_validation import committed
from thicket_runtime.line_tracing import sha
from thicket_runtime.visual_decisions import summarize
from thicket_runtime.visual_final import label_audit, paired
from thicket_runtime.visual_runtime import read, write
from thicket_runtime.transfer_aware import (METHODS, correlation, final_selection, fold_record, folds, rank,
                                            splits, validation_gate, verify_inputs)
from thicket_runtime.transfer_aware_audit import phase

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('lock', choices=['baseline', 'ranking', 'validation'])
p.add_argument('--root', type=Path, required=True)
p.add_argument('--protocol', type=Path, default=Path('experiments/transfer_aware_visual_protocol.json'))
a = p.parse_args()
committed(a.protocol)
protocol = read(a.protocol)
verify_inputs(protocol)
locks = {}
for name in {'baseline': [], 'ranking': ['baseline'], 'validation': ['baseline', 'ranking']}[a.lock]:
    committed(a.root/'locks'/(name+'.json'))
    locks[name] = read(a.root/'locks'/(name+'.json'))
sets = splits(protocol)
fold_index = folds(protocol, sets['search'])
common = {'protocol_sha256': sha(a.protocol), 'source': revision_info(), 'command': [sys.executable, *sys.argv]}

if a.lock == 'baseline':
    audit = phase(a.root/'baseline', protocol, a.protocol, {})
    base = audit['base']
    correct = np.asarray([o['correct'] for o in base['search']['outputs']], dtype=bool)
    value = {**common, 'base_id': audit['native']['base_id'], 'phase_manifest_sha256': sha(a.root/'baseline/run-manifest.json'),
             'phase_checksums_sha256': sha(a.root/'baseline/sha256.json'), 'base_path': 'baseline/base.json.gz',
             'output_identities': {s: v['output_identity'] for s, v in base.items()},
             'summary': {s: summarize(base[s]['outputs'], rows) for s, rows in sets.items()},
             'search_base': {'pooled_correct': int(correct.sum()), 'pooled_accuracy': float(correct.mean()),
                             'fold_accuracy': {f: float(correct[i].mean()) for f, i in fold_index.items()}},
             'no_base_capability_gate': True}
elif a.lock == 'ranking':
    shards = protocol['candidates']['count'] // protocol['candidates']['shard_size']
    records, checks = [], {}
    for k in range(shards):
        run = a.root/f'search-shard-{k:02d}'
        audit = phase(run, protocol, a.protocol, {'baseline': locks['baseline']}, shard=k)
        if audit['manifest']['phase'] != 'search': raise ValueError('ranking accepts search records only')
        checks[run.name] = {'manifest_sha256': sha(run/'run-manifest.json'), 'checksums_sha256': sha(run/'sha256.json')}
        for r in audit['records']:
            outs = audit['traces'][r['candidate']['candidate_id']]['splits']['search']['outputs']
            records.append({**r, 'shard': k, 'search': fold_record([o['correct'] for o in outs], fold_index, locks['baseline']['search_base'])})
    if len({r['candidate_state_id'] for r in records}) != 400: raise ValueError('search fingerprints are not distinct')
    k = protocol['validation']['top_k']
    ranked = {m: rank(records, m) for m in METHODS}
    top = {m: ranked[m][:k] for m in METHODS}
    union = list(top['VANILLA']) + [r for r in top['TRANSFER_MIN'] if r not in top['VANILLA']]
    value = {**common, 'search_phases': checks, 'records': records,
             'ranked_ids': {m: [r['candidate']['candidate_id'] for r in ranked[m]] for m in METHODS},
             'top': top, 'top_overlap': sum(r in top['VANILLA'] for r in top['TRANSFER_MIN']),
             'validation_union': union, 'ranking_inputs': 'audited search300 records only; no validation/test candidate output exists'}
else:
    audit = phase(a.root/'validation', protocol, a.protocol, locks)
    rows = sets['validation']
    base_out = read(a.root/locks['baseline']['base_path'])['validation']['outputs']
    base = summarize(base_out, rows)
    indices = np.random.default_rng(protocol['statistics']['bootstrap_seed']).integers(
        0, len(rows), size=(protocol['statistics']['bootstrap_replicates'], len(rows)), dtype=np.int64)
    per, gains = {}, {}
    for r in locks['ranking']['validation_union']:
        cid = r['candidate']['candidate_id']
        outs = audit['traces'][cid]['splits']['validation']['outputs']
        metric = summarize(outs, rows)
        gains[cid] = metric['accuracy'] - base['accuracy']
        per[cid] = {'candidate': r['candidate'], 'candidate_state_id': r['candidate_state_id'], 'search': r['search'],
                    'validation': metric, 'validation_gain': gains[cid],
                    'difficulty_gain': {l: metric['difficulty'][l]['accuracy'] - base['difficulty'][l]['accuracy'] for l in ('easy', 'medium', 'hard')},
                    'label_audit': label_audit(base_out, outs, rows, protocol['final_gate']['label_concentration_limit']),
                    'paired': paired(base_out, outs, indices), 'correct': [o['correct'] for o in outs]}
    gate = validation_gate(locks['ranking'], gains, protocol)
    union = locks['ranking']['validation_union']
    corr = {'union_pooled_search_gain': correlation([r['search']['pooled_gain'] for r in union], [gains[r['candidate']['candidate_id']] for r in union]),
            'union_transfer_score': correlation([r['search']['transfer_score'] for r in union], [gains[r['candidate']['candidate_id']] for r in union]),
            'vanilla_top20_pooled_search_gain': correlation([r['search']['pooled_gain'] for r in locks['ranking']['top']['VANILLA']],
                                                            [gains[r['candidate']['candidate_id']] for r in locks['ranking']['top']['VANILLA']]),
            'transfer_top20_transfer_score': correlation([r['search']['transfer_score'] for r in locks['ranking']['top']['TRANSFER_MIN']],
                                                         [gains[r['candidate']['candidate_id']] for r in locks['ranking']['top']['TRANSFER_MIN']])}
    value = {**common, 'validation_phase': {'manifest_sha256': sha(a.root/'validation/run-manifest.json'),
                                            'checksums_sha256': sha(a.root/'validation/sha256.json')},
             'base': base, 'base_correct': [o['correct'] for o in base_out], 'candidates': per,
             'transfer_correlation': corr, 'gate': gate}
    if gate['pass']:
        chosen, final = final_selection(locks['ranking'], gains)
        value.update(final_selection=chosen, final_union=final)
    else:
        value.update(final_selection=None, final_union=[], stop='NO_GO_TRANSFER_AWARE_VISUAL_SEARCH: final test forbidden')
out = a.root/'locks'/(a.lock+'.json')
write(out, value)
print({'lock': a.lock, 'sha256': sha(out), **({'decision': value['gate']['decision']} if a.lock == 'validation' else {})})

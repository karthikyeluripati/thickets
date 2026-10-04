"""Write one OmniSpatial lock from audited phases; the caller commits it next.

baseline:   base identities/summaries and the per-image official preprocessing record.
search:     all 400 audited SEARCH records, full ranking and frozen top 30.
validation: top-30 VALIDATION comparison, validation ranking, frozen rank 1 and top 5.
"""
import argparse
from pathlib import Path
import sys

import numpy as np

from thicket_runtime.cli import revision_info
from thicket_runtime.committee_validation import committed
from thicket_runtime.line_tracing import sha
from thicket_runtime.visual_runtime import read, write
from thicket_runtime.omnispatial import compare, counts, rank_search, rank_validation, splits, summarize, verify_inputs
from thicket_runtime.omnispatial_audit import phase

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('lock', choices=['baseline', 'search', 'validation'])
p.add_argument('--root', type=Path, required=True)
p.add_argument('--protocol', type=Path, default=Path('experiments/omnispatial_visual_expert_protocol.json'))
a = p.parse_args()
committed(a.protocol)
protocol = read(a.protocol)
verify_inputs(protocol)
locks = {}
for name in {'baseline': [], 'search': ['baseline'], 'validation': ['baseline', 'search']}[a.lock]:
    committed(a.root/'locks'/(name+'.json'))
    locks[name] = read(a.root/'locks'/(name+'.json'))
sets = splits(protocol)
common = {'protocol_sha256': sha(a.protocol), 'source': revision_info(), 'command': [sys.executable, *sys.argv]}
base_raw = read(a.root/'baseline/base.json.gz') if a.lock != 'baseline' else None
base_ref = {s: v['outputs'] for s, v in base_raw.items()} if base_raw else None

if a.lock == 'baseline':
    audit = phase(a.root/'baseline', protocol, a.protocol, {})
    base = audit['base']
    value = {**common, 'base_id': audit['native']['base_id'], 'phase_manifest_sha256': sha(a.root/'baseline/run-manifest.json'),
             'phase_checksums_sha256': sha(a.root/'baseline/sha256.json'), 'base_path': 'baseline/base.json.gz',
             'output_identities': {s: v['output_identity'] for s, v in base.items()},
             'summary': {s: summarize(base[s]['outputs'], rows) for s, rows in sets.items()},
             'preprocess': audit['preprocess'], 'no_base_capability_gate': True}
elif a.lock == 'search':
    if sha(a.root/'baseline/base.json.gz') != read(a.root/'baseline/sha256.json')['base.json.gz']: raise ValueError('baseline outputs changed')
    shards = protocol['candidates']['count'] // protocol['candidates']['shard_size']
    records, checks = [], {}
    for k in range(shards):
        run = a.root/f'search-shard-{k:02d}'
        audit = phase(run, protocol, a.protocol, {'baseline': locks['baseline']}, shard=k, base_reference=base_ref)
        if audit['manifest']['phase'] != 'search': raise ValueError('ranking accepts search records only')
        checks[run.name] = {'manifest_sha256': sha(run/'run-manifest.json'), 'checksums_sha256': sha(run/'sha256.json')}
        for r in audit['records']:
            outs = audit['traces'][r['candidate']['candidate_id']]['splits']['search']['outputs']
            records.append({**r, 'shard': k, 'search': summarize(outs, sets['search'])})
    if len({r['candidate_state_id'] for r in records}) != len(records): raise ValueError('search fingerprints are not distinct')
    ranked = rank_search(records, protocol)
    k = protocol['search']['top_k']
    value = {**common, 'search_phases': checks, 'records': records, 'ranked_ids': [r['candidate']['candidate_id'] for r in ranked],
             'best': ranked[0], 'top30': ranked[:k], 'base_search': locks['baseline']['summary']['search'],
             'ranking_inputs': 'audited SEARCH records only; no validation/test candidate output exists'}
else:
    audit = phase(a.root/'validation', protocol, a.protocol, locks, base_reference=base_ref)
    rows = sets['validation']
    base_out = base_raw['validation']['outputs']
    indices = np.random.default_rng(protocol['statistics']['bootstrap_seed']).integers(
        0, len(rows), size=(protocol['statistics']['bootstrap_replicates'], len(rows)), dtype=np.int64)
    top = locks['search']['top30']
    results, correct = {}, {}
    for r in top:
        cid = r['candidate']['candidate_id']
        outs = audit['traces'][cid]['splits']['validation']['outputs']
        results[cid] = compare(base_out, outs, rows, indices, protocol['gate']['answer_share_limit'])
        correct[cid] = results[cid]['summary']['correct_count']
    order = rank_validation(top, correct)
    value = {**common, 'validation_phase': {'manifest_sha256': sha(a.root/'validation/run-manifest.json'),
                                            'checksums_sha256': sha(a.root/'validation/sha256.json')},
             'base': summarize(base_out, rows), 'candidates': results,
             'validation_ranked_ids': [r['candidate']['candidate_id'] for r in order],
             'rank1': order[0], 'top5': order[:5],
             'secondary_top30_counts': counts([results[r['candidate']['candidate_id']]['gain'] for r in top]),
             'secondary_note': 'validation-enriched frozen top 30; not an unbiased expert density',
             'selection_inputs': 'VALIDATION outputs of the frozen search top 30 only; no test candidate output exists'}
out = a.root/'locks'/(a.lock+'.json')
write(out, value)
print({'lock': a.lock, 'sha256': sha(out)})

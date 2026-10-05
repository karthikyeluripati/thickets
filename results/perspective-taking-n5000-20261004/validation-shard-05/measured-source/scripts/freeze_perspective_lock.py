"""Write one Perspective-Taking N=5000 lock from audited phases; the caller commits it next.

baseline:   base identities/summaries, base fingerprints and per-image preprocessing record.
search:     all 5000 audited SEARCH records, complete ranking and frozen top 50.
validation: top-50 VALIDATION comparison, ranking, frozen rank 1/top 5/top 10,
            and the precommitted density audit (never used for selection).
"""
import argparse
from pathlib import Path
import sys

import numpy as np

from thicket_runtime.cli import revision_info
from thicket_runtime.committee_validation import committed
from thicket_runtime.line_tracing import sha
from thicket_runtime.visual_runtime import read, write
from thicket_runtime.transfer_aware import correlation
from thicket_runtime.perspective import compare, counts, density, manifest, rank_search, rank_validation, splits, summarize, verify_inputs
from thicket_runtime.perspective_audit import phase
from thicket_runtime.perspective_runtime import VALIDATION_SHARD_SIZE, validation_plan

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('lock', choices=['baseline', 'search', 'validation'])
p.add_argument('--root', type=Path, required=True)
p.add_argument('--protocol', type=Path, default=Path('experiments/perspective_taking_n5000_protocol.json'))
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
    value = {**common, 'base_id': audit['native']['base_id'], 'base_id_flat_sha256': audit['native']['base_id_flat_sha256'],
             'phase_manifest_sha256': sha(a.root/'baseline/run-manifest.json'), 'phase_checksums_sha256': sha(a.root/'baseline/sha256.json'),
             'base_path': 'baseline/base.json.gz', 'output_identities': {s: v['output_identity'] for s, v in base.items()},
             'summary': {s: summarize(base[s]['outputs'], rows) for s, rows in sets.items()},
             'preprocess': audit['preprocess'], 'memory': read(a.root/'baseline/memory.json'),
             'fast_control_equivalence': read(a.root/'baseline/fast-control-equivalence.json')['checks'],
             'no_base_capability_gate': True}
else:
    if sha(a.root/'baseline/base.json.gz') != read(a.root/'baseline/sha256.json')['base.json.gz']: raise ValueError('baseline outputs changed')

if a.lock == 'search':
    plan = manifest(protocol)
    index_of = {(r['seed'], r['sigma']): r['index'] for r in plan}
    shards = sorted({r['shard'] for r in plan})
    records, checks = [], {}
    for k in shards:
        run = a.root/f'search-shard-{k:03d}'
        audit = phase(run, protocol, a.protocol, {'baseline': locks['baseline']}, shard=k, base_reference=base_ref)
        if audit['manifest']['phase'] != 'search': raise ValueError('ranking accepts search records only')
        checks[run.name] = {'manifest_sha256': sha(run/'run-manifest.json'), 'checksums_sha256': sha(run/'sha256.json')}
        for r in audit['records']:
            outs = audit['traces'][r['candidate']['candidate_id']]['splits']['search']['outputs']
            records.append({**r, 'index': index_of[(r['candidate']['seed'], r['candidate']['sigma'])], 'shard': k,
                            'search': summarize(outs, sets['search'])})
    if sorted(r['index'] for r in records) != list(range(5000)): raise ValueError('missing or duplicate manifest candidates')
    if len({r['candidate_state_id'] for r in records}) != 5000: raise ValueError('search fingerprints are not distinct')
    for s in protocol['candidates']['sigma_mixture']:
        if sum(r['candidate']['sigma'] == s for r in records) != 1250: raise ValueError('sigma stratum incomplete')
    ranked = rank_search(records, protocol)
    k = protocol['search']['top_k']
    value = {**common, 'search_phases': checks, 'records': sorted(records, key=lambda r: r['index']),
             'ranked_ids': [r['candidate']['candidate_id'] for r in ranked], 'best': ranked[0], 'top50': ranked[:k],
             'base_search': locks['baseline']['summary']['search'],
             'population_checks': {'unique_candidates': 5000, 'per_sigma': 1250, 'missing': 0, 'duplicates': 0},
             'ranking_inputs': 'audited SEARCH records only; no VALIDATION/TEST candidate output exists'}
elif a.lock == 'validation':
    plan = validation_plan(locks, protocol)
    traces, checks = {}, {}
    for j in range(-(-len(plan) // VALIDATION_SHARD_SIZE)):
        run = a.root/f'validation-shard-{j:02d}'
        audit = phase(run, protocol, a.protocol, locks, shard=j, base_reference=base_ref)
        if set(traces) & set(audit['traces']): raise ValueError('candidate validated twice')
        traces.update(audit['traces'])
        checks[run.name] = {'manifest_sha256': sha(run/'run-manifest.json'), 'checksums_sha256': sha(run/'sha256.json')}
    if set(traces) != {r['candidate']['candidate_id'] for r in plan}: raise ValueError('validation does not cover the frozen plan exactly')
    rows = sets['validation']
    base_out = base_raw['validation']['outputs']
    indices = np.random.default_rng(protocol['statistics']['bootstrap_seed']).integers(
        0, len(rows), size=(protocol['statistics']['bootstrap_replicates'], len(rows)), dtype=np.int64)
    limit = protocol['gate']['answer_share_limit']
    top = locks['search']['top50']
    results = {r['candidate']['candidate_id']: compare(base_out, traces[r['candidate']['candidate_id']]['splits']['validation']['outputs'],
                                                       rows, indices, limit) for r in top}
    order = rank_validation(top, {c: x['summary']['correct_count'] for c, x in results.items()})
    base_search = locks['baseline']['summary']['search']['accuracy']
    sgain = [r['correct_count'] / 200 - base_search for r in top]
    vgain = [results[r['candidate']['candidate_id']]['gain'] for r in top]
    by_index = {r['index']: r for r in locks['search']['records']}
    base_acc = summarize(base_out, rows)['accuracy']
    audit_gains = {}
    for i in protocol['density_audit']['indices']:
        r = by_index[i]
        outs = traces[r['candidate']['candidate_id']]['splits']['validation']['outputs']
        audit_gains.setdefault(r['candidate']['sigma'], []).append(summarize(outs, rows)['accuracy'] - base_acc)
    top10_search = {r['candidate']['candidate_id'] for r in top[:10]}
    value = {**common, 'validation_phases': checks, 'base': summarize(base_out, rows), 'candidates': results,
             'validation_ranked_ids': [r['candidate']['candidate_id'] for r in order],
             'rank1': order[0], 'top5': order[:5], 'top10': order[:10],
             'search_to_validation': {'search_gain': sgain, 'validation_gain': vgain, **correlation(sgain, vgain),
                                      'validation_counts': counts(vgain),
                                      'search_top10_remaining_validation_top10': len(top10_search & {r['candidate']['candidate_id'] for r in order[:10]})},
             'density_audit': {'note': 'precommitted random 500 (125 per sigma), evaluated regardless of SEARCH score; VALIDATION gains only; not used for selection; not final transferable density',
                               'gains_by_sigma': {str(k): v for k, v in audit_gains.items()}, 'summary': density(audit_gains)},
             'selection_inputs': 'VALIDATION outputs of the frozen top 50 only; no TEST candidate output exists'}
out = a.root/'locks'/(a.lock+'.json')
write(out, value)
print({'lock': a.lock, 'sha256': sha(out)})

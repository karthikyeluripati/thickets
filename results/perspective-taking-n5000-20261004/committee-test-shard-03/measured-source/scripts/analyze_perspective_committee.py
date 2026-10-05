"""RandOpt top-50 ensemble re-evaluation of the frozen Perspective-Taking N=5000 population.

Reads only: the committed committee lock, the audited existing TEST phase (members
already tested) and the audited committee TEST shards. Writes a new directory.
"""
import argparse
from datetime import datetime
import hashlib
from pathlib import Path
import statistics
import subprocess
import sys

import numpy as np

from thicket_runtime.cli import revision_info
from thicket_runtime.committee_validation import committed
from thicket_runtime.line_tracing import sha
from thicket_runtime.visual_runtime import read, write
from thicket_runtime.perspective import EPS, compare, counts, majority, splits, summarize, verify_inputs
from thicket_runtime.perspective_audit import phase
from thicket_runtime.perspective_runtime import COMMITTEE_SHARD_SIZE

K_VALUES = (1, 5, 10, 25, 50)
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', type=Path, required=True)
p.add_argument('--protocol', type=Path, default=Path('experiments/perspective_taking_n5000_protocol.json'))
p.add_argument('--addendum', type=Path, default=Path('experiments/perspective_randopt_top50_addendum.json'))
a = p.parse_args()
out = a.root/'randopt-top50'
if out.exists(): raise FileExistsError(out)
for path in (a.protocol, a.addendum, *(a.root/'locks'/(n+'.json') for n in ('baseline', 'search', 'validation', 'committee'))):
    committed(path)
protocol, addendum = read(a.protocol), read(a.addendum)
proof = verify_inputs(protocol)
L = {n: read(a.root/'locks'/(n+'.json')) for n in ('baseline', 'search', 'validation', 'committee')}
c = L['committee']
if c['addendum_sha256'] != sha(a.addendum) or c['search_lock_sha256'] != sha(a.root/'locks/search.json'):
    raise ValueError('committee lock does not match addendum/search lock')
if [r['candidate']['candidate_id'] for r in c['committee']] != L['search']['ranked_ids'][:50]:
    raise ValueError('committee is not the committed SEARCH top 50')
rows = splits(protocol)['test']
base_raw = read(a.root/'baseline/base.json.gz')
ref = {s: v['outputs'] for s, v in base_raw.items()}
base_out = ref['test']
outputs, source, timeline = {}, {}, {}
existing = phase(a.root/'test', protocol, a.protocol, {k: L[k] for k in ('baseline', 'search', 'validation')}, base_reference=ref)
for cid in c['existing_test_outputs']['candidate_ids']:
    outputs[cid] = existing['traces'][cid]['splits']['test']['outputs']; source[cid] = 'test'
lock_commit = subprocess.check_output(['git', 'log', '-1', '--format=%H %cI', '--', str(a.root/'locks/committee.json')], text=True).strip().split()
for j in range(-(-len(c['to_generate']) // COMMITTEE_SHARD_SIZE)):
    d = a.root/f'committee-test-shard-{j:02d}'
    audit = phase(d, protocol, a.protocol, {k: L[k] for k in ('baseline', 'search', 'committee')}, shard=j, base_reference=ref)
    m = audit['manifest']; commit, when = m['lock_commits']['committee'].split()
    if commit != lock_commit[0] or datetime.fromisoformat(when) >= datetime.fromisoformat(m['started_utc']):
        raise ValueError('committee lock did not predate committee TEST inference')
    timeline[d.name] = [when, m['started_utc']]
    for cid, t in audit['traces'].items():
        if cid in outputs: raise ValueError('member evaluated twice')
        outputs[cid] = t['splits']['test']['outputs']; source[cid] = d.name
members = [r['candidate']['candidate_id'] for r in c['committee']]
if set(outputs) != set(members): raise ValueError('TEST outputs do not cover exactly the committee')
idx = np.random.default_rng(protocol['statistics']['bootstrap_seed']).integers(0, len(rows), size=(10000, len(rows)), dtype=np.int64)
base = summarize(base_out, rows)
vectors = [outputs[cid] for cid in members]
ensembles = {}
for k in K_VALUES:
    vote = majority(vectors, rows, k)
    res = compare(base_out, vote, rows, idx, 1.0)
    ensembles[str(k)] = {'k': k, 'members': members[:k], **res,
                         'invalid_or_empty_votes': sum(not v['valid_letter'] for v in vote)}
individual = []
for r, cid in zip(c['committee'], members):
    res = compare(base_out, outputs[cid], rows, idx, .8)
    individual.append({'search_rank': r['search_rank'], 'candidate': r['candidate'], 'search_correct': r['correct_count'],
                       'test_source': source[cid], **res})
gains = [x['gain'] for x in individual]
accs = [x['summary']['accuracy'] for x in individual]
indiv = {'best_accuracy': max(accs), 'best_gain': max(gains), 'best_search_rank': individual[int(np.argmax(accs))]['search_rank'],
         'mean_accuracy': statistics.mean(accs), 'median_accuracy': statistics.median(accs),
         'mean_gain': statistics.mean(gains), **counts(gains)}
g50 = ensembles['50']['gain']
decision = {'standalone_expert': 'NO_GO_STANDALONE_PERSPECTIVE_EXPERT',
            'standalone_source': 'previous study decision NO_GO_VISUAL_NEURAL_THICKET_PERSPECTIVE_N5000 (unchanged)',
            'randopt_ensemble': 'GO_RANDOPT_PERSPECTIVE_ENSEMBLE' if g50 > EPS else 'NO_GO_RANDOPT_PERSPECTIVE_ENSEMBLE',
            'rule': addendum['decision'], 'top50_gain': g50, 'top50_paired': ensembles['50']['paired']}
out.mkdir()
write(out/'manifest.json', {'source': revision_info(), 'command': [sys.executable, *sys.argv], 'addendum_sha256': sha(a.addendum),
      'protocol_sha256': sha(a.protocol), 'committee_lock_sha256': sha(a.root/'locks/committee.json'),
      'committee_lock_commit': lock_commit, 'committee_before_inference': timeline, 'dataset_proof': proof,
      'test_sources': source, 'bootstrap_indices_sha256': hashlib.sha256(idx.tobytes()).hexdigest(),
      'all_raw_generations_rescored': True})
write(out/'base_test.json', base)
write(out/'ensembles.json', ensembles)
write(out/'individual.json', {'summary': indiv, 'members': individual})
write(out/'decision.json', decision)
write(out/'per_question.json.gz', {'uids': [r['uid'] for r in rows], 'true_letter': [chr(65 + r['answer']) for r in rows],
      'sub_task': [r['sub_task_type'] for r in rows], 'base_prediction': [o['prediction'] for o in base_out],
      'members': members, 'predictions': [[o['prediction'] if o['valid_letter'] else '' for o in outputs[cid]] for cid in members],
      'ensemble_prediction': {str(k): [v['prediction'] for v in majority(vectors, rows, k)] for k in K_VALUES}})
print(decision['randopt_ensemble'], {k: round(100 * v['gain'], 2) for k, v in ensembles.items()})

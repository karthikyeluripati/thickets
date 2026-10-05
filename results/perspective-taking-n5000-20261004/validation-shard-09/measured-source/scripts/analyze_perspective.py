"""Audit the frozen Perspective-Taking N=5000 study and apply the precommitted decision."""
import argparse
from datetime import datetime
import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np

from thicket_runtime.cli import revision_info
from thicket_runtime.committee_validation import committed
from thicket_runtime.line_tracing import sha
from thicket_runtime.visual_runtime import read, write
from thicket_runtime.perspective import compare, decision, majority, manifest, splits, summarize, verify_inputs
from thicket_runtime.perspective_audit import phase

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', type=Path, required=True)
p.add_argument('--protocol', type=Path, default=Path('experiments/perspective_taking_n5000_protocol.json'))
a = p.parse_args()
out = a.root/'analysis'
if out.exists(): raise FileExistsError(out)
committed(a.protocol)
protocol = read(a.protocol)
proof = verify_inputs(protocol)
locks = {}
for name in ('baseline', 'search', 'validation'):
    committed(a.root/'locks'/(name+'.json'))
    locks[name] = read(a.root/'locks'/(name+'.json'))


def before(lock, phase_dir):
    m = read(phase_dir/'run-manifest.json')
    commit, when = m['lock_commits'][lock].split()
    path = a.root/'locks'/(lock+'.json')
    if subprocess.check_output(['git', 'show', commit+':'+path.as_posix()]) != path.read_bytes():
        raise ValueError('phase did not use committed lock bytes: '+lock)
    if datetime.fromisoformat(when) >= datetime.fromisoformat(m['started_utc']): raise ValueError(lock+' committed after phase start')
    return [commit, when, m['started_utc']]


timeline = {f'baseline<search-shard-{k:03d}': before('baseline', a.root/f'search-shard-{k:03d}')
            for k in sorted({r['shard'] for r in manifest(protocol)})}
timeline.update({f'search<{n}': before('search', a.root/n) for n in locks['validation']['validation_phases']})
timeline['validation<test'] = before('validation', a.root/'test')
base_raw = read(a.root/'baseline/base.json.gz')
audit = phase(a.root/'test', protocol, a.protocol, locks, base_reference={s: v['outputs'] for s, v in base_raw.items()})
rows = splits(protocol)['test']
base_out = base_raw['test']['outputs']
indices = np.random.default_rng(protocol['statistics']['bootstrap_seed']).integers(
    0, len(rows), size=(protocol['statistics']['bootstrap_replicates'], len(rows)), dtype=np.int64)
limit = protocol['gate']['answer_share_limit']
final, vectors = [], []
for rank, r in enumerate(locks['validation']['top10'], 1):
    cid = r['candidate']['candidate_id']
    outs = audit['traces'][cid]['splits']['test']['outputs']
    vectors.append(outs)
    final.append({'validation_rank': rank, 'candidate': r['candidate'], 'candidate_state_id': r['candidate_state_id'],
                  'search_correct': r['correct_count'], 'validation_gain': locks['validation']['candidates'][cid]['gain'],
                  **compare(base_out, outs, rows, indices, limit)})
result = decision(final, protocol, True)
ensembles = {}
for k in (5, 10):
    vote = majority(vectors, rows, k)
    ensembles[str(k)] = {**compare(base_out, vote, rows, indices, limit), 'at_least_7pp_secondary': None}
    ensembles[str(k)]['at_least_7pp_secondary'] = ensembles[str(k)]['gain'] >= .07 - 1e-12
write(out/'manifest.json', {'source': revision_info(), 'command': [sys.executable, *sys.argv], 'protocol_sha256': sha(a.protocol),
      'dataset_proof': proof, 'lock_sha256': {k: sha(a.root/'locks'/(k+'.json')) for k in locks}, 'timeline': timeline,
      'bootstrap_indices_sha256': hashlib.sha256(indices.tobytes()).hexdigest(), 'numpy': np.__version__, 'all_raw_generations_rescored': True})
write(out/'test_base.json', summarize(base_out, rows))
write(out/'final_candidates.json', final)
write(out/'ensembles.json', {**ensembles, 'note': 'secondary; never a substitute for an individual expert'})
write(out/'decision.json', result)
write(out/'per_question.json.gz', {'uids': [r['uid'] for r in rows], 'true_letter': [chr(65 + r['answer']) for r in rows],
      'sub_task': [r['sub_task_type'] for r in rows], 'base_correct': [o['correct'] for o in base_out],
      'base_prediction': [o['prediction'] for o in base_out],
      'candidates': {r['candidate']['candidate_id']: {'correct': [o['correct'] for o in v], 'prediction': [o['prediction'] for o in v]}
                     for r, v in zip(locks['validation']['top10'], vectors)}})
print(result['decision'])

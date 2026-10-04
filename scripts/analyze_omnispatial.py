"""Audit the frozen OmniSpatial study end to end and apply the precommitted decision."""
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
from thicket_runtime.omnispatial import compare, decision, splits, summarize, verify_inputs
from thicket_runtime.omnispatial_audit import phase

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', type=Path, required=True)
p.add_argument('--protocol', type=Path, default=Path('experiments/omnispatial_visual_expert_protocol.json'))
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
    path = (a.root/'locks'/(lock+'.json'))
    if subprocess.check_output(['git', 'show', commit+':'+path.as_posix()]) != path.read_bytes():
        raise ValueError('phase did not use committed lock bytes: '+lock)
    if datetime.fromisoformat(when) >= datetime.fromisoformat(m['started_utc']): raise ValueError(lock+' committed after phase start')
    return {'commit': commit, 'committed_at': when, 'phase_started_utc': m['started_utc']}


timeline = {f'baseline_before_search_shard_{k:02d}': before('baseline', a.root/f'search-shard-{k:02d}')
            for k in range(protocol['candidates']['count'] // protocol['candidates']['shard_size'])}
for j in range(len(locks['validation']['validation_phases'])):
    timeline[f'search_before_validation_shard_{j:02d}'] = before('search', a.root/f'validation-shard-{j:02d}')
timeline['validation_before_test'] = before('validation', a.root/'test')
base_raw = read(a.root/'baseline/base.json.gz')
base_ref = {s: v['outputs'] for s, v in base_raw.items()}
audit = phase(a.root/'test', protocol, a.protocol, locks, base_reference=base_ref)
rows = splits(protocol)['test']
base_out = base_raw['test']['outputs']
indices = np.random.default_rng(protocol['statistics']['bootstrap_seed']).integers(
    0, len(rows), size=(protocol['statistics']['bootstrap_replicates'], len(rows)), dtype=np.int64)
final = []
for rank, r in enumerate(locks['validation']['top5'], 1):
    cid = r['candidate']['candidate_id']
    res = compare(base_out, audit['traces'][cid]['splits']['test']['outputs'], rows, indices, protocol['gate']['answer_share_limit'])
    final.append({'validation_rank': rank, 'candidate': r['candidate'], 'candidate_state_id': r['candidate_state_id'],
                  'search_correct': r['correct_count'], 'validation_gain': locks['validation']['candidates'][cid]['gain'], **res})
result = decision(final, protocol, True)
write(out/'manifest.json', {'source': revision_info(), 'command': [sys.executable, *sys.argv], 'protocol_sha256': sha(a.protocol),
      'dataset_proof': proof, 'lock_sha256': {k: sha(a.root/'locks'/(k+'.json')) for k in locks}, 'timeline': timeline,
      'bootstrap_indices_sha256': hashlib.sha256(indices.tobytes()).hexdigest(), 'numpy': np.__version__,
      'all_raw_generations_rescored': True})
write(out/'test_base.json', summarize(base_out, rows))
write(out/'final_candidates.json', final)
write(out/'decision.json', result)
write(out/'per_question.json.gz', {'uids': [r['uid'] for r in rows], 'true_letter': [chr(65 + r['answer']) for r in rows],
      'sub_task': [r['sub_task_type'] for r in rows], 'base_correct': [o['correct'] for o in base_out],
      'candidate_correct': {r['candidate']['candidate_id']: [o['correct'] for o in audit['traces'][r['candidate']['candidate_id']]['splits']['test']['outputs']]
                            for r in locks['validation']['top5']}})
print(result['decision'])

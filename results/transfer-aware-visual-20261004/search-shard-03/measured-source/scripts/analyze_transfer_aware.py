"""Audit the frozen transfer-aware study and apply the precommitted decision.

If the committed validation gate failed, only search/validation summaries are
written and the decision is NO_GO; the test phase must not exist.
"""
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
from thicket_runtime.transfer_aware import METHODS, evaluate_final, final_decision, splits, verify_inputs
from thicket_runtime.transfer_aware_audit import phase

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', type=Path, required=True)
p.add_argument('--protocol', type=Path, default=Path('experiments/transfer_aware_visual_protocol.json'))
a = p.parse_args()
out = a.root/'analysis'
if out.exists(): raise FileExistsError(out)
committed(a.protocol)
protocol = read(a.protocol)
proof = verify_inputs(protocol)
locks = {}
for name in ('baseline', 'ranking', 'validation'):
    committed(a.root/'locks'/(name+'.json'))
    locks[name] = read(a.root/'locks'/(name+'.json'))


def before(lock, phase_dir):
    """The lock commit must precede the phase that consumed it."""
    path = (a.root/'locks'/(lock+'.json')).as_posix()
    commit, when = read(phase_dir/'run-manifest.json')['lock_commits'][lock].split()
    if subprocess.check_output(['git', 'show', commit+':'+path]) != (a.root/'locks'/(lock+'.json')).read_bytes():
        raise ValueError('phase did not use committed lock bytes: '+lock)
    started = read(phase_dir/'run-manifest.json')['started_utc']
    if datetime.fromisoformat(when) >= datetime.fromisoformat(started): raise ValueError(lock+' committed after phase start')
    return {'commit': commit, 'committed_at': when, 'phase_started_utc': started}


timeline = {'ranking_before_validation': before('ranking', a.root/'validation')}
for k in range(protocol['candidates']['count'] // protocol['candidates']['shard_size']):
    timeline[f'baseline_before_search_shard_{k:02d}'] = before('baseline', a.root/f'search-shard-{k:02d}')
sets = splits(protocol)
gate = locks['validation']['gate']
summary = {'validation_gate': gate, 'transfer_correlation': locks['validation']['transfer_correlation'],
           'base': locks['baseline']['summary'], 'search_base': locks['baseline']['search_base']}
recs = locks['ranking']['records']
summary['search_best'] = {m: {'pooled_gain': locks['ranking']['top'][m][0]['search']['pooled_gain'],
                              'transfer_score': locks['ranking']['top'][m][0]['search']['transfer_score']} for m in METHODS}
summary['search_distribution_by_sigma'] = {
    str(s): {'n': len(g), 'mean_pooled_gain': float(np.mean(g)), 'max_pooled_gain': float(np.max(g)),
             'at_least_3pp': int(np.sum(np.asarray(g) >= .03 - 1e-12))}
    for s in protocol['candidates']['sigma_mixture'] for g in [[r['search']['pooled_gain'] for r in recs if r['candidate']['sigma'] == s]]}
test_dir = a.root/'test'
if not gate['pass']:
    if test_dir.exists(): raise ValueError('protocol violation: test phase exists after a failed validation gate')
    decision = {'decision': 'NO_GO_TRANSFER_AWARE_VISUAL_SEARCH', 'stage': 'validation', 'final_test_run': False}
    per = None
else:
    timeline['validation_before_test'] = before('validation', test_dir)
    audit = phase(test_dir, protocol, a.protocol, locks)
    rows = sets['test']
    base_out = read(a.root/locks['baseline']['base_path'])['test']['outputs']
    n = len(rows)
    indices = np.random.default_rng(protocol['statistics']['bootstrap_seed']).integers(
        0, n, size=(protocol['statistics']['bootstrap_replicates'], n), dtype=np.int64)
    outs = {cid: t['splits']['test']['outputs'] for cid, t in audit['traces'].items()}
    selection = locks['validation']['final_selection']
    base, per = evaluate_final(selection, outs, base_out, rows, protocol, indices)
    decision = final_decision(selection, per, protocol, True) | {'stage': 'final_test', 'final_test_run': True}
    summary['test_base'] = base
    summary['bootstrap_indices_sha256'] = hashlib.sha256(indices.tobytes()).hexdigest()
    write(out/'per_question.json.gz', {'ids': [r['id'] for r in rows], 'true_labels': [r['answer'] for r in rows],
          'difficulty': [r['difficulty'] for r in rows], 'base_correct': [o['correct'] for o in base_out],
          'candidate_correct': {cid: [o['correct'] for o in v] for cid, v in outs.items()}})
    write(out/'final_candidates.json', per)
write(out/'manifest.json', {'source': revision_info(), 'command': [sys.executable, *sys.argv], 'protocol_sha256': sha(a.protocol),
      'dataset_proof': proof, 'lock_sha256': {k: sha(a.root/'locks'/(k+'.json')) for k in locks}, 'timeline': timeline,
      'numpy': np.__version__, 'all_raw_generations_rescored': True})
write(out/'summary.json', summary)
write(out/'decision.json', decision)
print(decision['decision'])

"""Freeze the RandOpt top-50 committee for the Perspective-Taking re-evaluation.

The committee is the committed SEARCH top 50, read byte-for-byte from the search
lock that predates validation. No validation or test output is read to choose it.
Members already evaluated on TEST by the completed study are reused (and audited);
only the rest are scheduled for TEST generation.
"""
import argparse
from pathlib import Path
import subprocess
import sys

from thicket_runtime.cli import revision_info
from thicket_runtime.committee_validation import committed
from thicket_runtime.line_tracing import sha
from thicket_runtime.visual_runtime import read, write

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', type=Path, required=True)
p.add_argument('--protocol', type=Path, default=Path('experiments/perspective_taking_n5000_protocol.json'))
p.add_argument('--addendum', type=Path, default=Path('experiments/perspective_randopt_top50_addendum.json'))
a = p.parse_args()
for path in (a.protocol, a.addendum, a.root/'locks/search.json', a.root/'locks/validation.json'):
    committed(path)
search, addendum = read(a.root/'locks/search.json'), read(a.addendum)
top50 = search['top50']
if len(top50) != 50 or [r['candidate']['candidate_id'] for r in top50] != search['ranked_ids'][:50]:
    raise ValueError('committee must be the committed SEARCH ranking prefix of length 50')
search_commit = subprocess.check_output(['git', 'log', '-1', '--format=%H %cI', '--', str(a.root/'locks/search.json')], text=True).strip()
# Membership of existing TEST outputs is read from the test phase's run plan (validation top-10 lock), not from any score.
tested = {r['candidate']['candidate_id'] for r in read(a.root/'locks/validation.json')['top10']}
existing = [r for r in top50 if r['candidate']['candidate_id'] in tested]
to_generate = [r for r in top50 if r['candidate']['candidate_id'] not in tested]
if len(existing) + len(to_generate) != 50:
    raise ValueError('committee partition error')
out = a.root/'locks/committee.json'
write(out, {'protocol_sha256': sha(a.protocol), 'addendum_sha256': sha(a.addendum), 'source': revision_info(),
            'command': [sys.executable, *sys.argv], 'search_lock_sha256': sha(a.root/'locks/search.json'),
            'search_lock_commit': search_commit, 'committee_rule': addendum['committee'],
            'committee': [{'search_rank': i + 1, **r} for i, r in enumerate(top50)],
            'existing_test_outputs': {'phase': 'test', 'candidate_ids': [r['candidate']['candidate_id'] for r in existing]},
            'to_generate': to_generate, 'shard_size': 5,
            'selection_inputs': 'committed SEARCH ranking only; no VALIDATION or TEST score read'})
print({'committee': 50, 'reuse_existing_test': len(existing), 'to_generate': len(to_generate), 'sha256': sha(out)})

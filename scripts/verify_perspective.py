"""Independently re-derive every Perspective-Taking N=5000 lock and the decision from raw generations."""
from datetime import datetime
from pathlib import Path
import subprocess
import sys

import numpy as np

from thicket_runtime.visual_runtime import read
from thicket_runtime.perspective import compare, decision, manifest, rank_search, rank_validation, splits, summarize, verify_inputs
from thicket_runtime.perspective_audit import phase
from thicket_runtime.perspective_runtime import VALIDATION_SHARD_SIZE, validation_plan

root = Path(sys.argv[1]); pp = Path('experiments/perspective_taking_n5000_protocol.json')
P = read(pp); proof = verify_inputs(P)
L = {k: read(root/'locks'/(k+'.json')) for k in ('baseline', 'search', 'validation')}
sets = splits(P)
base = phase(root/'baseline', P, pp, {})
assert base['native']['base_id'] == L['baseline']['base_id']
assert {s: v['output_identity'] for s, v in base['base'].items()} == L['baseline']['output_identities']
ref = {s: v['outputs'] for s, v in base['base'].items()}
recs = []
for k in sorted({r['shard'] for r in manifest(P)}):
    a = phase(root/f'search-shard-{k:03d}', P, pp, {'baseline': L['baseline']}, shard=k, base_reference=ref)
    recs += [(r['candidate']['candidate_id'], r['correct_count'], r['candidate_state_id']) for r in a['records']]
assert len(recs) == 5000 and len({c for c, _, _ in recs}) == 5000 and len({s for _, _, s in recs}) == 5000
assert {(r['candidate']['candidate_id'], r['correct_count'], r['candidate_state_id']) for r in L['search']['records']} == set(recs)
assert rank_search(L['search']['records'], P)[:50] == L['search']['top50']
plan = validation_plan(L, P); traces = {}
for j in range(-(-len(plan) // VALIDATION_SHARD_SIZE)):
    traces.update(phase(root/f'validation-shard-{j:02d}', P, pp, {'baseline': L['baseline'], 'search': L['search']}, shard=j, base_reference=ref)['traces'])
assert set(traces) == {r['candidate']['candidate_id'] for r in plan}
rows = sets['validation']; bo = ref['validation']
idx = np.random.default_rng(P['statistics']['bootstrap_seed']).integers(0, len(rows), size=(10000, len(rows)), dtype=np.int64)
vc = {r['candidate']['candidate_id']: summarize(traces[r['candidate']['candidate_id']]['splits']['validation']['outputs'], rows)['correct_count'] for r in L['search']['top50']}
assert rank_validation(L['search']['top50'], vc)[:10] == L['validation']['top10']
test = phase(root/'test', P, pp, L, base_reference=ref)
rows = sets['test']; bo = ref['test']
idx = np.random.default_rng(P['statistics']['bootstrap_seed']).integers(0, len(rows), size=(10000, len(rows)), dtype=np.int64)
final = [compare(bo, test['traces'][r['candidate']['candidate_id']]['splits']['test']['outputs'], rows, idx, .8) for r in L['validation']['top10']]
d = decision(final, P, True)
assert d['decision'] == read(root/'analysis/decision.json')['decision']
times = {}
for lock, dirs in (('baseline', [f'search-shard-{k:03d}' for k in range(50)]), ('search', [f'validation-shard-{j:02d}' for j in range(16)]), ('validation', ['test'])):
    for dname in dirs:
        m = read(root/dname/'run-manifest.json'); commit, when = m['lock_commits'][lock].split()
        assert subprocess.check_output(['git', 'show', f'{commit}:{(root/"locks"/(lock+".json")).as_posix()}']) == (root/'locks'/(lock+'.json')).read_bytes()
        assert datetime.fromisoformat(when) < datetime.fromisoformat(m['started_utc'])
        times[f'{lock}<{dname}'] = (when, m['started_utc'])
print({'verified': True, 'dataset_files': proof['files_verified'], 'search_candidates': 5000, 'validated': len(traces),
       'decision': d['decision'], 'rank1_test_gain': round(100 * final[0]['gain'], 2),
       'lock_order_checks': len(times), 'last_lock': times['validation<test']})

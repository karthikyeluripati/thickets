"""Independently re-derive every transfer-aware lock and decision from raw generations."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from thicket_runtime.line_tracing import sha
from thicket_runtime.visual_decisions import summarize
from thicket_runtime.visual_runtime import read
from thicket_runtime.transfer_aware import METHODS, fold_record, folds, rank, splits, validation_gate, verify_inputs
from thicket_runtime.transfer_aware_audit import phase

root = Path(sys.argv[1]); proto_path = Path('experiments/transfer_aware_visual_protocol.json')
P = read(proto_path); proof = verify_inputs(P)
locks = {k: read(root/'locks'/(k+'.json')) for k in ('baseline', 'ranking', 'validation')}
sets = splits(P); fi = folds(P, sets['search'])
base = phase(root/'baseline', P, proto_path, {})
assert base['native']['base_id'] == locks['baseline']['base_id']
assert {s: v['output_identity'] for s, v in base['base'].items()} == locks['baseline']['output_identities']
records = []
for k in range(8):
    a = phase(root/f'search-shard-{k:02d}', P, proto_path, {'baseline': locks['baseline']}, shard=k)
    for r in a['records']:
        outs = a['traces'][r['candidate']['candidate_id']]['splits']['search']['outputs']
        records.append({**r, 'shard': k, 'search': fold_record([o['correct'] for o in outs], fi, locks['baseline']['search_base'])})
assert records == locks['ranking']['records'] and len({r['candidate_state_id'] for r in records}) == 400
for m in METHODS:
    assert rank(records, m)[:20] == locks['ranking']['top'][m]
val = phase(root/'validation', P, proto_path, {'baseline': locks['baseline'], 'ranking': locks['ranking']})
rows = sets['validation']; b = summarize(base['base']['validation']['outputs'], rows)
gains = {cid: summarize(t['splits']['validation']['outputs'], rows)['accuracy'] - b['accuracy'] for cid, t in val['traces'].items()}
assert gains == {c: v['validation_gain'] for c, v in locks['validation']['candidates'].items()}
gate = validation_gate(locks['ranking'], gains, P)
assert gate == locks['validation']['gate'] and not (root/'test').exists()
times = {}
for lock, phase_dir in [('baseline', f'search-shard-{k:02d}') for k in range(8)] + [('ranking', 'validation')]:
    commit, when = read(root/phase_dir/'run-manifest.json')['lock_commits'][lock].split()
    assert subprocess.check_output(['git', 'show', f'{commit}:{root.as_posix()}/locks/{lock}.json']) == (root/'locks'/(lock+'.json')).read_bytes()
    times[phase_dir] = [when, read(root/phase_dir/'run-manifest.json')['started_utc']]
    from datetime import datetime
    assert datetime.fromisoformat(when) < datetime.fromisoformat(times[phase_dir][1])
print(json.dumps({'verified': True, 'dataset_files': proof['files_verified'], 'candidates': 400, 'distinct_fingerprints': 400,
                  'validation_candidates': len(gains), 'gate': gate['decision'], 'test_phase_exists': False,
                  'lock_before_phase': times}, indent=1))

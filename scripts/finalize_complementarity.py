"""Verify both experiments and preserve all earlier studies before indexing evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from thicket_runtime.cli import revision_info, write_json


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    root = args.root
    if (root / 'sha256.json').exists() or (root / 'integrity.json').exists():
        raise FileExistsError('final evidence is immutable')
    old = ['llm-bounded-20261003', 'work1-closure-20261003', 'shared-speculative-20261003', 'adaptive-evaluation-20261003']
    verified = {}
    for folder in [Path('results') / name for name in old] + [root / 'gpu-01']:
        index = json.loads((folder / 'sha256.json').read_bytes())
        for name, expected in index.items():
            if sha(folder / name) != expected:
                raise ValueError('artifact mismatch: ' + str(folder / name))
        verified[str(folder)] = len(index)
    changed = subprocess.check_output(['git', 'diff', '--name-only', '07d2d22b2157cc728c0425af9e1cd90244cea1a9', '--',
                                      *[str(Path('results') / name) for name in old]])
    if changed.strip():
        raise ValueError('previous evidence changed')
    main_ref = subprocess.check_output(['git', 'rev-parse', 'main'], text=True).strip()
    if main_ref != 'deb414253139cc2559d19cdfe7e6b4786e7c40db':
        raise ValueError('main changed')
    manifest = json.loads((root / 'gpu-01/manifest.json').read_bytes())
    source = manifest['source']['commit']
    measured = root / 'gpu-01/measured-source'
    files = [f for f in measured.rglob('*') if f.is_file()]
    for f in files:
        blob = subprocess.check_output(['git', 'show', source + ':' + f.relative_to(measured).as_posix()])
        if blob != f.read_bytes():
            raise ValueError('measured source mismatch: ' + str(f))
    frozen_lock = subprocess.check_output(['git', 'show', 'af4823c:results/complementarity-selection-20261003/offline/committee-lock.json'])
    if frozen_lock != (root / 'offline/committee-lock.json').read_bytes() or frozen_lock != (root / 'gpu-01/committee-lock.json').read_bytes():
        raise ValueError('pre-GPU committee lock changed')
    protocol_path = Path('experiments/complementarity_protocol.json')
    frozen_protocol = subprocess.check_output(['git', 'show', '8ebae53:experiments/complementarity_protocol.json'])
    if protocol_path.read_bytes() != frozen_protocol or (root / 'gpu-01/complementarity_protocol.json').read_bytes() != frozen_protocol:
        raise ValueError('frozen gate/protocol changed')
    lock = json.loads(frozen_lock)
    for pop, expected in lock['selection_files_sha256'].items():
        if sha(root / 'offline' / (pop + '.json')) != expected:
            raise ValueError('offline selection changed')
    if sha(root / 'offline/union.json') != lock['union_sha256']:
        raise ValueError('selected union changed')
    if not json.loads((root / 'analysis/manifest.json').read_bytes())['all_raw_generations_rescored']:
        raise ValueError('fresh trace scoring audit incomplete')
    write_json(root / 'integrity.json', {'source': revision_info(), 'command': [sys.executable, *sys.argv],
        'verified_checksum_files': verified, 'mismatches': 0, 'main_ref': main_ref,
        'historical_result_changes': [], 'measured_source_commit': source, 'source_files_match_git_blobs': len(files),
        'pre_gpu_committee_lock_unchanged': True, 'pre_gpu_protocol_unchanged': True,
        'all_fresh_generations_rescored': True})
    files = sorted(f for f in root.rglob('*') if f.is_file() and f != root / 'sha256.json')
    write_json(root / 'sha256.json', {f.relative_to(root).as_posix(): sha(f) for f in files})
    print(json.dumps({'indexed_files': len(files), 'all_integrity_checks_pass': True}))


if __name__ == '__main__':
    main()

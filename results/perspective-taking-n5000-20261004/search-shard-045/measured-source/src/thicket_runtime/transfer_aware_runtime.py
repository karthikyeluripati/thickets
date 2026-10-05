"""Transfer-aware first bridge phases on the unchanged upstream VLM/Ray executor.

baseline: base + zero perturbation on search/validation/test, base repeat.
search --shard k: 50 of the 400 frozen candidates on search300.
validation: the committed VANILLA/TRANSFER_MIN top-20 union on validation300.
test: the committed final union on test1000.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

from .candidate import CandidateSpec
from .cli import revision_info
from .committee_validation import committed
from .line_tracing import sha
from .upstream import verify_upstream
from .visual_runtime import VisualExecutor, evaluate_candidate, initialize_visual, launch, read, write
from .visual_v21 import population_id
from .transfer_aware import PHASE_SPLITS, shard_plan, splits, verify_inputs

ZERO_SEED = 5100000
LOCKS = {'baseline': (), 'search': ('baseline',), 'validation': ('baseline', 'ranking'),
         'test': ('baseline', 'ranking', 'validation')}


def recipe(base_id, seed, sigma, mask):
    spec = CandidateSpec(base_id, seed, sigma)
    return asdict(spec) | {'candidate_id': spec.candidate_id, 'parameter_mask': 'all-parameters-including-vision',
                           'parameter_mask_sha256': mask}


def planned(phase, protocol, locks, shard):
    """(seed, sigma, expected frozen record or None) for each candidate of the phase."""
    if phase == 'baseline':
        return []
    if phase == 'search':
        return [(seed, sigma, None) for seed, sigma in shard_plan(protocol, shard)]
    if phase == 'validation':
        union = locks['ranking']['validation_union']
    else:
        if locks['validation']['gate']['pass'] is not True:
            raise ValueError('validation gate failed: final test is forbidden')
        union = locks['validation']['final_union']
    return [(r['candidate']['seed'], r['candidate']['sigma'], r) for r in union]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=list(PHASE_SPLITS), required=True)
    p.add_argument('--shard', type=int)
    p.add_argument('--protocol', type=Path, default=Path('experiments/transfer_aware_visual_protocol.json'))
    p.add_argument('--locks', type=Path, required=True)
    p.add_argument('--upstream-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if (a.phase == 'search') != (a.shard is not None):
        raise ValueError('--shard is required for, and only for, search')
    committed(a.protocol)
    protocol = read(a.protocol)
    if protocol['schema'] != 'transfer-aware-visual-search-v1': raise ValueError('wrong protocol')
    proof = verify_inputs(protocol)
    locks, lock_commits = {}, {}
    for name in LOCKS[a.phase]:
        path = a.locks/(name+'.json')
        committed(path)
        locks[name] = read(path)
        if locks[name]['protocol_sha256'] != sha(a.protocol): raise ValueError('lock is for another protocol')
        lock_commits[name] = subprocess.check_output(['git','log','-1','--format=%H %cI','--',str(path)],text=True).strip()
    sets = splits(protocol)
    datasets = {s: sets[s] for s in PHASE_SPLITS[a.phase]}
    versions = {n: importlib.metadata.version(n) for n in protocol['required_versions']}
    if versions != protocol['required_versions']: raise ValueError('inference environment changed')
    upstream = verify_upstream(a.upstream_root)
    plan = planned(a.phase, protocol, locks, a.shard)
    root, repo = a.out, Path(__file__).resolve().parents[2]
    root.mkdir(parents=True, exist_ok=False)
    for source in [*Path(__file__).parent.glob('*.py'), a.protocol, *Path('scripts').glob('*transfer_aware*')]:
        target = root/'measured-source'/source.resolve().relative_to(repo)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for name in locks: shutil.copyfile(a.locks/(name+'.json'), root/('lock-'+name+'.json'))
    manifest = {'phase':a.phase,'shard':a.shard,'status':'running','source':revision_info(),'command':[sys.executable,*sys.argv],
        'started_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(a.protocol),'dataset_proof':proof,
        'model':protocol['model'],'model_revision':protocol['model_revision'],'processor_revision':protocol['processor_revision'],
        'upstream':upstream,'versions':versions,'lock_commits':lock_commits,
        'evaluation_sets':{s:{'examples':len(rows),'population_sha256':population_id(rows)} for s,rows in datasets.items()},
        'planned_candidates':len(plan),'completed_candidates':0}
    for cmd, name in ((['nvidia-smi','-q'],'gpu-before.txt'),([sys.executable,'-m','pip','freeze'],'pip-freeze.txt')):
        (root/name).write_bytes(subprocess.check_output(cmd))
    engine = pg = None
    records, controls = [], {}
    try:
        from transformers import AutoProcessor
        import ray
        engine, pg, path, kwargs = launch(protocol, a.upstream_root.resolve(), repo)
        write(root/'engine-config.json', kwargs)
        api = VisualExecutor(engine, AutoProcessor.from_pretrained(path), protocol['dataset'], protocol, root.name)
        native = api.rpc(initialize_visual)
        write(root/'native-state.json', native)
        base_id = manifest['base_id'] = native['base_id']
        if 'baseline' in locks and base_id != locks['baseline']['base_id']: raise ValueError('native base changed')
        controls['initial_base_exact'] = api.fingerprint() == base_id and api.drift()['exact_base']
        reference = None
        if a.phase == 'baseline':
            base = {s: api.generate(rows) for s, rows in datasets.items()}
            write(root/'base.json.gz', base)
            reference = {s: v['output_identity'] for s, v in base.items()}
        else:
            reference = {s: locks['baseline']['output_identities'][s] for s in datasets}
        zero = evaluate_candidate(api, recipe(base_id, ZERO_SEED, 0., native['parameter_mask_sha256']), datasets, base_id)
        write(root/'zero.json.gz', zero)
        controls['zero_reproduces_base'] = all(zero['splits'][s]['output_identity'] == reference[s] for s in datasets)
        controls['zero_restore_exact'] = zero['restoration']['exact_base']
        if not all(controls.values()): raise RuntimeError('base/zero control failed')
        first = None
        for seed, sigma, expected in plan:
            candidate = recipe(base_id, seed, sigma, native['parameter_mask_sha256'])
            if expected and candidate != expected['candidate']: raise ValueError('frozen recipe differs')
            state = expected['candidate_state_id'] if expected else None
            raw = evaluate_candidate(api, candidate, datasets, state)
            raw['expected_state_id'] = state
            raw['fingerprint_checked_before_generation'] = expected is not None
            write(root/'candidates'/(candidate['candidate_id']+'.json.gz'), raw)
            split = next(iter(datasets))
            outs = raw['splits'][split]['outputs']
            records.append({'candidate':candidate,'candidate_state_id':raw['candidate_state_id'],'split':split,
                            'examples':len(outs),'correct_count':sum(o['correct'] for o in outs),
                            'invalid_count':sum(not o['answer'] for o in outs),
                            'output_identity':raw['splits'][split]['output_identity']})
            first = first or raw
            manifest['completed_candidates'] = len(records)
            print(json.dumps({'phase':a.phase,'shard':a.shard,'completed':len(records),'planned':len(plan),
                              'seed':seed,'sigma':sigma,'correct_count':records[-1]['correct_count']}),flush=True)
        if a.phase == 'baseline':
            repeat = {s: api.generate(rows) for s, rows in datasets.items()}
            write(root/'base-repeat.json.gz', repeat)
            controls['base_repeat_equal'] = all(repeat[s]['output_identity'] == reference[s] for s in datasets)
        else:
            repeat = evaluate_candidate(api, first['candidate'], datasets, first['candidate_state_id'])
            write(root/'repeat.json.gz', repeat)
            controls['repeated_state_equal'] = repeat['candidate_state_id'] == first['candidate_state_id']
            controls['repeated_outputs_equal'] = all(repeat['splits'][s]['output_identity'] == first['splits'][s]['output_identity'] for s in datasets)
        api.rebase()
        controls['final_base_exact'] = api.fingerprint() == base_id and api.drift()['exact_base']
        write(root/'records.json', records)
        write(root/'controls.json', controls)
        if not all(controls.values()): raise RuntimeError('correctness controls failed')
        manifest['status'] = 'complete'
        print(json.dumps({'phase':a.phase,'shard':a.shard,'status':'complete','candidates':len(records)}),flush=True)
    except BaseException as exc:
        manifest.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        manifest['ended_utc'] = datetime.now(timezone.utc).isoformat()
        write(root/'run-manifest.json', manifest)
        if engine:
            from core.engine import cleanup_engines
            cleanup_engines([engine], [pg])
        elif 'ray' in locals(): ray.shutdown()
        (root/'gpu-after.txt').write_bytes(subprocess.check_output(['nvidia-smi','-q']))
        write(root/'sha256.json', {f.relative_to(root).as_posix(): sha(f) for f in sorted(root.rglob('*')) if f.is_file()})


if __name__ == '__main__': main()

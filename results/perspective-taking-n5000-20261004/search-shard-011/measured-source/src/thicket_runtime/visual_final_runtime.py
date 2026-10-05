"""Final v1 tail experiment using the unchanged upstream VLM/Ray executor."""
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
from .visual_final import evaluation_sets, population_id, summarize, verify_inputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['diagnostic','search','heldout'], required=True)
    p.add_argument('--protocol', type=Path, default=Path('experiments/visual_line_tracing_final_protocol.json'))
    p.add_argument('--sigma-lock', type=Path, required=True)
    p.add_argument('--selection-lock', type=Path)
    p.add_argument('--upstream-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    for path in (a.protocol, a.sigma_lock): committed(path)
    protocol, sigma_lock = read(a.protocol), read(a.sigma_lock)
    if protocol['schema'] != 'visual-line-tracing-final-v1': raise ValueError('wrong final protocol')
    proof = verify_inputs(protocol)
    if sigma_lock['protocol_sha256'] != sha(a.protocol) or sigma_lock['sigma'] != protocol['sigma']:
        raise ValueError('frozen sigma lock changed')
    datasets = evaluation_sets(protocol, a.phase)
    versions = {n:importlib.metadata.version(n) for n in protocol['required_versions']}
    if versions != protocol['required_versions']: raise ValueError('inference environment changed')
    upstream = verify_upstream(a.upstream_root)
    selected = None
    if a.phase == 'heldout':
        if not a.selection_lock: raise ValueError('committed selection lock required')
        committed(a.selection_lock)
        selected = read(a.selection_lock)
        if selected['protocol_sha256'] != sha(a.protocol) or len(selected['top10']) != 10:
            raise ValueError('wrong selection lock')
    elif a.selection_lock:
        raise ValueError('selection lock is only allowed in heldout')
    root, repo = a.out, Path(__file__).resolve().parents[2]
    root.mkdir(parents=True, exist_ok=False)
    sources = [*Path(__file__).parent.glob('*.py'), a.protocol,
               Path('scripts/run_visual_final_phase.sh'), Path('scripts/run_visual_final_study.sh'),
               Path('scripts/freeze_visual_final_selection.py')]
    for source in sources:
        target = root/'measured-source'/source.resolve().relative_to(repo)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source,target)
    for path in (a.protocol, a.sigma_lock, Path(protocol['dataset'])/'manifest.json'):
        shutil.copyfile(path,root/path.name)
    if a.selection_lock: shutil.copyfile(a.selection_lock,root/'selection.json')
    manifest = {'phase':a.phase,'status':'running','source':revision_info(),'command':[sys.executable,*sys.argv],
        'started_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(a.protocol),
        'dataset_manifest_sha256':proof['manifest_sha256'],'dataset_proof':proof,
        'model_revision':protocol['model_revision'],'processor_revision':protocol['processor_revision'],
        'upstream':upstream,'versions':versions,
        'evaluation_sets':{s:{'examples':len(rows),'population_sha256':population_id(rows)} for s,rows in datasets.items()},
        'completed_candidates':0,'reference_base_path':protocol['reference']['baseline']+'/base.json.gz',
        'selection_lock_commit':subprocess.check_output(['git','log','-1','--format=%H','--',str(a.selection_lock)],text=True).strip() if a.selection_lock else None}
    for cmd,name in ((['nvidia-smi','-q'],'gpu-before.txt'),([sys.executable,'-m','pip','freeze'],'pip-freeze.txt')):
        (root/name).write_bytes(subprocess.check_output(cmd))
    engine = pg = None
    records, controls = [], {}
    try:
        from transformers import AutoProcessor
        import ray
        engine,pg,path,kwargs = launch(protocol,a.upstream_root.resolve(),repo)
        write(root/'engine-config.json',kwargs)
        api = VisualExecutor(engine,AutoProcessor.from_pretrained(path),protocol['dataset'],protocol,root.name)
        native = api.rpc(initialize_visual)
        write(root/'native-state.json',native)
        manifest['base_id'] = native['base_id']
        if native['base_id'] != protocol['expected_base_id']: raise ValueError('native base changed')
        controls['initial_base_exact'] = api.fingerprint() == native['base_id'] and api.drift()['exact_base']
        historical = read(Path(protocol['reference']['baseline'])/'base.json.gz')
        if a.phase in ('search','heldout'):
            zero = CandidateSpec(native['base_id'],5100000,0.)
            raw = evaluate_candidate(api,asdict(zero)|{'candidate_id':zero.candidate_id},datasets,native['base_id'])
            write(root/'zero.json.gz',raw)
            controls['zero_matches_committed_base'] = all(raw['splits'][s]['output_identity'] == historical[s]['output_identity'] for s in datasets)
            controls['zero_restore_exact'] = raw['restoration']['exact_base']
        if not all(controls.values()): raise RuntimeError('base/zero control failed')
        if a.phase == 'diagnostic':
            item = protocol['diagnostic']['selected']
            plan = [(item['candidate']['seed'],item)]
        elif a.phase == 'search':
            plan = [(seed,None) for seed in sigma_lock['final_candidate_seeds']]
            if [s for s,_ in plan] != list(range(protocol['search']['seed_start'],protocol['search']['seed_start']+300)):
                raise ValueError('search must contain exactly 300 frozen fresh seeds')
        else:
            plan = [(item['candidate']['seed'],item) for item in selected['top10']]
        first = None
        for seed, expected in plan:
            spec = CandidateSpec(native['base_id'],seed,protocol['sigma'])
            candidate = asdict(spec)|{'candidate_id':spec.candidate_id,'parameter_mask':'all-parameters-including-vision',
                                    'parameter_mask_sha256':native['parameter_mask_sha256']}
            if expected and candidate != expected['candidate']: raise ValueError('frozen recipe differs')
            raw = evaluate_candidate(api,candidate,datasets,expected['candidate_state_id'] if expected else None)
            raw['expected_state_id'] = expected['candidate_state_id'] if expected else None
            raw['fingerprint_checked_before_generation'] = expected is not None
            write(root/'candidates'/(spec.candidate_id+'.json.gz'),raw)
            split = next(iter(datasets))
            metric = summarize(raw['splits'][split]['outputs'],datasets[split])
            record = {'candidate':candidate,'candidate_state_id':raw['candidate_state_id'],'split':split,
                      'population_sha256':population_id(datasets[split]),'examples':metric['n'],
                      'correct_count':metric['correct_count'],'invalid_count':metric['invalid_count'],
                      'output_identity':raw['splits'][split]['output_identity']}
            records.append(record)
            if first is None: first = raw
            manifest['completed_candidates'] = len(records)
            print(json.dumps({'phase':a.phase,'completed':len(records),'planned':len(plan),
                              'seed':seed,'correct_count':metric['correct_count']}),flush=True)
        if a.phase in ('search','heldout'):
            repeat = evaluate_candidate(api,first['candidate'],datasets,first['candidate_state_id'])
            write(root/'repeat.json.gz',repeat)
            controls['repeated_state_equal'] = repeat['candidate_state_id'] == first['candidate_state_id']
            controls['repeated_outputs_equal'] = all(repeat['splits'][s]['output_identity'] == first['splits'][s]['output_identity'] for s in datasets)
        api.rebase()
        controls['final_base_exact'] = api.fingerprint() == native['base_id'] and api.drift()['exact_base']
        write(root/'records.json',records)
        write(root/'controls.json',controls)
        if not all(controls.values()): raise RuntimeError('correctness controls failed')
        manifest['status'] = 'complete'
        print(json.dumps({'phase':a.phase,'status':'complete','candidates':len(records)}),flush=True)
    except BaseException as exc:
        manifest.update(status='failed',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        manifest['ended_utc'] = datetime.now(timezone.utc).isoformat()
        write(root/'run-manifest.json',manifest)
        if engine:
            from core.engine import cleanup_engines
            cleanup_engines([engine],[pg])
        elif 'ray' in locals(): ray.shutdown()
        (root/'gpu-after.txt').write_bytes(subprocess.check_output(['nvidia-smi','-q']))
        write(root/'sha256.json',{f.relative_to(root).as_posix():sha(f) for f in sorted(root.rglob('*')) if f.is_file()})


if __name__ == '__main__': main()

"""Versioned v2.1 orchestration; reuse the unchanged VLM/RandOpt executor."""
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
from .visual_v21 import PRIMARY, calibration_plan, evaluation_sets, population_id, summarize, verify_dataset


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['baseline','calibration','search','heldout'],required=True)
    p.add_argument('--protocol',type=Path,default=Path('experiments/visual_line_tracing_v21_protocol.json'))
    p.add_argument('--upstream-root',type=Path,required=True)
    p.add_argument('--baseline-lock',type=Path)
    p.add_argument('--sigma-lock',type=Path)
    p.add_argument('--selection-lock',type=Path)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    committed(args.protocol)
    protocol=read(args.protocol)
    if protocol['schema']!='visual-line-tracing-feasibility-v2.1': raise ValueError('v2.1 protocol required')
    dataset_proof=verify_dataset(protocol)
    datasets=evaluation_sets(protocol,args.phase)
    versions={name:importlib.metadata.version(name) for name in protocol['required_versions']}
    if versions!=protocol['required_versions']: raise ValueError('inference environment differs from frozen stack')
    locks={}
    for name,path in [('baseline',args.baseline_lock),('sigma',args.sigma_lock),('selection',args.selection_lock)]:
        if path:
            committed(path)
            locks[name]=read(path)
            if locks[name]['protocol_sha256']!=sha(args.protocol) or locks[name]['dataset_manifest_sha256']!=dataset_proof['manifest_sha256']:
                raise ValueError('lock differs from frozen protocol/dataset')
    if args.phase!='baseline' and not locks.get('baseline',{}).get('accepted'):
        raise ValueError('capability gate must pass before RandOpt')
    if args.phase=='search' and not locks.get('sigma',{}).get('continue_search'):
        raise ValueError('calibration gate must pass before search')
    if args.phase=='heldout' and len(locks.get('selection',{}).get('top10',[]))!=10:
        raise ValueError('committed top10 required before candidate held-out generation')
    upstream=verify_upstream(args.upstream_root)
    root,repo=args.out,Path(__file__).resolve().parents[2]
    root.mkdir(parents=True,exist_ok=False)
    for source in [*Path(__file__).parent.glob('*.py'),args.protocol]:
        target=root/'measured-source'/source.resolve().relative_to(repo)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
    for path in [args.protocol,Path(protocol['dataset'])/'manifest.json',*[v for v in (args.baseline_lock,args.sigma_lock,args.selection_lock) if v]]:
        shutil.copyfile(path,root/path.name)
    manifest={'phase':args.phase,'status':'running','source':revision_info(),
              'command':[sys.executable,*sys.argv],'started_utc':datetime.now(timezone.utc).isoformat(),
              'protocol_sha256':sha(args.protocol),'dataset_manifest_sha256':dataset_proof['manifest_sha256'],
              'dataset_proof':dataset_proof,'model_revision':protocol['model_revision'],
              'processor_revision':protocol['processor_revision'],'upstream':upstream,'versions':versions,
              'evaluation_sets':{k:{'examples':len(v),'population_sha256':population_id(v)} for k,v in datasets.items()},
              'completed_candidates':0}
    for command,name in [(['nvidia-smi','-q'],'gpu-before.txt'),([sys.executable,'-m','pip','freeze'],'pip-freeze.txt')]:
        (root/name).write_bytes(subprocess.check_output(command))
    engine=pg=None
    records,controls=[],{}
    try:
        from transformers import AutoProcessor
        import ray
        engine,pg,path,kwargs=launch(protocol,args.upstream_root.resolve(),repo)
        write(root/'engine-config.json',kwargs)
        api=VisualExecutor(engine,AutoProcessor.from_pretrained(path),protocol['dataset'],protocol,root.name)
        native=api.rpc(initialize_visual)
        write(root/'native-state.json',native)
        manifest['base_id']=native['base_id']
        if native['base_id']!=protocol['expected_base_id']:
            raise ValueError('native model state differs from the pinned reference executor')
        if 'baseline' in locks and locks['baseline']['base_id']!=native['base_id']:
            raise ValueError('base state differs across phases')
        bases={name:api.generate(rows) for name,rows in datasets.items()}
        write(root/'base.json.gz',bases)
        controls['base_state_unchanged']=api.fingerprint()==native['base_id']
        if 'baseline' in locks:
            for name,result in bases.items():
                controls['base_'+name+'_matches_baseline']=result['output_identity']==locks['baseline']['output_identities'][name]
        if not all(controls.values()): raise RuntimeError('cross-phase base equality failed')
        if args.phase=='baseline':
            zero=CandidateSpec(native['base_id'],5100000,0.)
            raw=evaluate_candidate(api,asdict(zero)|{'candidate_id':zero.candidate_id},datasets,native['base_id'])
            write(root/'zero.json.gz',raw)
            controls.update({name+'_zero_equal':raw['splits'][name]['output_identity']==bases[name]['output_identity'] for name in datasets})
            controls['zero_restore_exact']=raw['restoration']['exact_base']
            plan=[]
        elif args.phase=='calibration':
            plan=[(seed,sigma,None) for seed,sigma in calibration_plan(protocol['calibration'])]
        elif args.phase=='search':
            plan=[(protocol['search']['seed_start']+i,locks['sigma']['sigma'],None) for i in range(300)]
        else:
            plan=[(r['candidate']['seed'],r['candidate']['sigma'],r) for r in locks['selection']['top10']]
        first={}
        for seed,sigma,selected in plan:
            spec=CandidateSpec(native['base_id'],seed,sigma)
            candidate=asdict(spec)|{'candidate_id':spec.candidate_id,'parameter_mask':'all-parameters-including-vision',
                                  'parameter_mask_sha256':native['parameter_mask_sha256']}
            if selected and candidate!=selected['candidate']: raise ValueError('held-out recipe differs from selected recipe')
            raw=evaluate_candidate(api,candidate,datasets,selected['candidate_state_id'] if selected else None)
            write(root/'candidates'/(spec.candidate_id+'.json.gz'),raw)
            name=next(iter(datasets))
            metric=summarize(raw['splits'][name]['outputs'],datasets[name])['primary']
            primary=[r for r in datasets[name] if r['difficulty'] in PRIMARY]
            record={'candidate':candidate,'candidate_state_id':raw['candidate_state_id'],
                    'split':'heldout' if args.phase=='heldout' else 'selection','evaluation_set':name,
                    'population_sha256':population_id(primary),'examples':metric['n'],
                    'correct_count':metric['correct_count'],'invalid_count':metric['invalid_count'],
                    'output_identity':raw['splits'][name]['output_identity']}
            records.append(record)
            first.setdefault(sigma,(candidate,raw))
            manifest['completed_candidates']=len(records)
            print(json.dumps({'phase':args.phase,'completed':len(records),'planned':len(plan),
                              'sigma':sigma,'primary_correct_count':record['correct_count']}),flush=True)
        for sigma,(candidate,original) in first.items():
            repeated=evaluate_candidate(api,candidate,datasets,original['candidate_state_id'])
            write(root/('repeat-'+str(sigma)+'.json.gz'),repeated)
            controls['same_state_'+str(sigma)]=repeated['candidate_state_id']==original['candidate_state_id']
            controls['same_outputs_'+str(sigma)]=all(repeated['splits'][name]['output_identity']==original['splits'][name]['output_identity'] for name in datasets)
        api.rebase()
        repeated={name:api.generate(rows) for name,rows in datasets.items()}
        write(root/'base-repeat.json.gz',repeated)
        controls.update({name+'_base_repeat_equal':value['output_identity']==bases[name]['output_identity'] for name,value in repeated.items()})
        controls['final_base_exact']=api.fingerprint()==native['base_id'] and api.drift()['exact_base']
        write(root/'records.json',records)
        write(root/'controls.json',controls)
        if not all(controls.values()): raise RuntimeError('correctness controls failed')
        manifest['status']='complete'
        print(json.dumps({'phase':args.phase,'status':'complete','candidates':len(records)}),flush=True)
    except BaseException as exc:
        manifest.update(status='failed',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        manifest['ended_utc']=datetime.now(timezone.utc).isoformat()
        write(root/'run-manifest.json',manifest)
        if engine:
            from core.engine import cleanup_engines
            cleanup_engines([engine],[pg])
        elif 'ray' in locals(): ray.shutdown()
        (root/'gpu-after.txt').write_bytes(subprocess.check_output(['nvidia-smi','-q']))
        write(root/'sha256.json',{f.relative_to(root).as_posix():sha(f) for f in sorted(root.rglob('*')) if f.is_file()})


if __name__=='__main__': main()

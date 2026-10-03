"""Bounded VLM phases using pinned RandOpt RPCs and fresh vision computation."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .candidate import CandidateSpec
from .cli import revision_info
from .committee_validation import committed
from .line_tracing import load_split, parse_answer, sha
from .upstream import verify_upstream
from .vllm_profile import RayExecutor
from . import vllm_audit


def write(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2, allow_nan=False)+'\n').encode()
    path.write_bytes(gzip.compress(raw,mtime=0) if path.suffix=='.gz' else raw)


def read(path):
    path = Path(path)
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix=='.gz' else raw)


def output_identity(outputs):
    return hashlib.sha256(json.dumps(outputs,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def vision_hook(module, args, kwargs):
    grid = kwargs.get('grid_thw', args[1] if len(args)>1 else None)
    if grid is None:
        raise RuntimeError('cannot audit the number of encoded images')
    module._thicket_images_encoded += len(grid)
    module._thicket_encoder_calls += 1


def initialize_visual(worker):
    import torch
    torch.set_float32_matmul_precision('highest')
    model = worker.model_runner.model
    info = vllm_audit.initialize(worker)
    if not all(worker._should_perturb(n) for n,_ in model.named_parameters()):
        raise RuntimeError('all-parameter perturbation requires PERTURB_VISUAL=1 in worker')
    visual = model.visual
    visual._thicket_images_encoded = visual._thicket_encoder_calls = 0
    visual.register_forward_pre_hook(vision_hook, with_kwargs=True)
    info['perturb_visual'] = os.environ.get('PERTURB_VISUAL')
    info['all_parameters_perturbed'] = True
    info['parameter_mask_sha256'] = hashlib.sha256('\n'.join(n for n,_ in model.named_parameters()).encode()).hexdigest()
    return info


def encoder_audit(worker, clear=False):
    runner = worker.model_runner
    visual = runner.model.visual
    before = len(runner.encoder_cache)
    if clear:
        # All preceding requests have drained. UUIDs are never reused between
        # generation calls, so scheduler bookkeeping cannot request stale entries.
        runner.encoder_cache.clear()
    return {'cached_items_before':before, 'cached_items_after':len(runner.encoder_cache),
            'images_encoded':visual._thicket_images_encoded, 'encoder_calls':visual._thicket_encoder_calls}


class VisualExecutor(RayExecutor):
    def __init__(self, engine, processor, dataset, protocol, run_identity):
        super().__init__(engine, [], [], None)
        self.processor, self.dataset, self.protocol = processor, Path(dataset), protocol
        self.call_index, self.run_identity = 0, run_identity
        self.images, self.prompts = {}, {}

    def generate(self, rows):
        from PIL import Image
        from vllm import SamplingParams
        self.call_index += 1
        cache_before = self.rpc(encoder_audit, True)
        requests = []
        for row in rows:
            key = row['id']
            if key not in self.images:
                self.images[key] = Image.open(self.dataset/row['image']).convert('RGB')
                self.prompts[key] = self.processor.apply_chat_template(
                    [{'role':'user','content':[{'type':'image'},{'type':'text','text':row['prompt']}]}],
                    tokenize=False, add_generation_prompt=True)
            uuid = f"{self.run_identity}:{self.call_index}:{row['image_sha256']}"
            requests.append({'prompt':self.prompts[key], 'multi_modal_data':{'image':self.images[key]},
                             'multi_modal_uuids':{'image':[uuid]}})
        sampling = SamplingParams(temperature=0, seed=0, max_tokens=self.protocol['max_tokens'])
        started = time.perf_counter()
        result = self.ray.get(self.engine.generate.remote(requests,sampling,use_tqdm=False),timeout=1800)
        elapsed = time.perf_counter()-started
        after = self.rpc(encoder_audit)
        if after['images_encoded']-cache_before['images_encoded'] != len(rows):
            raise RuntimeError('vision embeddings were not freshly computed for every image')
        if len(result)!=len(rows) or any(len(r.outputs)!=1 for r in result):
            raise RuntimeError('generation cardinality mismatch')
        outputs = []
        for row,r in zip(rows,result):
            o = r.outputs[0]
            answer = parse_answer(o.text)
            outputs.append({'id':row['id'], 'image_sha256':row['image_sha256'], 'text':o.text,
                'token_ids':list(o.token_ids), 'prompt_token_ids':list(r.prompt_token_ids),
                'finish_reason':o.finish_reason, 'stop_reason':o.stop_reason,
                'answer':answer, 'correct':answer==row['answer'],
                'failure_category':'correct' if answer==row['answer'] else ('wrong_endpoint' if answer else 'invalid_format')})
        return {'outputs':outputs, 'output_identity':output_identity(outputs),
                'generation_seconds':elapsed, 'cache_audit':{'before':cache_before,'after':after,
                'fresh_images_encoded':after['images_encoded']-cache_before['images_encoded'],
                'uuid_namespace':f'{self.run_identity}:{self.call_index}'}}


def evaluate_candidate(api, candidate, datasets, expected_state=None):
    api.rebase()
    if not api.drift()['exact_base']:
        raise RuntimeError('candidate must begin at exact base')
    api.memory(reset=True)
    api.apply(candidate,'snapshot-copy')
    try:
        state = api.fingerprint()
        if expected_state is not None and state!=expected_state:
            raise RuntimeError('state mismatch BEFORE generation')
        output = {split:api.generate(rows) for split,rows in datasets.items()}
    finally:
        api.restore(candidate,'snapshot-copy')
    restored, memory = api.drift(), api.memory()
    if not restored['exact_base']:
        raise RuntimeError('snapshot restoration failed')
    return {'candidate':candidate, 'candidate_state_id':state, 'splits':output,
            'restoration':restored, 'memory':memory}


def launch(protocol, upstream_root, repo):
    import ray
    from huggingface_hub import snapshot_download
    from ray.util.placement_group import placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
    sys.path.insert(0,str(upstream_root))
    from core.engine import RandOptNcclLLM
    path = snapshot_download(protocol['model'], revision=protocol['model_revision'],
                             allow_patterns=['*.json','*.safetensors','*.txt','*.model'])
    if Path(path).name != protocol['model_revision']:
        raise ValueError('unpinned model snapshot')
    env = {'PYTHONPATH':os.pathsep.join([str(upstream_root),str(repo/'src')]), 'OMP_NUM_THREADS':'1',
           'VLLM_ENABLE_V1_MULTIPROCESSING':'0', 'PERTURB_VISUAL':'1', 'VLLM_USE_V1':'1'}
    os.environ.update(env)
    ray.init(num_cpus=4,num_gpus=1,include_dashboard=False,object_store_memory=2*1024**3,
             runtime_env={'env_vars':env})
    pg = placement_group([{'GPU':1,'CPU':0}])
    ray.get(pg.ready(),timeout=120)
    strategy = PlacementGroupSchedulingStrategy(placement_group=pg,placement_group_capture_child_tasks=True,placement_group_bundle_index=0)
    kwargs = dict(model=path,tokenizer=path,dtype=protocol['dtype'],tensor_parallel_size=1,
        distributed_executor_backend='ray',worker_extension_cls='utils.worker_extn.WorkerExtension',
        enforce_eager=True,enable_prefix_caching=False,gpu_memory_utilization=protocol['gpu_memory_utilization'],
        max_model_len=protocol['max_model_len'],max_num_seqs=protocol['max_num_seqs'],
        max_num_batched_tokens=protocol['max_num_batched_tokens'],disable_log_stats=True,
        limit_mm_per_prompt={'image':1,'video':0},disable_mm_preprocessor_cache=True,mm_processor_cache_gb=0,
        mm_processor_kwargs={'min_pixels':protocol['image_pixels'],'max_pixels':protocol['image_pixels']},seed=0)
    engine = ray.remote(num_cpus=0,num_gpus=0,scheduling_strategy=strategy)(RandOptNcclLLM).remote(**kwargs)
    ray.get(engine.collective_rpc.remote('store_base_weights',args=()),timeout=1800)
    return engine,pg,path,kwargs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['baseline','calibration','search','heldout'],required=True)
    p.add_argument('--protocol',type=Path,default=Path('experiments/visual_line_tracing_protocol.json'))
    p.add_argument('--upstream-root',type=Path,required=True)
    p.add_argument('--baseline-lock',type=Path)
    p.add_argument('--sigma-lock',type=Path)
    p.add_argument('--selection-lock',type=Path)
    p.add_argument('--out',type=Path,required=True)
    args = p.parse_args()
    protocol = read(args.protocol)
    committed(args.protocol)
    dataset = Path(protocol['dataset'])
    for path in (dataset/'manifest.json',dataset/'selection.jsonl',dataset/'heldout.jsonl'):
        committed(path)
    split_names = ['selection','heldout'] if args.phase=='baseline' else (['heldout'] if args.phase=='heldout' else ['selection'])
    datasets = {s:load_split(dataset,s) for s in split_names}
    locks = {}
    for name,path in (('baseline',args.baseline_lock),('sigma',args.sigma_lock),('selection',args.selection_lock)):
        if path:
            committed(path)
            locks[name]=read(path)
            if locks[name]['protocol_sha256']!=sha(args.protocol) or locks[name]['dataset_manifest_sha256']!=sha(dataset/'manifest.json'):
                raise ValueError('lock differs from frozen protocol/dataset')
    if args.phase!='baseline' and not locks.get('baseline',{}).get('accepted'):
        raise ValueError('baseline acceptance required before candidates')
    if args.phase=='search' and not locks.get('sigma',{}).get('scale_valid'):
        raise ValueError('frozen valid sigma required')
    if args.phase=='heldout' and len(locks.get('selection',{}).get('top10',[]))!=10:
        raise ValueError('committed top10 lock required before held-out candidates')
    upstream=verify_upstream(args.upstream_root)
    root,repo = args.out,Path(__file__).resolve().parents[2]
    root.mkdir(parents=True,exist_ok=False)
    for source in [*Path(__file__).parent.glob('*.py'),args.protocol]:
        target=root/'measured-source'/source.resolve().relative_to(repo)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
    for path in [args.protocol,dataset/'manifest.json',*[x for x in (args.baseline_lock,args.sigma_lock,args.selection_lock) if x]]:
        shutil.copyfile(path,root/path.name)
    manifest={'phase':args.phase,'status':'running','source':revision_info(),'command':[sys.executable,*sys.argv],
        'started_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(args.protocol),
        'dataset_manifest_sha256':sha(dataset/'manifest.json'),'model_revision':protocol['model_revision'],
        'processor_revision':protocol['processor_revision'],'upstream':upstream,
        'versions':{name:importlib.metadata.version(name) for name in ('torch','vllm','ray','transformers','numpy','Pillow','tokenizers','huggingface-hub')},
        'generation_splits':split_names,'completed_candidates':0}
    for command,name in ((['nvidia-smi','-q'],'gpu-before.txt'),([sys.executable,'-m','pip','freeze'],'pip-freeze.txt')):
        (root/name).write_bytes(subprocess.check_output(command))
    engine=pg=None
    records,controls=[],{}
    try:
        from transformers import AutoProcessor
        import ray
        engine,pg,path,kwargs=launch(protocol,args.upstream_root.resolve(),repo)
        write(root/'engine-config.json',kwargs)
        processor=AutoProcessor.from_pretrained(path)
        api=VisualExecutor(engine,processor,dataset,protocol,root.name)
        native=api.rpc(initialize_visual)
        write(root/'native-state.json',native)
        manifest['base_id']=native['base_id']
        if 'baseline' in locks and locks['baseline']['base_id']!=native['base_id']:
            raise ValueError('model state differs from baseline')
        bases={s:api.generate(rows) for s,rows in datasets.items()}
        write(root/'base.json.gz',bases)
        controls['base_state_unchanged']=api.fingerprint()==native['base_id']
        if 'baseline' in locks:
            for s,value in bases.items():
                controls['base_'+s+'_matches_baseline']=value['output_identity']==locks['baseline']['output_identities'][s]
        if not all(controls.values()):
            raise RuntimeError('baseline repeat control failed')
        if args.phase=='baseline':
            zero=CandidateSpec(native['base_id'],5100000,0.)
            raw=evaluate_candidate(api,asdict(zero)|{'candidate_id':zero.candidate_id},datasets,native['base_id'])
            write(root/'zero.json.gz',raw)
            controls.update({s+'_zero_equal':raw['splits'][s]['output_identity']==bases[s]['output_identity'] for s in datasets})
            controls['zero_restore_exact']=raw['restoration']['exact_base']
            plan=[]
        elif args.phase=='calibration':
            conf=protocol['calibration']
            plan=[(conf['seed_start']+j*conf['candidates_per_sigma']+i,sigma,None) for i in range(conf['candidates_per_sigma']) for j,sigma in enumerate(conf['sigmas'])]
        elif args.phase=='search':
            plan=[(protocol['search']['seed_start']+i,locks['sigma']['sigma'],None) for i in range(protocol['search']['candidates'])]
        else:
            plan=[(r['candidate']['seed'],r['candidate']['sigma'],r) for r in locks['selection']['top10']]
        first={}
        for seed,sigma,selected in plan:
            spec=CandidateSpec(native['base_id'],seed,sigma)
            candidate=asdict(spec)|{'candidate_id':spec.candidate_id,'parameter_mask':'all-parameters-including-vision',
                                  'parameter_mask_sha256':native['parameter_mask_sha256']}
            if selected and candidate!=selected['candidate']:
                raise ValueError('held-out recipe mismatch')
            raw=evaluate_candidate(api,candidate,datasets,selected['candidate_state_id'] if selected else None)
            write(root/'candidates'/(spec.candidate_id+'.json.gz'),raw)
            split=split_names[0]
            outputs=raw['splits'][split]['outputs']
            record={'candidate':candidate,'candidate_state_id':raw['candidate_state_id'],'split':split,
                    'correct_count':sum(o['correct'] for o in outputs),'examples':len(outputs),
                    'invalid_count':sum(not o['answer'] for o in outputs),'output_identity':raw['splits'][split]['output_identity']}
            records.append(record)
            first.setdefault(sigma,(candidate,raw))
            manifest['completed_candidates']=len(records)
            print(json.dumps({'phase':args.phase,'completed':len(records),'planned':len(plan),'sigma':sigma,
                              'correct_count':record['correct_count']}),flush=True)
        for sigma,(candidate,original) in first.items():
            repeated=evaluate_candidate(api,candidate,datasets,original['candidate_state_id'])
            write(root/('repeat-'+str(sigma)+'.json.gz'),repeated)
            controls['same_state_'+str(sigma)]=repeated['candidate_state_id']==original['candidate_state_id']
            controls['same_outputs_'+str(sigma)]=all(repeated['splits'][s]['output_identity']==original['splits'][s]['output_identity'] for s in datasets)
        api.rebase()
        repeat={s:api.generate(rows) for s,rows in datasets.items()}
        write(root/'base-repeat.json.gz',repeat)
        controls.update({s+'_base_repeat_equal':value['output_identity']==bases[s]['output_identity'] for s,value in repeat.items()})
        controls['final_base_exact']=api.fingerprint()==native['base_id'] and api.drift()['exact_base']
        write(root/'records.json',records)
        write(root/'controls.json',controls)
        manifest['status']='complete' if all(controls.values()) else 'invalid_controls'
        if manifest['status']!='complete':
            raise RuntimeError('phase correctness controls failed')
        print(json.dumps({'phase':args.phase,'status':'complete','candidates':len(records),
              'base_accuracy':{s:sum(o['correct'] for o in v['outputs'])/len(v['outputs']) for s,v in bases.items()}}),flush=True)
    except BaseException as exc:
        manifest.update(status='failed',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        manifest['ended_utc']=datetime.now(timezone.utc).isoformat()
        write(root/'run-manifest.json',manifest)
        if engine:
            from core.engine import cleanup_engines
            cleanup_engines([engine],[pg])
        elif 'ray' in locals():
            ray.shutdown()
        (root/'gpu-after.txt').write_bytes(subprocess.check_output(['nvidia-smi','-q']))
        write(root/'sha256.json',{f.relative_to(root).as_posix():sha(f) for f in sorted(root.rglob('*')) if f.is_file()})


if __name__=='__main__':
    main()

"""Perspective-Taking N=5000 phases: Qwen3-VL-8B on the unchanged upstream RandOpt worker.

baseline: base + zero perturbation on SEARCH/VALIDATION/TEST, base repeat.
search --shard k: the frozen manifest candidates of shard k on SEARCH200.
validation --shard j: slice j of [frozen top 50] + [precommitted density audit not in top 50] on VALIDATION200.
test: the frozen validation top 10 on the official TEST subset.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .cli import revision_info
from .committee_validation import committed
from .line_tracing import sha
from .upstream import verify_upstream
from .visual_runtime import VisualExecutor, encoder_audit, evaluate_candidate, initialize_visual, read, write
from .perspective import PHASE_SPLITS, ident, manifest, shard_plan, splits, verify_inputs
from .omnispatial_data import build_prompt, score, sha256
from .transfer_aware_runtime import recipe
from . import fast_state, vllm_audit

ZERO_SEED = 5100000
VALIDATION_SHARD_SIZE = 35
LOCKS = {'baseline': (), 'search': ('baseline',), 'validation': ('baseline', 'search'), 'test': ('baseline', 'search', 'validation')}


def output_identity(outputs):
    return hashlib.sha256(json.dumps(outputs, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def initialize(worker):
    info = initialize_visual(worker)
    info['base_id_flat_sha256'] = info['base_id']
    info['base_id'] = fast_state.fingerprint(worker)
    info['fingerprint_contract'] = 'state-sha256-tree-v1'
    return info


def engine_class():
    """Upstream RandOptNcclLLM plus an image cache in the actor. Ray does not release
    per-call request objects fast enough (each call carried ~0.9 GB of full-resolution
    images and the object store spilled to disk), so each process ships its images once
    and every generation sends only (image key, prompt, fresh UUID). The worker
    extension and all weight operations are unchanged."""
    from core.engine import RandOptNcclLLM

    class CachedImageLLM(RandOptNcclLLM):
        def cache_images(self, images):
            self._thicket_images = dict(getattr(self, '_thicket_images', {}), **images)
            return sorted(self._thicket_images)

        def generate_cached(self, items, sampling):
            requests = [{'prompt': prompt, 'multi_modal_data': {'image': self._thicket_images[key]},
                         'multi_modal_uuids': {'image': [uuid]}} for key, prompt, uuid in items]
            return self.generate(requests, sampling, use_tqdm=False)

    return CachedImageLLM


def validation_plan(locks, protocol):
    """Frozen top 50 in search order, then density-audit candidates not already in it (manifest order)."""
    records = {r['index']: r for r in locks['search']['records']}
    top = locks['search']['top50']
    ids = {r['candidate']['candidate_id'] for r in top}
    audit = [records[i] for i in protocol['density_audit']['indices'] if records[i]['candidate']['candidate_id'] not in ids]
    return top + audit


class PerspectiveExecutor(VisualExecutor):
    """Official OmniSpatial direct prompt/scorer; RGB conversion as qwen_vl_utils.fetch_image;
    the pinned Qwen3-VL processor performs its own resizing."""
    def __init__(self, engine, processor, images, protocol, run_identity):
        super().__init__(engine, processor, images, protocol, run_identity)
        self.preprocess, self.shipped = {}, set()

    def fingerprint(self):
        return self.rpc(fast_state.fingerprint)

    def drift(self):
        return self.rpc(fast_state.drift)

    def prepare(self, row):
        if row['uid'] in self.images:
            return
        from PIL import Image
        path = self.dataset / row['source_split'] / row['image_member'].rsplit('/', 1)[1]
        data = path.read_bytes()
        if sha256(data) != row['image_sha256']:
            raise ValueError('source image differs from frozen hash: ' + row['uid'])
        image = Image.open(path).convert('RGB')
        self.images[row['uid']] = image
        self.prompts[row['uid']] = self.processor.apply_chat_template(
            [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': build_prompt(row)}]}],
            tokenize=False, add_generation_prompt=True)
        self.preprocess[row['uid']] = {'rgb_size': list(image.size), 'pixels_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
                                       'prompt_sha256': hashlib.sha256(self.prompts[row['uid']].encode()).hexdigest()}

    def generate(self, rows):
        from vllm import SamplingParams
        self.call_index += 1
        for row in rows:
            self.prepare(row)
        new = {r['uid']: self.images[r['uid']] for r in rows if r['uid'] not in self.shipped}
        if new:
            cached = self.ray.get(self.engine.cache_images.remote(new), timeout=1800)
            if not set(new) <= set(cached): raise RuntimeError('actor image cache incomplete')
            self.shipped |= set(new)
        cache_before = self.rpc(encoder_audit, True)
        items = [(r['uid'], self.prompts[r['uid']], f"{self.run_identity}:{self.call_index}:{r['image_sha256']}:{r['uid']}") for r in rows]
        sampling = SamplingParams(temperature=0, seed=0, max_tokens=self.protocol['max_tokens'])
        started = time.perf_counter()
        result = self.ray.get(self.engine.generate_cached.remote(items, sampling), timeout=7200)
        elapsed = time.perf_counter() - started
        after = self.rpc(encoder_audit)
        if after['images_encoded'] - cache_before['images_encoded'] != len(rows):
            raise RuntimeError('vision embeddings were not freshly computed for every image')
        if len(result) != len(rows) or any(len(r.outputs) != 1 for r in result):
            raise RuntimeError('generation cardinality mismatch')
        outputs = []
        for row, r in zip(rows, result):
            o = r.outputs[0]
            outputs.append({'uid': row['uid'], 'image_sha256': row['image_sha256'], 'text': o.text,
                            'token_ids': list(o.token_ids), 'finish_reason': o.finish_reason, 'stop_reason': o.stop_reason,
                            'prompt_tokens': len(r.prompt_token_ids),
                            'prompt_token_ids_sha256': hashlib.sha256(json.dumps(list(r.prompt_token_ids)).encode()).hexdigest(),
                            **score(o.text, row['answer'])})
        return {'outputs': outputs, 'output_identity': output_identity(outputs), 'generation_seconds': elapsed,
                'cache_audit': {'before': cache_before, 'after': after,
                                'fresh_images_encoded': after['images_encoded'] - cache_before['images_encoded'],
                                'uuid_namespace': f'{self.run_identity}:{self.call_index}'}}


def launch(protocol, upstream_root, repo):
    import ray
    from huggingface_hub import snapshot_download
    from ray.util.placement_group import placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
    sys.path.insert(0, str(upstream_root))
    path = snapshot_download(protocol['model'], revision=protocol['model_revision'],
                             allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja'])
    if Path(path).name != protocol['model_revision']:
        raise ValueError('unpinned model snapshot')
    env = {'PYTHONPATH': os.pathsep.join([str(upstream_root), str(repo / 'src')]), 'OMP_NUM_THREADS': '1',
           'VLLM_ENABLE_V1_MULTIPROCESSING': '0', 'PERTURB_VISUAL': '1', 'VLLM_USE_V1': '1'}
    os.environ.update(env)
    ray.init(num_cpus=8, num_gpus=1, include_dashboard=False, object_store_memory=48 * 1024**3, runtime_env={'env_vars': env})
    pg = placement_group([{'GPU': 1, 'CPU': 0}])
    ray.get(pg.ready(), timeout=120)
    strategy = PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_capture_child_tasks=True, placement_group_bundle_index=0)
    e = protocol['engine']
    kwargs = dict(model=path, tokenizer=path, dtype=protocol['dtype'], tensor_parallel_size=1,
                  distributed_executor_backend='ray', worker_extension_cls='utils.worker_extn.WorkerExtension',
                  enforce_eager=True, enable_prefix_caching=False, gpu_memory_utilization=e['gpu_memory_utilization'],
                  max_model_len=e['max_model_len'], max_num_seqs=e['max_num_seqs'], max_num_batched_tokens=e['max_num_batched_tokens'],
                  disable_log_stats=True, limit_mm_per_prompt={'image': 1, 'video': 0}, mm_processor_cache_gb=0, seed=0)
    engine = ray.remote(num_cpus=0, num_gpus=0, scheduling_strategy=strategy)(engine_class()).remote(**kwargs)
    ray.get(engine.collective_rpc.remote('store_base_weights', args=()), timeout=1800)
    return engine, pg, path, kwargs


def planned(phase, protocol, locks, shard):
    if phase == 'baseline':
        return []
    if phase == 'search':
        return [(seed, sigma, None) for seed, sigma in shard_plan(protocol, shard)]
    if phase == 'validation':
        plan = validation_plan(locks, protocol)
        part = plan[shard * VALIDATION_SHARD_SIZE:(shard + 1) * VALIDATION_SHARD_SIZE]
        if shard < 0 or not part:
            raise ValueError('unknown validation shard')
    else:
        part = locks['validation']['top10']
    return [(r['candidate']['seed'], r['candidate']['sigma'], r) for r in part]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=list(PHASE_SPLITS), required=True)
    p.add_argument('--shard', type=int)
    p.add_argument('--protocol', type=Path, default=Path('experiments/perspective_taking_n5000_protocol.json'))
    p.add_argument('--locks', type=Path, required=True)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--upstream-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if (a.phase in ('search', 'validation')) != (a.shard is not None):
        raise ValueError('--shard is required for, and only for, search and validation')
    committed(a.protocol)
    protocol = read(a.protocol)
    if protocol['schema'] != 'perspective-taking-n5000-v1': raise ValueError('wrong protocol')
    proof = verify_inputs(protocol)
    locks, lock_commits = {}, {}
    for name in LOCKS[a.phase]:
        path = a.locks/(name+'.json')
        committed(path)
        locks[name] = read(path)
        if locks[name]['protocol_sha256'] != sha(a.protocol): raise ValueError('lock is for another protocol')
        lock_commits[name] = subprocess.check_output(['git', 'log', '-1', '--format=%H %cI', '--', str(path)], text=True).strip()
    sets = splits(protocol)
    datasets = {s: sets[s] for s in PHASE_SPLITS[a.phase]}
    versions = {n: importlib.metadata.version(n) for n in protocol['required_versions']}
    if versions != protocol['required_versions']: raise ValueError('inference environment changed')
    upstream = verify_upstream(a.upstream_root)
    plan = planned(a.phase, protocol, locks, a.shard)
    root, repo = a.out, Path(__file__).resolve().parents[2]
    root.mkdir(parents=True, exist_ok=False)
    for source in [*Path(__file__).parent.glob('*.py'), a.protocol, *Path('scripts').glob('*perspective*'),
                   repo/'third_party/omnispatial/system_prompts.py']:
        target = root/'measured-source'/source.resolve().relative_to(repo)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for name in locks: shutil.copyfile(a.locks/(name+'.json'), root/('lock-'+name+'.json'))
    m = {'phase': a.phase, 'shard': a.shard, 'status': 'running', 'source': revision_info(), 'command': [sys.executable, *sys.argv],
         'started_utc': datetime.now(timezone.utc).isoformat(), 'protocol_sha256': sha(a.protocol), 'dataset_proof': proof,
         'model': protocol['model'], 'model_revision': protocol['model_revision'], 'upstream': upstream, 'versions': versions,
         'lock_commits': lock_commits, 'evaluation_sets': {s: {'examples': len(r), 'population_sha256': ident(r)} for s, r in datasets.items()},
         'planned_candidates': len(plan), 'completed_candidates': 0, 'host': os.uname().nodename,
         'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
    for cmd, name in ((['nvidia-smi', '-q'], 'gpu-before.txt'), ([sys.executable, '-m', 'pip', 'freeze'], 'pip-freeze.txt')):
        (root/name).write_bytes(subprocess.check_output(cmd))
    engine = pg = None
    records, controls = [], {}
    try:
        from transformers import AutoProcessor
        import ray
        engine, pg, path, kwargs = launch(protocol, a.upstream_root.resolve(), repo)
        write(root/'engine-config.json', kwargs)
        api = PerspectiveExecutor(engine, AutoProcessor.from_pretrained(path), a.images, protocol, root.name)
        native = api.rpc(initialize)
        write(root/'native-state.json', native)
        base_id = m['base_id'] = native['base_id']
        if 'baseline' in locks and (base_id != locks['baseline']['base_id'] or native['base_id_flat_sha256'] != locks['baseline']['base_id_flat_sha256']):
            raise ValueError('native base changed')
        controls['initial_base_exact'] = api.fingerprint() == base_id and api.drift()['exact_base']
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
        if 'baseline' in locks:
            controls['preprocessing_matches_baseline'] = all(api.preprocess[r['uid']] == locks['baseline']['preprocess'][r['uid']]
                                                             for rows in datasets.values() for r in rows)
        if not all(controls.values()): raise RuntimeError('base/zero/preprocessing control failed')
        first = None
        for seed, sigma, expected in plan:
            candidate = recipe(base_id, seed, sigma, native['parameter_mask_sha256'])
            if expected and candidate != expected['candidate']: raise ValueError('frozen recipe differs')
            state = expected['candidate_state_id'] if expected else None
            t0 = time.perf_counter()
            raw = evaluate_candidate(api, candidate, datasets, state)
            raw['expected_state_id'] = state
            raw['fingerprint_checked_before_generation'] = expected is not None
            raw['candidate_seconds'] = time.perf_counter() - t0
            write(root/'candidates'/(candidate['candidate_id']+'.json.gz'), raw)
            split = next(iter(datasets))
            outs = raw['splits'][split]['outputs']
            records.append({'candidate': candidate, 'candidate_state_id': raw['candidate_state_id'], 'split': split,
                            'examples': len(outs), 'correct_count': sum(o['correct'] for o in outs),
                            'invalid_count': sum(not o['valid_letter'] for o in outs),
                            'output_identity': raw['splits'][split]['output_identity']})
            first = first or raw
            m['completed_candidates'] = len(records)
            print(json.dumps({'phase': a.phase, 'shard': a.shard, 'completed': len(records), 'planned': len(plan), 'seed': seed,
                              'sigma': sigma, 'correct_count': records[-1]['correct_count'], 'seconds': round(raw['candidate_seconds'], 1),
                              'generation_seconds': round(raw['splits'][split]['generation_seconds'], 1)}), flush=True)
        if a.phase == 'baseline':
            repeat = {s: api.generate(rows) for s, rows in datasets.items()}
            write(root/'base-repeat.json.gz', repeat)
            controls['base_repeat_equal'] = all(repeat[s]['output_identity'] == reference[s] for s in datasets)
            # One-time proof that the fast controls agree with the original ones on this model.
            api.rebase()
            slow_base = api.rpc(vllm_audit.drift)
            fast_base = api.drift()
            api.apply(recipe(base_id, protocol['preflight']['seed'], protocol['preflight']['sigma'], native['parameter_mask_sha256']), 'snapshot-copy')
            slow_pert, fast_pert = api.rpc(vllm_audit.drift), api.drift()
            pert_tree, pert_tree_again, pert_flat = api.fingerprint(), api.fingerprint(), api.rpc(vllm_audit.fingerprint)
            api.rebase()
            strip = lambda d: {k: v for k, v in d.items() if k != 'slow_path_tensors'}
            equivalence = {'base_drift_equal': strip(fast_base) == slow_base, 'perturbed_drift_equal': strip(fast_pert) == slow_pert,
                           'perturbed_not_exact': not fast_pert['exact_base'], 'tree_fingerprint_deterministic': pert_tree == pert_tree_again,
                           'tree_fingerprint_changes': pert_tree != base_id, 'flat_fingerprint_changes': pert_flat != native['base_id_flat_sha256'],
                           'base_restored_after_proof': api.fingerprint() == base_id and api.drift()['exact_base']}
            write(root/'fast-control-equivalence.json', {'base_slow': slow_base, 'base_fast': fast_base, 'perturbed_slow': slow_pert,
                                                         'perturbed_fast': fast_pert, 'checks': equivalence})
            controls['fast_controls_equivalent'] = all(equivalence.values())
        else:
            repeat = evaluate_candidate(api, first['candidate'], datasets, first['candidate_state_id'])
            write(root/'repeat.json.gz', repeat)
            controls['repeated_state_equal'] = repeat['candidate_state_id'] == first['candidate_state_id']
            controls['repeated_outputs_equal'] = all(repeat['splits'][s]['output_identity'] == first['splits'][s]['output_identity'] for s in datasets)
        api.rebase()
        controls['final_base_exact'] = api.fingerprint() == base_id and api.drift()['exact_base']
        write(root/'memory.json', api.memory())
        write(root/'preprocess.json', api.preprocess)
        write(root/'records.json', records)
        write(root/'controls.json', controls)
        if not all(controls.values()): raise RuntimeError('correctness controls failed')
        m['status'] = 'complete'
        print(json.dumps({'phase': a.phase, 'shard': a.shard, 'status': 'complete', 'candidates': len(records)}), flush=True)
    except BaseException as exc:
        m.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        m['ended_utc'] = datetime.now(timezone.utc).isoformat()
        write(root/'run-manifest.json', m)
        if engine:
            from core.engine import cleanup_engines
            cleanup_engines([engine], [pg])
        elif 'ray' in locals(): ray.shutdown()
        (root/'gpu-after.txt').write_bytes(subprocess.check_output(['nvidia-smi', '-q']))
        write(root/'sha256.json', {f.relative_to(root).as_posix(): sha(f) for f in sorted(root.rglob('*')) if f.is_file()})


if __name__ == '__main__': main()

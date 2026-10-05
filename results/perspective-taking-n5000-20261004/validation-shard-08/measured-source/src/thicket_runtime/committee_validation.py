"""One bounded fresh-test run of the precommitted selected expert union."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .candidate import CandidateSpec
from .cli import revision_info, write_json
from .evaluation_matrix import reward_and_extractor, write_gzip
from .shared_speculative import TraceExecutor, generate_workloads, score_workloads, sha256, GSM_BLOBS
from .upstream import verify_upstream
from .vllm_profile import MODEL, REVISION
from . import vllm_audit


def committed(path):
    name = path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    if subprocess.check_output(['git', 'show', 'HEAD:' + name]) != path.read_bytes():
        raise ValueError('GPU input is not committed byte-for-byte: ' + name)


def checked_candidate(api, plan, workloads, sampling):
    """Fingerprint rejection precedes generation; restoration also covers rejection."""
    candidate = plan['candidate']
    api.rebase()
    if not api.drift()['exact_base']:
        raise RuntimeError('candidate does not begin at exact base')
    api.memory(reset=True)
    started = time.perf_counter()
    api.apply(candidate, 'snapshot-copy')
    apply_seconds = time.perf_counter() - started
    try:
        state_id = api.fingerprint()
        if state_id != plan['expected_state_id']:
            raise RuntimeError('candidate fingerprint differs from original matrix BEFORE generation')
        outputs, timings = generate_workloads(api, workloads, sampling)
    finally:
        started = time.perf_counter()
        api.restore(candidate, 'snapshot-copy')
        restore_seconds = time.perf_counter() - started
    memory = api.memory()
    score_workloads(outputs, workloads, timings)
    restoration = api.drift()
    if not restoration['exact_base']:
        raise RuntimeError('candidate restoration is not exact')
    return {'candidate': candidate, 'expected_state_id': plan['expected_state_id'], 'candidate_state_id': state_id,
            'fingerprint_checked_before_generation': True, 'outputs': outputs,
            'descriptive_costs': {'apply_seconds': apply_seconds, 'restore_seconds': restore_seconds, 'workloads': timings},
            'memory': memory, 'restoration': restoration}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream-root', required=True)
    p.add_argument('--protocol', type=Path, default=Path('experiments/complementarity_protocol.json'))
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    protocol = json.loads(args.protocol.read_bytes())
    lock_path, union_path = args.selection / 'committee-lock.json', args.selection / 'union.json'
    lock, union = json.loads(lock_path.read_bytes()), json.loads(union_path.read_bytes())
    data_path, provenance_path = [Path(protocol['fresh_dataset'][key]) for key in ('path', 'provenance_path')]
    for path in (args.protocol, lock_path, union_path, data_path, provenance_path):
        committed(path)
    if lock['protocol_sha256'] != sha256(args.protocol) or lock['union_sha256'] != sha256(union_path):
        raise ValueError('selection lock changed')
    if lock['fresh_inputs_sha256'] != {'path': sha256(data_path), 'provenance_path': sha256(provenance_path)}:
        raise ValueError('fresh test inputs changed')
    required = {cid for pop in lock['populations'].values() for committee in pop.values() for cid in committee['candidate_ids']}
    if required != {item['candidate']['candidate_id'] for item in union} or len(union) != len(required) or len(union) >= 504:
        raise ValueError('GPU plan is not exactly the unique selected union')
    if (protocol['model'], protocol['model_revision']) != (MODEL, REVISION):
        raise ValueError('model revision mismatch')
    for item in union:
        c = item['candidate']
        spec = CandidateSpec(**{k: c[k] for k in ('base_id', 'seed', 'sigma', 'rng', 'sign')})
        if spec.candidate_id != c['candidate_id'] or spec.base_id != protocol['expected_native_base_id'] or spec.rng != 'randopt-per-tensor-v1' or spec.sign != 1:
            raise ValueError('unsupported or changed candidate recipe')
    provenance = json.loads(provenance_path.read_bytes())
    if provenance['frozen_jsonl_sha256'] != sha256(data_path):
        raise ValueError('fresh input provenance mismatch')
    data = [json.loads(line) for line in data_path.read_text(encoding='utf-8').splitlines()]
    if len(data) != 300 or [r['source_index'] for r in data] != provenance['indices'] or min(provenance['indices']) < 40:
        raise ValueError('expected 300 disjoint frozen test prompts')
    upstream = verify_upstream(args.upstream_root)
    reward, extract = reward_and_extractor(args.upstream_root)
    root, repo = args.out, Path(__file__).resolve().parents[2]
    root.mkdir(parents=True, exist_ok=False)
    (root / 'candidates').mkdir()
    for source in [*Path(__file__).parent.glob('*.py'), args.protocol, repo / 'scripts/run_committee_validation.sh']:
        target = root / 'measured-source' / source.resolve().relative_to(repo)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for source in (args.protocol, lock_path, union_path, data_path, provenance_path):
        shutil.copyfile(source, root / source.name)
    manifest = {'schema': 'selected-union-validation-v1', 'status': 'running',
        'started_utc': datetime.now(timezone.utc).isoformat(), 'source': revision_info(),
        'command': [sys.executable, *sys.argv], 'model': MODEL, 'model_revision': REVISION,
        'upstream': upstream, 'gsm_blobs': GSM_BLOBS, 'protocol_sha256': sha256(args.protocol),
        'committee_lock_sha256': sha256(lock_path), 'planned_candidates': len(union), 'prompts': len(data),
        'versions': {name: importlib.metadata.version(name) for name in ('torch', 'vllm', 'ray', 'transformers', 'numpy')},
        'execution': 'original RandOpt snapshot worker, native packed tensors, BF16 TP1 eager Ray/vLLM',
        'cache_isolation': 'prefix cache disabled; all requests drained before weight change',
        'fingerprint_contract': 'compare every selected state to original matrix BEFORE generation',
        'collection': 'only committed union; no candidate generation/search/selection on new test outputs',
        'correctness_audits_timed': False}
    write_json(root / 'manifest.json', manifest)
    for command, name in ((['nvidia-smi', '-q'], 'nvidia-smi-before.txt'), ([sys.executable, '-m', 'pip', 'freeze'], 'pip-freeze.txt')):
        with (root / name).open('w') as stream:
            subprocess.run(command, stdout=stream, check=True)
    engines, pgs, matrix = [], [], []
    try:
        import ray
        from huggingface_hub import snapshot_download
        from vllm import SamplingParams
        upstream_root = str(Path(args.upstream_root).resolve())
        sys.path.insert(0, upstream_root)
        from core.engine import launch_engines, cleanup_engines
        model_path = snapshot_download(MODEL, revision=REVISION, allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model'])
        if Path(model_path).name != REVISION:
            raise ValueError('model snapshot path mismatch')
        worker_path = os.pathsep.join([upstream_root, str(repo / 'src')])
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
        ray.init(num_cpus=4, num_gpus=1, include_dashboard=False, object_store_memory=512 * 1024**2,
                 runtime_env={'env_vars': {'PYTHONPATH': worker_path, 'OMP_NUM_THREADS': '1', 'VLLM_ENABLE_V1_MULTIPROCESSING': '0'}})
        engines, pgs = launch_engines(1, model_path, precision='bfloat16', tensor_parallel_size=1,
                                     enable_prefix_caching=False, gpu_memory_utilization=protocol['gpu_memory_utilization'])
        api = TraceExecutor(engines[0], [], [], None)
        state = api.rpc(vllm_audit.initialize)
        write_json(root / 'native-state.json', state)
        if state['base_id'] != protocol['expected_native_base_id']:
            raise ValueError('native base differs from original experiment')
        manifest.update(native_base_id=state['base_id'], environment=state['environment'])
        workloads = [({'name': 'fresh300', 'max_tokens': protocol['max_tokens'], 'stop': []}, data, reward)]
        base, times = generate_workloads(api, workloads, SamplingParams)
        score_workloads(base, workloads, times)
        base_rows = base['fresh300']
        write_gzip(root / 'base.json.gz', {'outputs': base_rows, 'costs': times})
        zero = CandidateSpec(state['base_id'], 10000, 0.)
        zero_plan = {'candidate': asdict(zero) | {'candidate_id': zero.candidate_id}, 'expected_state_id': state['base_id']}
        zero_raw = checked_candidate(api, zero_plan, workloads, SamplingParams)
        write_gzip(root / 'zero-control.json.gz', zero_raw)
        controls = {'zero_outputs_equal': zero_raw['outputs']['fresh300'] == base_rows,
                    'zero_restore_exact': zero_raw['restoration']['exact_base']}
        if not all(controls.values()):
            raise RuntimeError('zero control failed')
        first, started = {}, time.monotonic()
        for plan in union:
            raw = checked_candidate(api, plan, workloads, SamplingParams)
            c = plan['candidate']
            rows = raw['outputs'].pop('fresh300')
            if any(r['prompt_token_ids'] != b['prompt_token_ids'] for r, b in zip(rows, base_rows)):
                raise ValueError('prompt tokenization changed')
            raw['outputs'] = [{k: v for k, v in r.items() if k != 'prompt_token_ids'} |
                              {'prompt_id': item['id'], 'voting_answer': extract(r['text'])} for r, item in zip(rows, data)]
            if c['population'] not in first:
                first[c['population']] = (plan, raw)
            write_gzip(root / 'candidates' / (c['candidate_id'] + '.json.gz'), raw)
            matrix.append({'candidate': c, 'state_id': raw['candidate_state_id'],
                'rewards': [r['reward'] for r in rows], 'tokens': [len(r['token_ids']) for r in rows],
                'voting_answers': [r['voting_answer'] for r in raw['outputs']],
                'capped': [r['finish_reason'] == 'length' for r in rows]})
            print(json.dumps({'completed': len(matrix), 'planned': len(union), 'population': c['population'],
                              'elapsed_seconds': time.monotonic() - started}), flush=True)
        manifest['sweep_seconds'] = time.monotonic() - started
        write_json(root / 'matrix.json', {'prompt_ids': [r['id'] for r in data], 'answers': [str(r['answer']) for r in data], 'rows': matrix})
        api.rebase()
        base_repeat, times = generate_workloads(api, workloads, SamplingParams)
        score_workloads(base_repeat, workloads, times)
        write_gzip(root / 'base-repeat.json.gz', {'outputs': base_repeat['fresh300'], 'costs': times})
        controls['base_repeat_equal'] = base_repeat['fresh300'] == base_rows
        for pop, (plan, original) in first.items():
            repeated = checked_candidate(api, plan, workloads, SamplingParams)
            write_gzip(root / ('repeat-' + pop + '.json.gz'), repeated)
            controls[pop + '_repeat_equal'] = ([{k: v for k, v in r.items() if k != 'prompt_token_ids'} for r in repeated['outputs']['fresh300']]
                == [{k: v for k, v in r.items() if k not in ('prompt_id', 'voting_answer')} for r in original['outputs']])
        controls.update(all_selected_fingerprints_match=True, all_selected_restores_exact=True,
                        final_base_equal=api.fingerprint() == state['base_id'])
        write_json(root / 'controls.json', controls)
        manifest.update(completed_candidates=len(matrix), collection_generation_rpcs=len(matrix), control_generation_rpcs=6,
                        status='complete' if all(controls.values()) else 'invalid_controls')
        print(json.dumps({'status': manifest['status'], 'candidates': len(matrix)}), flush=True)
        return 0 if manifest['status'] == 'complete' else 2
    except BaseException as exc:
        manifest.update(status='failed', error_type=type(exc).__name__, error=str(exc), completed_candidates=len(matrix))
        raise
    finally:
        manifest['ended_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(root / 'manifest.json', manifest)
        if engines:
            cleanup_engines(engines, pgs)
        elif 'ray' in locals():
            ray.shutdown()
        with (root / 'nvidia-smi-after.txt').open('w') as stream:
            subprocess.run(['nvidia-smi', '-q'], stdout=stream, check=False)
        write_json(root / 'sha256.json', {f.relative_to(root).as_posix(): sha256(f)
                                         for f in sorted(root.rglob('*')) if f.is_file() and f.name != 'sha256.json'})


if __name__ == '__main__':
    raise SystemExit(main())

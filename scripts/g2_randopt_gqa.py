"""G2 runner (results/paper-analysis/g2-sameRun/plan_lock.md): randopt.py's loop for GQA with images, one in-process vLLM
per GPU. Phase 'select': worker w evaluates perturbations k with k % W == w on the selection set, appending
{k, seed, sigma, reward} to select_w.jsonl (resumable). Phase 'test': ranks perturbations by reward (desc; ties by k),
worker w evaluates top-K ranks r with r % W == w on the test set and saves vote keys; worker 0 also evaluates BASE."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

PROMPT = "Look at the image and answer the question.\n\nQuestion: {q}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."


def population(n):
    rng = np.random.default_rng(42)
    seeds = rng.choice(2 ** 31, size=n, replace=False).tolist()
    sig = rng.choice([0.0005, 0.001, 0.002], size=n).tolist()
    return seeds, sig


def _vision_changed(worker):
    import torch
    return sum(int(not torch.equal(p.data, worker._base_weights[n])) for n, p in worker.model_runner.model.named_parameters()
               if n.startswith(worker._VISUAL_PREFIXES))


def _restore_mismatch(worker):
    import torch
    return sum(int(not torch.equal(p.data, worker._base_weights[n])) for n, p in worker.model_runner.model.named_parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['select', 'test'], required=True); ap.add_argument('--worker', type=int, required=True)
    ap.add_argument('--workers', type=int, default=6); ap.add_argument('--n', type=int, default=5000); ap.add_argument('--k', type=int, default=50)
    ap.add_argument('--upstream', type=Path, default=Path('/workspace/RandOpt')); ap.add_argument('--root', type=Path, default=Path('/workspace/g2'))
    ap.add_argument('--out', default='out'); ap.add_argument('--model-path', required=True); ap.add_argument('--limit-test', type=int, default=0)
    ap.add_argument('--eager', type=int, default=1); ap.add_argument('--gpu-mem', type=float, default=.85)
    a = ap.parse_args()
    sys.path.insert(0, str(a.upstream))
    os.environ.update({'VLLM_ENABLE_V1_MULTIPROCESSING': '0', 'PERTURB_VISUAL': '0', 'OMP_NUM_THREADS': '2'})
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from data_handlers.gqa import GQAHandler
    h = GQAHandler(); items = json.loads((a.root / 'items.json').read_text()); out = a.root / a.out; out.mkdir(parents=True, exist_ok=True)
    llm = LLM(model=a.model_path, tokenizer=a.model_path, dtype='bfloat16', tensor_parallel_size=1, distributed_executor_backend='uni',
              worker_extension_cls='utils.worker_extn.WorkerExtension', enforce_eager=bool(a.eager), enable_prefix_caching=False,
              gpu_memory_utilization=a.gpu_mem, max_model_len=8192, max_num_seqs=256, max_num_batched_tokens=16384,
              limit_mm_per_prompt={'image': 1, 'video': 0}, mm_processor_cache_gb=0, seed=0, disable_log_stats=True)
    rpc = lambda f, *args: llm.collective_rpc(f, args=args)[0]
    rpc('store_base_weights'); proc = AutoProcessor.from_pretrained(a.model_path)
    sp = SamplingParams(temperature=0.0, seed=42, max_tokens=256)

    def reqs(rows):
        return [{'prompt': proc.apply_chat_template([{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': PROMPT.format(q=r['question'])}]}],
                                                    tokenize=False, add_generation_prompt=True),
                 'multi_modal_data': {'image': Image.open(a.root / 'images' / f"{r['imageId']}.jpg").convert('RGB')}} for r in rows]

    def gt(r):
        return {'answer': r['answer'], 'full_answer': r['fullAnswer']}
    seeds, sig = population(a.n)
    if a.phase == 'select':
        rows = items['selection']; R = reqs(rows); f = out / f'select_{a.worker}.jsonl'
        done = {json.loads(l)['k'] for l in f.read_text().splitlines() if l.strip()} if f.exists() else set()
        if a.worker == 0 and not (out / 'base_select.json').exists():
            o = llm.generate(R, sp, use_tqdm=False)
            (out / 'base_select.json').write_text(json.dumps({'reward': float(np.mean([h.compute_reward(x.outputs[0].text, gt(r)) for r, x in zip(rows, o)]))}))
        checked = 0
        for k in range(a.worker, a.n, a.workers):
            if k in done: continue
            t0 = time.time(); rpc('apply_perturbation', seeds[k], sig[k])
            if checked < 2:
                assert rpc(_vision_changed) == 0, 'vision changed'; checked += 1
            o = llm.generate(R, sp, use_tqdm=False)
            rew = float(np.mean([h.compute_reward(x.outputs[0].text, gt(r)) for r, x in zip(rows, o)]))
            with open(f, 'a') as fh:
                fh.write(json.dumps({'k': k, 'seed': seeds[k], 'sigma': sig[k], 'reward': rew, 'sec': round(time.time() - t0, 2)}) + '\n')
        rpc('reset_to_base_weights'); assert rpc(_restore_mismatch) == 0, 'restore'
        print('SELECT_DONE', a.worker, flush=True)
    else:
        perf = {}
        for w in range(a.workers):
            for l in (out / f'select_{w}.jsonl').read_text().splitlines():
                if l.strip():
                    d = json.loads(l); perf[d['k']] = d['reward']
        assert len(perf) == a.n, (len(perf), a.n)
        order = sorted(range(a.n), key=lambda k: (-perf[k], k))[: a.k]
        if a.worker == 0:
            (out / 'topk.json').write_text(json.dumps([{'rank': i, 'k': k, 'seed': seeds[k], 'sigma': sig[k], 'reward': perf[k]} for i, k in enumerate(order)]))
        rows = items['test'][: a.limit_test] if a.limit_test else items['test']; R = reqs(rows)
        jobs = ([('base', None)] if a.worker == 0 else []) + [(f'rank{i}', order[i]) for i in range(a.worker, a.k, a.workers)]
        for name, k in jobs:
            if (out / f'test_{name}.json').exists(): continue
            if k is None: rpc('reset_to_base_weights')
            else: rpc('apply_perturbation', seeds[k], sig[k])
            o = llm.generate(R, sp, use_tqdm=False)
            (out / f'test_{name}.json').write_text(json.dumps({'k': k, 'ans': [h.extract_answer_for_voting(x.outputs[0].text) for x in o],
                                                               'correct': [bool(h.is_answer_correct(x.outputs[0].text, gt(r))) for r, x in zip(rows, o)]}))
        rpc('reset_to_base_weights'); assert rpc(_restore_mismatch) == 0, 'restore'
        print('TEST_DONE', a.worker, flush=True)


if __name__ == '__main__':
    main()

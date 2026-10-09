"""GB runner (results/paper-analysis/gb-gsm8k-boxed-search/plan_lock.md): randopt.py's GSM8K loop with a chosen prompt,
one in-process vLLM per worker (CUDA graphs), several workers per GPU. Faithful to randopt.py: population from
np.random.default_rng(pop_seed) (seeds, then σ ∈ {0.0005, 0.001, 0.002}); selection reward = mean of
GSM8KHandler.compute_reward over the first 200 train rows, greedy (temperature 0, seed 42, max_tokens 1024); ranking
by reward, ties by population index; test = greedy answers of the top-K on the 1319 test rows, RandOpt's vote and
scorer. Deviation (stated): weights reset exactly from a stored base copy (apply_perturbation) instead of
randopt.py's subtract-to-restore. Phase 'select': worker w scores perturbations k ≡ w (mod W), appending to
select_w.jsonl (resumable). Phase 'test': worker w evaluates ranks r ≡ w (mod W); worker 0 also BASE. Texts saved."""
import argparse
import gzip
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from o2_eval import PROMPTS  # noqa: E402  (randopt / plain / boxed, identical to O2/Q2/PS)


def population(n, pop_seed):
    rng = np.random.default_rng(pop_seed)
    return rng.choice(2 ** 31, size=n, replace=False).tolist(), rng.choice([0.0005, 0.001, 0.002], size=n).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['select', 'test'], required=True); ap.add_argument('--worker', type=int, required=True)
    ap.add_argument('--workers', type=int, required=True); ap.add_argument('--n', type=int, default=5000); ap.add_argument('--k', type=int, default=50)
    ap.add_argument('--pop-seed', type=int, default=42); ap.add_argument('--prompt', choices=list(PROMPTS), required=True)
    ap.add_argument('--first', type=int, default=0)  # select only k < first (fidelity check); 0 = all
    ap.add_argument('--upstream', type=Path, default=Path('/workspace/RandOpt')); ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', default='out'); ap.add_argument('--model-path', required=True); ap.add_argument('--limit-test', type=int, default=0)
    ap.add_argument('--gpu-mem', type=float, default=0.3)
    a = ap.parse_args()
    sys.path.insert(0, str(a.upstream)); os.environ.update({'VLLM_ENABLE_V1_MULTIPROCESSING': '0', 'OMP_NUM_THREADS': '2'})
    import datasets
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from data_handlers.gsm8k import GSM8KHandler
    h = GSM8KHandler(); out = a.root / a.out; out.mkdir(parents=True, exist_ok=True)
    up = a.upstream / 'data/gsm8k'
    sel_gt = [t['ground_truth'] for t in h.load_data(str(up / 'train.parquet'), split='train', max_samples=200)]
    test_gt = [t['ground_truth'] for t in h.load_data(str(up / 'test.parquet'), split='test')]
    sel_q = [r['question'] for r in datasets.load_dataset('openai/gsm8k', 'main', split='train').select(range(200))]
    test_q = [r['question'] for r in datasets.load_dataset('openai/gsm8k', 'main', split='test')]
    tok = AutoTokenizer.from_pretrained(a.model_path)
    req = lambda qs: [tok.apply_chat_template([{'role': 'user', 'content': PROMPTS[a.prompt](q)}], add_generation_prompt=True, tokenize=False) for q in qs]
    llm = LLM(model=a.model_path, tokenizer=a.model_path, dtype='bfloat16', seed=0, gpu_memory_utilization=a.gpu_mem, max_model_len=4096,
              max_num_seqs=512, enable_prefix_caching=False, worker_extension_cls='utils.worker_extn.WorkerExtension', disable_log_stats=True)
    rpc = lambda f, *args: llm.collective_rpc(f, args=args)[0]
    rpc('store_base_weights'); sp = SamplingParams(temperature=0.0, seed=42, max_tokens=1024)
    print('ENGINE_READY', a.worker, flush=True)  # the launcher starts the next worker on this GPU only after this
    seeds, sig = population(a.n, a.pop_seed)
    if a.phase == 'select':
        R = req(sel_q); f = out / f'select_{a.worker}.jsonl'
        done = {json.loads(l)['k'] for l in f.read_text().splitlines() if l.strip()} if f.exists() else set()
        if a.worker == 0 and not (out / 'base_select.json').exists():
            o = llm.generate(R, sp, use_tqdm=False)
            (out / 'base_select.json').write_text(json.dumps({'reward': float(np.mean([h.compute_reward(x.outputs[0].text, g) for x, g in zip(o, sel_gt)]))}))
        last = a.first if a.first else a.n
        for k in range(a.worker, last, a.workers):
            if k in done: continue
            t0 = time.time(); rpc('apply_perturbation', seeds[k], sig[k]); o = llm.generate(R, sp, use_tqdm=False)
            rew = float(np.mean([h.compute_reward(x.outputs[0].text, g) for x, g in zip(o, sel_gt)]))
            with open(f, 'a') as fh:
                fh.write(json.dumps({'k': k, 'seed': seeds[k], 'sigma': sig[k], 'reward': rew, 'sec': round(time.time() - t0, 2)}) + '\n')
        rpc('reset_to_base_weights'); print('SELECT_DONE', a.worker, flush=True)
    else:
        perf = {}
        for fsel in sorted(out.glob('select_*.jsonl')):  # selection may have used a different worker count
            for l in fsel.read_text().splitlines():
                if l.strip():
                    d = json.loads(l); perf[d['k']] = d['reward']
        assert len(perf) == a.n, (len(perf), a.n)
        order = sorted(range(a.n), key=lambda k: (-perf[k], k))[: a.k]
        if a.worker == 0:
            (out / 'topk.json').write_text(json.dumps([{'rank': i, 'k': k, 'seed': seeds[k], 'sigma': sig[k], 'reward': perf[k]} for i, k in enumerate(order)]))
        n_t = a.limit_test or len(test_q); R = req(test_q[:n_t])
        jobs = ([('base', None)] if a.worker == 0 else []) + [(f'rank{i}', order[i]) for i in range(a.worker, a.k, a.workers)]
        for name, k in jobs:
            if (out / f'test_{name}.json').exists(): continue
            if k is None: rpc('reset_to_base_weights')
            else: rpc('apply_perturbation', seeds[k], sig[k])
            o = llm.generate(R, sp, use_tqdm=False); texts = [x.outputs[0].text for x in o]; ans = [h.extract_answer(t) for t in texts]
            (out / f'test_{name}.json').write_text(json.dumps({'k': k, 'ans': ans, 'correct': [bool(v) and bool(h.is_answer_correct(h.format_answer_for_check(v), test_gt[i]))
                                                                                               for i, v in enumerate(ans)]}))
            (out / f'test_{name}.texts.json.gz').write_bytes(gzip.compress(json.dumps(texts).encode()))
        rpc('reset_to_base_weights'); print('TEST_DONE', a.worker, flush=True)


if __name__ == '__main__':
    main()

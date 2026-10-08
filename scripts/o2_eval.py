"""O2 runner (results/paper-analysis/o2-olmo-prompt/plan_lock.md): GSM8K / OLMo-2-1B under several prompts. One
in-process vLLM (CUDA graphs) with RandOpt's WorkerExtension. Arms: base (greedy), sc (50 samples, T = 0.7, seed
20261007, as p2_sc.py), members (O1's top-50 perturbations, rebuilt with apply_perturbation from stored base weights).
Saves per generation the extracted answer, correctness (RandOpt's GSM8K scorer), finish reason, token count and format
flags, plus all texts (gzipped). Resumable: existing output files are skipped."""
import argparse
import gzip
import json
import os
from pathlib import Path
import sys
import time

GSM_INSTR = 'Let\'s think step by step and output the final answer after "####".'  # = p2_sc.py / RandOpt's verl prompt
PROMPTS = {'randopt': lambda q: q + ' ' + GSM_INSTR,
           'plain': lambda q: q,
           'boxed': lambda q: q + '\nPlease reason step by step, and put your final answer within \\boxed{}.'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompt', choices=list(PROMPTS), required=True); ap.add_argument('--arms', required=True)  # e.g. base,sc,members
    ap.add_argument('--ranks', default='0-49'); ap.add_argument('--top50', type=Path, required=True)  # ranks: e.g. 0-4,45-49
    ap.add_argument('--upstream', type=Path, default=Path('/workspace/RandOpt')); ap.add_argument('--model-path', required=True)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    sys.path.insert(0, str(a.upstream)); os.environ.update({'VLLM_ENABLE_V1_MULTIPROCESSING': '0', 'OMP_NUM_THREADS': '4'})
    import datasets
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from data_handlers.gsm8k import GSM8KHandler
    h = GSM8KHandler(); a.out.mkdir(parents=True, exist_ok=True)
    ds = datasets.load_dataset('openai/gsm8k', 'main', split='test')
    items = [{'q': r['question'], 'gt': r['answer'].split('####')[-1].strip().replace(',', '')} for r in ds]
    items = items[: a.limit] if a.limit else items
    tok = AutoTokenizer.from_pretrained(a.model_path)
    R = [tok.apply_chat_template([{'role': 'user', 'content': PROMPTS[a.prompt](it['q'])}], add_generation_prompt=True, tokenize=False) for it in items]
    llm = LLM(model=a.model_path, tokenizer=a.model_path, dtype='bfloat16', seed=0, gpu_memory_utilization=.85, max_model_len=4096,
              max_num_seqs=512, enable_prefix_caching=False, worker_extension_cls='utils.worker_extn.WorkerExtension', disable_log_stats=True)
    rpc = lambda f, *args: llm.collective_rpc(f, args=args)[0]
    rpc('store_base_weights')
    greedy = SamplingParams(temperature=0.0, max_tokens=1024, n=1)

    def rec(c, i):
        ans = h.extract_answer(c.text)
        return {'a': ans, 'c': bool(ans) and bool(h.is_answer_correct(h.format_answer_for_check(ans), '#### ' + items[i]['gt'])),
                'fin': c.finish_reason, 'ntok': len(c.token_ids), 'hash4': '####' in c.text, 'boxed': '\\boxed{' in c.text}

    def save(name, recs, texts):
        (a.out / f'{name}.json').write_text(json.dumps(recs))
        (a.out / f'{name}.texts.json.gz').write_bytes(gzip.compress(json.dumps(texts).encode()))
    top = json.loads(a.top50.read_text())
    ranks = [r for part in a.ranks.split(',') for r in range(int(part.split('-')[0]), int(part.split('-')[-1]) + 1)]
    for arm in a.arms.split(','):
        t0 = time.time()
        if arm == 'base' and not (a.out / f'{a.prompt}_base.json').exists():
            rpc('reset_to_base_weights'); o = llm.generate(R, greedy, use_tqdm=False)
            save(f'{a.prompt}_base', [rec(x.outputs[0], i) for i, x in enumerate(o)], [x.outputs[0].text for x in o])
        elif arm == 'sc' and not (a.out / f'{a.prompt}_sc.json').exists():
            rpc('reset_to_base_weights')
            o = llm.generate(R, SamplingParams(temperature=0.7, top_p=1.0, seed=20261007, max_tokens=1024, n=50), use_tqdm=False)
            save(f'{a.prompt}_sc', [[rec(c, i) for c in x.outputs] for i, x in enumerate(o)], [[c.text for c in x.outputs] for x in o])
        elif arm == 'members':
            for m in [top[r] for r in ranks]:
                name = f"{a.prompt}_rank{m['rank']}"
                if (a.out / f'{name}.json').exists(): continue
                rpc('apply_perturbation', m['seed'], m['sigma']); o = llm.generate(R, greedy, use_tqdm=False)
                save(name, [rec(x.outputs[0], i) for i, x in enumerate(o)], [x.outputs[0].text for x in o])
            rpc('reset_to_base_weights')
        print(json.dumps({'prompt': a.prompt, 'arm': arm, 'seconds': round(time.time() - t0, 1)}), flush=True)


if __name__ == '__main__':
    main()

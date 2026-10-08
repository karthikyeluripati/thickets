"""G3 runner (results/paper-analysis/g3-termination/plan_lock.md). One in-process vLLM (CUDA graphs). For each budget,
evaluates BASE, the G2 top-50 MEMBERS (by rank) and SC@50 on G2's test questions, saving per generation the vote key,
correctness, finish reason, token count and whether a \\boxed{} answer is present (texts for BASE/MEMBERS, gzipped).
Resumable: existing output files are skipped."""
import argparse
import gzip
import json
import os
from pathlib import Path
import sys
import time

PROMPT = "Look at the image and answer the question.\n\nQuestion: {q}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--budgets', default='256,1024'); ap.add_argument('--arms', default='base,sc,members')
    ap.add_argument('--k', type=int, default=50); ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--items', default='results/paper-analysis/g2-sameRun/pod/g2/items.json')
    ap.add_argument('--topk', default='results/paper-analysis/g2-sameRun/pod/g2/out/topk.json')
    ap.add_argument('--images', type=Path, default=Path('/workspace/g2/images'))
    ap.add_argument('--upstream', type=Path, default=Path('/workspace/RandOpt')); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--model-path', required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(a.upstream))
    os.environ.update({'VLLM_ENABLE_V1_MULTIPROCESSING': '0', 'PERTURB_VISUAL': '0', 'OMP_NUM_THREADS': '4'})
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from data_handlers.gqa import GQAHandler
    h = GQAHandler(); a.out.mkdir(parents=True, exist_ok=True)
    rows = json.loads(Path(a.items).read_text())['test']; rows = rows[: a.limit] if a.limit else rows
    topk = json.loads(Path(a.topk).read_text())[: a.k]
    llm = LLM(model=a.model_path, tokenizer=a.model_path, dtype='bfloat16', tensor_parallel_size=1, distributed_executor_backend='uni',
              worker_extension_cls='utils.worker_extn.WorkerExtension', enforce_eager=False, enable_prefix_caching=False,
              gpu_memory_utilization=0.80, max_model_len=8192, max_num_seqs=256, max_num_batched_tokens=16384,
              limit_mm_per_prompt={'image': 1, 'video': 0}, mm_processor_cache_gb=0, seed=0, disable_log_stats=True)
    rpc = lambda f, *args: llm.collective_rpc(f, args=args)[0]
    rpc('store_base_weights'); proc = AutoProcessor.from_pretrained(a.model_path)
    R = [{'prompt': proc.apply_chat_template([{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': PROMPT.format(q=r['question'])}]}],
                                             tokenize=False, add_generation_prompt=True),
          'multi_modal_data': {'image': Image.open(a.images / f"{r['imageId']}.jpg").convert('RGB')}} for r in rows]
    gt = [{'answer': r['answer'], 'full_answer': r['fullAnswer']} for r in rows]

    def rec(c, i):
        t = c.text
        return {'key': h.extract_answer_for_voting(t), 'c': bool(h.is_answer_correct(t, gt[i])), 'fin': c.finish_reason,
                'ntok': len(c.token_ids), 'boxed': '\\boxed{' in t}

    def save(name, outs, texts):
        (a.out / f'{name}.json').write_text(json.dumps(outs))
        if texts is not None: (a.out / f'{name}.texts.json.gz').write_bytes(gzip.compress(json.dumps(texts).encode()))
    for b in [int(x) for x in a.budgets.split(',')]:
        greedy = SamplingParams(temperature=0.0, seed=42, max_tokens=b)
        for arm in a.arms.split(','):
            t0 = time.time()
            if arm == 'base':
                name = f'b{b}_base'
                if (a.out / f'{name}.json').exists(): continue
                rpc('reset_to_base_weights'); o = llm.generate(R, greedy, use_tqdm=False)
                save(name, [rec(x.outputs[0], i) for i, x in enumerate(o)], [x.outputs[0].text for x in o])
            elif arm == 'sc':
                name = f'b{b}_sc'
                if (a.out / f'{name}.json').exists(): continue
                rpc('reset_to_base_weights')
                o = llm.generate(R, SamplingParams(temperature=0.7, top_p=1.0, seed=20261007, max_tokens=b, n=50), use_tqdm=False)
                save(name, [[rec(c, i) for c in x.outputs] for i, x in enumerate(o)], None)
            else:
                for m in topk:
                    name = f"b{b}_rank{m['rank']}"
                    if (a.out / f'{name}.json').exists(): continue
                    rpc('apply_perturbation', m['seed'], m['sigma']); o = llm.generate(R, greedy, use_tqdm=False)
                    save(name, [rec(x.outputs[0], i) for i, x in enumerate(o)], [x.outputs[0].text for x in o])
                rpc('reset_to_base_weights')
            print(json.dumps({'budget': b, 'arm': arm, 'seconds': round(time.time() - t0, 1)}), flush=True)
    print('G3_DONE', flush=True)


if __name__ == '__main__':
    main()

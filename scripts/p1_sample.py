"""P1 (results/paper-analysis/p1/plan_lock.md): temperature sampling of the UNPERTURBED base model on the same GQA items,
with RandOpt's CoT prompt and RandOpt's GQAHandler scorer. For each T in the grid: n samples per item (fixed seed).
Output: samples_T{T}.json.gz = {item_id: [{'pred','correct'} x n]}.
"""
import argparse
import gzip
import json
import os
from pathlib import Path
import sys
import time

COT = "Look at the image and answer the question.\n\nQuestion: {q}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', type=Path, required=True); ap.add_argument('--upstream', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--model-path', required=True)
    ap.add_argument('--items', default='results/paper-analysis/p1/items.json')
    ap.add_argument('--temps', default='0.3,0.5,0.7,1.0'); ap.add_argument('--n', type=int, default=32)
    a = ap.parse_args()
    sys.path.insert(0, str(a.upstream)); os.environ.update({'VLLM_ENABLE_V1_MULTIPROCESSING': '0', 'OMP_NUM_THREADS': '2'})
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from data_handlers.gqa import GQAHandler
    handler = GQAHandler(); a.out.mkdir(parents=True, exist_ok=True)
    it = json.loads(Path(a.items).read_text(encoding='utf-8')); rows = it['selection'] + it['heldout']
    llm = LLM(model=a.model_path, tokenizer=a.model_path, dtype='bfloat16', tensor_parallel_size=1, distributed_executor_backend='uni',
              enforce_eager=True, enable_prefix_caching=True, gpu_memory_utilization=.85, max_model_len=8192, max_num_seqs=256,
              max_num_batched_tokens=16384, limit_mm_per_prompt={'image': 1, 'video': 0}, seed=0, disable_log_stats=True)
    proc = AutoProcessor.from_pretrained(a.model_path)
    req = [{'prompt': proc.apply_chat_template([{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': COT.format(q=r['question'])}]}],
                                               tokenize=False, add_generation_prompt=True),
            'multi_modal_data': {'image': Image.open(a.images / f"{r['imageId']}.jpg").convert('RGB')}} for r in rows]
    for T in [float(x) for x in a.temps.split(',')]:
        t0 = time.time(); res = {}
        outs = llm.generate(req, SamplingParams(n=a.n, temperature=T, top_p=1.0, seed=20261007, max_tokens=256), use_tqdm=False)
        for r, o in zip(rows, outs):
            res[r['id']] = [{'pred': handler.extract_answer(c.text), 'correct': bool(handler.compute_reward(c.text, {'answer': r['answer']}) > 0)} for c in o.outputs]
        (a.out / f'samples_T{T}.json.gz').write_bytes(gzip.compress(json.dumps(res).encode()))
        acc = sum(x['correct'] for v in res.values() for x in v) / (len(res) * a.n)
        print(json.dumps({'T': T, 'n': a.n, 'items': len(rows), 'mean_acc': round(100 * acc, 2), 'seconds': round(time.time() - t0, 1)}), flush=True)
    print('P1_SAMPLES_DONE', flush=True)


if __name__ == '__main__':
    main()

"""P2 (results/paper-analysis/p2/plan_lock.md): base-model greedy and self-consistency (SC@50) on RandOpt's own tasks with
RandOpt's own prompts and scorers, to compare against the Neural Thickets paper's reported Base / TT-MV / RandOpt numbers.
Tasks: gsm8k (openai/gsm8k main test, 1319; verl preprocessing prompt; GSM8KHandler; max_tokens 1024) and gqa (2000-item
hash sample of testdev-balanced; RandOpt GQA CoT prompt; GQAHandler; max_tokens 256).
Output per condition: {task}_{tag}_{cond}.json.gz with per-item answers/correctness; summary printed.
"""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

GSM_INSTR = 'Let\'s think step by step and output the final answer after "####".'
GQA_COT = "Look at the image and answer the question.\n\nQuestion: {q}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, choices=['gsm8k', 'gqa', 'math500']); ap.add_argument('--model', required=True)
    ap.add_argument('--revision', default='main'); ap.add_argument('--tag', required=True)
    ap.add_argument('--upstream', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--temps', default='0.7,0.3'); ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--gqa-n', type=int, default=2000); ap.add_argument('--images', type=Path, default=Path('/workspace/gqa-all'))
    a = ap.parse_args()
    sys.path.insert(0, str(a.upstream)); os.environ.update({'VLLM_ENABLE_V1_MULTIPROCESSING': '0', 'OMP_NUM_THREADS': '2'})
    from huggingface_hub import hf_hub_download, snapshot_download
    from vllm import LLM, SamplingParams
    a.out.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(a.model, revision=a.revision, allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja'])
    meta = {'task': a.task, 'model': a.model, 'revision': a.revision}
    if a.task == 'gsm8k':
        import datasets
        from data_handlers.gsm8k import GSM8KHandler
        h = GSM8KHandler(); ds = datasets.load_dataset('openai/gsm8k', 'main', split='test')
        meta['dataset_fingerprint'] = ds._fingerprint
        items = [{'id': str(i), 'q': r['question'], 'gt': r['answer'].split('####')[-1].strip().replace(',', '')} for i, r in enumerate(ds)]
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(path)
        req = [tok.apply_chat_template([{'role': 'user', 'content': it['q'] + ' ' + GSM_INSTR}], add_generation_prompt=True, tokenize=False) for it in items]
        max_tokens = 1024
        score = lambda t, it: h.compute_reward(t, it['gt']) > 0
        llm = LLM(model=path, tokenizer=path, dtype='bfloat16', seed=0, gpu_memory_utilization=.88, max_model_len=4096, max_num_seqs=512,
                  enable_prefix_caching=True, disable_log_stats=True)
    elif a.task == 'math500':
        import datasets
        from data_handlers.math500 import INSTRUCTION, MATH500Handler
        h = MATH500Handler(); ds = datasets.load_dataset('HuggingFaceH4/MATH-500', split='test')
        meta['dataset_fingerprint'] = ds._fingerprint
        items = [{'id': str(r['unique_id']), 'q': r['problem'], 'gt': str(r['answer'])} for r in ds]
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(path)
        req = [tok.apply_chat_template([{'role': 'user', 'content': it['q'] + chr(10) * 2 + INSTRUCTION}], add_generation_prompt=True, tokenize=False) for it in items]
        max_tokens = 2048
        score = lambda t, it: h.compute_reward(t, it['gt']) > 0
        llm = LLM(model=path, tokenizer=path, dtype='bfloat16', seed=0, gpu_memory_utilization=.88, max_model_len=4096, max_num_seqs=512,
                  enable_prefix_caching=True, disable_log_stats=True)
    else:
        import pandas as pd
        from PIL import Image
        from transformers import AutoProcessor
        from data_handlers.gqa import GQAHandler
        h = GQAHandler(); REV = 'a6e72d6e1b912da88af8b2f9eba05d5ea8ec2dd8'
        q = pd.read_parquet(hf_hub_download('lmms-lab-encoder/GQA', 'testdev_balanced_instructions/testdev-00000-of-00001.parquet', repo_type='dataset', revision=REV))
        q['h'] = [hashlib.sha256(('p2-gqa-v1:' + str(i)).encode()).hexdigest() for i in q.id]; q = q.sort_values('h').head(a.gqa_n)
        imgs = pd.read_parquet(hf_hub_download('lmms-lab-encoder/GQA', 'testdev_balanced_images/testdev-00000-of-00001.parquet', repo_type='dataset', revision=REV))
        need = set(q.imageId); a.images.mkdir(parents=True, exist_ok=True)
        for _, r in imgs.iterrows():
            if str(r['id']) in need:
                v = r['image']; b = v['bytes'] if isinstance(v, dict) else v; (a.images / f"{r['id']}.jpg").write_bytes(b)
        items = [{'id': str(r.id), 'q': r.question, 'gt': str(r.answer).strip().lower(), 'img': r.imageId} for r in q.itertuples()]
        proc = AutoProcessor.from_pretrained(path)
        req = [{'prompt': proc.apply_chat_template([{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': GQA_COT.format(q=it['q'])}]}],
                                                   tokenize=False, add_generation_prompt=True),
                'multi_modal_data': {'image': Image.open(io.BytesIO((a.images / f"{it['img']}.jpg").read_bytes())).convert('RGB')}} for it in items]
        max_tokens = 256
        score = lambda t, it: h.compute_reward(t, {'answer': it['gt']}) > 0
        llm = LLM(model=path, tokenizer=path, dtype='bfloat16', seed=0, gpu_memory_utilization=.88, max_model_len=8192, max_num_seqs=256,
                  enable_prefix_caching=True, limit_mm_per_prompt={'image': 1, 'video': 0}, disable_log_stats=True)
    meta['n_items'] = len(items); (a.out / f'{a.task}_{a.tag}_meta.json').write_text(json.dumps(meta))
    conds = [('greedy', SamplingParams(temperature=0.0, max_tokens=max_tokens, n=1))] + \
            [(f'T{T}', SamplingParams(temperature=float(T), top_p=1.0, seed=20261007, max_tokens=max_tokens, n=a.n)) for T in a.temps.split(',')]
    for name, sp in conds:
        t0 = time.time(); res = {}
        for it, o in zip(items, llm.generate(req, sp, use_tqdm=False)):
            res[it['id']] = [{'a': str(h.extract_answer(c.text)), 'c': bool(score(c.text, it))} for c in o.outputs]
        (a.out / f'{a.task}_{a.tag}_{name}.json.gz').write_bytes(gzip.compress(json.dumps({'gt': {it['id']: it['gt'] for it in items}, 'res': res}).encode()))
        acc1 = sum(v[0]['c'] for v in res.values()) / len(res)
        print(json.dumps({'task': a.task, 'tag': a.tag, 'cond': name, 'acc_first_sample': round(100 * acc1, 2), 'seconds': round(time.time() - t0, 1)}), flush=True)
    print('P2_DONE', a.task, a.tag, flush=True)


if __name__ == '__main__':
    main()

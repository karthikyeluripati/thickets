"""G1 (results/paper-analysis/g1/plan_lock.md): GQA in the RandOpt setting. One worker per GPU (vLLM in-process, `uni`),
pinned RandOpt WorkerExtension with PERTURB_VISUAL=0 (language model only, as in the Neural Thickets GQA experiment).
Workers claim candidates from a shared queue (O_EXCL claim files) so no GPU idles.

Per candidate (and BASE): RandOpt's GQA chain-of-thought prompt + boxed answer (greedy, max_tokens 256), scored with
RandOpt's own GQAHandler.compute_reward; plus a direct one-word prompt on yes/no items recording first-token log-probs
of 'Yes' (9454) and 'No' (2753). Saves one file per candidate; exact base restore verified after each.
"""
import argparse
import gzip
import json
import os
from pathlib import Path
import sys
import time

COT = "Look at the image and answer the question.\n\nQuestion: {q}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
DIRECT = "Look at the image and answer the question.\n\nQuestion: {q}\n\nAnswer with a single word."
YES, NO = 9454, 2753


def _restored_exact(worker):
    import torch
    return sum(int(not torch.equal(p.data, worker._base_weights[n].to(p.device))) for n, p in worker.model_runner.model.named_parameters())


def _vision_untouched(worker):
    import torch
    return sum(int(not torch.equal(p.data, worker._base_weights[n].to(p.device)))
               for n, p in worker.model_runner.model.named_parameters() if n.startswith(('visual.', 'model.visual.')))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--worker', required=True); ap.add_argument('--images', type=Path, required=True)
    ap.add_argument('--upstream', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--model-path', required=True)
    ap.add_argument('--usd-per-hour', type=float, required=True); ap.add_argument('--pod-start-epoch', type=float, required=True)
    ap.add_argument('--cap-usd', type=float, required=True)
    ap.add_argument('--items', default='results/paper-analysis/g1/frozen_items.json')
    ap.add_argument('--cands', default='results/paper-analysis/g1/frozen_candidates.json')
    a = ap.parse_args()
    sys.path.insert(0, str(a.upstream))
    os.environ.update({'VLLM_ENABLE_V1_MULTIPROCESSING': '0', 'PERTURB_VISUAL': '0', 'OMP_NUM_THREADS': '2'})
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from data_handlers.gqa import GQAHandler
    handler = GQAHandler()
    out = a.out; (out / 'claims').mkdir(parents=True, exist_ok=True); (out / 'cands').mkdir(parents=True, exist_ok=True)
    spent = lambda: (time.time() - a.pod_start_epoch) / 3600 * a.usd_per_hour
    logf = out / f'log_{a.worker}.jsonl'

    def note(step, **kw):
        kw.update(step=step, worker=a.worker, spent_usd=round(spent(), 3), t=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        with open(logf, 'a') as f: f.write(json.dumps(kw) + '\n')
        print(json.dumps(kw)[:300], flush=True)
    items = json.loads(Path(a.items).read_text(encoding='utf-8')); rows = items['selection'] + items['heldout']
    yn = [r for r in rows if r['yesno']]
    cands = json.loads(Path(a.cands).read_text())['candidates']
    llm = LLM(model=a.model_path, tokenizer=a.model_path, dtype='bfloat16', tensor_parallel_size=1, distributed_executor_backend='uni',
              worker_extension_cls='utils.worker_extn.WorkerExtension', enforce_eager=True, enable_prefix_caching=False,
              gpu_memory_utilization=.85, max_model_len=8192, max_num_seqs=256, max_num_batched_tokens=16384, max_logprobs=20,
              limit_mm_per_prompt={'image': 1, 'video': 0}, mm_processor_cache_gb=0, seed=0, disable_log_stats=True)
    rpc = lambda f, *args: llm.collective_rpc(f, args=args)[0]
    rpc('store_base_weights'); proc = AutoProcessor.from_pretrained(a.model_path)

    def req(r, tmpl):
        text = proc.apply_chat_template([{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': tmpl.format(q=r['question'])}]}],
                                        tokenize=False, add_generation_prompt=True)
        return {'prompt': text, 'multi_modal_data': {'image': Image.open(a.images / f"{r['imageId']}.jpg").convert('RGB')}}
    cot_req = [req(r, COT) for r in rows]; dir_req = [req(r, DIRECT) for r in yn]
    sp_cot = SamplingParams(temperature=0.0, seed=42, max_tokens=256)
    sp_dir = SamplingParams(temperature=0.0, seed=42, max_tokens=1, logprobs=20)

    def score():
        res = {}
        for r, o in zip(rows, llm.generate(cot_req, sp_cot, use_tqdm=False)):
            t = o.outputs[0].text
            res[r['id']] = {'correct': bool(handler.compute_reward(t, {'answer': r['answer']}) > 0), 'pred': handler.extract_answer(t),
                            'n_tokens': len(o.outputs[0].token_ids), 'text': t}
        for r, o in zip(yn, llm.generate(dir_req, sp_dir, use_tqdm=False)):
            lp = {int(k): float(v.logprob) for k, v in (o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}).items()}
            floor = min(lp.values()) if lp else None
            res[r['id']].update({'lp_yes': lp.get(YES), 'lp_no': lp.get(NO), 'lp_floor': floor, 'direct_text': o.outputs[0].text})
        return res
    note('loaded', n_items=len(rows), n_yesno=len(yn))
    t0 = time.time(); base = score()
    (out / f'base_{a.worker}.json.gz').write_bytes(gzip.compress(json.dumps(base).encode()))
    note('base_scored', correct_sel=sum(base[r['id']]['correct'] for r in items['selection']),
         correct_ho=sum(base[r['id']]['correct'] for r in items['heldout']), seconds=round(time.time() - t0, 1))
    per = None
    for c in cands:
        claim = out / 'claims' / f"{c['index']}"
        try:
            fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, a.worker.encode()); os.close(fd)
        except FileExistsError:
            continue
        if per and spent() + per * 1.15 / 3600 * a.usd_per_hour + 0.30 > a.cap_usd:
            note('STOP_BUDGET', index=c['index']); os.remove(claim); break
        t0 = time.time()
        rpc('apply_perturbation', c['seed'], c['sigma'])
        vis = rpc(_vision_untouched)
        res = score()
        rpc('reset_to_base_weights'); rb = rpc(_restored_exact)
        (out / 'cands' / f"cand_{c['index']}.json.gz").write_bytes(gzip.compress(json.dumps({'candidate': c, 'worker': a.worker, 'res': res,
                                                                                          'vision_changed': vis, 'restore_mismatch': rb}).encode()))
        per = time.time() - t0
        note('cand', index=c['index'], sigma=c['sigma'], correct_sel=sum(res[r['id']]['correct'] for r in items['selection']),
             correct_ho=sum(res[r['id']]['correct'] for r in items['heldout']), vision_changed=vis, restore_mismatch=rb, seconds=round(per, 1))
        if vis or rb:
            note('STOP_FIDELITY', index=c['index']); raise SystemExit('fidelity')
    note('DONE')


if __name__ == '__main__':
    main()

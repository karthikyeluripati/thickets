"""M1 matched-split transfer (results/paper-analysis/m1/plan_lock.md): BASE + 61 frozen candidates on the fresh
compass-format OmniSpatial-train hold-out (gpu/holdout.jsonl). Same engine/prompt/perturbation as the original study
(vLLM 0.11 in-process, pinned RandOpt worker, fingerprint per candidate, exact base restore). Records the first generated
token (max_tokens=1) and top-20 first-token A-D log-probs. Saves after every candidate; in-script budget guard.
"""
import argparse
import gzip
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, 'src')
import causal_diag_9504111 as cd  # noqa: E402

BASE_ID = '5bc2499cb980429deee42ccabf91358e85a699f25b67ab5f7bdbf196457c8485'


def _restored_exact(worker):
    import torch
    return sum(int(not torch.equal(p.data, worker._base_weights[n].to(p.device))) for n, p in worker.model_runner.model.named_parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', type=Path, required=True); ap.add_argument('--upstream', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--usd-per-hour', type=float, required=True)
    ap.add_argument('--pod-start-epoch', type=float, required=True); ap.add_argument('--cap-usd', type=float, required=True)
    ap.add_argument('--candidates', default='results/paper-analysis/m1/frozen_candidates.json')
    ap.add_argument('--holdout', default='results/paper-analysis/m1/gpu/holdout.jsonl')
    a = ap.parse_args()
    sys.path.insert(0, str(a.upstream))
    os.environ.update({'VLLM_ENABLE_V1_MULTIPROCESSING': '0', 'PERTURB_VISUAL': '1', 'OMP_NUM_THREADS': '1'})
    from huggingface_hub import snapshot_download
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from thicket_runtime import fast_state
    a.out.mkdir(parents=True, exist_ok=True)
    spent = lambda: (time.time() - a.pod_start_epoch) / 3600 * a.usd_per_hour
    log = {'steps': []}

    def note(step, **kw):
        kw.update(step=step, spent_usd=round(spent(), 3), t=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        log['steps'].append(kw); (a.out / 'cost_log.json').write_text(json.dumps(log, indent=1)); print(json.dumps(kw)[:400], flush=True)
    cands = json.loads(Path(a.candidates).read_text())['candidates']
    rows = [json.loads(l) for l in Path(a.holdout).read_text(encoding='utf-8').splitlines() if l.strip()]
    spec = importlib.util.spec_from_file_location('sp', 'third_party/omnispatial/system_prompts.py'); sp = importlib.util.module_from_spec(spec); spec.loader.exec_module(sp)
    path = snapshot_download(cd.MODEL, revision=cd.REV, allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja'])
    llm = LLM(model=path, tokenizer=path, dtype='bfloat16', tensor_parallel_size=1, distributed_executor_backend='uni',
              worker_extension_cls='utils.worker_extn.WorkerExtension', enforce_eager=True, enable_prefix_caching=False,
              gpu_memory_utilization=.5, max_model_len=16384, max_num_seqs=64, max_num_batched_tokens=16384, max_logprobs=20,
              limit_mm_per_prompt={'image': 1, 'video': 0}, mm_processor_cache_gb=0, seed=0, disable_log_stats=True)
    rpc = lambda f, *args: llm.collective_rpc(f, args=args)[0]
    rpc('store_base_weights'); proc = AutoProcessor.from_pretrained(path)

    def prompt(r):
        t = sp.SYS_PROMPTS['none'] + '\n' + sp.FORMAT_PROMPTS['direct'] + '\n\n' + r['question']
        for i, o in enumerate(r['options']):
            t += f'\n{cd.LETTERS[i]}. {o}'
        return proc.apply_chat_template([{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': t}]}], tokenize=False, add_generation_prompt=True)
    req = [{'prompt': prompt(r), 'multi_modal_data': {'image': Image.open(a.images / r['source_split'] / r['image_member'].rsplit('/', 1)[1]).convert('RGB')}} for r in rows]
    spp = SamplingParams(temperature=0, seed=0, max_tokens=1, logprobs=20)

    def run():
        out, miss = [], 0
        for r, o in zip(rows, llm.generate(req, spp, use_tqdm=False)):
            g = o.outputs[0]
            lp0 = {int(k): float(v.logprob) for k, v in (g.logprobs[0] if g.logprobs else {}).items()}
            sc, bound = cd.letter_scores(lp0); miss += sum(v is None for v in sc.values())
            first = g.text.strip()[:1].upper()
            out.append({'uid': r['uid'], 'text': g.text, 'parsed': first if first in cd.LETTERS else '', 'letter_logprobs': sc, 'top20_floor': bound})
        return out, miss
    note('model_loaded', n_items=len(rows))
    if rpc(fast_state.fingerprint) != BASE_ID: raise SystemExit('base fingerprint mismatch')
    t0 = time.time(); base, miss = run(); note('base_scored', missing=miss, seconds=round(time.time() - t0, 1),
                                              correct=sum(x['parsed'] == cd.LETTERS[r['answer']] for x, r in zip(base, rows)))
    (a.out / 'base.json.gz').write_bytes(gzip.compress(json.dumps(base).encode()))
    per = None
    for i, c in enumerate(cands):
        if per and spent() + per * 1.15 / 3600 * a.usd_per_hour + 0.10 > a.cap_usd:
            note('STOP_BUDGET_before', index=i); break
        t0 = time.time()
        rpc('apply_perturbation', c['seed'], c['sigma'])
        fp = rpc(fast_state.fingerprint)
        if fp != c['expected_state_id']:
            note('FINGERPRINT_MISMATCH', index=i, seed=c['seed']); rpc('reset_to_base_weights'); raise SystemExit('fingerprint mismatch')
        out, miss = run()
        rpc('reset_to_base_weights'); rb = rpc(_restored_exact)
        (a.out / f"cand_{c['seed']}.json.gz").write_bytes(gzip.compress(json.dumps({'candidate': c, 'HOLDOUT': out, 'missing': miss, 'restore_mismatch': rb}).encode()))
        per = time.time() - t0
        note(f'cand_{i}', seed=c['seed'], role=c['role'], correct=sum(x['parsed'] == cd.LETTERS[r['answer']] for x, r in zip(out, rows)),
             missing=miss, restore_mismatch=rb, seconds=round(per, 1))
        if rb: note('STOP_FIDELITY', index=i); raise SystemExit('restore mismatch')
    note('DONE')


if __name__ == '__main__':
    main()

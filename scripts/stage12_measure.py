"""Stage 1 measurement (results/paper-analysis/stage12/plan_lock.md): answer-content tilt of each frozen perturbation,
FULL and NOV (vision reset to BASE), on the 96 frozen probe items. Engine and fidelity checks as random_control_transfer.py.
Saves after every perturbation; in-script budget guard; stops on any fidelity failure.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, 'src')
import causal_diag_9504111 as cd  # noqa: E402
from stage12_gradient import is_front  # noqa: E402

BASE_ID = '5bc2499cb980429deee42ccabf91358e85a699f25b67ab5f7bdbf196457c8485'


def _restored_exact(worker):
    import torch
    return sum(int(not torch.equal(p.data, worker._base_weights[n].to(p.device))) for n, p in worker.model_runner.model.named_parameters())


def _reset_vision(worker):
    """Copy BASE into every vision tensor; return (#vision tensors, #vision mismatches vs BASE, #non-vision equal to BASE)."""
    import torch
    nv = bad = same_nonv = 0
    for n, p in worker.model_runner.model.named_parameters():
        if cd.assign_group(n) == 'vision':
            p.data.copy_(worker._base_weights[n].to(p.device)); nv += 1
    torch.cuda.synchronize()
    for n, p in worker.model_runner.model.named_parameters():
        eq = torch.equal(p.data, worker._base_weights[n].to(p.device))
        if cd.assign_group(n) == 'vision':
            bad += int(not eq)
        else:
            same_nonv += int(eq)
    return nv, bad, same_nonv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', type=Path, required=True); ap.add_argument('--upstream', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--usd-per-hour', type=float, required=True)
    ap.add_argument('--pod-start-epoch', type=float, required=True); ap.add_argument('--cap-usd', type=float, required=True)
    ap.add_argument('--inputs', type=Path, default=Path('results/paper-analysis/stage12/frozen_inputs.json'))
    a = ap.parse_args()
    sys.path.insert(0, str(a.upstream))
    os.environ.update({'VLLM_ENABLE_V1_MULTIPROCESSING': '0', 'PERTURB_VISUAL': '1', 'OMP_NUM_THREADS': '1'})
    import importlib.util
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
    fin = json.loads(a.inputs.read_text()); probe = fin['probe_uids']
    ex = {}
    for s in ('search', 'validation'):
        for l in (cd.DATA / f'{s}.jsonl').read_text(encoding='utf-8').splitlines():
            r = json.loads(l); ex[r['uid']] = r
    rows = [ex[u] for u in probe]
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

    def score():
        out, miss = [], 0
        for r, o in zip(rows, llm.generate(req, spp, use_tqdm=False)):
            g = o.outputs[0]
            lp0 = {int(k): float(v.logprob) for k, v in (g.logprobs[0] if g.logprobs else {}).items()}
            sc, bound = cd.letter_scores(lp0)
            m = sum(v is None for v in sc.values()); miss += m
            z = [sc[L] if sc[L] is not None else bound for L in cd.LETTERS]
            zc = [x - sum(z) / 4 for x in z]; F = [is_front(o_) for o_ in r['options']]
            B = sum(x for x, f in zip(zc, F) if f) / sum(F) - sum(x for x, f in zip(zc, F) if not f) / (4 - sum(F))
            out.append({'uid': r['uid'], 'z': z, 'B': B, 'parsed': g.text.strip()[:1].upper(), 'missing': m})
        return out, miss
    note('model_loaded')
    fb = rpc(fast_state.fingerprint); note('base_fingerprint', ok=fb == BASE_ID)
    if fb != BASE_ID: raise SystemExit('base fingerprint mismatch')
    t0 = time.time(); base, miss = score(); note('base_scored', missing=miss, seconds=round(time.time() - t0, 1))
    res = {'base': base, 'perturbations': []}
    per = None
    for i, p in enumerate(fin['perturbations']):
        if per and spent() + per * 1.15 / 3600 * a.usd_per_hour + 0.15 > a.cap_usd:
            note('STOP_BUDGET_before', index=i); break
        t0 = time.time()
        rpc('apply_perturbation', p['seed'], p['sigma'])
        fp = rpc(fast_state.fingerprint)
        if fp != p['expected_state_id']:
            note('FINGERPRINT_MISMATCH', index=i, seed=p['seed']); rpc('reset_to_base_weights'); raise SystemExit('fingerprint mismatch')
        full, m1 = score()
        nv, vbad, nonv_same = rpc(_reset_vision)
        nov, m2 = score()
        rpc('reset_to_base_weights'); rb = rpc(_restored_exact)
        rec = {**p, 'FULL': full, 'NOV': nov, 'missing': m1 + m2, 'vision_tensors_reset': nv, 'vision_reset_mismatch': vbad,
               'nonvision_equal_base_after_reset': nonv_same, 'restore_mismatch': rb}
        res['perturbations'].append(rec)
        (a.out / 'measurements.json').write_text(json.dumps(res))
        per = time.time() - t0
        T = lambda xs: sum(x['B'] - b['B'] for x, b in zip(xs, base)) / len(base)
        note(f'pert_{i}', seed=p['seed'], role=p['role'], T_FULL=round(T(full), 4), T_NOV=round(T(nov), 4), missing=m1 + m2, vision_reset_mismatch=vbad,
             restore_mismatch=rb, seconds=round(per, 1))
        if vbad or rb:
            note('STOP_FIDELITY', index=i); raise SystemExit('fidelity')
    note('DONE', n=len(res['perturbations']))


if __name__ == '__main__':
    main()

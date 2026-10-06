"""R2 job 01 (Qwen2.5-VL-7B): stream-structure check + vLLM/HF tensor inventories (adapted from geometry_vllm_check.py).

For each seed: apply the pinned upstream apply_perturbation, then for EVERY vLLM tensor compare the realized weights
with bf16(base + bf16(sigma * eps[:numel].view(shape))) where eps = randn(N_MAX, bf16, cuda, Generator(seed)).
Records per-tensor SHA-256 of the realized 9504111 tensors (raw bytes) for the HF layout check, then resets.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
import geometry_lib as G  # noqa: E402

MODEL = 'Qwen/Qwen2.5-VL-7B-Instruct'


def _check(worker, seed, sigma, record_sha):
    import torch
    g = torch.Generator(device='cuda'); g.manual_seed(int(seed))
    nmax = max(p.numel() for p in worker.model_runner.model.parameters())
    eps = torch.randn(nmax, dtype=torch.bfloat16, device='cuda', generator=g)
    res = {'tensors': 0, 'exact': 0, 'mismatch_names': [], 'sha': {}}
    for n, p in worker.model_runner.model.named_parameters():
        base = worker._base_weights[n].to(p.device)
        expected = base + float(sigma) * eps[: p.numel()].view(p.shape)
        res['tensors'] += 1
        ok = torch.equal(p.data, expected)
        res['exact'] += int(ok)
        if not ok:
            res['mismatch_names'].append(n)
        if record_sha:
            res['sha'][n] = hashlib.sha256(p.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--upstream', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seeds', default='9504111:0.002,9500059:0.002')
    a = ap.parse_args()
    sys.path.insert(0, str(a.upstream)); sys.path.insert(0, 'src')
    os.environ.update({'VLLM_ENABLE_V1_MULTIPROCESSING': '0', 'PERTURB_VISUAL': '1', 'OMP_NUM_THREADS': '1'})
    from huggingface_hub import snapshot_download
    from vllm import LLM
    path = snapshot_download(MODEL, allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja'])
    llm = LLM(model=path, tokenizer=path, dtype='bfloat16', tensor_parallel_size=1, distributed_executor_backend='uni',
              worker_extension_cls='utils.worker_extn.WorkerExtension', enforce_eager=True, enable_prefix_caching=False,
              gpu_memory_utilization=.5, max_model_len=16384, max_num_seqs=64, max_num_batched_tokens=16384,
              limit_mm_per_prompt={'image': 1, 'video': 0}, mm_processor_cache_gb=0, seed=0, disable_log_stats=True)
    rpc = lambda f, *args: llm.collective_rpc(f, args=args)[0]
    rpc('store_base_weights')
    out = {'seeds': {}}
    for spec in a.seeds.split(','):
        seed, sigma = spec.split(':'); seed, sigma = int(seed), float(sigma)
        rpc('apply_perturbation', seed, sigma)
        out['seeds'][str(seed)] = {**rpc(_check, seed, sigma, seed == 9504111)}
        rpc('reset_to_base_weights')
        print(json.dumps({k: v for k, v in out['seeds'][str(seed)].items() if k != 'sha'})[:400], flush=True)
    out['names_shapes'] = rpc(lambda w: {n: list(p.shape) for n, p in w.model_runner.model.named_parameters()})
    out['dtypes'] = rpc(lambda w: sorted({str(p.dtype) for p in w.model_runner.model.parameters()}))
    del llm
    import gc, torch; gc.collect(); torch.cuda.empty_cache()
    from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration
    from accelerate import init_empty_weights
    with init_empty_weights():
        hf = Qwen2_5_VLForConditionalGeneration._from_config(AutoConfig.from_pretrained(path))
    out['hf_shapes'] = {n: list(p.shape) for n, p in hf.named_parameters()}
    out['hf_tied'] = bool(getattr(hf.config, 'tie_word_embeddings', False))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'vllm_stream_check.json').write_text(json.dumps(out))
    print('VLLM_CHECK_DONE', flush=True)


if __name__ == '__main__':
    main()

"""Worker-side state controls that avoid one host sync per 1M values.

drift(): bitwise equality per tensor on GPU (torch.equal on byte views) plus a
finiteness check. When a tensor is bitwise equal and finite the result is
identical to vllm_audit.drift for that tensor (0 changed values, max_abs 0.0,
finite). Any unequal or non-finite tensor falls back to the original
vllm_audit.tensor_difference, so non-exact states report identical detail.

fingerprint(): 'state-sha256-tree-v1' = SHA256 over canonical
[kind:name, shape, dtype, SHA256(tensor bytes)] for every parameter and buffer
in named order, with per-tensor hashing in a thread pool (hashlib releases the
GIL). It is a deterministic function of the exact bytes, as collision-resistant
as the original flat SHA256, and is used for every phase of this study.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib

import torch

from .candidate import canonical_json
from .vllm_audit import tensor_difference


def _bytes_equal(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return torch.equal(a.detach().contiguous().view(torch.uint8), b.detach().contiguous().view(torch.uint8))


@torch.no_grad()
def drift(worker):
    model = worker.model_runner.model
    changed, count, maximum, finite, changed_buffers, fallbacks = 0, 0, 0.0, True, 0, 0
    for name, p in model.named_parameters():
        base = worker._base_weights[name]
        count += p.numel()
        if _bytes_equal(p, base) and bool(torch.isfinite(p).all()):
            continue
        fallbacks += 1
        d = tensor_difference(p, base)
        changed += d['changed_values']
        finite = finite and d['finite']
        if d['max_abs'] is not None:
            maximum = max(maximum, d['max_abs'])
    for name, b in model.named_buffers():
        base = worker._thicket_base_buffers[name]
        if _bytes_equal(b, base) and (not b.is_floating_point() or bool(torch.isfinite(b).all())):
            continue
        fallbacks += 1
        d = tensor_difference(b, base)
        changed_buffers += d['changed_values']
        finite = finite and d['finite']
    return {'changed_values': changed, 'parameter_values': count, 'changed_fraction': changed / max(count, 1),
            'changed_buffer_values': changed_buffers, 'max_abs': maximum if finite else None, 'finite': finite,
            'exact_base': finite and changed == 0 and changed_buffers == 0, 'slow_path_tensors': fallbacks}


def _digest(item):
    name, t = item
    t = t.detach().cpu().contiguous()
    return [name, list(t.shape), str(t.dtype), hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()]


@torch.no_grad()
def fingerprint(worker, threads=16):
    model = worker.model_runner.model
    entries = [('parameter:' + n, t) for n, t in model.named_parameters()]
    entries += [('buffer:' + n, t) for n, t in model.named_buffers()]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    with ThreadPoolExecutor(threads) as pool:
        leaves = list(pool.map(_digest, entries))
    return hashlib.sha256(canonical_json(['state-sha256-tree-v1', leaves]).encode()).hexdigest()

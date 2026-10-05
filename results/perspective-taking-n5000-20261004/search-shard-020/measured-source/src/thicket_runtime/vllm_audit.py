"""Read-only audit/memory callables for vLLM collective_rpc.

Weight operations come from the unchanged, hash-verified upstream extension.
These callables do not replace its perturb, restore, or snapshot methods.
"""
import hashlib
import torch

from .benchmark import environment
from .candidate import canonical_json
from .execution import state_hash


def initialize(worker):
    model = worker.model_runner.model
    if not hasattr(worker, "_base_weights"):
        raise RuntimeError("upstream base snapshot must exist before audit setup")
    worker._thicket_base_buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
    layout = [(n, list(p.shape), str(p.dtype)) for n, p in model.named_parameters()]
    return {"base_id": state_hash(model), "parameter_layout": layout,
            "layout_sha256": hashlib.sha256(canonical_json(layout).encode()).hexdigest(),
            "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
            "snapshot_parameter_bytes": sum(p.numel() * p.element_size() for p in worker._base_weights.values()),
            "environment": environment(torch.device("cuda", torch.cuda.current_device()))}


@torch.no_grad()
def noise_layout_probe(worker, seed=42):
    """Show why unchanged upstream RNG on packed tensors is not an HF candidate.

    Qwen2-only, TP=1: compare one packed QKV and gate/up tensor to independent
    tensor-local draws with the corresponding HF logical shapes. Audit-only.
    """
    model = worker.model_runner.model
    config = model.config
    head_dim = config.hidden_size // config.num_attention_heads
    parts = {"qkv_proj.weight": [config.num_attention_heads * head_dim,
                                 config.num_key_value_heads * head_dim,
                                 config.num_key_value_heads * head_dim],
             "gate_up_proj.weight": [config.intermediate_size, config.intermediate_size]}
    result = []
    for suffix, sizes in parts.items():
        name, p = next((n, p) for n, p in model.named_parameters()
                       if n.endswith(suffix))
        if p.shape[0] != sum(sizes) or p.ndim != 2:
            raise ValueError("unexpected packed Qwen layout; refusing a misleading comparison")
        def draw(shape):
            gen = torch.Generator(device=p.device).manual_seed(seed)
            return torch.randn(shape, dtype=p.dtype, device=p.device, generator=gen)
        native = draw(p.shape)
        logical = torch.cat([draw((size, p.shape[1])) for size in sizes])
        changed = int((native != logical).sum().item())
        result.append({"native_parameter": name, "native_shape": list(p.shape),
                       "hf_logical_row_counts": sizes, "seed": seed,
                       "noise_values": p.numel(), "different_noise_values": changed,
                       "noise_exact": changed == 0})
    return {"contract": "randopt-per-tensor-v1", "probes": result,
            "same_seed_is_same_cross_backend_perturbation": all(p["noise_exact"] for p in result)}


def fingerprint(worker):
    return state_hash(worker.model_runner.model)


def memory(worker, reset=False):
    torch.cuda.synchronize()
    if reset:
        torch.cuda.reset_peak_memory_stats()
    free, total = torch.cuda.mem_get_info()
    return {"allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "device_used_bytes_at_boundary": total - free,
            "device_total_bytes": total}


def tensor_difference(current, reference):
    """Bitwise equality includes signed zeros; scratch is bounded to 1M values."""
    if current.shape != reference.shape or current.dtype != reference.dtype:
        return {"changed_values": max(current.numel(), reference.numel(), 1),
                "max_abs": None, "finite": False}
    changed, maximum, finite = 0, 0.0, True
    current, reference = current.detach().reshape(-1), reference.reshape(-1)
    for offset in range(0, current.numel(), 1_000_000):
        cur, ref = current[offset:offset + 1_000_000], reference[offset:offset + 1_000_000]
        a = cur.contiguous().view(torch.uint8).reshape(-1, cur.element_size())
        b = ref.contiguous().view(torch.uint8).reshape(-1, ref.element_size())
        changed += int((a != b).any(dim=1).sum().item())
        diff = (cur.double() - ref.double()).abs()
        ok = bool(torch.isfinite(diff).all().item())
        finite = finite and ok
        if ok and diff.numel():
            maximum = max(maximum, float(diff.max().item()))
    return {"changed_values": changed, "max_abs": maximum if finite else None, "finite": finite}


@torch.no_grad()
def drift(worker):
    changed, count, maximum, finite, changed_buffers = 0, 0, 0.0, True, 0
    for name, p in worker.model_runner.model.named_parameters():
        d = tensor_difference(p, worker._base_weights[name])
        changed += d["changed_values"]
        count += p.numel()
        finite = finite and d["finite"]
        if d["max_abs"] is not None:
            maximum = max(maximum, d["max_abs"])
    for name, b in worker.model_runner.model.named_buffers():
        d = tensor_difference(b, worker._thicket_base_buffers[name])
        changed_buffers += d["changed_values"]
        finite = finite and d["finite"]
    return {"changed_values": changed, "parameter_values": count,
            "changed_fraction": changed / max(count, 1),
            "changed_buffer_values": changed_buffers,
            "max_abs": maximum if finite else None, "finite": finite,
            "exact_base": finite and changed == 0 and changed_buffers == 0}

"""Memory-scalable alternative to upstream's GPU-resident `store_base_weights()`/
`reset_to_base_weights()` (external/RandOpt/utils/worker_extn.py, vendored/gitignored, NEVER
edited by this project -- see run_global_visual_thicket_pilot.py's own docstrings). Follows the
SAME Callable-dispatch convention thicket/worker_rpc.py already established for extending worker
capability without touching `external/RandOpt` or `worker_extension_cls`: every function below
takes `worker_self` (the real vLLM worker, mixed with upstream's WorkerExtension) as its first
argument and is a plain importable Callable, dispatched via `engine.collective_rpc.remote(fn,
args=...)`, fully unit-testable against a duck-typed fake or a small real `torch.nn.Module` --
no GPU/ray/external-RandOpt import anywhere in this module.

WHY THIS EXISTS (Stage-11 32B readiness): upstream's `store_base_weights()` does
`self._base_weights[name] = p.data.clone()` -- `.clone()` preserves the source tensor's device,
so on a GPU worker this clones EVERY parameter A SECOND TIME on the SAME GPU, doubling
GPU-resident model-weight storage for the lifetime of the run. At 32B (~65 GiB of BF16 weights,
see stage11_32b_readiness.py's parameter-count estimate) this is infeasible on a single L40S
(~44.39 GiB usable) even before any vLLM/KV-cache overhead. The scientific fixed-base
requirement -- every candidate begins from and restores to EXACT theta_0, restoration verified
bit-exactly -- does NOT require theta_0 to live on GPU a second time; it only requires an
UNCHANGED, addressable, exact-precision (BF16, never re-serialized/re-quantized) copy that can
be copied back parameter-by-parameter. Storing it on CPU (optionally pinned, for faster
CPU->GPU copies) satisfies the same invariant with GPU memory overhead bounded by the single
LARGEST parameter tensor's size during restoration, never 2x the whole model.

`_base_weights_cpu` is a NEW attribute name (distinct from upstream's own `_base_weights`) so
both mechanisms can coexist on the same worker without collision -- exactly what the Section-5
equivalence gate (tests/test_thicket_cpu_base_snapshot.py) needs to compare them side by side.

Restoration verification against a CPU-resident reference needs its OWN chunked routine
(`_chunked_cross_device_drift` below), not thicket.worker_rpc's existing `measure_drift`:
PyTorch raises on a bare CPU-vs-GPU tensor subtraction (it never silently moves devices), so
comparing a GPU-resident live parameter against a CPU-resident snapshot requires explicitly
moving each bounded CHUNK (never the whole reference tensor) to the parameter's device
immediately before the diff -- this was caught by reading `chunked_abs_stats`'s implementation
before writing any code against it, not discovered by a failing test.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import torch

BASE_SNAPSHOT_MODE_LEGACY_GPU = "store_base_weights"  # upstream's own, unchanged -- see module docstring
BASE_SNAPSHOT_MODE_CPU = "cpu_base_weights"


def store_base_weights_cpu_rpc(worker_self, *, pin_memory: bool = True) -> Dict:
    """Clones every named parameter to CPU, in the model's NATIVE dtype (no cast, no lossy
    serialization -- `tensor.to("cpu")` on a BF16 source tensor yields a BF16 CPU tensor with
    bit-identical values), and stores it under `worker_self._base_weights_cpu`. `pin_memory`
    defaults to True (faster async CPU->GPU copies during restoration) but falls back silently
    to regular (unpinned) CPU memory if pinned allocation fails (e.g. an exhausted pinned pool)
    -- pinning is a performance optimization, never a correctness requirement, so its failure is
    never fatal.
    """
    base_weights_cpu: Dict[str, torch.Tensor] = {}
    total_bytes = 0
    for name, p in worker_self.model_runner.model.named_parameters():
        cpu_tensor = p.data.detach().to("cpu", copy=True)
        if pin_memory:
            try:
                cpu_tensor = cpu_tensor.pin_memory()
            except RuntimeError:
                pass  # pinned-memory allocation failure -- keep the unpinned (still exact) copy
        base_weights_cpu[name] = cpu_tensor
        total_bytes += cpu_tensor.numel() * cpu_tensor.element_size()
    worker_self._base_weights_cpu = base_weights_cpu
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return {"n_parameters": len(base_weights_cpu), "total_bytes": total_bytes}


def reset_to_base_weights_cpu_rpc(worker_self) -> Dict:
    """Restores every parameter from `worker_self._base_weights_cpu`, ONE TENSOR AT A TIME --
    transient GPU memory for this operation is bounded by the single largest parameter tensor's
    size (the same bound upstream's own per-parameter-loop `reset_to_base_weights` already has),
    never 2x the full model. `non_blocking=True` on the CPU->GPU copy is safe here because the
    subsequent `torch.cuda.synchronize()` (and every later use of `p.data`) waits for it to land
    before any dependent computation reads the tensor.
    """
    if not hasattr(worker_self, "_base_weights_cpu"):
        raise RuntimeError("reset_to_base_weights_cpu_rpc: store_base_weights_cpu_rpc() must be called first -- no CPU base snapshot on this worker.")
    for name, p in worker_self.model_runner.model.named_parameters():
        cpu_tensor = worker_self._base_weights_cpu[name]
        p.data.copy_(cpu_tensor.to(device=p.device, non_blocking=True))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return {"ok": True}


def clear_base_weights_cpu_rpc(worker_self) -> Dict:
    """Frees the CPU base snapshot -- symmetric with upstream's own `clear_base_weights`."""
    had_snapshot = hasattr(worker_self, "_base_weights_cpu")
    if had_snapshot:
        del worker_self._base_weights_cpu
    return {"freed": had_snapshot}


def _chunked_cross_device_drift(model: torch.nn.Module, cpu_reference: Dict[str, torch.Tensor], param_filter: Optional[Callable[[str], bool]], chunk_elements: int) -> Dict:
    """Cross-device analog of diagnostics/perturb_restore_drift.py's `measure_drift`.

    `measure_drift`'s own chunked reductions (thicket.memory_bounded_ops.chunked_abs_stats /
    chunked_count_differing) operate on two SAME-DEVICE tensors -- PyTorch raises on a bare
    CPU-vs-GPU tensor subtraction, it does not silently move devices. Reusing `measure_drift`
    unmodified against a CPU-resident reference is therefore NOT correct (this was caught before
    any test was written, not discovered by one failing). This function reimplements the exact
    same per-chunk arithmetic, adding only an explicit `.to(p.device)` on each CPU-resident
    CHUNK (never the whole reference tensor) immediately before the comparison -- transient GPU
    memory for that one chunk is bounded by `chunk_elements`, exactly preserving the "no
    whole-tensor giant temporaries" requirement.
    """
    max_abs = 0.0
    n_differing = 0
    n_total = 0
    for name, p in model.named_parameters():
        if param_filter is not None and not param_filter(name):
            continue
        ref = cpu_reference[name]
        flat_p, flat_ref = p.detach().reshape(-1), ref.reshape(-1)
        n = flat_p.numel()
        n_total += n
        for start in range(0, n, chunk_elements):
            end = start + chunk_elements
            p_chunk = flat_p[start:end]
            ref_chunk = flat_ref[start:end].to(device=p_chunk.device)
            if p_chunk.numel() == 0:
                continue
            diff = (p_chunk.double() - ref_chunk.double()).abs()
            max_abs = max(max_abs, diff.max().item())
            n_differing += int((p_chunk != ref_chunk).sum().item())
    return {
        "max_abs_drift": max_abs, "n_differing": n_differing, "n_total": n_total,
        "fraction_elements_differing": (n_differing / n_total) if n_total else 0.0,
    }


def verify_exact_fixed_base_restoration_cpu_rpc(worker_self, *, chunk_elements: int = 4_194_304) -> Dict:
    """CPU-snapshot analog of thicket.worker_rpc.verify_exact_fixed_base_restoration_rpc --
    same exact-equality invariant (zero differing elements after reset_to_base_weights_cpu_rpc),
    computed via `_chunked_cross_device_drift` (above) since the reference now lives on CPU
    while the live parameters live on GPU.
    """
    if not hasattr(worker_self, "_base_weights_cpu"):
        raise RuntimeError(
            "verify_exact_fixed_base_restoration_cpu_rpc: worker_self has no _base_weights_cpu -- "
            "store_base_weights_cpu_rpc() must be called exactly once before any fixed-base "
            "perturb/reset/verify cycle."
        )
    drift = _chunked_cross_device_drift(worker_self.model_runner.model, worker_self._base_weights_cpu, worker_self._should_perturb, chunk_elements)
    return {"ok": drift["max_abs_drift"] == 0.0 and drift["n_differing"] == 0, "max_abs_drift": drift["max_abs_drift"], "fraction_elements_differing": drift["fraction_elements_differing"]}


# =================================================================================================
# Section 5 (task spec): legacy-GPU-clone vs CPU-snapshot equivalence classification
# =================================================================================================

EQUIVALENCE_BIT_EXACT = "bit_decision_equivalent"          # (A) required before 32B may use cpu_base_weights
EQUIVALENCE_SCIENTIFICALLY_EQUIVALENT = "scientifically_equivalent_not_bit_equivalent"  # (B)
EQUIVALENCE_SEMANTICS_CHANGED = "semantics_changed"          # (C) -- STOP


def classify_snapshot_equivalence(
    initial_snapshots_equal: bool, perturbed_weights_equal: bool, restored_weights_equal: bool, n_differing_after_restore: int,
) -> str:
    """Deterministic classification (A/B/C) from four already-computed boolean/count facts --
    never a judgment call made inline at a call site. `n_differing_after_restore` is the element
    count that differs between the two paths' final restored weights (0 required for class A).
    """
    if initial_snapshots_equal and perturbed_weights_equal and restored_weights_equal and n_differing_after_restore == 0:
        return EQUIVALENCE_BIT_EXACT
    if perturbed_weights_equal is False and initial_snapshots_equal:
        # Same starting point, same seed/scale inputs, yet a different perturbed result implies
        # the storage location itself altered arithmetic (e.g. a silent dtype/precision change)
        # -- this is a semantics change, not mere floating-point path noise, because both paths
        # apply the IDENTICAL add() to an IDENTICAL starting tensor with an IDENTICAL delta.
        return EQUIVALENCE_SEMANTICS_CHANGED
    if restored_weights_equal and n_differing_after_restore == 0:
        return EQUIVALENCE_SCIENTIFICALLY_EQUIVALENT
    return EQUIVALENCE_SEMANTICS_CHANGED


def ensure_bit_exact_before_32b(equivalence_class: str) -> None:
    """32B may use cpu_base_weights ONLY if classify_snapshot_equivalence() returned class A.
    Hard stop otherwise -- see task spec Section 5 ("We require A before using this for 32B. If
    not A: STOP.").
    """
    if equivalence_class != EQUIVALENCE_BIT_EXACT:
        raise RuntimeError(
            f"CPU base-snapshot equivalence class is {equivalence_class!r}, not {EQUIVALENCE_BIT_EXACT!r} -- "
            f"refusing to use cpu_base_weights for 32B until bit/decision equivalence is proven."
        )

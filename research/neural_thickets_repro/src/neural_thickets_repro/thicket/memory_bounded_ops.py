"""Bounded-memory tensor reduction primitives for GPU worker-side perturbation/measurement code.

ROOT CAUSE (live Stage-11 7B whole_model --smoke run, real pod evidence): apply_anatomical_
relative_l2 (thicket/perturbation.py) and measure_drift (diagnostics/perturb_restore_drift.py)
both cast an INDIVIDUAL parameter tensor to float32 IN FULL (`p.detach().float()`) before
reducing (`.pow(2).sum()`, `.abs().max()`, etc). At 3B scale every single parameter tensor is
small enough for this to be a negligible temporary. At 7B scale, Qwen2.5-VL-7B-Instruct's
vocabulary-sized projection (~151,936 x 3584 = 544,538,624 elements) cast to float32 needs
~2.03 GiB -- exactly the allocation the live smoke crashed on, against a GPU already left with
well under 1 GiB free (bf16 model weights + vLLM's own reserved KV-cache pool + the base-weights
snapshot together already consume ~97% of the 44.39 GiB device at gpu_memory_utilization=0.60).
This module replaces every such full-tensor cast with a fixed-size CHUNKED reduction: only one
small chunk is ever materialized in a higher-precision dtype at a time, regardless of the source
tensor's total size -- so peak temporary VRAM for these operations no longer scales with
parameter count at all, which is required for the unified 3B->7B->32B->72B scaling experiment
(a 72B model's largest single tensor is bigger still).

These are the SAME mathematical quantities as before (sum_j x_j^2, sum_j |x_j|, max_j |x_j|,
elementwise-difference-then-square-sum across two tensors) -- never a different reduction and
never a different summation ORDER claim beyond ordinary floating-point non-associativity (the
same caveat that already applies to any GPU reduction split across kernel launches). Per the
Stage-11 OOM-fix task spec's own guidance ("accumulate ... in FP64 or existing scientifically
appropriate accumulator"), each chunk is upcast to float64 -- STRICTLY HIGHER precision than the
legacy single-shot float32 reduction it replaces, never lower -- so this implementation is more
numerically faithful to the true mathematical sum, not less. The resulting (tiny, expected)
discrepancy against the legacy float32 path is quantified in tests/test_memory_bounded_ops.py and
tests/test_thicket_perturbation.py, and shown to be many orders of magnitude below the frozen
radius-acceptance tolerances (RADIUS_REALIZATION_TOLERANCE / QUANTIZATION_PLATEAU_RELATIVE_
TOLERANCE in scoped_anatomical_perturbation.py) and to never change any v3 acceptance decision.
"""
from __future__ import annotations

from typing import Tuple

import torch

# 4M elements/chunk -> a float64 chunk temporary is <= 32 MiB regardless of the source tensor's
# total size. A conservative FIXED bound, deliberately NOT tuned per-model/per-scale/per-region
# (the OOM-fix task spec: "the first attempt should solve the transient-memory problem rather
# than alter runtime/science") -- never changed based on which scale happens to be running.
DEFAULT_CHUNK_ELEMENTS = 4_194_304


def _flatten_view(tensor: torch.Tensor) -> torch.Tensor:
    """A VIEW (never a copy) of `tensor` as a flat 1-D tensor. Every caller in this module passes
    an nn.Parameter's own `.detach()` output or a freshly-generated same-shape tensor, both of
    which are always contiguous in this project's usage (nn.Parameter storage and freshly
    allocated torch.randn output are never non-contiguous views) -- `.reshape(-1)` therefore never
    silently falls back to a copying path here.
    """
    return tensor.reshape(-1)


def chunked_squared_l2_sum(tensor: torch.Tensor, *, chunk_elements: int = DEFAULT_CHUNK_ELEMENTS) -> float:
    """sum_j tensor_j^2, computed by upcasting only one fixed-size CHUNK at a time to float64 --
    never a full-tensor float32/float64 materialization of `tensor` itself.
    """
    flat = _flatten_view(tensor)
    n = flat.numel()
    total = 0.0
    for start in range(0, n, chunk_elements):
        chunk = flat[start:start + chunk_elements]
        total += chunk.double().pow(2).sum().item()
    return total


def chunked_squared_l2_diff_sum(a: torch.Tensor, b: torch.Tensor, *, chunk_elements: int = DEFAULT_CHUNK_ELEMENTS) -> float:
    """sum_j (a_j - b_j)^2 for two same-shape tensors, chunk-at-a-time in float64 -- never a
    full-tensor (a.float() - b.float()) difference tensor.
    """
    if a.shape != b.shape:
        raise ValueError(f"chunked_squared_l2_diff_sum requires matching shapes, got {tuple(a.shape)} and {tuple(b.shape)}")
    flat_a, flat_b = _flatten_view(a), _flatten_view(b)
    n = flat_a.numel()
    total = 0.0
    for start in range(0, n, chunk_elements):
        end = start + chunk_elements
        diff = flat_a[start:end].double() - flat_b[start:end].double()
        total += diff.pow(2).sum().item()
    return total


def chunked_abs_stats(a: torch.Tensor, b: torch.Tensor, *, chunk_elements: int = DEFAULT_CHUNK_ELEMENTS) -> Tuple[float, float]:
    """(max_j |a_j - b_j|, sum_j |a_j - b_j|) for two same-shape tensors, chunk-at-a-time -- used
    by measure_drift's max_abs_drift/mean_abs_drift, never a full-tensor abs-diff tensor.
    """
    if a.shape != b.shape:
        raise ValueError(f"chunked_abs_stats requires matching shapes, got {tuple(a.shape)} and {tuple(b.shape)}")
    flat_a, flat_b = _flatten_view(a), _flatten_view(b)
    n = flat_a.numel()
    max_abs = 0.0
    sum_abs = 0.0
    for start in range(0, n, chunk_elements):
        end = start + chunk_elements
        diff = (flat_a[start:end].double() - flat_b[start:end].double()).abs()
        if diff.numel() == 0:
            continue
        max_abs = max(max_abs, diff.max().item())
        sum_abs += diff.sum().item()
    return max_abs, sum_abs

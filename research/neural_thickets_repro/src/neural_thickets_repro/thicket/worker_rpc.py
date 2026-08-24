"""Worker-side RPC callables for the Stage-6 Global Visual-Thicket Pilot.

ROOT CAUSE this module fixes: under vLLM V1, `LLMEngine` has no `.model_executor` attribute
reachable from the frontend/driver process (confirmed: `AttributeError: 'LLMEngine' object has
no attribute 'model_executor'` on the real RunPod smoke, commit 630bc34) -- the model's real
parameters only ever exist inside the worker process. There is no alternate frontend attribute
to try; vLLM's own supported mechanism for driver -> worker access is `LLM.collective_rpc
(method, args)`, which can dispatch either a STRING naming a method already present on the
worker (e.g. upstream's own `perturb_self_weights`/`restore_self_weights`, unchanged), or an
arbitrary CALLABLE, invoked inside the worker process as `callable(worker_self, *args)`.

This project already established the Callable-dispatch half of that pattern BEFORE this stage
(`scoped_perturbation.scoped_apply_perturbation`, dispatched via
`collective_rpc(scoped_apply_perturbation, args=(...))`) -- the functions below follow the
IDENTICAL convention for the two NEW capabilities Stage 6 needs (mask-hash/inventory and
restoration verification) that upstream's `utils.worker_extn.WorkerExtension` does not provide.
Using a plain Callable, rather than a custom `worker_extension_cls` subclass, means
`external/RandOpt/core/engine.py`'s `launch_engines()` (which hardcodes
`worker_extension_cls="utils.worker_extn.WorkerExtension"`) needs ZERO changes and stays
completely unmodified -- exactly the "reuse the existing integration" instruction.

Every function here takes `worker_self` (the real vLLM worker instance, already mixed with
upstream's `WorkerExtension` by `launch_engines()`) as its first argument and is fully
unit-testable against a plain duck-typed fake exposing `.model_runner.model.named_parameters()`
and `._should_perturb(name)` -- no GPU/vllm/ray/external-RandOpt import needed anywhere in this
module.

THE RESTORATION INVARIANT (documented precisely, per Stage-6 Task 2's requirement): for every
parameter tensor `_should_perturb` selects (the SAME mask `global_gaussian_upstream`
perturbs), the CURRENT per-tensor L2 norm must equal the BASE (pre-perturbation) per-tensor L2
norm within an `atol + rtol * |base_norm|` tolerance (the same additive+relative convention
`torch.allclose` uses, chosen because a fixed absolute tolerance is not meaningful across
tensors whose norms range from O(1) to O(100+) in this model):

    for every perturbable tensor name n:
        | ||theta_current[n]||_2 - ||theta_base[n]||_2 |  <=  atol + rtol * ||theta_base[n]||_2

This is a lightweight fingerprint -- one float per tensor, never a second full-model clone (no
extra ~3B-parameter copy in GPU memory): an actually-accumulated perturbation residue changes
at least one tensor's L2 norm by an amount that would not exactly cancel across that tensor's
own independently-reseeded Gaussian noise; an exact false negative would require the
accumulated residual to be a precisely zero-norm vector, not a realistic floating-point
coincidence.
"""
from __future__ import annotations

import hashlib
from typing import Dict


def compute_perturbable_mask_info_rpc(worker_self) -> Dict:
    """The `global_gaussian_upstream` mask is exactly the set of named parameters
    `_should_perturb` (upstream, unmodified, mixed in via `worker_extension_cls`) selects --
    every parameter NOT prefixed `visual.`/`model.visual.` (or, with `PERTURB_VISUAL=1`, every
    parameter). Returns a small summary -- mask_hash, param_count, total_elements -- never the
    parameter names or tensors themselves, keeping the RPC payload tiny.
    """
    names = []
    total_elements = 0
    for name, p in worker_self.model_runner.model.named_parameters():
        if worker_self._should_perturb(name):
            names.append(name)
            total_elements += p.numel()
    canonical = "\n".join(sorted(names))
    mask_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {"mask_hash": mask_hash, "param_count": len(names), "total_elements": total_elements}


def compute_restoration_fingerprint_rpc(worker_self) -> Dict[str, float]:
    """Per-tensor L2 norm of every perturbable parameter -- see module docstring's
    restoration invariant. Deliberately NOT a state-dict clone: only one float per tensor is
    computed and returned, no second copy of the model's weights is ever created.
    """
    fingerprint: Dict[str, float] = {}
    for name, p in worker_self.model_runner.model.named_parameters():
        if worker_self._should_perturb(name):
            fingerprint[name] = float(p.detach().float().norm().item())
    return fingerprint


def verify_restoration_rpc(worker_self, base_fingerprint: Dict[str, float], atol: float, rtol: float = 1e-3) -> Dict:
    """Checks the restoration invariant (module docstring) entirely INSIDE the worker;
    returns a small diagnostic dict (ok, max_diff, worst offenders) -- never full tensors.
    Callers (evaluate_one_perturbation_rpc) MUST abort the whole experiment when `ok` is
    False, never continue to the next perturbation.
    """
    current = compute_restoration_fingerprint_rpc(worker_self)
    diffs: Dict[str, float] = {}
    failing: Dict[str, float] = {}
    for name, base_norm in base_fingerprint.items():
        current_norm = current.get(name, float("nan"))
        diff = abs(current_norm - base_norm)
        diffs[name] = diff
        threshold = atol + rtol * abs(base_norm)
        if not (diff <= threshold):  # NaN-safe: a missing/NaN current_norm always fails
            failing[name] = diff

    worst_offenders = dict(sorted(failing.items(), key=lambda kv: -kv[1])[:5])
    max_diff = max(diffs.values()) if diffs else 0.0
    return {
        "ok": len(failing) == 0, "max_diff": max_diff, "n_checked": len(diffs),
        "n_failing": len(failing), "worst_offenders": worst_offenders,
    }

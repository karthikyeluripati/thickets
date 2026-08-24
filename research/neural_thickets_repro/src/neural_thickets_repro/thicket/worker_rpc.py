"""Worker-side RPC callables for the Stage-6 Global Visual-Thicket Pilot.

ROOT CAUSE (original, commit 630bc34): under vLLM V1, `LLMEngine` has no `.model_executor`
attribute reachable from the frontend/driver process -- the model's real parameters only ever
exist inside the worker process. vLLM's own supported mechanism for driver -> worker access is
`LLM.collective_rpc(method, args)`, which can dispatch either a STRING naming a method already
present on the worker (e.g. upstream's own `perturb_self_weights`/`store_base_weights`/
`reset_to_base_weights`, unchanged), or an arbitrary CALLABLE, invoked inside the worker
process as `callable(worker_self, *args)`.

This project already established the Callable-dispatch half of that pattern BEFORE this stage
(`scoped_perturbation.scoped_apply_perturbation`, dispatched via
`collective_rpc(scoped_apply_perturbation, args=(...))`) -- the functions below follow the
IDENTICAL convention for the capabilities Stage 6 needs that upstream's `utils.worker_extn.
WorkerExtension` does not provide (mask-hash/inventory, exact restoration verification). Using
a plain Callable, rather than a custom `worker_extension_cls` subclass, means `external/
RandOpt/core/engine.py` needs ZERO changes and stays completely unmodified.

Every function here takes `worker_self` (the real vLLM worker instance, already mixed with
upstream's `WorkerExtension`) as its first argument and is fully unit-testable against a plain
duck-typed fake (or a tiny real `torch.nn.Module`, for the exact-restoration check, which needs
real tensor arithmetic) -- no GPU/ray/external-RandOpt import needed anywhere in this module.

RESTORATION-MODE HISTORY (read this before touching the restoration check below): an earlier
version of this module verified restoration via a per-tensor L2-NORM fingerprint with an
`atol + rtol*|base_norm|` tolerance, paired with `restore_self_weights`'s native-BF16
regenerate-and-subtract restoration. A real 384-candidate RunPod run proved that restoration
is NOT reliably invertible after bf16 rounding: it aborted at candidate
`5a417b7937eca5ad522e9c6b` (seed=1480723517, sigma=0.01) with a real, non-tolerance-passing
norm discrepancy of 0.0473 on `language_model.model.layers.5.self_attn.o_proj.weight`. Stage 6
now uses upstream's OWN `store_base_weights()`/`reset_to_base_weights()` (a direct GPU-resident
tensor COPY from a frozen snapshot, not add-then-subtract) for restoration -- `restore_self_
weights` is never called by Stage 6 anymore -- so restoration can and must be held to an EXACT
(not tolerance-based) standard: `verify_exact_fixed_base_restoration_rpc` below.
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


def verify_exact_fixed_base_restoration_rpc(worker_self) -> Dict:
    """THE RESTORATION INVARIANT (fixed-base, this repair pass -- replaces the earlier
    tolerance-based fingerprint check): after `reset_to_base_weights()`, every perturbable
    parameter tensor must be EXACTLY (bitwise, in the model's native dtype) equal to upstream's
    own stored `_base_weights[name]` snapshot -- zero changed tensors, zero differing elements.
    `reset_to_base_weights` performs a direct `p.data.copy_(self._base_weights[name])`, never
    an add-then-subtract, so a correct reset is exact by construction; this check exists to
    PROVE that happened, not to tolerate any amount of drift.

    Reuses the SAME exact-equality invariant already established and GPU-validated by
    `diagnostics/scope_isolation_gpu_check.py`'s Test A-G ("reset_exact":
    `max_abs_drift == 0.0` after `reset_to_base_weights`), via `diagnostics/perturb_restore_
    drift.py`'s already-unit-tested `measure_drift` -- not a new, third measurement.

    Raises `RuntimeError` if `store_base_weights()` was never called on this worker (nothing
    to compare against). Returns a small diagnostic dict (ok, max_abs_drift,
    fraction_elements_differing) -- never full tensors. Callers MUST abort the whole
    experiment when `ok` is False.
    """
    if not hasattr(worker_self, "_base_weights"):
        raise RuntimeError(
            "verify_exact_fixed_base_restoration_rpc: worker_self has no _base_weights -- "
            "store_base_weights() must be called exactly once before any fixed-base "
            "perturb/reset/verify cycle."
        )
    from ..diagnostics.perturb_restore_drift import measure_drift

    drift = measure_drift(worker_self.model_runner.model, worker_self._base_weights, param_filter=worker_self._should_perturb)
    return {
        "ok": drift["max_abs_drift"] == 0.0,
        "max_abs_drift": drift["max_abs_drift"],
        "fraction_elements_differing": drift["fraction_elements_differing"],
    }

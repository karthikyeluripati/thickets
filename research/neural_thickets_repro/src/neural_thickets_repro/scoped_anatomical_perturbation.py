"""GPU worker-side dispatch wrapper around `thicket.perturbation.apply_anatomical_relative_l2`
-- the Stage 7B anatomical-calibration analog of `scoped_perturbation.scoped_apply_perturbation`
(which wraps the older, expectation-only `scopes.compute_relative_l2_sigma` scale mode instead).

Deliberately a SEPARATE module, not an extension of scoped_perturbation.py: apply_anatomical_
relative_l2 is a distinct, exact-rescale scientific choice (VISUAL_THICKET_EXPERIMENT_SPEC.md
section C2/C3), and the region vocabulary here is thicket.anatomy's L1 atlas regions ("vision",
"multimodal_connector_or_merger", "language"), not scopes.py's PERTURBATION_SCOPES registry
("vision_encoder", "vision_merger", "full_lm") -- the two name the same underlying parameter
sets (see tests/test_scoped_anatomical_perturbation.py), but this module speaks the anatomy
vocabulary directly, since that is what Stage 7's calibration plan and calibration runner use.

Dispatched via vLLM's collective_rpc(Callable, ...), the same mechanism scoped_perturbation.py
already established -- never touches external/RandOpt, never calls perturb_self_weights.
"""
from __future__ import annotations

from typing import Dict, Sequence

from .diagnostics.perturb_restore_drift import measure_drift
from .thicket.perturbation import apply_anatomical_relative_l2


def scoped_apply_anatomical_perturbation(
    worker_self, seed: int, r: float, region_name: str, region_param_names: Sequence[str],
) -> Dict:
    """Dispatched via collective_rpc(scoped_apply_anatomical_perturbation, args=(seed, r,
    region_name, region_param_names)). Defensive restore-then-perturb (same discipline as
    scoped_perturbation.scoped_apply_perturbation): always applies from the exact stored base,
    never from whatever the current (possibly already-perturbed) state happens to be.
    """
    worker_self.reset_to_base_weights()
    model = worker_self.model_runner.model
    record = apply_anatomical_relative_l2(model, region_name, region_param_names, seed, r)
    realized_r = record.realized_epsilon_l2_norm / record.theta_l2_norm if record.theta_l2_norm > 0 else 0.0
    return {
        "region": record.region,
        "seed": record.seed,
        "requested_relative_l2": record.requested_r,
        "realized_relative_l2": realized_r,
        "theta_l2_norm": record.theta_l2_norm,
        "raw_noise_l2_norm": record.raw_noise_l2_norm,
        "scale": record.scale,
        "realized_epsilon_l2_norm": record.realized_epsilon_l2_norm,
        "region_param_count": record.param_count,
    }


def diag_snapshot_base(worker_self) -> str:
    """Diagnostic-only in-worker base snapshot for drift measurement -- same shape as
    diagnostics/scope_isolation_gpu_check.py's _diag_snapshot_base, duplicated rather than
    cross-imported (consistent with this project's per-diagnostic-module convention).
    """
    model = worker_self.model_runner.model
    worker_self._anatomical_diag_base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return f"snapshotted {len(worker_self._anatomical_diag_base_state)} tensors"


def diag_region_drift(worker_self, region_param_names: Sequence[str]) -> Dict:
    """Splits drift into in-region (the perturbed region's own parameters) vs out-of-region
    (the literal complement across the entire runtime model) -- reuses measure_drift's
    param_filter, same pattern as scope_isolation_gpu_check.py's _diag_scope_drift. Must be
    called after diag_snapshot_base and before reset_to_base_weights.
    """
    if not hasattr(worker_self, "_anatomical_diag_base_state"):
        raise RuntimeError("diag_snapshot_base was never called on this worker before diag_region_drift")
    model = worker_self.model_runner.model
    base_state = worker_self._anatomical_diag_base_state
    selected_names = set(region_param_names)

    in_region = measure_drift(model, base_state, param_filter=lambda n: n in selected_names)
    out_of_region = measure_drift(model, base_state, param_filter=lambda n: n not in selected_names)
    total_params = sum(1 for _ in model.named_parameters())
    return {
        "in_region": in_region,
        "out_of_region": out_of_region,
        "region_param_count": len(selected_names),
        "out_of_region_param_count": total_params - len(selected_names),
    }


def diag_full_model_drift(worker_self) -> Dict:
    if not hasattr(worker_self, "_anatomical_diag_base_state"):
        raise RuntimeError("diag_snapshot_base was never called on this worker before diag_full_model_drift")
    model = worker_self.model_runner.model
    return measure_drift(model, worker_self._anatomical_diag_base_state)

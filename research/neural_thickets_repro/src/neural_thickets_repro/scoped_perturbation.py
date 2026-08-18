"""Scoped perturbation -- the local package-side extension to WorkerExtension, in place of
modifying external/RandOpt (see SCOPED_PERTURBATION_DESIGN.md's Phase 1 investigation:
neither apply_perturbation nor any other WorkerExtension method takes a scope/component
argument, so scoped perturbation cannot be expressed through the upstream mechanism at all).

scoped_apply_perturbation() is dispatched via vLLM's own
collective_rpc(method: Union[str, Callable], ...) -- the same Callable-dispatch mechanism
already proven working in diagnostics/gate2_restoration_ab.py's
_diag_snapshot_base/_diag_diff_from_base. It never calls, wraps, or reimplements
perturb_self_weights/apply_perturbation/_should_perturb; the only WorkerExtension method it
calls is worker_self.reset_to_base_weights() -- the real, unmodified upstream method,
invoked as an ordinary Python method call on the same worker object, to defensively land on
the exact stored base before perturbing (mirrors upstream apply_perturbation's own
"restore-then-perturb" discipline, described not copied).

Noise generation reuses perturb_cpu._generate_noise UNCHANGED -- the exact per-tensor-reseed
convention already validated against REPRO_SPEC.md's "Perturbation formula" row: for each
selected tensor, a FRESH torch.Generator().manual_seed(candidate_seed) (the same candidate
seed reused per tensor, not a continuous stream across the concatenated scope), shape-matched
torch.randn(...), added in the tensor's native dtype. Scoped perturbation changes ONLY which
tensors are eligible (via scopes.py) and, for relative_l2 scale mode, the scalar sigma
applied -- never the noise-generation scheme itself. No independent/isotropic-across-tensors
Gaussian mode is introduced.
"""
from __future__ import annotations

from typing import Dict

import torch

from .perturb_cpu import _generate_noise
from .scopes import build_scope_manifest, compute_relative_l2_sigma

NOISE_SEMANTICS = "upstream_per_tensor_reseed"
PERTURBATION_SCALE_MODES = ("raw_sigma", "relative_l2")


def _try_named_parameters_no_duplicate(model):
    """Best-effort, diagnostic-only untied view for alias reporting -- remove_duplicate is
    not guaranteed available on every torch/transformers model class, so this degrades to
    None (no alias reporting) rather than failing the whole perturbation.
    """
    try:
        return list(model.named_parameters(remove_duplicate=False))
    except TypeError:
        return None


def scoped_apply_perturbation(worker_self, seed: int, sigma_or_r: float, scope_name: str, scale_mode: str) -> Dict:
    """Dispatched via collective_rpc(scoped_apply_perturbation, args=(seed, sigma_or_r,
    scope_name, scale_mode)) -- runs INSIDE the vLLM worker process. Returns a small
    JSON-serializable dict (never per-tensor data) recording exactly what was done, for the
    driver to fold into the candidate ledger / run metadata.
    """
    if scale_mode not in PERTURBATION_SCALE_MODES:
        raise ValueError(f"Unknown perturbation_scale_mode {scale_mode!r}, expected one of {PERTURBATION_SCALE_MODES}")

    # Defensive restore-then-perturb, mirroring upstream apply_perturbation's own discipline
    # (described in SCOPED_PERTURBATION_DESIGN.md, not copied) -- guarantees perturbation is
    # always applied from the exact stored base, never from whatever the current (possibly
    # already-perturbed) state happens to be, regardless of caller ordering.
    worker_self.reset_to_base_weights()

    model = worker_self.model_runner.model
    named_parameters = list(model.named_parameters())
    alias_view = _try_named_parameters_no_duplicate(model)

    manifest = build_scope_manifest(scope_name, named_parameters, alias_named_parameters=alias_view)

    if scale_mode == "relative_l2":
        derived_sigma = compute_relative_l2_sigma(manifest.base_l2_norm, manifest.total_element_count, sigma_or_r)
        requested_relative_l2 = sigma_or_r
    else:
        derived_sigma = sigma_or_r
        requested_relative_l2 = None

    selected_by_name = {name: p for name, p in named_parameters if name in set(manifest.selected_param_names)}

    sum_sq_delta = 0.0
    with torch.no_grad():
        for name in manifest.selected_param_names:
            p = selected_by_name[name]
            noise = _generate_noise(p, seed)
            delta = derived_sigma * noise
            p.add_(delta)
            sum_sq_delta += delta.detach().float().pow(2).sum().item()

    actual_perturbation_l2 = sum_sq_delta ** 0.5

    return {
        "scope": scope_name,
        "scale_mode": scale_mode,
        "seed": seed,
        "requested_relative_l2": requested_relative_l2,
        "derived_sigma": derived_sigma,
        "actual_perturbation_l2": actual_perturbation_l2,
        "scope_param_count": manifest.selected_param_count,
        "scope_total_element_count": manifest.total_element_count,
        "scope_base_l2_norm": manifest.base_l2_norm,
        "detected_lm_convention": manifest.detected_lm_convention,
        "representative_names": manifest.representative_names,
        "aliases": manifest.aliases,
        "noise_semantics": NOISE_SEMANTICS,
    }

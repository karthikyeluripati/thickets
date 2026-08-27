"""Legacy-vs-memory-bounded numerical equivalence tests for the Stage-11 7B whole_model OOM fix
(thicket/perturbation.py's apply_anatomical_relative_l2, thicket/memory_bounded_ops.py).

Reimplements the ORIGINAL (pre-fix) apply_anatomical_relative_l2 body here, UNCHANGED except for
being renamed, purely as a reference oracle for equivalence testing -- this is the only place
that implementation is preserved. It full-tensor-casts to float32 exactly as the live code did
before this repair pass; the CURRENT implementation (imported, not reimplemented) instead uses
thicket.memory_bounded_ops's chunked float64 accumulators. Both must:
  (a) apply the IDENTICAL Gaussian direction (same seed -> _generate_noise is bit-reproducible,
      already relied upon elsewhere in this codebase for undo_anatomical_relative_l2);
  (b) compute the SAME norm quantities to within legacy float32's OWN rounding error (the new
      float64-chunked path is MORE precise, so any discrepancy is attributable to the legacy
      path's rounding, not the new one -- quantified below, never hidden);
  (c) produce BIT-IDENTICAL final BF16 parameter values (the scale factor differs by at most a
      few ULPs of float64 near a value BF16 cannot resolve to begin with).
"""
from typing import Dict, Optional, Sequence

import pytest
import torch

from neural_thickets_repro.perturb_cpu import _generate_noise
from neural_thickets_repro.run_stage8_coarse_anatomical_atlas import STAGE8_RADII
from neural_thickets_repro.scoped_anatomical_perturbation import (
    QUANTIZATION_AWARE_METHOD_V3,
    scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3,
)
from neural_thickets_repro.thicket.perturbation import AnatomicalRelativeL2Record, apply_anatomical_relative_l2


@torch.no_grad()
def _legacy_apply_anatomical_relative_l2(
    model: torch.nn.Module, region: str, region_param_names: Sequence[str], seed: int, r: float,
    *, base_state: Optional[Dict[str, torch.Tensor]] = None,
) -> AnatomicalRelativeL2Record:
    """The PRE-FIX implementation, preserved verbatim (modulo the rename and this decorator --
    the REAL apply_anatomical_relative_l2 was always @torch.no_grad()-decorated; this reference
    oracle needs the same decorator to be usable at all, since test-fixture parameters here
    default to requires_grad=True) as a reference oracle. Never used outside this test file.
    """
    region_param_names = tuple(sorted(set(region_param_names)))
    named = dict(model.named_parameters())
    theta_sq_sum = 0.0
    noises: Dict[str, torch.Tensor] = {}
    noise_sq_sum = 0.0
    theta_before: Dict[str, torch.Tensor] = {}
    for name in region_param_names:
        p = named[name]
        theta_before[name] = base_state[name] if base_state is not None else p.detach().clone()
        theta_sq_sum += p.detach().float().pow(2).sum().item()
        noise = _generate_noise(p, seed)
        noises[name] = noise
        noise_sq_sum += noise.detach().float().pow(2).sum().item()

    theta_l2_norm = theta_sq_sum ** 0.5
    raw_noise_l2_norm = noise_sq_sum ** 0.5
    scale = (r * theta_l2_norm) / raw_noise_l2_norm

    designed_sq_sum = 0.0
    realized_sq_sum = 0.0
    for name in region_param_names:
        p = named[name]
        delta = scale * noises[name]
        designed_sq_sum += delta.detach().float().pow(2).sum().item()
        p.add_(delta.to(dtype=p.dtype))
        realized_delta = p.detach().float() - theta_before[name].float()
        realized_sq_sum += realized_delta.pow(2).sum().item()

    return AnatomicalRelativeL2Record(
        region=region, seed=seed, requested_r=r, theta_l2_norm=theta_l2_norm, raw_noise_l2_norm=raw_noise_l2_norm,
        scale=scale, designed_epsilon_l2_norm=designed_sq_sum ** 0.5, realized_epsilon_l2_norm=realized_sq_sum ** 0.5,
        region_param_names=region_param_names,
    )


class _MultiTensorModel(torch.nn.Module):
    """A handful of differently-shaped BF16 tensors -- exercises a REGION composed of several
    tensors of different sizes (matching the real whole_model/anatomy shape), not a single tensor.
    """

    def __init__(self, shapes: Dict[str, tuple]):
        super().__init__()
        for name, shape in shapes.items():
            setattr(self, name, torch.nn.Parameter(torch.empty(shape)))


REGION_SHAPE_SETS = [
    {"w1": (37,), "w2": (5, 5)},
    {"embed": (997,), "proj": (64, 64)},
    {"big": (200, 300)},  # 60,000 elements
    {"tiny": (3,), "small": (1,), "medium": (128, 4)},
]
SEEDS = [0, 1, 12345, 2_147_483_647]
MAGNITUDES = [0.001, 1.0, 50.0]


def _build_model_pair(shapes: Dict[str, tuple], magnitude: float, init_seed: int):
    torch.manual_seed(init_seed)
    model_legacy = _MultiTensorModel(shapes).to(torch.bfloat16)
    with torch.no_grad():
        for name, p in model_legacy.named_parameters():
            p.copy_((torch.randn(p.shape) * magnitude).to(torch.bfloat16))

    model_new = _MultiTensorModel(shapes).to(torch.bfloat16)
    with torch.no_grad():
        for (n1, p1), (n2, p2) in zip(model_legacy.named_parameters(), model_new.named_parameters()):
            assert n1 == n2
            p2.copy_(p1)
    return model_legacy, model_new


# =================================================================================================
# Legacy-vs-streaming numerical equivalence (Section 5 of the task spec)
# =================================================================================================


@pytest.mark.parametrize("shapes", REGION_SHAPE_SETS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("magnitude", MAGNITUDES)
@pytest.mark.parametrize("radius", STAGE8_RADII)
def test_legacy_vs_streaming_norms_and_final_weights_agree(shapes, seed, magnitude, radius):
    model_legacy, model_new = _build_model_pair(shapes, magnitude, init_seed=17)
    region_names = [n for n, _ in model_legacy.named_parameters()]

    for (n1, p1), (n2, p2) in zip(model_legacy.named_parameters(), model_new.named_parameters()):
        assert torch.equal(p1, p2), f"pre-condition failed: {n1} did not start identical"

    legacy_record = _legacy_apply_anatomical_relative_l2(model_legacy, "region", region_names, seed, radius)
    new_record = apply_anatomical_relative_l2(model_new, "region", region_names, seed, radius)

    # (b): norms agree to within legacy float32's own rounding error -- far below the frozen
    # 1e-6 absolute / 1e-3 relative radius-acceptance tolerances.
    assert new_record.theta_l2_norm == pytest.approx(legacy_record.theta_l2_norm, rel=1e-6)
    assert new_record.raw_noise_l2_norm == pytest.approx(legacy_record.raw_noise_l2_norm, rel=1e-6)
    assert new_record.scale == pytest.approx(legacy_record.scale, rel=1e-6)
    assert new_record.designed_epsilon_l2_norm == pytest.approx(legacy_record.designed_epsilon_l2_norm, rel=1e-6)
    assert new_record.realized_epsilon_l2_norm == pytest.approx(legacy_record.realized_epsilon_l2_norm, rel=1e-6)

    # (c): final BF16 weights are BIT-IDENTICAL -- same noise (regenerated, not cached, but
    # exactly reproducible), same scale-computation formula, and the scale itself differs by an
    # amount many orders of magnitude below BF16's own ~2^-8 relative resolution.
    for (n1, p1), (n2, p2) in zip(model_legacy.named_parameters(), model_new.named_parameters()):
        assert torch.equal(p1, p2), f"final BF16 mismatch for {n1}"


def test_legacy_vs_streaming_realized_relative_l2_discrepancy_is_far_below_radius_tolerance():
    """Directly quantifies the discrepancy in the SCIENTIFICALLY RELEVANT derived quantity
    (realized_relative_l2 = realized_epsilon_l2_norm / theta_l2_norm) across the full grid above,
    and asserts the worst case observed is far below RADIUS_REALIZATION_TOLERANCE (1e-6, absolute)
    -- not merely "small" but explicitly bounded against the actual frozen gate.
    """
    from neural_thickets_repro.scoped_anatomical_perturbation import RADIUS_REALIZATION_TOLERANCE

    max_abs_discrepancy = 0.0
    for shapes in REGION_SHAPE_SETS:
        for seed in SEEDS:
            for radius in STAGE8_RADII:
                model_legacy, model_new = _build_model_pair(shapes, magnitude=1.0, init_seed=17)
                region_names = [n for n, _ in model_legacy.named_parameters()]
                legacy_record = _legacy_apply_anatomical_relative_l2(model_legacy, "region", region_names, seed, radius)
                new_record = apply_anatomical_relative_l2(model_new, "region", region_names, seed, radius)
                legacy_realized_r = legacy_record.realized_epsilon_l2_norm / legacy_record.theta_l2_norm
                new_realized_r = new_record.realized_epsilon_l2_norm / new_record.theta_l2_norm
                max_abs_discrepancy = max(max_abs_discrepancy, abs(legacy_realized_r - new_realized_r))

    assert max_abs_discrepancy < RADIUS_REALIZATION_TOLERANCE / 100.0  # 100x safety margin below the frozen gate


# =================================================================================================
# v3 solver-level equivalence: acceptance mode / accepted scalar / realized value unaffected
# =================================================================================================


class _V3TwoTensorModel(torch.nn.Module):
    def __init__(self, region_elements: int, outside_elements: int = 100):
        super().__init__()
        self.region_layer = torch.nn.Linear(region_elements, 1, bias=False)
        self.outside_layer = torch.nn.Linear(outside_elements, 1, bias=False)


def _v3_worker(region_elements: int, outside_elements: int = 100, init_seed: int = 0):
    torch.manual_seed(init_seed)
    model = _V3TwoTensorModel(region_elements, outside_elements).to(torch.bfloat16)
    base_weights = {name: p.detach().clone() for name, p in model.named_parameters()}

    def _reset():
        with torch.no_grad():
            for name, p in model.named_parameters():
                p.copy_(base_weights[name])

    worker = type("W", (), {})()
    worker.model_runner = type("MR", (), {"model": model})()
    worker.reset_to_base_weights = _reset
    worker._base_weights = base_weights
    return worker


@pytest.mark.parametrize("region_elements,seed,radius", [
    (500_000, 12345, 0.035698828543799424),
    (500_000, 999, 0.0035698828543799426),
    (2_000, 12345, 0.035698828543799424),  # the known plateau case
])
def test_v3_acceptance_decision_identical_legacy_vs_streaming(monkeypatch, region_elements, seed, radius):
    """Runs the REAL v3 solver (unmodified acceptance/bracket-expansion logic) twice: once
    against the current (memory-bounded) apply_anatomical_relative_l2, once with
    scoped_anatomical_perturbation's OWN reference to apply_anatomical_relative_l2 monkeypatched
    to the legacy float32 implementation -- proving the solver's ACCEPTANCE DECISION (mode,
    accepted scalar, realized value) is unaffected by which norm-computation path feeds it.
    """
    import neural_thickets_repro.scoped_anatomical_perturbation as sap

    worker_new = _v3_worker(region_elements, init_seed=0)
    result_new = scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3(worker_new, seed, radius, "region", ["region_layer.weight"])

    worker_legacy = _v3_worker(region_elements, init_seed=0)
    monkeypatch.setattr(sap, "apply_anatomical_relative_l2", _legacy_apply_anatomical_relative_l2)
    result_legacy = scoped_apply_anatomical_perturbation_bf16_quantization_aware_v3(worker_legacy, seed, radius, "region", ["region_layer.weight"])

    assert result_new["radius_realization_method"] == QUANTIZATION_AWARE_METHOD_V3
    assert result_new["radius_acceptance_mode"] == result_legacy["radius_acceptance_mode"]
    assert result_new["quantization_limited"] == result_legacy["quantization_limited"]
    assert result_new["realized_relative_l2"] == pytest.approx(result_legacy["realized_relative_l2"], abs=1e-8)
    assert result_new["accepted_scalar"] == pytest.approx(result_legacy["accepted_scalar"], rel=1e-6)
    # final BF16 weights bit-identical between the two acceptance paths too
    assert torch.equal(worker_new.model_runner.model.region_layer.weight, worker_legacy.model_runner.model.region_layer.weight)


# =================================================================================================
# measure_drift: full-function legacy-vs-streaming equivalence (second OOM fix -- exact-difference
# counting). Reimplements the ORIGINAL (pre-either-repair-pass) measure_drift body here, verbatim
# except for the rename, as a second reference oracle.
# =================================================================================================


def _legacy_measure_drift(model, original_state, param_filter=None) -> dict:
    max_abs = 0.0
    sum_abs = 0.0
    n_elems = 0
    sq_diff_sum = 0.0
    orig_sq_sum = 0.0
    n_differing = 0
    for name, p in model.named_parameters():
        if param_filter is not None and not param_filter(name):
            continue
        orig = original_state[name]
        diff = (p.detach().float() - orig.float())
        max_abs = max(max_abs, diff.abs().max().item())
        sum_abs += diff.abs().sum().item()
        n_elems += diff.numel()
        sq_diff_sum += diff.pow(2).sum().item()
        orig_sq_sum += orig.float().pow(2).sum().item()
        n_differing += int((p.detach() != orig).sum().item())
    mean_abs = sum_abs / n_elems if n_elems else 0.0
    rel_norm = (sq_diff_sum**0.5) / (orig_sq_sum**0.5) if orig_sq_sum > 0 else 0.0
    fraction_differing = n_differing / n_elems if n_elems else 0.0
    return {
        "max_abs_drift": max_abs, "mean_abs_drift": mean_abs,
        "relative_norm_drift": rel_norm, "fraction_elements_differing": fraction_differing,
    }


class _DriftMultiTensorModel(torch.nn.Module):
    def __init__(self, shapes):
        super().__init__()
        for name, shape in shapes.items():
            setattr(self, name, torch.nn.Parameter(torch.empty(shape)))


DRIFT_SHAPE_SETS = [
    {"a": (41,), "b": (7, 7)},
    {"big": (150, 220)},  # 33,000 elements
    {"tiny": (2,), "mid": (64, 8)},
]


@pytest.mark.parametrize("shapes", DRIFT_SHAPE_SETS)
def test_measure_drift_matches_legacy_when_unchanged(shapes):
    from neural_thickets_repro.diagnostics.perturb_restore_drift import measure_drift

    torch.manual_seed(21)
    model = _DriftMultiTensorModel(shapes).to(torch.bfloat16)
    with torch.no_grad():
        for _, p in model.named_parameters():
            p.copy_(torch.randn(p.shape).to(torch.bfloat16))
    original = {n: p.detach().clone() for n, p in model.named_parameters()}

    legacy = _legacy_measure_drift(model, original)
    new = measure_drift(model, original)
    assert new == legacy  # zero drift is exact in both implementations, no precision-dependent path taken


@pytest.mark.parametrize("shapes", DRIFT_SHAPE_SETS)
def test_measure_drift_matches_legacy_with_sparse_and_dense_changes(shapes):
    from neural_thickets_repro.diagnostics.perturb_restore_drift import measure_drift

    torch.manual_seed(22)
    model = _DriftMultiTensorModel(shapes).to(torch.bfloat16)
    with torch.no_grad():
        for _, p in model.named_parameters():
            p.copy_(torch.randn(p.shape).to(torch.bfloat16))
    original = {n: p.detach().clone() for n, p in model.named_parameters()}

    with torch.no_grad():
        for _, p in model.named_parameters():
            flat = p.view(-1)
            flat[0].add_(1.0)  # sparse: one element per tensor
            if flat.numel() > 3:
                flat[1:3].add_(0.5)  # a couple more, dense-ish

    legacy = _legacy_measure_drift(model, original)
    new = measure_drift(model, original)
    # fraction_elements_differing and n_differing-derived quantities are EXACT integers/counts --
    # must match exactly; max/mean/relative drift may differ by the already-documented,
    # far-below-tolerance float64-vs-float32 precision improvement.
    assert new["fraction_elements_differing"] == legacy["fraction_elements_differing"]
    assert new["max_abs_drift"] == pytest.approx(legacy["max_abs_drift"], abs=1e-6)
    assert new["mean_abs_drift"] == pytest.approx(legacy["mean_abs_drift"], abs=1e-6)
    assert new["relative_norm_drift"] == pytest.approx(legacy["relative_norm_drift"], abs=1e-6)

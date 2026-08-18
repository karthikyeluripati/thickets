"""Unit tests for the parts of gate2_restoration_ab.py that don't need ray/vllm/GPU: the
diagnostic snapshot/diff functions (exercised directly against the Gate-0 synthetic DummyVLM
fixture, standing in for a vLLM worker's worker_self.model_runner.model), and the pure
report-shaping helpers (_drift_summary, _visual_frozen_every_cycle). The ray-dependent
collective_rpc plumbing itself is exercised on the pod, same limitation as
tests/test_vlm_adapter.py.
"""
from types import SimpleNamespace

import pytest
import torch

from neural_thickets_repro.diagnostics.gate2_gpu_preflight import N_PREFLIGHT_EXAMPLES, TEST_SEED, TEST_SIGMA
from neural_thickets_repro.diagnostics.gate2_restoration_ab import (
    PATH_A,
    PATH_B,
    _diag_diff_from_base,
    _diag_snapshot_base,
    _drift_summary,
    _validate_collective_rpc_results,
    _verify_call_shape,
    _verify_worker_extension_signatures,
    _visual_frozen_every_cycle,
)
from neural_thickets_repro.perturb_cpu import perturb, restore


def _fake_worker(model):
    return SimpleNamespace(model_runner=SimpleNamespace(model=model))


def test_reuses_gate2_gpu_preflight_constants_not_redeclared():
    # Not a strong assertion by itself -- the real point is these names import cleanly from
    # gate2_restoration_ab's own import line (`from .gate2_gpu_preflight import ...`), so a
    # future edit to gate2_gpu_preflight.py's constants can't silently diverge unnoticed.
    assert isinstance(N_PREFLIGHT_EXAMPLES, int) and N_PREFLIGHT_EXAMPLES > 0
    assert isinstance(TEST_SEED, int)
    assert isinstance(TEST_SIGMA, float)


def test_diag_diff_from_base_zero_when_unchanged(dummy_vlm_factory):
    worker = _fake_worker(dummy_vlm_factory())
    _diag_snapshot_base(worker)

    drift = _diag_diff_from_base(worker)

    assert drift["max_abs_drift"] == 0.0
    assert drift["fraction_elements_differing"] == 0.0
    assert drift["vision_frozen"] is True


def test_diag_diff_from_base_detects_drift_after_perturb_restore(dummy_vlm_factory):
    """Mirrors what the real diagnostic is checking for: perturb_cpu's own restore() is a
    reseed-and-subtract, not a stored-copy restore -- REPRO_SPEC.md's documented mechanism --
    so after a perturb/restore cycle at bf16 there is no guarantee of exact equality even in
    our own reimplementation. This does not test the REAL WorkerExtension (unavailable
    locally); it confirms _diag_diff_from_base correctly reports nonzero drift when the
    underlying model genuinely differs from its snapshot, rather than silently reporting zero.
    """
    model = dummy_vlm_factory().to(torch.bfloat16)
    worker = _fake_worker(model)
    _diag_snapshot_base(worker)

    perturb(model, seed=TEST_SEED, sigma=TEST_SIGMA)
    restore(model, seed=TEST_SEED, sigma=TEST_SIGMA)

    drift = _diag_diff_from_base(worker)

    assert drift["max_abs_drift"] >= 0.0
    assert drift["vision_frozen"] is True  # perturb_cpu.perturb never touches visual.* params


def test_diag_diff_from_base_flags_visual_drift_if_it_ever_happens(dummy_vlm_factory):
    """Regression guard for the check itself: if some future change to how the diagnostic
    invokes perturb caused visual.* tensors to change too, vision_frozen must go False rather
    than silently staying True.
    """
    model = dummy_vlm_factory()
    worker = _fake_worker(model)
    _diag_snapshot_base(worker)

    with torch.no_grad():
        next(model.visual.parameters()).add_(1.0)

    drift = _diag_diff_from_base(worker)

    assert drift["vision_frozen"] is False
    assert drift["max_abs_drift"] > 0.0


def test_diag_diff_from_base_requires_snapshot_first(dummy_vlm_factory):
    worker = _fake_worker(dummy_vlm_factory())
    with pytest.raises(RuntimeError, match="_diag_snapshot_base was never called"):
        _diag_diff_from_base(worker)


def test_drift_summary_none_when_no_cycles():
    assert _drift_summary([]) is None


def test_drift_summary_none_when_drift_unavailable():
    cycles = [{"cycle": 1, "drift": None}]
    assert _drift_summary(cycles) is None


def test_drift_summary_reports_first_and_last_and_ratios():
    cycles = [
        {"cycle": 1, "drift": {"max_abs_drift": 0.001, "fraction_elements_differing": 0.01, "vision_frozen": True}},
        {"cycle": 2, "drift": {"max_abs_drift": 0.002, "fraction_elements_differing": 0.02, "vision_frozen": True}},
        {"cycle": 3, "drift": {"max_abs_drift": 0.004, "fraction_elements_differing": 0.04, "vision_frozen": True}},
    ]
    summary = _drift_summary(cycles)
    assert summary["first_cycle"]["max_abs_drift"] == 0.001
    assert summary["last_cycle"]["max_abs_drift"] == 0.004
    assert summary["max_abs_drift_ratio_last_over_first"] == pytest.approx(4.0)
    assert summary["fraction_elements_differing_ratio_last_over_first"] == pytest.approx(4.0)


def test_drift_summary_ratio_is_none_when_first_cycle_had_zero_drift():
    cycles = [
        {"cycle": 1, "drift": {"max_abs_drift": 0.0, "fraction_elements_differing": 0.0, "vision_frozen": True}},
        {"cycle": 2, "drift": {"max_abs_drift": 0.002, "fraction_elements_differing": 0.02, "vision_frozen": True}},
    ]
    summary = _drift_summary(cycles)
    assert summary["max_abs_drift_ratio_last_over_first"] is None
    assert summary["fraction_elements_differing_ratio_last_over_first"] is None


def test_visual_frozen_every_cycle_true_when_all_true():
    cycles = [{"drift": {"vision_frozen": True}}, {"drift": {"vision_frozen": True}}]
    assert _visual_frozen_every_cycle(cycles) is True


def test_visual_frozen_every_cycle_false_if_any_cycle_flags_it():
    cycles = [{"drift": {"vision_frozen": True}}, {"drift": {"vision_frozen": False}}]
    assert _visual_frozen_every_cycle(cycles) is False


def test_visual_frozen_every_cycle_none_when_drift_unavailable():
    cycles = [{"drift": None}, {"drift": None}]
    assert _visual_frozen_every_cycle(cycles) is None


# --- _validate_collective_rpc_results: the TP=1 unwrap/validate fix ---


def test_validate_collective_rpc_results_unwraps_single_worker_list():
    assert _validate_collective_rpc_results(["a value"], label="some_method") == "a value"


def test_validate_collective_rpc_results_unwraps_single_worker_dict():
    # The exact shape that triggered the original bug: collective_rpc(_diag_diff_from_base)
    # returns [drift_dict] under TP=1, not drift_dict itself.
    drift_dict = {"max_abs_drift": 0.001, "vision_frozen": True}
    assert _validate_collective_rpc_results([drift_dict], label="_diag_diff_from_base") is drift_dict


def test_validate_collective_rpc_results_rejects_non_list():
    with pytest.raises(RuntimeError, match="expected vLLM's own list-of-per-worker-results"):
        _validate_collective_rpc_results({"max_abs_drift": 0.001}, label="_diag_diff_from_base")


def test_validate_collective_rpc_results_rejects_empty_list():
    with pytest.raises(RuntimeError, match="TP=1-only"):
        _validate_collective_rpc_results([], label="_diag_diff_from_base")


def test_validate_collective_rpc_results_rejects_multi_worker_list():
    with pytest.raises(RuntimeError, match="TP=1-only"):
        _validate_collective_rpc_results(["shard_0", "shard_1"], label="_diag_diff_from_base")


# --- PATH_A/PATH_B: each restoration path uses its OWN real upstream call shape, not one
# identical args tuple forced onto both -- this is exactly what the pod-side crash was ---


def test_path_a_perturb_receives_three_runtime_args():
    assert PATH_A.perturb_method == "perturb_self_weights"
    assert PATH_A.perturb_args == (TEST_SEED, TEST_SIGMA, False)
    assert len(PATH_A.perturb_args) == 3


def test_path_a_restore_receives_three_runtime_args():
    assert PATH_A.restore_method == "restore_self_weights"
    assert PATH_A.restore_args == (TEST_SEED, TEST_SIGMA, False)
    assert len(PATH_A.restore_args) == 3


def test_path_b_perturb_receives_two_runtime_args():
    assert PATH_B.perturb_method == "apply_perturbation"
    assert PATH_B.perturb_args == (TEST_SEED, TEST_SIGMA)
    assert len(PATH_B.perturb_args) == 2


def test_path_b_reset_receives_zero_runtime_args():
    assert PATH_B.restore_method == "reset_to_base_weights"
    assert PATH_B.restore_args == ()
    assert len(PATH_B.restore_args) == 0


# --- _verify_call_shape / _verify_worker_extension_signatures: the driver-side WorkerExtension
# signature check, now validating each method's ACTUAL call shape against the real pinned
# source instead of comparing parameter counts across two fundamentally different APIs ---


class _FakeWorkerExtensionRealShape:
    """Mirrors the ACTUAL confirmed pinned utils/worker_extn.py signatures (from the pod)."""

    def perturb_self_weights(self, seed, noise_scale, negate=False): ...
    def restore_self_weights(self, seed, SIGMA, negate=False): ...
    def apply_perturbation(self, seed, sigma): ...
    def reset_to_base_weights(self): ...


class _FakeWorkerExtensionUnexpectedApplySignature:
    """An upstream that changed apply_perturbation to require a third arg -- PATH_B's 2-arg
    call must be rejected before engine launch, not silently sent anyway.
    """

    def perturb_self_weights(self, seed, noise_scale, negate=False): ...
    def restore_self_weights(self, seed, SIGMA, negate=False): ...
    def apply_perturbation(self, seed, sigma, extra): ...  # now requires 3, PATH_B only sends 2
    def reset_to_base_weights(self): ...


class _FakeWorkerExtensionUnexpectedResetSignature:
    """An upstream where reset_to_base_weights unexpectedly takes an argument -- PATH_B's
    zero-arg call must be rejected rather than silently under-calling it.
    """

    def perturb_self_weights(self, seed, noise_scale, negate=False): ...
    def restore_self_weights(self, seed, SIGMA, negate=False): ...
    def apply_perturbation(self, seed, sigma): ...
    def reset_to_base_weights(self, seed): ...  # now requires 1, PATH_B sends 0


class _FakeWorkerExtensionMissingMethod:
    def perturb_self_weights(self, seed, noise_scale, negate=False): ...
    def restore_self_weights(self, seed, SIGMA, negate=False): ...
    def apply_perturbation(self, seed, sigma): ...
    # reset_to_base_weights not defined at all


def test_verify_call_shape_accepts_exact_required_arg_count():
    _verify_call_shape(_FakeWorkerExtensionRealShape.apply_perturbation, (TEST_SEED, TEST_SIGMA), label="apply_perturbation")


def test_verify_call_shape_accepts_zero_args_for_zero_arg_method():
    _verify_call_shape(_FakeWorkerExtensionRealShape.reset_to_base_weights, (), label="reset_to_base_weights")


def test_verify_call_shape_accepts_omitted_default_arg():
    # perturb_self_weights(seed, noise_scale, negate=False) -- calling with just 2 args
    # (omitting the defaulted negate) must be valid, not just the 3-arg form PATH_A uses.
    _verify_call_shape(_FakeWorkerExtensionRealShape.perturb_self_weights, (TEST_SEED, TEST_SIGMA), label="perturb_self_weights")


def test_verify_call_shape_rejects_too_few_args():
    with pytest.raises(RuntimeError, match="cannot be called with 1 positional"):
        _verify_call_shape(_FakeWorkerExtensionRealShape.apply_perturbation, (TEST_SEED,), label="apply_perturbation")


def test_verify_call_shape_rejects_too_many_args():
    with pytest.raises(RuntimeError, match="cannot be called with 3 positional"):
        _verify_call_shape(_FakeWorkerExtensionRealShape.reset_to_base_weights, (TEST_SEED, TEST_SIGMA, False), label="reset_to_base_weights")


def test_verify_worker_extension_signatures_passes_for_real_pinned_shape_with_path_a_and_b(capsys):
    _verify_worker_extension_signatures(_FakeWorkerExtensionRealShape, (PATH_A, PATH_B))  # should not raise
    printed = capsys.readouterr().out
    assert "apply_perturbation" in printed
    assert "reset_to_base_weights" in printed
    assert f"args={PATH_B.perturb_args!r}" in printed
    assert f"args={PATH_B.restore_args!r}" in printed


def test_verify_worker_extension_signatures_hard_fails_on_unexpected_apply_perturbation_signature():
    """An unexpected upstream signature change must hard-fail BEFORE engine launch, not
    surface only after GPU time has already been spent inside Path B's cycle loop.
    """
    with pytest.raises(RuntimeError, match="apply_perturbation"):
        _verify_worker_extension_signatures(_FakeWorkerExtensionUnexpectedApplySignature, (PATH_A, PATH_B))


def test_verify_worker_extension_signatures_hard_fails_on_unexpected_reset_signature():
    with pytest.raises(RuntimeError, match="reset_to_base_weights"):
        _verify_worker_extension_signatures(_FakeWorkerExtensionUnexpectedResetSignature, (PATH_A, PATH_B))


def test_verify_worker_extension_signatures_flags_missing_method():
    with pytest.raises(RuntimeError, match="missing method"):
        _verify_worker_extension_signatures(_FakeWorkerExtensionMissingMethod, (PATH_A, PATH_B))

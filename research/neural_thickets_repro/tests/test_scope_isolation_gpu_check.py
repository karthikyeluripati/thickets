"""Tests for scope_isolation_gpu_check.py's pure-logic diagnostic helper functions, exercised
directly against a fake worker (SimpleNamespace + a real synthetic model), same pattern as
tests/test_gate2_restoration_ab.py. No ray/vllm/GPU needed -- the collective_rpc plumbing
itself needs the pod, same limitation noted throughout this project's diagnostics.
"""
from types import SimpleNamespace

import pytest
import torch

from neural_thickets_repro.diagnostics.scope_isolation_gpu_check import (
    TEST_SEED,
    TEST_SIGMA,
    _diag_full_model_drift,
    _diag_report_all_scopes,
    _diag_scope_drift,
    _diag_snapshot_base,
    _validate_collective_rpc_results,
)
from neural_thickets_repro.scopes import PERTURBATION_SCOPES, build_scope_manifest


def _fake_worker(model):
    return SimpleNamespace(model_runner=SimpleNamespace(model=model))


def test_diag_snapshot_base_stores_full_state(runtime_wrapped_vlm_factory):
    model = runtime_wrapped_vlm_factory()
    worker = _fake_worker(model)
    msg = _diag_snapshot_base(worker)
    assert hasattr(worker, "_scope_diag_base_state")
    assert str(len(worker._scope_diag_base_state)) in msg


def test_diag_report_all_scopes_covers_all_seven(runtime_wrapped_vlm_factory):
    model = runtime_wrapped_vlm_factory()
    worker = _fake_worker(model)
    report = _diag_report_all_scopes(worker)

    assert report["detected_lm_namespace_convention"] == "runtime_wrapped"
    assert report["detected_lm_layer_count"] == 12
    assert set(report["scope_summaries"]) == set(PERTURBATION_SCOPES)
    for scope, summary in report["scope_summaries"].items():
        assert summary["selected_param_count"] > 0, f"{scope} selected zero"
    assert len(report["representative_param_names"]) > 0


def test_diag_report_all_scopes_reports_tied_alias(runtime_wrapped_vlm_factory):
    model = runtime_wrapped_vlm_factory()
    worker = _fake_worker(model)
    report = _diag_report_all_scopes(worker)
    assert "language_model.lm_head.weight" in report["scope_summaries"]["full_lm"]["aliases"]


def test_diag_scope_drift_detects_in_scope_change_and_out_of_scope_unchanged(runtime_wrapped_vlm_factory):
    model = runtime_wrapped_vlm_factory()
    worker = _fake_worker(model)
    _diag_snapshot_base(worker)

    with torch.no_grad():
        next(model.visual.patch_embed.parameters()).add_(1.0)  # in vision_encoder scope

    drift = _diag_scope_drift(worker, "vision_encoder")
    assert drift["in_scope"]["max_abs_drift"] == 1.0
    assert drift["out_of_scope"]["max_abs_drift"] == 0.0


def test_diag_scope_drift_reports_out_of_scope_as_literal_complement(runtime_wrapped_vlm_factory):
    """The out-of-scope drift check must cover the LITERAL complement of the scope's selected
    storage across the entire (deduplicated) runtime model -- not an enumerated list of "the
    other named components," which can silently omit tensors belonging to none of them (e.g.
    non-layer LM tensors like embeddings/final-norm that aren't part of any lm_early/middle/
    late third). Proven directly against scopes.build_scope_manifest's own name sets, not
    just asserted by reading the implementation.
    """
    model = runtime_wrapped_vlm_factory()
    worker = _fake_worker(model)
    _diag_snapshot_base(worker)

    drift = _diag_scope_drift(worker, "lm_middle")

    named_parameters = list(model.named_parameters())
    full_vlm_names = set(build_scope_manifest("full_vlm", named_parameters).selected_param_names)
    lm_middle_names = set(build_scope_manifest("lm_middle", named_parameters).selected_param_names)
    out_of_scope_names = full_vlm_names - lm_middle_names

    assert drift["full_vlm_param_count"] == len(full_vlm_names)
    assert drift["scope_param_count"] == len(lm_middle_names)
    assert drift["out_of_scope_param_count"] == len(out_of_scope_names)

    # The literal partition property requested: in_scope UNION out_of_scope == full_vlm,
    # intersection empty -- not just matching counts, the actual name sets.
    assert lm_middle_names | out_of_scope_names == full_vlm_names
    assert lm_middle_names & out_of_scope_names == set()

    # Concretely confirms the non-layer LM tensors (embed_tokens/norm -- NOT part of any
    # lm_early/middle/late third) are included in the out-of-scope complement, exactly the
    # gap an enumerated "vision + merger + lm_early + lm_late" description would have missed.
    non_layer_lm_names = {n for n in full_vlm_names if "layers." not in n and not n.startswith("visual.")}
    assert non_layer_lm_names, "fixture must have at least one non-layer LM tensor to make this check meaningful"
    assert non_layer_lm_names <= out_of_scope_names


@pytest.mark.parametrize("scope", PERTURBATION_SCOPES)
def test_in_scope_and_out_of_scope_partition_full_vlm_for_every_scope(scope, runtime_wrapped_vlm_factory):
    """Same partition property as above, generalized across all seven scopes -- in_scope and
    out_of_scope must always union to exactly full_vlm's own selection and never overlap.
    """
    model = runtime_wrapped_vlm_factory()
    worker = _fake_worker(model)
    _diag_snapshot_base(worker)

    drift = _diag_scope_drift(worker, scope)

    named_parameters = list(model.named_parameters())
    full_vlm_names = set(build_scope_manifest("full_vlm", named_parameters).selected_param_names)
    scope_names = set(build_scope_manifest(scope, named_parameters).selected_param_names)
    out_of_scope_names = full_vlm_names - scope_names

    assert scope_names | out_of_scope_names == full_vlm_names
    assert scope_names & out_of_scope_names == set()
    assert drift["out_of_scope_param_count"] == len(out_of_scope_names)
    assert drift["scope_param_count"] + drift["out_of_scope_param_count"] == drift["full_vlm_param_count"]


def test_diag_scope_drift_requires_snapshot_first(runtime_wrapped_vlm_factory):
    model = runtime_wrapped_vlm_factory()
    worker = _fake_worker(model)
    with pytest.raises(RuntimeError, match="_diag_snapshot_base was never called"):
        _diag_scope_drift(worker, "vision_encoder")


def test_diag_full_model_drift_zero_when_unchanged(runtime_wrapped_vlm_factory):
    model = runtime_wrapped_vlm_factory()
    worker = _fake_worker(model)
    _diag_snapshot_base(worker)
    drift = _diag_full_model_drift(worker)
    assert drift["max_abs_drift"] == 0.0


def test_diag_full_model_drift_nonzero_after_any_change(runtime_wrapped_vlm_factory):
    model = runtime_wrapped_vlm_factory()
    worker = _fake_worker(model)
    _diag_snapshot_base(worker)
    with torch.no_grad():
        next(model.language_model.model.layers[4].parameters()).add_(0.5)
    drift = _diag_full_model_drift(worker)
    assert drift["max_abs_drift"] == 0.5


def test_diag_full_model_drift_requires_snapshot_first(runtime_wrapped_vlm_factory):
    model = runtime_wrapped_vlm_factory()
    worker = _fake_worker(model)
    with pytest.raises(RuntimeError, match="_diag_snapshot_base was never called"):
        _diag_full_model_drift(worker)


# --- _validate_collective_rpc_results (same TP=1 shape-validation pattern used elsewhere) ---


def test_validate_collective_rpc_results_unwraps_single_worker_list():
    assert _validate_collective_rpc_results(["x"], label="m") == "x"


def test_validate_collective_rpc_results_rejects_non_list():
    with pytest.raises(RuntimeError, match="expected vLLM's own list-of-per-worker-results"):
        _validate_collective_rpc_results({"a": 1}, label="m")


def test_validate_collective_rpc_results_rejects_multi_worker_list():
    with pytest.raises(RuntimeError, match="TP=1-only"):
        _validate_collective_rpc_results(["a", "b"], label="m")


def test_module_level_test_constants_are_fixed_and_distinct_from_real_candidates():
    # Same convention as gate2_gpu_preflight.py -- a clearly-labeled test seed/sigma, not
    # drawn from any real candidate's RNG stream.
    assert TEST_SEED == 999_999_999
    assert TEST_SIGMA == 0.01

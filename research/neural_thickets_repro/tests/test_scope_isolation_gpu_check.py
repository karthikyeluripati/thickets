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
    """_diag_snapshot_base now ALIASES worker_self._base_weights (upstream's own
    store_base_weights() clone) rather than making its own separate copy -- see that
    function's own docstring for why (a third full-size GPU-resident clone genuinely OOMs a
    48GB L40S at 7B). Every fake worker here therefore comes pre-populated with a
    `_base_weights` dict shaped exactly like upstream's real store_base_weights() would build
    it (name -> cloned tensor, named_parameters()-derived, never buffers).
    """
    return SimpleNamespace(model_runner=SimpleNamespace(model=model), _base_weights={n: p.data.clone() for n, p in model.named_parameters()})


def test_diag_snapshot_base_stores_full_state(runtime_wrapped_vlm_factory):
    model = runtime_wrapped_vlm_factory()
    worker = _fake_worker(model)
    msg = _diag_snapshot_base(worker)
    assert hasattr(worker, "_scope_diag_base_state")
    assert worker._scope_diag_base_state is worker._base_weights  # aliased, not a separate clone
    assert str(len(worker._scope_diag_base_state)) in msg


def test_diag_snapshot_base_requires_base_weights_first():
    worker = SimpleNamespace(model_runner=SimpleNamespace(model=None))  # no _base_weights set
    with pytest.raises(RuntimeError, match="no _base_weights"):
        _diag_snapshot_base(worker)


def test_diag_report_all_scopes_covers_every_registered_scope(runtime_wrapped_vlm_32vision_factory):
    # Needs the 32-vision-block fixture: PERTURBATION_SCOPES includes vision_early/middle/
    # late and vision_late_a/b, whose fixed partitions hard-require the complete block set.
    model = runtime_wrapped_vlm_32vision_factory()
    worker = _fake_worker(model)
    report = _diag_report_all_scopes(worker)

    assert report["detected_lm_namespace_convention"] == "runtime_wrapped"
    assert report["detected_lm_layer_count"] == 12
    assert set(report["scope_summaries"]) == set(PERTURBATION_SCOPES)
    for scope, summary in report["scope_summaries"].items():
        assert summary["applicable"] is True, f"{scope} unexpectedly marked not applicable: {summary}"
        assert summary["selected_param_count"] > 0, f"{scope} selected zero"
    assert len(report["representative_param_names"]) > 0


def test_diag_report_all_scopes_marks_architecture_inapplicable_scopes_without_crashing(runtime_wrapped_vlm_32vision_factory, monkeypatch):
    """Confirmed live against Qwen2.5-VL-7B-Instruct (28 LM layers, not divisible by 3):
    build_scope_manifest("lm_middle", ...) hard-raises ScopeSelectionError.
    _diag_report_all_scopes must catch this PER SCOPE and keep reporting every other scope,
    never crash the whole report (which would also block Tests A/H/I from ever running).
    """
    import neural_thickets_repro.diagnostics.scope_isolation_gpu_check as module
    from neural_thickets_repro.scopes import ScopeSelectionError

    model = runtime_wrapped_vlm_32vision_factory()
    worker = _fake_worker(model)

    real_build_scope_manifest = module.build_scope_manifest

    def _fake_build_scope_manifest(scope_name, *a, **k):
        if scope_name == "lm_middle":
            raise ScopeSelectionError("Cannot partition 28 LM layers into three equal contiguous thirds")
        return real_build_scope_manifest(scope_name, *a, **k)

    monkeypatch.setattr(module, "build_scope_manifest", _fake_build_scope_manifest)
    report = _diag_report_all_scopes(worker)

    assert report["scope_summaries"]["lm_middle"]["applicable"] is False
    assert "28 LM layers" in report["scope_summaries"]["lm_middle"]["reason"]
    # every OTHER scope still reported normally -- the one failure didn't crash the sweep
    assert report["scope_summaries"]["vision_encoder"]["applicable"] is True
    assert report["scope_summaries"]["vision_encoder"]["selected_param_count"] > 0
    assert report["scope_summaries"]["full_lm"]["applicable"] is True
    assert report["scope_summaries"]["full_vlm"]["applicable"] is True


def test_diag_report_all_scopes_reports_tied_alias(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
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
    # measure_drift's chunked float64 accumulation (Stage-11 7B whole_model OOM fix -- see
    # thicket/memory_bounded_ops.py) is STRICTLY MORE precise than the float32 diff it replaces,
    # so it can reveal a genuine ~2^-24 (~5.96e-8) bf16-rounding discrepancy that float32
    # subtraction happened to round away to exactly 1.0 via round-half-to-even.
    assert drift["in_scope"]["max_abs_drift"] == pytest.approx(1.0, abs=1e-6)
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
def test_in_scope_and_out_of_scope_partition_full_vlm_for_every_scope(scope, runtime_wrapped_vlm_32vision_factory):
    """Same partition property as above, generalized across every registered scope -- in_scope
    and out_of_scope must always union to exactly full_vlm's own selection and never overlap.
    """
    model = runtime_wrapped_vlm_32vision_factory()
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


# --- main()'s own test coverage (source inspection -- main() itself needs a real GPU/ray/vllm
# engine and is not otherwise unit-testable, matching this project's established convention) ---


def test_run_isolation_test_or_skip_catches_scope_selection_error(monkeypatch):
    import neural_thickets_repro.diagnostics.scope_isolation_gpu_check as module
    from neural_thickets_repro.scopes import ScopeSelectionError

    def _fake_run_isolation_test(engine, label, scope):
        raise ScopeSelectionError(f"Cannot partition 28 LM layers ({scope})")

    monkeypatch.setattr(module, "_run_isolation_test", _fake_run_isolation_test)
    result = module._run_isolation_test_or_skip(engine=object(), test_label="B", scope="lm_middle")
    assert result["applicable"] is False
    assert result["pass"] is None
    assert "28 LM layers" in result["skip_reason"]
    assert result["scope"] == "lm_middle"


def test_run_isolation_test_or_skip_passes_through_a_real_result(monkeypatch):
    import neural_thickets_repro.diagnostics.scope_isolation_gpu_check as module

    fake_result = {"scope": "vision_early", "pass": True, "in_scope_changed": True, "out_of_scope_unchanged": True, "reset_exact": True}
    monkeypatch.setattr(module, "_run_isolation_test", lambda engine, label, scope: dict(fake_result))
    result = module._run_isolation_test_or_skip(engine=object(), test_label="C", scope="vision_early")
    assert result["applicable"] is True
    assert result["pass"] is True


def test_main_covers_all_three_iclr_causal_density_preregistered_scopes():
    """The iclr_causal_density pilot's preregistration.md requires vision_encoder/full_lm/
    full_vlm as its scope-isolation precondition -- vision_encoder was already Test A; this
    proves Tests H/I (full_lm/full_vlm) were added, additively, alongside every pre-existing
    test (never replacing them). Tests A/H/I (this pilot's required scopes) must be dispatched
    via the STRICT _run_isolation_test helper, never the _or_skip wrapper (see that wrapper's
    own docstring for why it exists ONLY for the pre-existing, architecture-dependent legacy
    scopes B-G) -- a real isolation failure on any of the three must never be silently
    downgraded to a skip.
    """
    import inspect

    from neural_thickets_repro.diagnostics import scope_isolation_gpu_check as module

    source = inspect.getsource(module.main)
    for label, scope in (("A", "vision_encoder"), ("H", "full_lm"), ("I", "full_vlm")):
        assert f'_run_isolation_test(engine, "{label}", "{scope}")' in source, f"Required Test {label} ({scope}) missing, or not strictly dispatched, in main()"
    for label, scope in (("B", "lm_middle"), ("C", "vision_early"), ("D", "vision_middle"), ("E", "vision_late"), ("F", "vision_late_a"), ("G", "vision_late_b")):
        assert f'_run_isolation_test_or_skip(engine, "{label}", "{scope}")' in source, f"Legacy Test {label} ({scope}) missing from main()"
    assert "test_h" in source and "test_i" in source
    assert "required_tests = {\"A\": test_a, \"H\": test_h, \"I\": test_i}" in source
    assert "required_pass = all(t[\"pass\"] for t in required_tests.values())" in source


def test_main_uses_launch_stage6_engine_not_launch_engines():
    """external/RandOpt's own launch_engines() accepts no max_model_len at all, so vLLM falls
    back to Qwen2.5-VL's full native context -- at 7B this OOMs the KV cache regardless of
    gpu_memory_utilization (confirmed live: still OOMs even at 0.60). launch_stage6_engine is
    this repo's own already-established fix for exactly this failure mode (STAGE6_GPU_MEMORY_
    UTILIZATION=0.60 + STAGE6_MAX_MODEL_LEN=4096) -- this diagnostic must use it, never
    external/RandOpt's own launch_engines (which is also never modified, per this repo's
    reproduction-integrity rule).
    """
    import inspect

    from neural_thickets_repro.diagnostics import scope_isolation_gpu_check as module

    source = inspect.getsource(module.main)
    assert "engines, pgs = launch_stage6_engine(" in source
    assert "engines, pgs = launch_engines(" not in source  # the actual dispatch call -- prose mentioning launch_engines() by name (explaining why it's NOT used) is fine
    assert "store_base_weights_via_rpc(engine)" in source  # launch_stage6_engine does NOT auto-store base weights on creation, unlike launch_engines


def test_main_module_imports_launch_stage6_engine_from_the_local_package_not_external():
    from neural_thickets_repro.diagnostics import scope_isolation_gpu_check as module

    assert module.launch_stage6_engine.__module__ == "neural_thickets_repro.run_global_visual_thicket_pilot"

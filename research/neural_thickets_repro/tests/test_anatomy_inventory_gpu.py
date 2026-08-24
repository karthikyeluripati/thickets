"""Tests for diagnostics/anatomy_inventory_gpu.py's pure-logic helpers -- exercised directly
against a fake worker/engine (SimpleNamespace + a real synthetic model), same no-GPU-needed
pattern as tests/test_scope_isolation_gpu_check.py. The real collective_rpc/vLLM/ray plumbing
needs the pod, same limitation noted throughout this project's diagnostics.

Also covers the RunPod KV-cache-OOM regression (max_model_len defaulting to the real model's
128000 via upstream launch_engines()) -- proves this diagnostic reuses Stage 6's own safe
launcher/engine-config instead of ever calling upstream launch_engines().
"""
import inspect
from types import SimpleNamespace

import pytest
import torch

import neural_thickets_repro.diagnostics.anatomy_inventory_gpu as anatomy_inventory_gpu
import neural_thickets_repro.run_global_visual_thicket_pilot as run_global_visual_thicket_pilot
from neural_thickets_repro.diagnostics.anatomy_inventory_gpu import (
    EMPIRICAL_CHECK_SCOPE,
    EMPIRICAL_CHECK_SEED,
    EMPIRICAL_CHECK_SIGMA,
    RestorationFailedError,
    _report_anatomy_and_upstream_scope,
    _run_empirical_norm_sanity_check,
    _validate_collective_rpc_results,
    maybe_run_empirical_check,
)
from neural_thickets_repro.perturb_cpu import should_perturb


def _identity_ray_get(x):
    return x


def _fake_worker(model):
    return SimpleNamespace(model_runner=SimpleNamespace(model=model))


def test_report_anatomy_and_upstream_scope_returns_both_sections(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    worker = _fake_worker(model)

    report = _report_anatomy_and_upstream_scope(worker)

    assert "anatomy_inventory" in report
    assert "upstream_scope_inventory" in report
    assert set(report["anatomy_inventory"]["regions"]) >= {"vision", "multimodal_connector_or_merger", "language"}
    assert report["upstream_scope_inventory"]["upstream_scope_vs_language_region"]["equals_language_region"] is True


def test_report_anatomy_and_upstream_scope_raises_on_degenerate_atlas(flat_checkpoint_vlm_factory):
    """FlatCheckpointVLM's synthetic vision tower has only 2 blocks -- not enough vision blocks
    to build the full atlas -- must propagate the underlying AnatomyDiscoveryError rather than
    silently returning a partial report.
    """
    from neural_thickets_repro.thicket.anatomy import AnatomyDiscoveryError

    model = flat_checkpoint_vlm_factory()
    worker = _fake_worker(model)
    with pytest.raises(AnatomyDiscoveryError):
        _report_anatomy_and_upstream_scope(worker)


# --- _validate_collective_rpc_results: same TP=1 shape-validation convention as elsewhere -----


def test_validate_collective_rpc_results_unwraps_single_worker_list():
    assert _validate_collective_rpc_results(["x"], label="m") == "x"


def test_validate_collective_rpc_results_rejects_non_list():
    with pytest.raises(RuntimeError, match="expected a list of per-worker results"):
        _validate_collective_rpc_results({"a": 1}, label="m")


def test_validate_collective_rpc_results_rejects_multi_worker_list():
    with pytest.raises(RuntimeError, match="TP=1-only"):
        _validate_collective_rpc_results(["a", "b"], label="m")


# =================================================================================================
# KV-cache-OOM regression: prove Stage 7A reuses Stage 6's safe launcher, never upstream
# launch_engines(), and never falls back to the real model's 128000-token default.
# =================================================================================================


def test_never_imports_or_calls_upstream_launch_engines():
    """Checks main()'s own CODE for the actual import/call shape (not prose comments, which
    may legitimately name "launch_engines" when explaining why it's avoided).
    """
    main_source = inspect.getsource(anatomy_inventory_gpu.main)
    assert "launch_stage6_engine(" in main_source  # the REPLACEMENT must actually be called
    assert "import launch_engines" not in main_source  # the upstream function must never be imported
    assert "launch_engines(" not in main_source  # nor called under any alias


def test_module_never_imports_upstream_launch_engines_anywhere():
    full_source = inspect.getsource(anatomy_inventory_gpu)
    assert "import launch_engines" not in full_source


def test_reuses_stage6_launcher_and_config_by_identity_not_duplication():
    """Same function OBJECTS as run_global_visual_thicket_pilot's own -- proves reuse, not a
    second, independently-maintained copy of the engine-construction logic.
    """
    assert anatomy_inventory_gpu.launch_stage6_engine is run_global_visual_thicket_pilot.launch_stage6_engine
    assert anatomy_inventory_gpu.build_stage6_engine_config is run_global_visual_thicket_pilot.build_stage6_engine_config
    assert anatomy_inventory_gpu.resolve_and_report_model_snapshot is run_global_visual_thicket_pilot.resolve_and_report_model_snapshot
    assert anatomy_inventory_gpu.store_base_weights_via_rpc is run_global_visual_thicket_pilot.store_base_weights_via_rpc
    assert anatomy_inventory_gpu.reset_to_base_weights_via_rpc is run_global_visual_thicket_pilot.reset_to_base_weights_via_rpc
    assert anatomy_inventory_gpu.verify_exact_fixed_base_restoration_via_rpc is run_global_visual_thicket_pilot.verify_exact_fixed_base_restoration_via_rpc


def test_stage6_engine_config_never_falls_back_to_model_default_context_length():
    """Regression test for the exact OOM: 128000 (the real model's own max_position_embeddings)
    must never appear as this diagnostic's max_model_len -- it must be the Stage-6-frozen 4096,
    and gpu_memory_utilization must be the Stage-6-frozen 0.60, not upstream launch_engines()'s
    implicit 0.75 default.
    """
    engine_config = anatomy_inventory_gpu.build_stage6_engine_config()
    assert engine_config["max_model_len"] == 4096
    assert engine_config["max_model_len"] != 128000
    assert engine_config["gpu_memory_utilization"] == 0.60
    assert engine_config["tensor_parallel_size"] == 1


def test_no_dataset_evaluation_is_introduced():
    """Source-level guard: this diagnostic must never gain a dependency on the benchmark/
    dataset machinery -- Sections 1-5 are pure weight inspection plus one optional
    single-perturbation norm measurement, never a capability evaluation.
    """
    source = inspect.getsource(anatomy_inventory_gpu)
    forbidden = ("run_benchmark", "build_d_map_context", "benchmarks.runner", "CapabilityContext", "D_map", "SamplingParams")
    for token in forbidden:
        assert token not in source, f"found forbidden dataset-evaluation reference {token!r}"


def test_empirical_check_constants_are_frozen_single_seed_single_sigma():
    assert EMPIRICAL_CHECK_SIGMA == 0.001
    assert isinstance(EMPIRICAL_CHECK_SEED, int)
    assert EMPIRICAL_CHECK_SCOPE == "full_lm"


# =================================================================================================
# maybe_run_empirical_check: base-snapshot lifecycle (store_base_weights only when enabled,
# exactly once; never stored at all when skipped)
# =================================================================================================


class _FakeStage7AEngine:
    """Persistent-worker-shaped fake (worker_self built ONCE, not per-call) using a REAL small
    synthetic model -- same philosophy as test_run_stage7b_anatomical_calibration.py's
    _FakeCalibrationEngine. Tracks store/reset call counts explicitly.
    """

    def __init__(self, model, *, break_reset: bool = False):
        self._model = model
        self._base_weights = None
        self._break_reset = break_reset
        self.store_base_weights_calls = 0
        self.reset_calls = 0
        self._worker_self = SimpleNamespace(
            model_runner=SimpleNamespace(model=model),
            reset_to_base_weights=self._reset_to_base_weights,
            _should_perturb=should_perturb,
        )
        self.collective_rpc = SimpleNamespace(remote=self._collective_rpc)

    def _store_base_weights(self):
        self.store_base_weights_calls += 1
        self._base_weights = {name: p.detach().clone() for name, p in self._model.named_parameters()}
        self._worker_self._base_weights = self._base_weights

    def _reset_to_base_weights(self):
        self.reset_calls += 1
        if self._base_weights is None:
            raise RuntimeError("store_base_weights not called")
        if self._break_reset:
            return  # simulate a broken reset that leaves weights perturbed
        with torch.no_grad():
            for name, p in self._model.named_parameters():
                p.copy_(self._base_weights[name])

    def _collective_rpc(self, method, args=()):
        if method == "store_base_weights":
            self._store_base_weights()
            return [True]
        if method == "reset_to_base_weights":
            self._reset_to_base_weights()
            return [True]
        if callable(method):
            return [method(self._worker_self, *args)]
        raise ValueError(f"unsupported method {method!r}")


def test_maybe_run_empirical_check_returns_none_and_never_stores_base_when_skipped(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    engine = _FakeStage7AEngine(model)

    result = maybe_run_empirical_check(
        engine, skip_empirical_check=True, gpu_memory_utilization=0.60, base_snapshot_mode="store_base_weights", ray_get=_identity_ray_get,
    )

    assert result is None
    assert engine.store_base_weights_calls == 0


def test_maybe_run_empirical_check_stores_base_exactly_once_when_enabled(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    engine = _FakeStage7AEngine(model)

    result = maybe_run_empirical_check(
        engine, skip_empirical_check=False, gpu_memory_utilization=0.60, base_snapshot_mode="store_base_weights", ray_get=_identity_ray_get,
    )

    assert engine.store_base_weights_calls == 1
    assert result is not None
    assert result["seed"] == EMPIRICAL_CHECK_SEED
    assert result["sigma"] == EMPIRICAL_CHECK_SIGMA
    assert result["scope"] == EMPIRICAL_CHECK_SCOPE
    assert result["restoration_verified_exact"] is True


def test_maybe_run_empirical_check_resets_to_base_after_perturbation(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    engine = _FakeStage7AEngine(model)

    maybe_run_empirical_check(
        engine, skip_empirical_check=False, gpu_memory_utilization=0.60, base_snapshot_mode="store_base_weights", ray_get=_identity_ray_get,
    )

    assert engine.reset_calls >= 1
    for name, p in model.named_parameters():
        assert torch.equal(p.detach(), engine._base_weights[name])


def test_maybe_run_empirical_check_raises_on_restoration_failure(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    engine = _FakeStage7AEngine(model, break_reset=True)

    with pytest.raises(RestorationFailedError):
        maybe_run_empirical_check(
            engine, skip_empirical_check=False, gpu_memory_utilization=0.60, base_snapshot_mode="store_base_weights", ray_get=_identity_ray_get,
        )


def test_run_empirical_norm_sanity_check_requires_base_weights_stored(runtime_wrapped_vlm_32vision_factory):
    """Calling the empirical check WITHOUT store_base_weights first must fail loudly (the
    restoration-verification step has nothing to compare against) rather than silently skipping
    verification.
    """
    model = runtime_wrapped_vlm_32vision_factory()
    engine = _FakeStage7AEngine(model)  # store_base_weights never called
    with pytest.raises(RuntimeError, match="_base_weights"):
        _run_empirical_norm_sanity_check(engine, ray_get=_identity_ray_get)


def test_run_empirical_norm_sanity_check_realized_close_to_analytical(runtime_wrapped_vlm_32vision_factory):
    """A raw_sigma perturbation's realized L2 norm is a random sample (not an exact rescale --
    that's the anatomical_relative_l2 mode's job), so this only checks the reported fields are
    internally consistent and in the right ballpark, not bit-exact equality.
    """
    model = runtime_wrapped_vlm_32vision_factory()
    engine = _FakeStage7AEngine(model)
    engine._store_base_weights()

    result = _run_empirical_norm_sanity_check(engine, ray_get=_identity_ray_get)

    assert result["seed"] == EMPIRICAL_CHECK_SEED
    assert result["sigma"] == EMPIRICAL_CHECK_SIGMA
    assert result["scope"] == EMPIRICAL_CHECK_SCOPE
    assert result["theta_l2_norm"] > 0
    assert result["realized_epsilon_l2_norm"] > 0
    assert result["realized_relative_l2"] == pytest.approx(result["realized_epsilon_l2_norm"] / result["theta_l2_norm"])
    assert result["absolute_difference"] == pytest.approx(abs(result["realized_relative_l2"] - result["analytical_expected_relative_l2"]))

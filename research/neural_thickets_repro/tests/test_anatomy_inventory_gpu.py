"""Tests for diagnostics/anatomy_inventory_gpu.py's pure-logic helpers -- exercised directly
against a fake worker/engine (SimpleNamespace + a real synthetic model), same no-GPU-needed
pattern as tests/test_scope_isolation_gpu_check.py. The real collective_rpc/vLLM/ray plumbing
needs the pod, same limitation noted throughout this project's diagnostics.
"""
from types import SimpleNamespace

import pytest

from neural_thickets_repro.diagnostics.anatomy_inventory_gpu import (
    EMPIRICAL_CHECK_SEED,
    EMPIRICAL_CHECK_SIGMA,
    _report_anatomy_and_upstream_scope,
    _run_empirical_norm_sanity_check,
    _validate_collective_rpc_results,
)


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


# --- _run_empirical_norm_sanity_check: real math against a real (small) synthetic model -------


class _FakeSanityCheckEngine:
    """Executes a Callable method against a real worker_self wrapping a real synthetic model --
    just enough to exercise scoped_perturbation.scoped_apply_perturbation's real math, and
    dispatches the string "reset_to_base_weights" as a no-op (this diagnostic never calls
    store_base_weights, so there is no stored base to reset to -- matches the real
    scoped_apply_perturbation contract, which itself calls worker_self.reset_to_base_weights()
    defensively before perturbing).
    """

    def __init__(self, model):
        self._model = model
        self.collective_rpc = SimpleNamespace(remote=self._collective_rpc)

    def _collective_rpc(self, method, args=()):
        worker_self = SimpleNamespace(
            model_runner=SimpleNamespace(model=self._model),
            reset_to_base_weights=lambda: None,
        )
        if method == "reset_to_base_weights":
            return [True]
        if callable(method):
            return [method(worker_self, *args)]
        raise ValueError(f"unsupported method {method!r}")


def test_run_empirical_norm_sanity_check_realized_close_to_analytical(runtime_wrapped_vlm_32vision_factory):
    """A raw_sigma perturbation's realized L2 norm is a random sample (not an exact rescale --
    that's the anatomical_relative_l2 mode's job), so this only checks the reported fields are
    internally consistent and in the right ballpark, not bit-exact equality.
    """
    model = runtime_wrapped_vlm_32vision_factory()
    engine = _FakeSanityCheckEngine(model)

    result = _run_empirical_norm_sanity_check(engine, ray_get=_identity_ray_get)

    assert result["seed"] == EMPIRICAL_CHECK_SEED
    assert result["sigma"] == EMPIRICAL_CHECK_SIGMA
    assert result["scope"] == "full_lm"
    assert result["theta_l2_norm"] > 0
    assert result["realized_epsilon_l2_norm"] > 0
    assert result["realized_relative_l2"] == pytest.approx(result["realized_epsilon_l2_norm"] / result["theta_l2_norm"])
    assert result["absolute_difference"] == pytest.approx(abs(result["realized_relative_l2"] - result["analytical_expected_relative_l2"]))

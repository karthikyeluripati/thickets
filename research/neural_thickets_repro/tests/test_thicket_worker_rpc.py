"""Tests for thicket/worker_rpc.py against a plain duck-typed fake worker -- no GPU/vllm/ray/
external-RandOpt import needed, matching the module's own zero-heavy-dependency design.
"""
from types import SimpleNamespace

import pytest

from neural_thickets_repro.thicket.worker_rpc import (
    compute_perturbable_mask_info_rpc,
    compute_restoration_fingerprint_rpc,
    verify_restoration_rpc,
)


class _FakeTensor:
    def __init__(self, values):
        self._values = values

    def numel(self):
        return len(self._values)

    def detach(self):
        return self

    def float(self):
        return self

    def norm(self):
        return _FakeScalar(sum(v * v for v in self._values) ** 0.5)


class _FakeScalar:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


def _fake_worker(named_params, visual_prefixes=("visual.",)):
    model = SimpleNamespace(named_parameters=lambda: [(name, _FakeTensor(values)) for name, values in named_params])
    model_runner = SimpleNamespace(model=model)
    return SimpleNamespace(
        model_runner=model_runner,
        _should_perturb=lambda name: not name.startswith(visual_prefixes),
    )


def _params():
    return [
        ("visual.blocks.0.weight", [1.0, 2.0]),
        ("model.layers.0.weight", [3.0, 4.0]),
        ("model.layers.1.weight", [1.0, 1.0, 1.0]),
    ]


# --- compute_perturbable_mask_info_rpc -------------------------------------------------------


def test_mask_info_excludes_visual_params():
    worker = _fake_worker(_params())
    info = compute_perturbable_mask_info_rpc(worker)
    assert info["param_count"] == 2  # only the two model.layers.* tensors
    assert info["total_elements"] == 2 + 3


def test_mask_info_hash_is_deterministic_and_order_independent():
    worker_a = _fake_worker(_params())
    worker_b = _fake_worker(list(reversed(_params())))
    assert compute_perturbable_mask_info_rpc(worker_a)["mask_hash"] == compute_perturbable_mask_info_rpc(worker_b)["mask_hash"]


def test_mask_info_hash_changes_with_membership():
    worker_a = _fake_worker(_params())
    worker_b = _fake_worker(_params() + [("model.layers.2.weight", [9.0])])
    assert compute_perturbable_mask_info_rpc(worker_a)["mask_hash"] != compute_perturbable_mask_info_rpc(worker_b)["mask_hash"]


def test_mask_info_never_returns_raw_parameter_names_or_tensors():
    worker = _fake_worker(_params())
    info = compute_perturbable_mask_info_rpc(worker)
    assert set(info) == {"mask_hash", "param_count", "total_elements"}


# --- compute_restoration_fingerprint_rpc -----------------------------------------------------


def test_fingerprint_only_covers_perturbable_params():
    worker = _fake_worker(_params())
    fingerprint = compute_restoration_fingerprint_rpc(worker)
    assert set(fingerprint) == {"model.layers.0.weight", "model.layers.1.weight"}


def test_fingerprint_values_are_l2_norms():
    worker = _fake_worker(_params())
    fingerprint = compute_restoration_fingerprint_rpc(worker)
    assert fingerprint["model.layers.0.weight"] == pytest.approx((3.0 ** 2 + 4.0 ** 2) ** 0.5)
    assert fingerprint["model.layers.1.weight"] == pytest.approx(3.0 ** 0.5)


# --- verify_restoration_rpc: the restoration invariant ----------------------------------------


def test_verify_restoration_passes_when_unchanged():
    worker = _fake_worker(_params())
    base = compute_restoration_fingerprint_rpc(worker)
    result = verify_restoration_rpc(worker, base, atol=1e-9, rtol=0.0)
    assert result["ok"] is True
    assert result["n_failing"] == 0


def test_verify_restoration_fails_on_drift_beyond_tolerance():
    worker = _fake_worker(_params())
    base = compute_restoration_fingerprint_rpc(worker)
    drifted = _fake_worker([("visual.blocks.0.weight", [1.0, 2.0]), ("model.layers.0.weight", [30.0, 40.0]), ("model.layers.1.weight", [1.0, 1.0, 1.0])])
    result = verify_restoration_rpc(drifted, base, atol=1e-6, rtol=1e-6)
    assert result["ok"] is False
    assert "model.layers.0.weight" in result["worst_offenders"]


def test_verify_restoration_tolerates_within_rtol_band():
    worker = _fake_worker(_params())
    base = compute_restoration_fingerprint_rpc(worker)
    # model.layers.0.weight base norm = 5.0; nudge to 5.0005 -- within a generous rtol band.
    nudged = _fake_worker([("visual.blocks.0.weight", [1.0, 2.0]), ("model.layers.0.weight", [3.0003, 4.0004]), ("model.layers.1.weight", [1.0, 1.0, 1.0])])
    result = verify_restoration_rpc(nudged, base, atol=1e-4, rtol=1e-3)
    assert result["ok"] is True


def test_verify_restoration_never_returns_full_tensors():
    worker = _fake_worker(_params())
    base = compute_restoration_fingerprint_rpc(worker)
    result = verify_restoration_rpc(worker, base, atol=1e-9, rtol=0.0)
    assert set(result) == {"ok", "max_diff", "n_checked", "n_failing", "worst_offenders"}


def test_verify_restoration_missing_tensor_counts_as_failing():
    worker = _fake_worker(_params())
    base = compute_restoration_fingerprint_rpc(worker)
    base["a_tensor_that_no_longer_exists"] = 42.0
    result = verify_restoration_rpc(worker, base, atol=1e-9, rtol=0.0)
    assert result["ok"] is False
    assert result["n_failing"] == 1

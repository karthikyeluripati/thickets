"""Tests for thicket/worker_rpc.py. compute_perturbable_mask_info_rpc uses a plain duck-typed
fake worker (no GPU/vllm/ray/external-RandOpt import needed). verify_exact_fixed_base_
restoration_rpc needs real tensor arithmetic (measure_drift calls .pow()/!=/.abs()/.max() on
real torch tensors), so it's tested against the existing synthetic dummy_vlm fixture instead
(same no-GPU-needed philosophy, just real small CPU tensors rather than a duck-typed double).
"""
from types import SimpleNamespace

import pytest
import torch

from neural_thickets_repro.thicket.worker_rpc import (
    compute_perturbable_mask_info_rpc,
    verify_exact_fixed_base_restoration_rpc,
)


class _FakeTensor:
    def __init__(self, values):
        self._values = values

    def numel(self):
        return len(self._values)


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


# --- verify_exact_fixed_base_restoration_rpc: the fixed-base restoration invariant ------------
#
# Root cause this replaces: a real 384-candidate RunPod run proved the OLD tolerance-based
# per-tensor-norm check (paired with restore_self_weights' native-BF16 regenerate-and-subtract
# restoration) was insufficient -- restoration is now via store_base_weights/
# reset_to_base_weights (a direct tensor copy), which must be held to an EXACT standard.


def _worker_from_model(model, visual_prefixes=("visual.",)):
    return SimpleNamespace(model_runner=SimpleNamespace(model=model), _should_perturb=lambda name: not name.startswith(visual_prefixes))


def test_verify_exact_restoration_raises_without_stored_base_weights(dummy_vlm_factory):
    model = dummy_vlm_factory()
    worker = _worker_from_model(model)  # no _base_weights attribute at all
    with pytest.raises(RuntimeError, match="store_base_weights"):
        verify_exact_fixed_base_restoration_rpc(worker)


def test_verify_exact_restoration_passes_when_current_exactly_matches_base(dummy_vlm_factory):
    model = dummy_vlm_factory()
    worker = _worker_from_model(model)
    worker._base_weights = {name: p.detach().clone() for name, p in model.named_parameters()}

    result = verify_exact_fixed_base_restoration_rpc(worker)

    assert result["ok"] is True
    assert result["max_abs_drift"] == 0.0
    assert result["fraction_elements_differing"] == 0.0


def test_verify_exact_restoration_fails_on_any_drift_in_a_perturbable_tensor(dummy_vlm_factory):
    model = dummy_vlm_factory()
    worker = _worker_from_model(model)
    worker._base_weights = {name: p.detach().clone() for name, p in model.named_parameters()}

    with torch.no_grad():
        for name, p in model.named_parameters():
            if not name.startswith("visual."):
                p.add_(0.001)
                break

    result = verify_exact_fixed_base_restoration_rpc(worker)

    assert result["ok"] is False
    assert result["max_abs_drift"] > 0.0


def test_verify_exact_restoration_ignores_drift_in_non_perturbable_visual_tensors(dummy_vlm_factory):
    """The check is restricted to `_should_perturb`-selected tensors, matching the exact mask
    global_gaussian_upstream perturbs -- a visual-tower tensor drifting (which should never
    happen in practice, since it's never perturbed) must not itself trip a false failure here.
    """
    model = dummy_vlm_factory()
    worker = _worker_from_model(model)
    worker._base_weights = {name: p.detach().clone() for name, p in model.named_parameters()}

    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.startswith("visual."):
                p.add_(1.0)
                break

    result = verify_exact_fixed_base_restoration_rpc(worker)
    assert result["ok"] is True


def test_verify_exact_restoration_never_returns_full_tensors(dummy_vlm_factory):
    model = dummy_vlm_factory()
    worker = _worker_from_model(model)
    worker._base_weights = {name: p.detach().clone() for name, p in model.named_parameters()}

    result = verify_exact_fixed_base_restoration_rpc(worker)
    assert set(result) == {"ok", "max_abs_drift", "fraction_elements_differing"}

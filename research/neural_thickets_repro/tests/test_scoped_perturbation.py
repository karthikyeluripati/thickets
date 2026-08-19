"""Tests for scoped_perturbation.py against a fake worker (SimpleNamespace + a
reset_to_base_weights stand-in), same fake-worker pattern used in
tests/test_gate2_restoration_ab.py. No ray/vllm/GPU needed -- scoped_apply_perturbation()
only ever calls worker_self.reset_to_base_weights() (an ordinary Python method here, exactly
as it is a real WorkerExtension method on the pod) and reads worker_self.model_runner.model.
"""
from types import SimpleNamespace

import pytest
import torch

from neural_thickets_repro.scoped_perturbation import NOISE_SEMANTICS, scoped_apply_perturbation
from neural_thickets_repro.scopes import compute_relative_l2_sigma


class _FakeWorker:
    """Stands in for a real vLLM worker + mixed-in WorkerExtension: exposes
    model_runner.model (a real, synthetic nn.Module) and a reset_to_base_weights() that
    copies a stored base state_dict back -- exactly what the real upstream method does
    (SCOPED_PERTURBATION_DESIGN.md's WorkerExtension investigation), just implemented here
    with plain torch so it works without ray/vllm.
    """

    def __init__(self, model):
        self.model_runner = SimpleNamespace(model=model)
        self._base_state = {k: v.clone() for k, v in model.state_dict().items()}

    def reset_to_base_weights(self) -> None:
        with torch.no_grad():
            for k, v in self.model_runner.model.state_dict().items():
                v.copy_(self._base_state[k])


def test_raw_sigma_only_changes_scope_selected_params(runtime_wrapped_vlm_factory):
    model = runtime_wrapped_vlm_factory()
    worker = _FakeWorker(model)
    base_state = {k: v.clone() for k, v in model.state_dict().items()}

    scoped_apply_perturbation(worker, seed=42, sigma_or_r=0.1, scope_name="vision_encoder", scale_mode="raw_sigma")

    for name, p in model.named_parameters():
        changed = not torch.equal(p.detach(), base_state[name])
        if name.startswith("visual.") and "merger" not in name:
            assert changed, f"expected {name} (vision_encoder) to change"
        else:
            assert not changed, f"expected {name} to be UNCHANGED (outside vision_encoder scope)"


def test_lm_middle_only_changes_middle_layer_params(runtime_wrapped_vlm_factory):
    model = runtime_wrapped_vlm_factory()
    worker = _FakeWorker(model)
    base_state = {k: v.clone() for k, v in model.state_dict().items()}

    scoped_apply_perturbation(worker, seed=42, sigma_or_r=0.1, scope_name="lm_middle", scale_mode="raw_sigma")

    middle_layer_prefixes = tuple(f"language_model.model.layers.{i}." for i in range(4, 8))
    for name, p in model.named_parameters():
        changed = not torch.equal(p.detach(), base_state[name])
        if name.startswith(middle_layer_prefixes):
            assert changed, f"expected {name} (lm_middle) to change"
        else:
            assert not changed, f"expected {name} to be UNCHANGED (outside lm_middle scope)"


def test_reset_to_base_weights_restores_exactly(runtime_wrapped_vlm_factory):
    model = runtime_wrapped_vlm_factory()
    worker = _FakeWorker(model)
    base_state = {k: v.clone() for k, v in model.state_dict().items()}

    scoped_apply_perturbation(worker, seed=42, sigma_or_r=0.1, scope_name="lm_early", scale_mode="raw_sigma")
    worker.reset_to_base_weights()

    for name, p in model.named_parameters():
        assert torch.equal(p.detach(), base_state[name]), f"{name} did not restore exactly"


def test_defensive_reset_before_perturb_ignores_prior_manual_drift(runtime_wrapped_vlm_factory):
    """scoped_apply_perturbation calls reset_to_base_weights() itself before perturbing --
    even if the model was manually drifted beforehand (simulating a caller-discipline bug),
    the result must be identical to perturbing from a clean instance.
    """
    model_a = runtime_wrapped_vlm_factory()
    worker_a = _FakeWorker(model_a)
    scoped_apply_perturbation(worker_a, seed=7, sigma_or_r=0.05, scope_name="vision_merger", scale_mode="raw_sigma")

    model_b = runtime_wrapped_vlm_factory()
    worker_b = _FakeWorker(model_b)
    with torch.no_grad():
        for p in model_b.parameters():
            p.add_(999.0)  # simulate prior drift that scoped_apply_perturbation must undo first
    scoped_apply_perturbation(worker_b, seed=7, sigma_or_r=0.05, scope_name="vision_merger", scale_mode="raw_sigma")

    for (name_a, p_a), (name_b, p_b) in zip(model_a.named_parameters(), model_b.named_parameters()):
        assert name_a == name_b
        assert torch.equal(p_a.detach(), p_b.detach()), f"{name_a} differs despite identical seed/scope/scale"


def test_relative_l2_derives_sigma_from_manifest(runtime_wrapped_vlm_factory):
    model = runtime_wrapped_vlm_factory()
    worker = _FakeWorker(model)

    result = scoped_apply_perturbation(worker, seed=1, sigma_or_r=0.02, scope_name="vision_encoder", scale_mode="relative_l2")

    expected_sigma = compute_relative_l2_sigma(
        result["scope_base_l2_norm"], result["scope_total_element_count"], r=0.02,
    )
    assert result["derived_sigma"] == pytest.approx(expected_sigma)
    assert result["requested_relative_l2"] == 0.02


def test_raw_sigma_mode_derived_sigma_equals_requested(runtime_wrapped_vlm_factory):
    model = runtime_wrapped_vlm_factory()
    worker = _FakeWorker(model)
    result = scoped_apply_perturbation(worker, seed=1, sigma_or_r=0.03, scope_name="full_lm", scale_mode="raw_sigma")
    assert result["derived_sigma"] == 0.03
    assert result["requested_relative_l2"] is None


def test_result_dict_records_noise_semantics_and_scope_info(runtime_wrapped_vlm_factory):
    model = runtime_wrapped_vlm_factory()
    worker = _FakeWorker(model)
    result = scoped_apply_perturbation(worker, seed=1, sigma_or_r=0.01, scope_name="lm_late", scale_mode="raw_sigma")
    assert result["noise_semantics"] == NOISE_SEMANTICS == "upstream_per_tensor_reseed"
    assert result["scope"] == "lm_late"
    assert result["scale_mode"] == "raw_sigma"
    assert result["scope_param_count"] > 0
    assert result["actual_perturbation_l2"] > 0.0


def test_unknown_scale_mode_raises(runtime_wrapped_vlm_factory):
    model = runtime_wrapped_vlm_factory()
    worker = _FakeWorker(model)
    with pytest.raises(ValueError, match="Unknown perturbation_scale_mode"):
        scoped_apply_perturbation(worker, seed=1, sigma_or_r=0.01, scope_name="full_lm", scale_mode="not_a_mode")


def test_same_seed_scope_scale_deterministic(runtime_wrapped_vlm_factory):
    model_a = runtime_wrapped_vlm_factory()
    worker_a = _FakeWorker(model_a)
    scoped_apply_perturbation(worker_a, seed=123, sigma_or_r=0.01, scope_name="lm_early", scale_mode="raw_sigma")

    model_b = runtime_wrapped_vlm_factory()
    worker_b = _FakeWorker(model_b)
    scoped_apply_perturbation(worker_b, seed=123, sigma_or_r=0.01, scope_name="lm_early", scale_mode="raw_sigma")

    for (name_a, p_a), (name_b, p_b) in zip(model_a.named_parameters(), model_b.named_parameters()):
        assert torch.equal(p_a.detach(), p_b.detach()), f"{name_a} not deterministic"


@pytest.mark.parametrize("scope,in_scope_blocks", [
    ("vision_early", range(0, 11)),
    ("vision_middle", range(11, 22)),
    ("vision_late", range(22, 32)),
])
def test_vision_thirds_only_change_their_own_selected_params(scope, in_scope_blocks, runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    worker = _FakeWorker(model)
    base_state = {k: v.clone() for k, v in model.state_dict().items()}
    in_scope_blocks = set(in_scope_blocks)

    scoped_apply_perturbation(worker, seed=42, sigma_or_r=0.1, scope_name=scope, scale_mode="raw_sigma")

    for name, p in model.named_parameters():
        changed = not torch.equal(p.detach(), base_state[name])
        if name.startswith("visual.blocks."):
            block_idx = int(name.split(".")[2])
            expected_change = block_idx in in_scope_blocks
        elif scope == "vision_early" and (name.startswith("visual.patch_embed.") or name.startswith("visual.rotary_pos_emb.")):
            expected_change = True
        else:
            expected_change = False
        assert changed == expected_change, f"{name}: changed={changed}, expected={expected_change} (scope={scope})"


def test_vision_thirds_restore_exactly(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    worker = _FakeWorker(model)
    base_state = {k: v.clone() for k, v in model.state_dict().items()}

    scoped_apply_perturbation(worker, seed=42, sigma_or_r=0.1, scope_name="vision_middle", scale_mode="raw_sigma")
    worker.reset_to_base_weights()

    for name, p in model.named_parameters():
        assert torch.equal(p.detach(), base_state[name]), f"{name} did not restore exactly"


def test_vision_thirds_relative_l2_sigma_derived_independently(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    worker = _FakeWorker(model)

    results = {}
    for scope in ("vision_early", "vision_middle", "vision_late"):
        results[scope] = scoped_apply_perturbation(worker, seed=1, sigma_or_r=0.04, scope_name=scope, scale_mode="relative_l2")

    # Each scope's derived sigma must come from ITS OWN manifest (norm/dimension), not a
    # shared/inherited vision_encoder-level value -- proven by checking each result's sigma
    # matches the closed form applied to that same result's own reported norm/count, and that
    # not all three scopes (different sizes: 11/11/10 blocks, early also has patch_embed +
    # rotary_pos_emb) landed on the identical sigma.
    for scope, result in results.items():
        expected = compute_relative_l2_sigma(result["scope_base_l2_norm"], result["scope_total_element_count"], r=0.04)
        assert result["derived_sigma"] == pytest.approx(expected)

    sigmas = {scope: r["derived_sigma"] for scope, r in results.items()}
    assert len(set(sigmas.values())) > 1, f"expected scope-specific sigmas, got identical values: {sigmas}"


def test_different_seed_produces_different_perturbation(runtime_wrapped_vlm_factory):
    model_a = runtime_wrapped_vlm_factory()
    worker_a = _FakeWorker(model_a)
    scoped_apply_perturbation(worker_a, seed=1, sigma_or_r=0.05, scope_name="lm_early", scale_mode="raw_sigma")

    model_b = runtime_wrapped_vlm_factory()
    worker_b = _FakeWorker(model_b)
    scoped_apply_perturbation(worker_b, seed=2, sigma_or_r=0.05, scope_name="lm_early", scale_mode="raw_sigma")

    any_diff = any(
        not torch.equal(p_a.detach(), p_b.detach())
        for (_, p_a), (_, p_b) in zip(model_a.named_parameters(), model_b.named_parameters())
    )
    assert any_diff

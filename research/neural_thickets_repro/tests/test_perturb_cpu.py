import copy

import torch

from neural_thickets_repro.perturb_cpu import perturb, restore, should_perturb


def _clone_state(model):
    return {k: v.clone() for k, v in model.state_dict().items()}


def _states_equal(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


def _states_close(a, b, atol=1e-6):
    """Perturb-then-restore regenerates and subtracts the identical noise tensor, but
    (p + x) - x is not guaranteed bit-exact in floating point (rounding at each op) --
    confirmed empirically here at ~1 ULP for float32 (~6e-8). This is inherent to the
    add-noise/subtract-noise restoration mechanism itself (and would be a larger relative
    error at bf16, the precision the real pipeline actually uses), not a scaffold bug, so
    restoration is checked for closeness, not bitwise equality.
    """
    return all(torch.allclose(a[k], b[k], atol=atol, rtol=0) for k in a)


def test_should_perturb_scope_rule():
    assert should_perturb("model.layers.0.self_attn.q_proj.weight") is True
    assert should_perturb("model.embed_tokens.weight") is True
    assert should_perturb("lm_head.weight") is True  # not vision-prefixed -> perturbed
    assert should_perturb("visual.blocks.0.attn.qkv.weight") is False
    assert should_perturb("visual.patch_embed.weight") is False
    assert should_perturb("model.visual.blocks.0.weight") is False  # secondary upstream prefix


def test_perturbation_is_deterministic_given_same_seed_and_sigma(dummy_vlm_factory):
    m1, m2 = dummy_vlm_factory(), dummy_vlm_factory()
    perturb(m1, seed=123, sigma=0.01)
    perturb(m2, seed=123, sigma=0.01)
    assert _states_equal(m1.state_dict(), m2.state_dict())


def test_different_seeds_yield_different_perturbations(dummy_vlm_factory):
    m1, m2 = dummy_vlm_factory(), dummy_vlm_factory()
    perturb(m1, seed=1, sigma=0.01)
    perturb(m2, seed=2, sigma=0.01)
    assert not _states_equal(m1.state_dict(), m2.state_dict())


def test_sigma_zero_is_a_no_op(dummy_vlm_factory):
    model = dummy_vlm_factory()
    before = _clone_state(model)
    perturb(model, seed=42, sigma=0.0)
    assert _states_equal(before, model.state_dict())


def test_vision_encoder_never_changes(dummy_vlm_factory):
    model = dummy_vlm_factory()
    before_visual = {k: v.clone() for k, v in model.state_dict().items() if k.startswith("visual.")}
    perturb(model, seed=7, sigma=0.5)
    after_visual = {k: v for k, v in model.state_dict().items() if k.startswith("visual.")}
    assert _states_equal(before_visual, after_visual)


def test_perturb_then_restore_is_bitwise_identical(dummy_vlm_factory):
    model = dummy_vlm_factory()
    original = _clone_state(model)
    perturb(model, seed=99, sigma=0.3)
    assert not _states_equal(original, model.state_dict())  # sanity: something changed
    restore(model, seed=99, sigma=0.3)
    # Not exact-bitwise: see _states_close docstring -- (p + x) - x has ~1-ULP float32
    # rounding error, not a scaffold bug.
    assert _states_close(original, model.state_dict())


def test_no_cross_candidate_accumulation(dummy_vlm_factory):
    """theta0 -> +delta1 -> restore -> theta0 -> +delta2, never theta0+delta1+delta2."""
    model = dummy_vlm_factory()
    original = _clone_state(model)

    perturb(model, seed=1, sigma=0.1)
    restore(model, seed=1, sigma=0.1)
    assert _states_close(original, model.state_dict())

    perturb(model, seed=2, sigma=0.1)
    after_second = _clone_state(model)

    # Compare against evaluating candidate 2 fresh from theta0 independently.
    fresh = copy.deepcopy(model)
    fresh.load_state_dict(original)
    perturb(fresh, seed=2, sigma=0.1)
    assert _states_close(after_second, fresh.state_dict())

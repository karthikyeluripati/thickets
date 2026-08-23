import numpy as np
import pytest

from neural_thickets_repro.thicket.geometry import PerturbationVectorHandle, effective_rank, principal_angles
from neural_thickets_repro.thicket.perturbation import PerturbationManifest


def test_effective_rank_of_a_single_singular_value_is_one():
    assert effective_rank(np.array([5.0])) == pytest.approx(1.0)


def test_effective_rank_of_k_equal_singular_values_is_k():
    assert effective_rank(np.array([2.0, 2.0, 2.0, 2.0])) == pytest.approx(4.0)


def test_effective_rank_of_all_zeros_is_zero():
    assert effective_rank(np.array([0.0, 0.0])) == pytest.approx(0.0)


def test_effective_rank_is_between_one_and_full_rank_for_a_skewed_spectrum():
    s = np.array([10.0, 1.0, 0.5, 0.1])
    er = effective_rank(s)
    assert 1.0 < er < len(s)


def test_principal_angles_of_identical_subspaces_are_zero():
    rng = np.random.default_rng(0)
    basis = rng.normal(size=(20, 3))
    angles = principal_angles(basis, basis)
    assert np.allclose(angles, 0.0, atol=1e-6)


def test_principal_angles_of_orthogonal_subspaces_are_pi_over_two():
    basis_a = np.eye(4)[:, :2]
    basis_b = np.eye(4)[:, 2:]
    angles = principal_angles(basis_a, basis_b)
    assert np.allclose(angles, np.pi / 2, atol=1e-8)


def test_perturbation_vector_handle_stores_manifest_not_a_raw_vector():
    manifest = PerturbationManifest(
        seed=1, perturbation_mode="anatomical_relative_l2", model_family="qwen2_5_vl", model_scale="3B",
        model_revision="rev1", parameter_mask_hash="hash1", anatomy_region="vision_early", radius=0.05,
    )
    handle = PerturbationVectorHandle(perturbation_id=manifest.perturbation_id, manifest=manifest, capability_deltas={"counting": 0.02})
    assert handle.manifest.seed == 1
    assert handle.capability_deltas["counting"] == 0.02
    # the handle itself carries no array/tensor field -- reconstruction happens elsewhere.
    assert not any(isinstance(v, np.ndarray) for v in vars(handle).values())

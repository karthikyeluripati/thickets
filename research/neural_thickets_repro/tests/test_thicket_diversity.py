import numpy as np
import pytest

from neural_thickets_repro.thicket.diversity import (
    CapabilitySignatureMatrix,
    DiversityInputError,
    cross_capability_transfer_matrix,
    expert_overlap_matrix,
    jaccard,
    percentile_rank_matrix,
    spectral_discordance,
    task_rank_correlation_matrix,
    top_q_indices,
)


def test_percentile_rank_matrix_shape_and_range():
    m = np.array([[1.0, 5.0], [2.0, 4.0], [3.0, 3.0]])
    p = percentile_rank_matrix(m)
    assert p.shape == (3, 2)
    assert np.all((p > 0) & (p < 1))
    # column 0 is monotonically increasing -> percentile ranks strictly increasing too.
    assert p[0, 0] < p[1, 0] < p[2, 0]


def test_percentile_rank_matrix_rejects_non_2d():
    with pytest.raises(DiversityInputError):
        percentile_rank_matrix(np.array([1.0, 2.0, 3.0]))


def test_task_rank_correlation_identical_columns_gives_perfect_correlation():
    n = 50
    rng = np.random.default_rng(0)
    col = rng.normal(size=n)
    m = np.column_stack([col, col, col])
    c = task_rank_correlation_matrix(m)
    assert c.shape == (3, 3)
    off_diag = c[~np.eye(3, dtype=bool)]
    assert np.allclose(off_diag, 1.0, atol=1e-9)


def test_task_rank_correlation_requires_at_least_two_tasks():
    m = np.array([[1.0], [2.0], [3.0]])
    with pytest.raises(DiversityInputError):
        task_rank_correlation_matrix(m)


# --- spectral_discordance: known reference points from the paper's own bounds ------------------


def test_spectral_discordance_is_zero_for_perfectly_correlated_tasks():
    """D -> 0 implies parallel rankings (generalists) -- two tasks with identical (perfectly
    rank-correlated) deltas across every perturbation.
    """
    n = 100
    rng = np.random.default_rng(1)
    col = rng.normal(size=n)
    m = np.column_stack([col, col])
    d = spectral_discordance(m)
    assert d == pytest.approx(0.0, abs=1e-9)


def test_spectral_discordance_hits_the_theoretical_upper_bound_for_two_anticorrelated_tasks():
    """The paper reports D bounded in [0, M/(M-1)]; for M=2 perfectly anti-correlated task
    rankings (task 2 = -task 1), C_12 = -1 exactly, so D = 1 - (-1) = 2 = M/(M-1).
    """
    n = 100
    rng = np.random.default_rng(2)
    col = rng.normal(size=n)
    m = np.column_stack([col, -col])
    d = spectral_discordance(m)
    assert d == pytest.approx(2.0, abs=1e-9)


def test_spectral_discordance_equals_manual_formula_for_a_small_fixture():
    m = np.array([[1.0, 2.0, 9.0], [2.0, 1.0, 8.0], [3.0, 3.0, 7.0], [4.0, 0.0, 6.0]])
    c = task_rank_correlation_matrix(m)
    manual = 1.0 - (c.sum() - np.trace(c)) / (3 * 2)
    assert spectral_discordance(m) == pytest.approx(manual)


# --- H3: expert overlap (Jaccard) --------------------------------------------------------------


def test_jaccard_identical_sets():
    assert jaccard([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_jaccard_disjoint_sets():
    assert jaccard([1, 2], [3, 4]) == pytest.approx(0.0)


def test_jaccard_partial_overlap():
    assert jaccard([1, 2, 3], [2, 3, 4]) == pytest.approx(2 / 4)


def test_jaccard_both_empty_is_zero_by_convention():
    assert jaccard([], []) == 0.0


def test_top_q_indices_as_fraction():
    deltas = [0.1, 0.9, 0.5, 0.2, 0.8]
    top = top_q_indices(deltas, q=0.4, q_is_fraction=True)  # 40% of 5 -> round to 2
    assert list(top) == sorted([1, 4])  # indices of the two largest values (0.9, 0.8)


def test_top_q_indices_as_absolute_k():
    deltas = [0.1, 0.9, 0.5, 0.2, 0.8]
    top = top_q_indices(deltas, q=3, q_is_fraction=False)
    assert set(top) == {1, 2, 4}


def test_expert_overlap_matrix_diagonal_is_one():
    rng = np.random.default_rng(0)
    m = rng.normal(size=(50, 3))
    overlap = expert_overlap_matrix(m, q=0.2)
    assert np.allclose(np.diag(overlap), 1.0)


def test_expert_overlap_matrix_is_symmetric():
    rng = np.random.default_rng(0)
    m = rng.normal(size=(50, 3))
    overlap = expert_overlap_matrix(m, q=0.2)
    assert np.allclose(overlap, overlap.T)


# --- H4: cross-capability transfer matrix ------------------------------------------------------


def test_transfer_matrix_diagonal_equals_mean_of_own_top_q():
    m = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [-5.0, -50.0]])
    transfer = cross_capability_transfer_matrix(m, q=2, q_is_fraction=False)
    top_task0 = top_q_indices(m[:, 0], q=2, q_is_fraction=False)
    assert transfer[0, 0] == pytest.approx(m[top_task0, 0].mean())


def test_transfer_matrix_can_be_asymmetric_for_specialists():
    """T[t, u] is a genuinely directional statistic (mean of task u's delta restricted to
    task t's own top-q rows) -- not silently symmetrized -- exercised here against random
    continuous data, where an exact symmetric coincidence is vanishingly unlikely.
    """
    rng = np.random.default_rng(3)
    m = rng.normal(size=(30, 2))
    transfer = cross_capability_transfer_matrix(m, q=5, q_is_fraction=False)
    assert transfer[0, 1] != pytest.approx(transfer[1, 0])


# --- H5: capability signature -------------------------------------------------------------------


def test_capability_signature_matrix_row_and_column_lookup():
    matrix = np.array([[0.1, 0.2], [0.3, 0.4]])
    sig = CapabilitySignatureMatrix(perturbation_ids=("p0", "p1"), task_names=("t0", "t1"), matrix=matrix)
    assert np.array_equal(sig.row("p1"), [0.3, 0.4])
    assert np.array_equal(sig.column("t0"), [0.1, 0.3])


def test_capability_signature_matrix_rejects_shape_mismatch():
    with pytest.raises(DiversityInputError):
        CapabilitySignatureMatrix(perturbation_ids=("p0", "p1"), task_names=("t0",), matrix=np.zeros((3, 2)))

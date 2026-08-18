"""Regression test: candidate_sampling.sample_candidate_seeds must reproduce
run_randopt_image_aware.py:sample_candidates()'s seed draw exactly, across several
(N, global_seed) combinations -- pinned by test, not asserted by inspection. Neither function
calls the other; run_randopt_image_aware.py is not imported here except read-only, to compare
against, never modified.
"""
import pytest

from neural_thickets_repro.candidate_sampling import sample_candidate_seeds
from neural_thickets_repro.run_randopt_image_aware import sample_candidates


@pytest.mark.parametrize(
    "n,seed,sigma_values",
    [
        (20, 42, [0.001, 0.002]),
        (5, 0, [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01]),
        (50, 12345, [0.005]),
        (1, 999999999, [0.01, 0.02]),
        (100, 7, [0.001, 0.002, 0.005]),
    ],
)
def test_sample_candidate_seeds_matches_sample_candidates_seed_draw(n, seed, sigma_values):
    seeds_only = sample_candidate_seeds(n, seed)
    seeds_from_existing = [s for s, _ in sample_candidates(n, sigma_values, seed=seed)]
    assert seeds_only == seeds_from_existing


def test_sample_candidate_seeds_returns_n_unique_seeds():
    seeds = sample_candidate_seeds(30, seed=1)
    assert len(seeds) == 30
    assert len(set(seeds)) == 30


def test_sample_candidate_seeds_deterministic():
    assert sample_candidate_seeds(20, seed=42) == sample_candidate_seeds(20, seed=42)


def test_sample_candidate_seeds_different_seeds_differ():
    assert sample_candidate_seeds(20, seed=1) != sample_candidate_seeds(20, seed=2)


def test_sample_candidate_seeds_types():
    for s in sample_candidate_seeds(5, seed=42):
        assert isinstance(s, int)


def test_sample_candidate_seeds_independent_of_sigma_values_choice():
    """relative_l2 mode never draws sigma_values at all -- confirm the seed sequence a
    raw_sigma run WOULD have produced doesn't vary with which sigma_candidate set that run
    happened to use, since sample_candidate_seeds never even sees sigma_values.
    """
    seeds_a = [s for s, _ in sample_candidates(20, [0.001], seed=42)]
    seeds_b = [s for s, _ in sample_candidates(20, [0.001, 0.002, 0.005, 0.01], seed=42)]
    assert seeds_a == seeds_b == sample_candidate_seeds(20, seed=42)

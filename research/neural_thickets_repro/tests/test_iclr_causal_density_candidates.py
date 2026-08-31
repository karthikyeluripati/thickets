"""Tests for iclr_causal_density.candidates -- seed reuse across capabilities (item 8),
scope/radius/seed pairing (item 9). CPU-only.
"""
from __future__ import annotations

import pytest

from neural_thickets_repro.iclr_causal_density.candidates import (
    DuplicateCandidateError,
    build_candidate_population,
    build_cell_seeds,
    compute_candidate_id,
    compute_population_hash,
    validate_candidate_population,
    write_candidate_manifest,
)
from neural_thickets_repro.iclr_causal_density.design import N_SEEDS_PER_CELL, N_UNIQUE_PERTURBATIONS, RADII, SCOPES


def test_population_has_exactly_600_unique_candidates():
    pop = build_candidate_population()
    assert len(pop) == N_UNIQUE_PERTURBATIONS == 600
    validate_candidate_population(pop)  # must not raise


def test_six_cells_each_with_exactly_100_seeds():
    pop = build_candidate_population()
    by_cell = {}
    for c in pop:
        by_cell.setdefault((c.scope, c.radius), []).append(c)
    assert len(by_cell) == 6
    for cell, candidates in by_cell.items():
        assert len(candidates) == N_SEEDS_PER_CELL == 100


def test_seed_sequence_is_shared_across_capabilities_by_construction():
    """Item 8: the population is built ONCE, capability-agnostic -- the SAME 100-seed
    sequence for a (scope, radius) cell is therefore the same seed list regardless of which
    capability will later be evaluated against it (there is no per-capability seed draw at all).
    """
    pop = build_candidate_population()
    cell_seeds = [c.seed for c in pop if c.scope == "vision_encoder" and c.radius == 0.02]
    # Building the SAME cell's seeds independently (as evaluator.py would for ANY capability)
    # reproduces the identical ordered sequence.
    reproduced = build_cell_seeds("vision_encoder", 0.02)
    assert cell_seeds == reproduced


def test_scope_radius_seed_triple_is_the_candidate_identity():
    """Item 9: candidate_id is a pure, deterministic function of (scope, radius, seed)."""
    cid_a = compute_candidate_id("full_lm", 0.02, 12345)
    cid_b = compute_candidate_id("full_lm", 0.02, 12345)
    cid_c = compute_candidate_id("full_lm", 0.04, 12345)
    cid_d = compute_candidate_id("full_vlm", 0.02, 12345)
    assert cid_a == cid_b
    assert cid_a != cid_c  # radius changes identity
    assert cid_a != cid_d  # scope changes identity


def test_different_cells_never_produce_duplicate_candidate_ids():
    pop = build_candidate_population()
    ids = [c.candidate_id for c in pop]
    assert len(ids) == len(set(ids))


def test_validate_rejects_wrong_total_count():
    pop = build_candidate_population(radii=(0.02,))  # only 3 cells instead of 6
    with pytest.raises(ValueError):
        validate_candidate_population(pop)  # expects the full 6-cell/600-candidate shape by default


def test_population_hash_is_deterministic_and_order_sensitive():
    pop = build_candidate_population()
    h1 = compute_population_hash(pop)
    h2 = compute_population_hash(build_candidate_population())
    assert h1 == h2
    reversed_pop = list(reversed(pop))
    assert compute_population_hash(reversed_pop) != h1


def test_duplicate_candidate_error_type_exists_and_is_raised_on_manual_construction():
    from neural_thickets_repro.iclr_causal_density.candidates import PerturbationCandidate

    dup = [
        PerturbationCandidate(candidate_id="x", scope="full_lm", radius=0.02, seed=1, seed_index=0),
        PerturbationCandidate(candidate_id="x", scope="full_lm", radius=0.02, seed=2, seed_index=1),
    ]
    with pytest.raises(DuplicateCandidateError):
        validate_candidate_population(dup, expected_scopes=("full_lm",), expected_radii=(0.02,), expected_n_per_cell=2)


def test_write_candidate_manifest(tmp_path):
    pop = build_candidate_population()
    path = tmp_path / "candidates.json"
    write_candidate_manifest(pop, path)
    import json

    data = json.loads(path.read_text())
    assert data["n_candidates"] == 600
    assert data["population_hash"] == compute_population_hash(pop)

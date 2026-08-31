"""Tests for iclr_causal_density.subsets -- deterministic subset generation (item 1),
selection/audit disjointness (item 2). CPU-only, no dataset/network access.
"""
from __future__ import annotations

import pytest

from neural_thickets_repro.benchmarks.base import Example
from neural_thickets_repro.iclr_causal_density.subsets import (
    CapabilitySubsetManifest,
    InsufficientExamplesError,
    SubsetManifestMismatchError,
    build_selection_and_audit_subsets,
    ensure_subset_manifest_unchanged,
    load_subset_manifest,
    write_subset_manifest,
)


def _pool(n=500):
    return [Example(example_id=f"ex_{i}", image=None, image_ref=f"img_{i}", prompt_input={}, target=i) for i in range(n)]


def test_deterministic_subset_generation_is_reproducible():
    pool = _pool()
    sel1, aud1, m1 = build_selection_and_audit_subsets(pool, "visual_grounding", seed=42)
    sel2, aud2, m2 = build_selection_and_audit_subsets(pool, "visual_grounding", seed=42)
    assert [e.example_id for e in sel1] == [e.example_id for e in sel2]
    assert [e.example_id for e in aud1] == [e.example_id for e in aud2]
    assert m1 == m2


def test_different_seed_gives_different_subset():
    pool = _pool()
    _, _, m1 = build_selection_and_audit_subsets(pool, "visual_grounding", seed=42)
    _, _, m2 = build_selection_and_audit_subsets(pool, "visual_grounding", seed=43)
    assert m1.selection_hash != m2.selection_hash


def test_selection_and_audit_are_disjoint():
    pool = _pool()
    selection, audit, manifest = build_selection_and_audit_subsets(pool, "counting", seed=1)
    selection_ids = {e.example_id for e in selection}
    audit_ids = {e.example_id for e in audit}
    assert selection_ids.isdisjoint(audit_ids)
    assert set(manifest.selection_example_ids).isdisjoint(manifest.audit_example_ids)
    assert len(selection) == 200
    assert len(audit) == 200


def test_insufficient_examples_raises():
    pool = _pool(n=399)
    with pytest.raises(InsufficientExamplesError):
        build_selection_and_audit_subsets(pool, "ocr_text_recognition", seed=1)


def test_exactly_400_examples_is_sufficient():
    pool = _pool(n=400)
    selection, audit, _ = build_selection_and_audit_subsets(pool, "spatial_reasoning", seed=1)
    assert len(selection) == 200 and len(audit) == 200


def test_manifest_roundtrip_write_load(tmp_path):
    pool = _pool()
    _, _, manifest = build_selection_and_audit_subsets(pool, "relational_reasoning", seed=1)
    path = tmp_path / "subsets.json"
    write_subset_manifest({"relational_reasoning": manifest}, path)
    loaded = load_subset_manifest(path)
    assert loaded["relational_reasoning"] == manifest


def test_manifest_mismatch_detected():
    pool = _pool()
    _, _, m1 = build_selection_and_audit_subsets(pool, "counting", seed=1)
    _, _, m2 = build_selection_and_audit_subsets(pool, "counting", seed=2)
    with pytest.raises(SubsetManifestMismatchError):
        ensure_subset_manifest_unchanged(m1, m2)
    ensure_subset_manifest_unchanged(m1, m1)  # identical -- must not raise

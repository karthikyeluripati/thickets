"""Tests for benchmarks/subset_selection.py -- pure stdlib, no GPU/ray/vllm needed."""
import pytest

from neural_thickets_repro.benchmarks.base import Example
from neural_thickets_repro.benchmarks.subset_selection import (
    SubsetSelectionError,
    build_or_load_subset,
    load_persisted_ids,
    persist_subset_ids,
    select_fixed_subset,
)


def _examples(n):
    return [Example(example_id=str(i)) for i in range(n)]


def test_prefix_rule_takes_exact_prefix():
    examples = _examples(10)
    subset = select_fixed_subset(examples, 3, "prefix")
    assert [e.example_id for e in subset] == ["0", "1", "2"]


def test_shuffled_prefix_is_deterministic_across_independent_calls():
    examples = _examples(50)
    a = select_fixed_subset(examples, 10, "shuffled_prefix", seed=42)
    b = select_fixed_subset(examples, 10, "shuffled_prefix", seed=42)
    assert [e.example_id for e in a] == [e.example_id for e in b]


def test_shuffled_prefix_different_seed_gives_different_subset():
    examples = _examples(50)
    a = select_fixed_subset(examples, 10, "shuffled_prefix", seed=1)
    b = select_fixed_subset(examples, 10, "shuffled_prefix", seed=2)
    assert [e.example_id for e in a] != [e.example_id for e in b]


def test_shuffled_prefix_requires_seed():
    with pytest.raises(SubsetSelectionError, match="requires a seed"):
        select_fixed_subset(_examples(10), 3, "shuffled_prefix")


def test_unknown_rule_raises():
    with pytest.raises(SubsetSelectionError, match="Unknown subset_selection_rule"):
        select_fixed_subset(_examples(10), 3, "not_a_rule")


def test_oversized_request_raises():
    with pytest.raises(SubsetSelectionError, match="exceeds pool size"):
        select_fixed_subset(_examples(3), 10, "prefix")


def test_non_positive_n_raises():
    with pytest.raises(SubsetSelectionError, match="must be positive"):
        select_fixed_subset(_examples(10), 0, "prefix")


def test_persist_and_load_ids_round_trip(tmp_path):
    examples = _examples(5)
    path = tmp_path / "subset.json"
    persist_subset_ids(examples, path)
    assert load_persisted_ids(path) == ["0", "1", "2", "3", "4"]


def test_load_persisted_ids_missing_file_raises(tmp_path):
    with pytest.raises(SubsetSelectionError, match="No persisted subset IDs"):
        load_persisted_ids(tmp_path / "does_not_exist.json")


def test_build_or_load_subset_persists_then_reloads_identical_ids_even_if_pool_order_changes(tmp_path):
    path = tmp_path / "subset.json"
    examples = _examples(20)
    first = build_or_load_subset(examples, 5, "shuffled_prefix", seed=7, ids_path=path)
    first_ids = [e.example_id for e in first]

    # Simulate a freshly-reloaded pool with a different row order (e.g. dataset re-downloaded)
    reordered = list(reversed(examples))
    second = build_or_load_subset(reordered, 5, "shuffled_prefix", seed=7, ids_path=path)
    assert [e.example_id for e in second] == first_ids


def test_build_or_load_subset_raises_on_dataset_drift(tmp_path):
    path = tmp_path / "subset.json"
    examples = _examples(20)
    build_or_load_subset(examples, 5, "prefix", seed=None, ids_path=path)

    # A fresh pool missing one of the persisted IDs -- must hard-fail, not silently resample.
    persisted_ids = load_persisted_ids(path)
    drifted_pool = [e for e in examples if e.example_id != persisted_ids[0]]
    with pytest.raises(SubsetSelectionError, match="dataset drift"):
        build_or_load_subset(drifted_pool, 5, "prefix", seed=None, ids_path=path)

"""Tests for adapters/gqa_raw_schema.py -- synthetic raw-GQA-shaped rows matching the field
NAMES confirmed live this session (types.semantic/types.structural, a "semantic" reasoning
program with "relate" operation steps, semantic value "rel" for relational questions, raw id
field "id") -- no real dataset download / GPU / ray / vllm needed.
"""
import pytest

from neural_thickets_repro.benchmarks.adapters.gqa_raw_schema import (
    RELATION_SEMANTIC_VALUE,
    build_spatial_relational_filters,
    inspect_raw_schema,
    is_spatial_relation,
    load_persisted_filter_ids,
    persist_filter_ids,
)


def _relation_row(qid, relation_argument):
    return {
        "id": qid,
        "types": {"semantic": RELATION_SEMANTIC_VALUE, "structural": "query"},
        "semantic": [{"operation": "select", "argument": "chair"}, {"operation": "relate", "argument": relation_argument}],
    }


def _non_relation_row(qid, semantic_type="attr"):
    return {
        "id": qid,
        "types": {"semantic": semantic_type, "structural": "query"},
        "semantic": [{"operation": "select", "argument": "chair"}],
    }


def test_relation_semantic_value_is_the_confirmed_abbreviation():
    # "rel", not "relation" -- corrected after live HF viewer inspection this session.
    assert RELATION_SEMANTIC_VALUE == "rel"


def test_inspect_raw_schema_reports_found_values():
    rows = [_relation_row("1", "left of,s"), _non_relation_row("2", "attr")]
    report = inspect_raw_schema(rows)
    assert report["assumed_schema_confirmed"] is True
    assert RELATION_SEMANTIC_VALUE in report["semantic_type_values_found"]
    assert "query" in report["structural_type_values_found"]


def test_inspect_raw_schema_reports_unconfirmed_when_fields_absent():
    rows = [{"id": "1", "some_other_field": True}]
    report = inspect_raw_schema(rows)
    assert report["assumed_schema_confirmed"] is False


@pytest.mark.parametrize("relation_name,expected", [
    ("to the left of,s", True), ("above,s", True), ("holding,o", False), ("wearing,s", False), (None, False), ("", False),
])
def test_is_spatial_relation_classification(relation_name, expected):
    assert is_spatial_relation(relation_name) == expected


def test_spatial_and_experimental_relational_are_disjoint_by_explicit_construction():
    rows = [
        _relation_row("s1", "to the left of,s"),
        _relation_row("s2", "above,s"),
        _relation_row("r1", "holding,o"),
        _relation_row("r2", "wearing,s"),
        _non_relation_row("a1"),
    ]
    spatial_ids, relational_ids, stats = build_spatial_relational_filters(rows)

    assert spatial_ids == {"s1", "s2"}
    assert relational_ids == {"r1", "r2"}
    assert spatial_ids & relational_ids == set()  # disjoint by the explicit exclusion, not assumed
    assert "a1" not in spatial_ids and "a1" not in relational_ids


def test_natural_relational_category_contains_spatial_as_a_subset_not_assumed_zero_overlap():
    """The core scientific-honesty requirement: prove the OVERLAP COMPUTATION actually runs
    and reports the true natural containment (spatial questions ARE relation-type questions),
    rather than a filter design that makes overlap structurally impossible to observe.
    """
    rows = [
        _relation_row("s1", "to the left of,s"),
        _relation_row("s2", "above,s"),
        _relation_row("r1", "holding,o"),
        _non_relation_row("a1"),
    ]
    spatial_ids, relational_ids, stats = build_spatial_relational_filters(rows)

    assert stats["n_total_rows"] == 4
    assert stats["n_spatial"] == 2
    assert stats["n_relational"] == 1
    assert stats["n_intersection"] == 0  # the two FINAL, disjoint capability sets
    assert stats["n_spatial_only"] == 2
    assert stats["n_relational_only"] == 1
    assert stats["n_neither"] == 1  # a1 -- not a "rel"-type question at all
    assert stats["n_natural_relational_category"] == 3  # ALL relation-type questions, spatial included
    assert stats["n_natural_intersection"] == 2  # spatial is fully contained in the natural category
    assert stats["natural_intersection_over_spatial"] == pytest.approx(1.0)


def test_stats_documents_the_experimental_choice_explicitly():
    rows = [_relation_row("s1", "left,s")]
    _, _, stats = build_spatial_relational_filters(rows)
    assert "explicit experimental choice" in stats["experimental_relational_definition"]
    assert "GQA's actual category structure" in stats["natural_relational_note"]


def test_no_relation_questions_at_all_gives_empty_sets_no_crash():
    rows = [_non_relation_row("a1"), _non_relation_row("a2", "cat")]
    spatial_ids, relational_ids, stats = build_spatial_relational_filters(rows)
    assert spatial_ids == set()
    assert relational_ids == set()
    assert stats["natural_intersection_over_spatial"] == 0.0
    assert stats["n_neither"] == 2


def test_persist_and_load_filter_ids_round_trip(tmp_path):
    spatial_path = tmp_path / "spatial.json"
    relational_path = tmp_path / "relational.json"
    stats_path = tmp_path / "stats.json"

    persist_filter_ids({"s1", "s2"}, {"r1"}, {"n_spatial": 2}, spatial_path, relational_path, stats_path)

    assert load_persisted_filter_ids(spatial_path) == ["s1", "s2"]
    assert load_persisted_filter_ids(relational_path) == ["r1"]

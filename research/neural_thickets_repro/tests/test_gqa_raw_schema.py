"""Tests for adapters/gqa_raw_schema.py -- the FINAL high-purity spatial/relational/mixed
partition (this repair pass), built from a real full predicate inventory over GQA
testdev-balanced. Synthetic raw-GQA-shaped rows matching the REAL confirmed schema
(types.semantic/types.structural, a "semantic" reasoning program whose relate/verify rel/
choose rel steps carry a `<object_or_placeholder>,<predicate>,<flag>` argument) -- no real
dataset download / GPU / ray / vllm needed. Several fixtures reproduce the exact real
question-ID examples given in the task (device,on top of,s ; _,flying above,o ; etc).
"""
import pytest

from neural_thickets_repro.benchmarks.adapters.gqa_raw_schema import (
    RELATION_SEMANTIC_VALUE,
    SPATIAL_PREDICATE_WHITELIST,
    build_spatial_relational_filters,
    classify_relation_question_predicates,
    describe_question_classification,
    inspect_raw_schema,
    load_persisted_filter_ids,
    persist_filter_ids,
)


def _relation_row(qid, steps, semantic_type=RELATION_SEMANTIC_VALUE, structural_type="query"):
    """`steps`: list of (operation, argument) tuples, e.g. [("relate", "device,on top of,s")]
    -- matches the REAL confirmed argument shape (object_or_placeholder, predicate, flag).
    """
    return {
        "id": qid,
        "types": {"semantic": semantic_type, "structural": structural_type},
        "semantic": [{"operation": op, "argument": arg} for op, arg in steps],
    }


def _non_relation_row(qid, semantic_type="attr"):
    return {
        "id": qid,
        "types": {"semantic": semantic_type, "structural": "query"},
        "semantic": [{"operation": "select", "argument": "chair"}],
    }


def test_relation_semantic_value_is_the_confirmed_abbreviation():
    assert RELATION_SEMANTIC_VALUE == "rel"


def test_inspect_raw_schema_reports_found_values():
    rows = [_relation_row("1", [("relate", "chair,left of,s")]), _non_relation_row("2", "attr")]
    report = inspect_raw_schema(rows)
    assert report["assumed_schema_confirmed"] is True
    assert RELATION_SEMANTIC_VALUE in report["semantic_type_values_found"]
    assert "query" in report["structural_type_values_found"]


def test_inspect_raw_schema_reports_unconfirmed_when_fields_absent():
    rows = [{"id": "1", "some_other_field": True}]
    report = inspect_raw_schema(rows)
    assert report["assumed_schema_confirmed"] is False


# ---------------------------------------------------------------------------------------
# Real, individually-audited question examples from the task
# ---------------------------------------------------------------------------------------

def test_real_example_device_on_top_of_is_spatial():
    # 201902997: device,on top of,s -> spatial
    row = _relation_row("201902997", [("relate", "device,on top of,s")])
    assert classify_relation_question_predicates(row) == "spatial"


def test_real_example_flying_above_is_not_spatial_under_exact_matching():
    """20567512: _,flying above,o -- "flying above" is NOT in the strict whitelist. Must NOT
    be classified spatial via substring "above" -- this is the exact case the task calls out
    as a trap for a substring-based (or careless exact-but-untested) implementation.
    """
    row = _relation_row("20567512", [("relate", "_,flying above,o")])
    assert "flying above" not in SPATIAL_PREDICATE_WHITELIST
    assert classify_relation_question_predicates(row) == "relational"


def test_real_example_to_the_right_of_is_spatial():
    # 201079958: drapes,to the right of,s -> spatial
    row = _relation_row("201079958", [("relate", "drapes,to the right of,s")])
    assert classify_relation_question_predicates(row) == "spatial"


def test_real_example_verify_rel_platter_on_is_spatial():
    # 20609782: verify rel: platter,on,o (-) -> spatial ("on", ignoring the trailing "(-)")
    row = _relation_row("20609782", [("verify rel", "platter,on,o (-)")])
    assert classify_relation_question_predicates(row) == "spatial"


def test_real_example_verify_rel_wetsuit_wearing_is_non_spatial():
    # 2062325: verify rel: wetsuit,wearing,o (12) -> non-spatial relational
    row = _relation_row("2062325", [("verify rel", "wetsuit,wearing,o (12)")])
    assert classify_relation_question_predicates(row) == "relational"


def test_real_example_around_is_spatial():
    # 201079951: _,around,s -> spatial
    row = _relation_row("201079951", [("relate", "_,around,s")])
    assert classify_relation_question_predicates(row) == "spatial"


def test_real_example_wearing_is_non_spatial():
    # 201640614: person,wearing,s -> non-spatial relational
    row = _relation_row("201640614", [("relate", "person,wearing,s")])
    assert classify_relation_question_predicates(row) == "relational"


def test_real_example_mixed_question_is_excluded_from_both():
    # 201757757: "to the right of" (spatial) + "wearing" (non-spatial) -> mixed / excluded
    row = _relation_row("201757757", [
        ("relate", "cup,to the right of,s"),
        ("relate", "person,wearing,s"),
    ])
    assert classify_relation_question_predicates(row) == "mixed"


# ---------------------------------------------------------------------------------------
# Operation forms: relate / verify rel / choose rel
# ---------------------------------------------------------------------------------------

def test_relate_operation_extracts_predicate():
    row = _relation_row("1", [("relate", "chair,near,s")])
    assert classify_relation_question_predicates(row) == "spatial"


def test_verify_rel_operation_extracts_predicate_not_null():
    """Do not return null just because operation != "relate" -- verify rel must still yield
    a real predicate.
    """
    row = _relation_row("1", [("verify rel", "table,under,o (3)")])
    assert classify_relation_question_predicates(row) == "spatial"


def test_choose_rel_all_alternatives_spatial_is_spatial():
    row = _relation_row("1", [("choose rel", "chair,to the left of|to the right of,s")])
    assert classify_relation_question_predicates(row) == "spatial"


def test_choose_rel_mixed_alternatives_is_not_spatial():
    # one alternative whitelisted, one not -- ALL alternatives must be spatial to count.
    row = _relation_row("1", [("choose rel", "chair,on|wearing,s")])
    assert classify_relation_question_predicates(row) == "relational"


def test_unknown_operation_name_is_never_treated_as_a_relation_step():
    row = _relation_row("1", [("select", "chair")])  # not relate/verify rel/choose rel
    assert classify_relation_question_predicates(row) == "no_extractable_predicate"


def test_malformed_argument_missing_object_field_is_skipped_not_guessed():
    # only 2 comma-separated fields, not the expected 3 -- skipped, not misparsed.
    row = _relation_row("1", [("relate", "on,s")])
    assert classify_relation_question_predicates(row) == "no_extractable_predicate"


# ---------------------------------------------------------------------------------------
# Exact matching (never substring)
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("predicate", ["person", "along", "onion"])
def test_exact_matching_never_substring_matches_on(predicate):
    row = _relation_row("1", [("relate", f"x,{predicate},s")])
    assert classify_relation_question_predicates(row) == "relational"


@pytest.mark.parametrize("action_modified_predicate", [
    "sitting on", "standing on", "sitting atop", "sitting on top of", "hanging above",
    "sitting beside", "lying on top of", "standing behind", "riding on", "walking on",
    "resting on", "mounted on",
])
def test_action_modified_relations_are_not_promoted_to_spatial(action_modified_predicate):
    row = _relation_row("1", [("relate", f"x,{action_modified_predicate},s")])
    assert action_modified_predicate not in SPATIAL_PREDICATE_WHITELIST
    assert classify_relation_question_predicates(row) == "relational"


def test_no_extractable_predicate_when_semantic_program_has_no_relation_steps():
    row = {"id": "1", "types": {"semantic": RELATION_SEMANTIC_VALUE, "structural": "query"}, "semantic": []}
    assert classify_relation_question_predicates(row) == "no_extractable_predicate"


# ---------------------------------------------------------------------------------------
# build_spatial_relational_filters -- pure spatial / pure relational / mixed exclusion /
# zero overlap
# ---------------------------------------------------------------------------------------

def test_pure_spatial_and_pure_relational_are_disjoint_by_construction():
    rows = [
        _relation_row("s1", [("relate", "chair,to the left of,s")]),
        _relation_row("s2", [("relate", "chair,above,s")]),
        _relation_row("r1", [("relate", "person,holding,o")]),
        _relation_row("r2", [("relate", "person,wearing,s")]),
        _non_relation_row("a1"),
    ]
    spatial_ids, relational_ids, mixed_ids, stats = build_spatial_relational_filters(rows)

    assert spatial_ids == {"s1", "s2"}
    assert relational_ids == {"r1", "r2"}
    assert mixed_ids == set()
    assert spatial_ids & relational_ids == set(), "the two final benchmark sets must be disjoint"
    assert "a1" not in spatial_ids and "a1" not in relational_ids


def test_mixed_questions_are_excluded_from_both_sets_but_persisted_separately():
    rows = [
        _relation_row("s1", [("relate", "chair,on,s")]),
        _relation_row("m1", [("relate", "cup,to the right of,s"), ("relate", "person,wearing,s")]),
    ]
    spatial_ids, relational_ids, mixed_ids, stats = build_spatial_relational_filters(rows)

    assert spatial_ids == {"s1"}
    assert relational_ids == set()
    assert mixed_ids == {"m1"}
    assert "m1" not in spatial_ids and "m1" not in relational_ids
    assert stats["n_mixed_excluded"] == 1


def test_stats_reports_the_real_inventory_shaped_counts_and_sum_check():
    rows = (
        [_relation_row(f"s{i}", [("relate", "chair,on,s")]) for i in range(3)]
        + [_relation_row(f"r{i}", [("relate", "person,wearing,s")]) for i in range(2)]
        + [_relation_row("m1", [("relate", "cup,near,s"), ("relate", "person,riding on,s")])]
        + [_non_relation_row(f"a{i}") for i in range(4)]
    )
    spatial_ids, relational_ids, mixed_ids, stats = build_spatial_relational_filters(rows)

    assert stats["n_total_rows"] == 10
    assert stats["n_not_relation_type"] == 4
    assert stats["n_relation_type_rows"] == 6
    assert stats["n_pure_spatial"] == 3
    assert stats["n_pure_relational"] == 2
    assert stats["n_mixed_excluded"] == 1
    assert stats["n_no_extractable_predicate"] == 0
    assert stats["sum_check_ok"] is True


def test_no_relation_questions_at_all_gives_empty_sets_no_crash():
    rows = [_non_relation_row("a1"), _non_relation_row("a2", "cat")]
    spatial_ids, relational_ids, mixed_ids, stats = build_spatial_relational_filters(rows)
    assert spatial_ids == set()
    assert relational_ids == set()
    assert mixed_ids == set()
    assert stats["n_not_relation_type"] == 2
    assert stats["n_relation_type_rows"] == 0


def test_stats_documents_the_high_purity_experimental_partition_explicitly():
    rows = [_relation_row("s1", [("relate", "chair,on,s")])]
    _, _, _, stats = build_spatial_relational_filters(rows)
    assert "EXPERIMENTAL" in stats["partition_definition_note"]
    assert "NOT an official GQA taxonomy" in stats["partition_definition_note"]
    assert "NOT automatically promoted to spatial" in stats["action_modified_relation_note"]


# ---------------------------------------------------------------------------------------
# describe_question_classification
# ---------------------------------------------------------------------------------------

def test_describe_question_classification_spatial_case_with_predicate_detail():
    rows = [_relation_row("s1", [("relate", "device,on top of,s")])]
    detail = describe_question_classification(rows, "s1")

    assert detail["found"] is True
    assert detail["types_semantic"] == RELATION_SEMANTIC_VALUE
    assert detail["is_relation_type_question"] is True
    assert detail["extracted_predicates"] == ["on top of"]
    assert detail["predicate_classification_detail"] == [{"predicate": "on top of", "is_spatial": True}]
    assert detail["classification"] == "spatial"


def test_describe_question_classification_mixed_case_with_full_predicate_detail():
    rows = [_relation_row("m1", [("relate", "cup,to the right of,s"), ("relate", "person,wearing,s")])]
    detail = describe_question_classification(rows, "m1")

    assert detail["classification"] == "mixed"
    assert detail["extracted_predicates"] == ["to the right of", "wearing"]
    assert detail["predicate_classification_detail"] == [
        {"predicate": "to the right of", "is_spatial": True},
        {"predicate": "wearing", "is_spatial": False},
    ]


def test_describe_question_classification_neither_case():
    rows = [_non_relation_row("a1", semantic_type="attr")]
    detail = describe_question_classification(rows, "a1")
    assert detail["classification"] == "neither"
    assert detail["is_relation_type_question"] is False
    assert detail["extracted_predicates"] == []


def test_describe_question_classification_not_found():
    rows = [_relation_row("s1", [("relate", "chair,on,s")])]
    detail = describe_question_classification(rows, "does-not-exist")
    assert detail == {"question_id": "does-not-exist", "found": False}


def test_describe_question_classification_exposes_question_and_semantic_str():
    row = _relation_row("s1", [("relate", "chair,above,s")])
    row["question"] = "What is above the table?"
    row["semanticStr"] = "relate(above,s)"
    detail = describe_question_classification([row], "s1")
    assert detail["question"] == "What is above the table?"
    assert detail["semantic_str"] == "relate(above,s)"
    assert detail["semantic_program"] == row["semantic"]


# ---------------------------------------------------------------------------------------
# persist_filter_ids / load_persisted_filter_ids -- now includes mixed_ids
# ---------------------------------------------------------------------------------------

def test_persist_and_load_filter_ids_round_trip(tmp_path):
    spatial_path = tmp_path / "spatial.json"
    relational_path = tmp_path / "relational.json"
    mixed_path = tmp_path / "mixed.json"
    stats_path = tmp_path / "stats.json"

    persist_filter_ids({"s1", "s2"}, {"r1"}, {"m1"}, {"n_pure_spatial": 2}, spatial_path, relational_path, mixed_path, stats_path)

    assert load_persisted_filter_ids(spatial_path) == ["s1", "s2"]
    assert load_persisted_filter_ids(relational_path) == ["r1"]
    assert load_persisted_filter_ids(mixed_path) == ["m1"]

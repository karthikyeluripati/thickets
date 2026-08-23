"""Raw GQA annotation schema investigation + spatial/relational subset filter construction,
shared by spatial_reasoning_gqa.py and relational_reasoning_gqa.py.

GQAHandler.load_data()'s own output (question_id, image_path, messages, ground_truth) does
NOT expose GQA's own question-type metadata (semantic/structural type, relation predicate)
needed to build a defensible spatial-vs-relational filter -- that metadata lives only in the
RAW GQA annotation file, which must be loaded a SECOND time, independently of GQAHandler,
purely to read it. This module operates on that already-loaded raw row list; it does not
itself call datasets.load_dataset (that stays in each adapter's own load_examples(),
consistent with every other adapter's lazy-import convention).

FIELD NAMES CONFIRMED (live HF dataset-viewer inspection of `lmms-lab-encoder/GQA`,
`testdev_balanced_instructions` config): the raw rows carry `types` (nested
`structural`/`semantic`/`detailed`), `semantic` (a list of reasoning-program operation
steps), `semanticStr`, `groups`, `isBalanced`, `entailed`, `equivalent` -- i.e.
`prepare_gqa_data.py`'s parquet (`id`/`imageId`/`question`/`answer`/`fullAnswer` only) is a
deliberately NARROWED projection of these same rows for GQAHandler's own needs, not evidence
the richer fields don't exist upstream. `types.semantic`'s value for relational questions is
`"rel"`, not `"relation"` -- GQA's semantic categories are
`{"object", "attr", "cat", "global", "rel"}`.

HIGH-PURITY PARTITION (this repair pass, FINAL definition, superseding the previous
substring/single-predicate-based version): a REAL full predicate inventory over GQA
testdev-balanced (12578 rows; 7270 not relation-type; 5308 relation-type = 2862 pure
spatial + 2028 pure non-spatial relational + 418 mixed + 0 with no extractable predicate)
was used to freeze this EXPLICIT, EXPERIMENTAL high-purity capability partition -- it is NOT
claimed to be an official GQA taxonomy:

  - SPATIAL: every extracted relation predicate in the question is in
    SPATIAL_PREDICATE_WHITELIST below, via EXACT normalized-string matching -- NEVER
    substring matching (e.g. the compound predicate "flying above" is NOT spatial just
    because "above" is a substring/suffix of it -- see the ACTION-MODIFIED RELATIONS note).
  - RELATIONAL (non-spatial): none of the question's extracted predicates is in the
    whitelist.
  - MIXED: the question has at least one whitelisted AND at least one non-whitelisted
    predicate (e.g. "person to the right of cup wearing jeans") -- EXCLUDED from both
    benchmark capabilities (never assigned to either), but its ID is still persisted
    (gqa_mixed_ids.json) for auditability, not silently dropped.
  - NO_EXTRACTABLE_PREDICATE: a relation-type row whose semantic program yields zero
    predicates the extractor can parse -- real inventory found zero of these; the bucket
    exists so a nonzero count on a different data snapshot is visible, not silently ignored.

ACTION-MODIFIED RELATIONS ARE NOT PROMOTED TO SPATIAL (explicit high-purity decision, not an
oversight): predicates like "sitting on", "standing on", "flying above", "riding on", "walking
on", "resting on", "mounted on" conflate an action/pose/interaction with a spatial relation.
Promoting them to spatial by matching a spatial WORD inside them (e.g. treating "flying above"
as spatial because it contains "above") would blur the very distinction this partition exists
to draw. They are classified NON-SPATIAL unless the exact compound string is itself
explicitly added to SPATIAL_PREDICATE_WHITELIST -- which it is not, by design, in this pass.

RELATION OPERATION FORMS (all three must be inspected, not just "relate" -- a real inventory
found `relate`: 6201, `verify rel`: 769, `choose rel`: 211 occurrences): the argument string
for all three follows the same `<object_or_placeholder>,<predicate>,<flag>` shape (commas
separate exactly 3 fields; a `verify rel` argument's flag field may carry a trailing
`" (12)"`/`" (-)"` annotation, which is simply part of the ignored 3rd field). Examples
(all real, confirmed): `relate` "device,on top of,s" -> predicate "on top of"; `verify rel`
"platter,on,o (-)" -> predicate "on"; `verify rel` "wetsuit,wearing,o (12)" -> predicate
"wearing"; `choose rel` alternatives are pipe-separated within the same middle field (e.g.
"to the left of|to the right of") and classified spatial only if ALL alternatives are
individually whitelisted. `_extract_predicates_from_row()` inspects every step whose
`operation` is exactly one of these three names (RELATION_OPERATION_NAMES, exact match, not
`"relate" in operation` -- the old check silently never matched `verify rel`/`choose rel` at
all, since "relate" is not a substring of "verify rel").

CANNOT INDEPENDENTLY VERIFY THE AGGREGATE 2862/2028/418 COUNTS FROM THIS ENVIRONMENT: this
module implements the rule exactly as specified and is tested against the individually-audited
real question IDs given in the task (including the "flying above" non-spatial case, verified
NOT to match the whitelist under exact matching). If a fresh real-data run of
`prepare_gqa_capability_filters.py` produces different aggregate counts than 2862/2028/418,
that discrepancy must be investigated and explained (e.g. a difference in the original
inventory script's own normalization), never silently forced to match by adjusting the rule.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

# --- Field NAMES confirmed live this session ---
SEMANTIC_TYPE_FIELD = "types.semantic"
STRUCTURAL_TYPE_FIELD = "types.structural"
RELATION_SEMANTIC_VALUE = "rel"  # CORRECTED from an initial "relation" guess -- confirmed via live HF viewer inspection

# GQAHandler's own output records expose the question id under the key "question_id"
# (see eval_base_image_aware.py's own usage: d["question_id"]), but the RAW row (and
# prepare_gqa_data.py's parquet) uses the key "id" -- both trace back to the identical
# underlying GQA question-id string, just under different key names at different pipeline
# stages. build_spatial_relational_filters()'s default question_id_field matches the RAW
# row's own key ("id"), NOT GQAHandler's renamed "question_id" -- do not conflate the two
# key names, only their values are expected to match.
RAW_QUESTION_ID_FIELD = "id"

# The three GQA semantic-program operation names that carry a relation predicate -- EXACT
# match against this set (never `"relate" in operation`, which silently never matches
# "verify rel"/"choose rel" -- see module docstring's RELATION OPERATION FORMS section).
RELATION_OPERATION_NAMES: frozenset = frozenset({"relate", "verify rel", "choose rel"})

# FROZEN, real-data-audited spatial predicate whitelist (this repair pass) -- EXACT
# normalized-string matching only, never substring. Not "every possible spatial-sounding
# word" -- deliberately excludes action-modified compounds like "sitting on"/"flying above"
# (see module docstring's ACTION-MODIFIED RELATIONS note). Expanding this list is a real
# scientific decision requiring a documented reason, not a casual edit.
SPATIAL_PREDICATE_WHITELIST: frozenset = frozenset({
    "in front of", "to the right of", "to the left of", "on", "behind", "on top of", "near",
    "above", "below", "next to", "inside", "in", "underneath", "under", "beside", "beneath",
    "around", "surrounding", "surrounded by", "in between", "over", "across from",
    "on the edge of", "on the front of", "on the bottom of", "touching", "higher than",
})


class GQASchemaError(RuntimeError):
    """The raw GQA dataset's actual schema doesn't match the assumed field names -- refuses
    to silently build a filter against fields that don't exist.
    """


def _get_nested(row: Dict[str, Any], dotted_path: str) -> Any:
    value: Any = row
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def inspect_raw_schema(raw_rows: Sequence[Dict[str, Any]], sample_size: int = 5) -> Dict[str, Any]:
    """Reports the actual key set and assumed-field values found in a sample of the raw
    dataset -- the required first pod-side diagnostic step before trusting
    build_spatial_relational_filters() below. Never raises; a report with
    assumed_schema_confirmed=False is itself the useful, actionable output.
    """
    sample = list(raw_rows[:sample_size])
    top_level_keys = sorted({k for row in sample for k in row.keys()})
    semantic_values = sorted({v for row in sample if (v := _get_nested(row, SEMANTIC_TYPE_FIELD)) is not None})
    structural_values = sorted({v for row in sample if (v := _get_nested(row, STRUCTURAL_TYPE_FIELD)) is not None})

    return {
        "n_sampled": len(sample),
        "top_level_keys": top_level_keys,
        "semantic_type_field": SEMANTIC_TYPE_FIELD,
        "semantic_type_values_found": semantic_values,
        "structural_type_field": STRUCTURAL_TYPE_FIELD,
        "structural_type_values_found": structural_values,
        "assumed_schema_confirmed": bool(semantic_values) and bool(structural_values),
    }


def _normalize_predicate(predicate: str) -> str:
    """Minimal, deterministic normalization -- lowercase + strip whitespace only. No
    stemming, no punctuation stripping, no substring reduction: EXACT matching against
    SPATIAL_PREDICATE_WHITELIST depends on this staying minimal (see module docstring).
    """
    return predicate.strip().lower()


def _extract_predicates_from_row(row: Dict[str, Any]) -> List[str]:
    """Returns every relation predicate string found in the row's semantic program, across
    ALL relation-bearing operation steps (relate/verify rel/choose rel) -- a question can
    have more than one (e.g. "person to the right of cup wearing jeans" has two). A step
    whose argument doesn't parse into the expected 3-field
    `<object_or_placeholder>,<predicate>,<flag>` shape is skipped, not guessed at. A
    `choose rel` predicate field may itself contain pipe-separated alternatives (e.g.
    "to the left of|to the right of") -- returned here as a single un-split string; splitting
    into alternatives is _is_spatial_predicate()'s job, so this function's output always
    means "one semantic-program relation step", not "one predicate word".
    """
    program = row.get("semantic") or row.get("semantic_program") or []
    predicates: List[str] = []
    for step in program:
        if not isinstance(step, dict):
            continue
        if step.get("operation") not in RELATION_OPERATION_NAMES:
            continue
        argument = step.get("argument")
        if not argument:
            continue
        parts = str(argument).split(",")
        if len(parts) < 3:
            continue  # malformed/unexpected shape -- not guessed at
        predicate = parts[1].strip()
        if predicate:
            predicates.append(predicate)
    return predicates


def _is_spatial_predicate(predicate: str) -> bool:
    """EXACT normalized matching only -- never substring (see module docstring). A
    `choose rel` multi-choice predicate (pipe-separated alternatives, e.g. "to the left
    of|to the right of") is spatial iff ALL alternatives are individually whitelisted.
    """
    alternatives = [_normalize_predicate(p) for p in predicate.split("|")]
    return bool(alternatives) and all(alt in SPATIAL_PREDICATE_WHITELIST for alt in alternatives)


def classify_relation_question_predicates(row: Dict[str, Any]) -> str:
    """Returns one of "spatial" / "relational" / "mixed" / "no_extractable_predicate" for a
    single relation-type ("rel") row, based on EVERY extracted predicate (not just the
    first) -- see module docstring's HIGH-PURITY PARTITION section for the exact rule.
    """
    predicates = _extract_predicates_from_row(row)
    if not predicates:
        return "no_extractable_predicate"
    flags = [_is_spatial_predicate(p) for p in predicates]
    if all(flags):
        return "spatial"
    if not any(flags):
        return "relational"
    return "mixed"


def build_spatial_relational_filters(
    raw_rows: Sequence[Dict[str, Any]], question_id_field: str = RAW_QUESTION_ID_FIELD,
) -> Tuple[Set[str], Set[str], Set[str], Dict[str, Any]]:
    """Returns (spatial_ids, relational_ids, mixed_ids, stats) -- the FINAL high-purity
    partition (see module docstring). spatial_ids and relational_ids are disjoint BY
    CONSTRUCTION (classify_relation_question_predicates() assigns each relation-type question
    to exactly one of "spatial"/"relational"/"mixed"/"no_extractable_predicate", never more
    than one bucket) -- never merely by set subtraction. mixed_ids is real, non-empty in
    practice, and persisted for auditability, not silently discarded.
    """
    spatial_ids: Set[str] = set()
    relational_ids: Set[str] = set()
    mixed_ids: Set[str] = set()
    n_no_extractable_predicate = 0
    n_not_relation_type = 0

    for row in raw_rows:
        if _get_nested(row, SEMANTIC_TYPE_FIELD) != RELATION_SEMANTIC_VALUE:
            n_not_relation_type += 1
            continue
        qid = str(row[question_id_field])
        classification = classify_relation_question_predicates(row)
        if classification == "spatial":
            spatial_ids.add(qid)
        elif classification == "relational":
            relational_ids.add(qid)
        elif classification == "mixed":
            mixed_ids.add(qid)
        else:
            n_no_extractable_predicate += 1

    n_total_rows = len(raw_rows)
    n_relation_type_rows = n_total_rows - n_not_relation_type

    stats = {
        "n_total_rows": n_total_rows,
        "n_not_relation_type": n_not_relation_type,
        "n_relation_type_rows": n_relation_type_rows,
        "n_pure_spatial": len(spatial_ids),
        "n_pure_relational": len(relational_ids),
        "n_mixed_excluded": len(mixed_ids),
        "n_no_extractable_predicate": n_no_extractable_predicate,
        "sum_check_ok": (
            n_not_relation_type + len(spatial_ids) + len(relational_ids) + len(mixed_ids) + n_no_extractable_predicate
        ) == n_total_rows,
        "spatial_predicate_whitelist_size": len(SPATIAL_PREDICATE_WHITELIST),
        "partition_definition_note": (
            "EXPLICIT high-purity EXPERIMENTAL partition of GQA relation-type questions, NOT "
            "an official GQA taxonomy: spatial = every extracted predicate exactly matches "
            "SPATIAL_PREDICATE_WHITELIST; relational = none do; mixed = both -- excluded from "
            "BOTH benchmark capabilities, not assigned to either. Exact normalized-string "
            "matching only, never substring matching."
        ),
        "action_modified_relation_note": (
            "Action-modified relations (e.g. 'sitting on', 'flying above', 'riding on') are "
            "NOT automatically promoted to spatial by matching a spatial word inside them -- "
            "classified non-spatial unless the exact compound predicate string is itself in "
            "SPATIAL_PREDICATE_WHITELIST, which it is not by design in this pass."
        ),
    }
    return spatial_ids, relational_ids, mixed_ids, stats


def describe_question_classification(
    raw_rows: Sequence[Dict[str, Any]], question_id: Any, question_id_field: str = RAW_QUESTION_ID_FIELD,
) -> Dict[str, Any]:
    """CPU-side manual-audit utility (Task: GQA capability taxonomy audit): for ONE question
    ID, returns the full real raw record plus every extracted predicate, each predicate's
    individual spatial/non-spatial classification, and the overall question classification
    ("spatial"/"relational"/"mixed"/"no_extractable_predicate"/"neither") -- so a decision can
    be verified against the real record rather than guessed from the English question text.
    Never fabricates or guesses a record: {"found": False} if the ID isn't present in
    `raw_rows`. Used by prepare_gqa_capability_filters.py's `--audit-question-ids` flag.
    """
    qid = str(question_id)
    for row in raw_rows:
        if str(row.get(question_id_field)) != qid:
            continue
        semantic_type = _get_nested(row, SEMANTIC_TYPE_FIELD)
        structural_type = _get_nested(row, STRUCTURAL_TYPE_FIELD)
        is_relation_type = semantic_type == RELATION_SEMANTIC_VALUE

        if is_relation_type:
            predicates = _extract_predicates_from_row(row)
            predicate_detail = [{"predicate": p, "is_spatial": _is_spatial_predicate(p)} for p in predicates]
            classification = classify_relation_question_predicates(row)
        else:
            predicates = []
            predicate_detail = []
            classification = "neither"

        return {
            "question_id": qid,
            "found": True,
            "question": row.get("question"),
            "types_semantic": semantic_type,
            "types_structural": structural_type,
            "semantic_program": row.get("semantic"),
            "semantic_str": row.get("semanticStr"),
            "is_relation_type_question": is_relation_type,
            "extracted_predicates": predicates,
            "predicate_classification_detail": predicate_detail,
            "classification": classification,
        }
    return {"question_id": qid, "found": False}


def persist_filter_ids(
    spatial_ids: Set[str], relational_ids: Set[str], mixed_ids: Set[str], stats: Dict[str, Any],
    spatial_path: "str | Path", relational_path: "str | Path", mixed_path: "str | Path", stats_path: "str | Path",
) -> None:
    """Persists all four artifacts -- including mixed_ids, for auditability, even though
    mixed questions are never evaluated by either benchmark capability.
    """
    for path in (spatial_path, relational_path, mixed_path, stats_path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(spatial_path).write_text(json.dumps(sorted(spatial_ids), indent=2))
    Path(relational_path).write_text(json.dumps(sorted(relational_ids), indent=2))
    Path(mixed_path).write_text(json.dumps(sorted(mixed_ids), indent=2))
    Path(stats_path).write_text(json.dumps(stats, indent=2))


def load_persisted_filter_ids(path: "str | Path") -> List[str]:
    path = Path(path)
    if not path.exists():
        raise GQASchemaError(f"No persisted GQA spatial/relational filter IDs found at {path}")
    return json.loads(path.read_text())

"""Raw GQA annotation schema investigation + spatial/relational subset filter construction,
shared by spatial_reasoning_gqa.py and relational_reasoning_gqa.py.

GQAHandler.load_data()'s own output (question_id, image_path, messages, ground_truth) does
NOT expose GQA's own question-type metadata (semantic/structural type, relation name) needed
to build a defensible spatial-vs-relational filter -- that metadata lives only in the RAW GQA
annotation file, which must be loaded a SECOND time, independently of GQAHandler, purely to
read it. This module operates on that already-loaded raw row list; it does not itself call
datasets.load_dataset (that stays in each adapter's own load_examples(), consistent with
every other adapter's lazy-import convention).

FIELD NAMES CONFIRMED (live HF dataset-viewer inspection of `lmms-lab-encoder/GQA`,
`testdev_balanced_instructions` config, this session -- see CAPABILITY_BENCHMARK_GATE.md):
the raw rows DO carry `types` (nested `structural`/`semantic`/`detailed`), `semantic` (a list
of reasoning-program operation steps), `semanticStr`, `groups`, `isBalanced`, `entailed`,
`equivalent` -- i.e. `prepare_gqa_data.py`'s parquet (`id`/`imageId`/`question`/`answer`/
`fullAnswer` only) is a deliberately NARROWED projection of these same rows for GQAHandler's
own needs, not evidence the richer fields don't exist upstream. `types.semantic`'s actual
value for relational questions is **`"rel"`, not `"relation"`** (corrected from the initial
public-documentation-based guess after this live check) -- GQA's semantic categories are
`{"object", "attr", "cat", "global", "rel"}`.

STILL UNRESOLVED, pod-side investigation required (`inspect_raw_schema()` below, or the
`prepare_gqa_capability_filters.py` CLI's own printed report) before trusting the filter for
real: the EXACT shape/argument-encoding of `semantic`'s "relate" operation steps
(`_extract_relation_name`) was not independently confirmed at the individual-value level
(only the field names and semantic-category value set were) -- run
`prepare_gqa_capability_filters.py --config configs/gqa_repro.yaml` on the pod and inspect
its printed counts before treating the resulting ID files as final.

SCIENTIFIC NOTE on spatial vs. relational (per explicit instruction -- do not pretend these
are naturally disjoint): GQA's own "relation" semantic-type category is a single category
that NATURALLY CONTAINS both spatial relations (left/right/above/etc.) and non-spatial
relations (holding/wearing/riding/etc.) -- spatial is a SUB-FILTER of it, not a separate
GQA-provided category. build_spatial_relational_filters() reports this natural containment
explicitly (spatial ⊆ natural-relational, so their natural intersection ratio is 1.0 relative
to spatial, by GQA's own structure) and ALSO reports the final, disjoint sets actually used by
the two adapters -- which are disjoint only because of an explicit experimental choice
(relational_reasoning = natural-relational MINUS spatial, so the two capability benchmarks
measure distinct skills), not because GQA itself hands us two separate categories.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

# --- Field NAMES confirmed live this session (see module docstring); the exact "relate"
# argument shape is still pod-side-unconfirmed. ---
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

# Closed, explicit, documented spatial-relation keyword list -- classifies a "relation"-type
# question as spatial iff its extracted relation name contains one of these. Not "every
# relation," and not silently expanded later without updating this list and this docstring.
SPATIAL_RELATION_KEYWORDS = (
    "left", "right", "above", "below", "behind", "front", "near", "next to", "on", "under",
    "underneath", "inside", "outside", "between", "beside", "atop", "over", "top of", "bottom of",
)


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


def _extract_relation_name(row: Dict[str, Any]) -> "str | None":
    """Extracts the specific relation name (e.g. "to the left of", "holding") from the
    question's semantic reasoning program's "relate" operation step. Exact field/shape TBD
    against real data (see module docstring) -- this reads the publicly documented shape (a
    list of {"operation": ..., "argument": ...} steps under "semantic").
    """
    program = row.get("semantic") or row.get("semantic_program") or []
    for step in program:
        if not isinstance(step, dict):
            continue
        if "relate" in str(step.get("operation", "")):
            argument = step.get("argument")
            return str(argument) if argument else None
    return None


def is_spatial_relation(relation_name: "str | None") -> bool:
    if not relation_name:
        return False
    lowered = relation_name.lower()
    return any(keyword in lowered for keyword in SPATIAL_RELATION_KEYWORDS)


def build_spatial_relational_filters(
    raw_rows: Sequence[Dict[str, Any]], question_id_field: str = RAW_QUESTION_ID_FIELD,
) -> Tuple[Set[str], Set[str], Dict[str, Any]]:
    """Returns (spatial_ids, experimental_relational_ids, stats). See module docstring's
    SCIENTIFIC NOTE: stats reports BOTH the natural containment (spatial is a subset of GQA's
    own "rel" category) and the final experimental exclusion, so the disjointness of the two
    returned ID sets is never mistaken for something GQA itself provides. Also reports the
    literal spatial/relational/intersection/neither breakdown against the FULL row set (not
    just relation-type rows), so "neither" (attr/cat/global/object questions) is visible too.
    """
    natural_relational_ids: Set[str] = set()
    spatial_ids: Set[str] = set()

    for row in raw_rows:
        if _get_nested(row, SEMANTIC_TYPE_FIELD) != RELATION_SEMANTIC_VALUE:
            continue
        qid = str(row[question_id_field])
        natural_relational_ids.add(qid)
        if is_spatial_relation(_extract_relation_name(row)):
            spatial_ids.add(qid)

    experimental_relational_ids = natural_relational_ids - spatial_ids
    natural_intersection = spatial_ids & natural_relational_ids
    experimental_intersection = spatial_ids & experimental_relational_ids

    n_total_rows = len(raw_rows)
    n_neither = n_total_rows - len(natural_relational_ids)  # not a "rel"-type question at all (attr/cat/global/object)

    stats = {
        "n_total_rows": n_total_rows,
        "n_spatial": len(spatial_ids),
        "n_relational": len(experimental_relational_ids),
        "n_intersection": len(experimental_intersection),  # spatial vs. the FINAL relational set -- 0 by explicit construction
        "n_spatial_only": len(spatial_ids - experimental_relational_ids),
        "n_relational_only": len(experimental_relational_ids - spatial_ids),
        "n_neither": n_neither,
        "n_natural_relational_category": len(natural_relational_ids),
        "n_natural_intersection": len(natural_intersection),  # spatial vs. GQA's OWN "rel" category -- equals n_spatial, by GQA's real structure
        "natural_intersection_over_spatial": (len(natural_intersection) / len(spatial_ids)) if spatial_ids else 0.0,
        "natural_relational_note": (
            "GQA's own 'rel' semantic-type category naturally CONTAINS the spatial "
            "subset (every spatial question IS a relation-type question) -- this is GQA's "
            "actual category structure, not an artifact of this filter. n_intersection above "
            "(0 by construction) describes the two DISJOINT capability sets actually used; "
            "n_natural_intersection (== n_spatial) describes GQA's real, non-disjoint structure."
        ),
        "experimental_relational_definition": (
            "The relational_reasoning capability benchmark = natural relation-type questions "
            "EXCLUDING the spatial subset -- an explicit experimental choice made so the two "
            "capability benchmarks measure distinct skills, not a claim that GQA itself "
            "provides these as naturally disjoint categories."
        ),
    }
    return spatial_ids, experimental_relational_ids, stats


def persist_filter_ids(
    spatial_ids: Set[str], relational_ids: Set[str], stats: Dict[str, Any],
    spatial_path: "str | Path", relational_path: "str | Path", stats_path: "str | Path",
) -> None:
    for path in (spatial_path, relational_path, stats_path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(spatial_path).write_text(json.dumps(sorted(spatial_ids), indent=2))
    Path(relational_path).write_text(json.dumps(sorted(relational_ids), indent=2))
    Path(stats_path).write_text(json.dumps(stats, indent=2))


def load_persisted_filter_ids(path: "str | Path") -> List[str]:
    path = Path(path)
    if not path.exists():
        raise GQASchemaError(f"No persisted GQA spatial/relational filter IDs found at {path}")
    return json.loads(path.read_text())

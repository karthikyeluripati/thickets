"""Derives the spatial_reasoning / relational_reasoning capability-benchmark ID filters from
GQA's RAW annotation metadata (types.semantic/types.structural/the "semantic" reasoning
program -- see benchmarks/adapters/gqa_raw_schema.py).

Two conceptually SEPARATE sources of GQA data (see CAPABILITY_BENCHMARK_GATE.md -- do not
conflate them):
  A. GQAHandler evaluation parquet/images (prepare_gqa_data.py) -- what the model is actually
     shown and scored against; a NARROWER column projection (id/imageId/question/answer/
     fullAnswer only) of the same underlying rows, with historical GQA-reproduction scoring
     semantics UNCHANGED by this script.
  B. Raw GQA annotations (this script) -- loaded a SECOND time, independently, read ONLY for
     question-type metadata to build the spatial/relational filter. Never used for
     prompting/scoring -- run_capability_benchmark_gate.py never imports this module.
The two are joined by GQA's own stable question ID: the raw row's "id" field and
GQAHandler's own output "question_id" field are confirmed to carry the identical underlying
GQA question-id string (see gqa_raw_schema.py's RAW_QUESTION_ID_FIELD docstring), even though
the two pipeline stages expose it under different key names.

Prerequisite: none, technically (this script only reads the raw HF dataset, not
prepare_gqa_data.py's parquet) -- but the resulting ID filters are only USEFUL once
prepare_gqa_data.py has also been run, since the spatial/relational adapters filter
GQAHandler's own loaded records by these same IDs.

Usage:
    python -m neural_thickets_repro.prepare_gqa_capability_filters --config configs/gqa_repro.yaml
    python -m neural_thickets_repro.prepare_gqa_capability_filters --config configs/gqa_repro.yaml --inspect-only
    python -m neural_thickets_repro.prepare_gqa_capability_filters --config configs/gqa_repro.yaml --audit-question-ids 201640614,201902997

`--audit-question-ids`: prints the full real raw record -- question, types.semantic/
structural, the raw `semantic` program, `semanticStr`, every extracted predicate with its
individual spatial/non-spatial classification, and the overall question classification
("spatial"/"relational"/"mixed"/"no_extractable_predicate"/"neither") -- for each given
question ID (see gqa_raw_schema.describe_question_classification()), and exits WITHOUT
persisting or even building the aggregate filters. A pure read-only diagnostic, used to
verify real semantic-program predicate values before trusting or regenerating the persisted
filter ID files.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .benchmarks.adapters.gqa_raw_schema import (
    build_spatial_relational_filters,
    describe_question_classification,
    inspect_raw_schema,
    persist_filter_ids,
)
from .config import load_config
from .prepare_gqa_data import DATASET_NAME, TEST_INSTRUCTIONS_CONFIG, _split_name

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "benchmark_subsets"


def load_raw_testdev_rows(sample_size: Optional[int] = None) -> List[dict]:
    """Loads the SAME testdev_balanced_instructions config/split prepare_gqa_data.py uses for
    testdev.parquet -- but keeps every raw field (including types/semantic), instead of
    prepare_gqa_data.py's own narrower id/imageId/question/answer/fullAnswer projection.
    """
    from datasets import load_dataset

    split_name = _split_name(TEST_INSTRUCTIONS_CONFIG)
    ds = load_dataset(DATASET_NAME, TEST_INSTRUCTIONS_CONFIG, split=split_name)
    if sample_size is not None:
        ds = ds.select(range(min(sample_size, len(ds))))
    return list(ds)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "gqa_repro.yaml"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--inspect-only", action="store_true",
        help="print the raw-schema report and filter stats, but do not persist any ID files",
    )
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="load only the first N raw rows (for a quick schema check); default loads the full testdev split",
    )
    parser.add_argument(
        "--audit-question-ids", default=None,
        help="comma-separated GQA question IDs to print full raw-record classification detail "
        "for (question, types.semantic/structural, semantic program, semanticStr, extracted "
        "relation name, and the spatial/relational decision + matched keyword) -- read-only, "
        "never persists or regenerates any filter file",
    )
    args = parser.parse_args(argv)

    load_config(args.config)  # confirms the config loads cleanly; not otherwise used here

    print(f"Loading raw GQA annotations (source={DATASET_NAME}, config={TEST_INSTRUCTIONS_CONFIG}) ...")
    raw_rows = load_raw_testdev_rows(args.sample_size)
    print(f"  loaded {len(raw_rows)} raw rows")

    if args.audit_question_ids:
        ids = [s.strip() for s in args.audit_question_ids.split(",") if s.strip()]
        print(f"\n=== Auditing {len(ids)} question ID(s) (read-only -- no filters persisted) ===")
        for qid in ids:
            detail = describe_question_classification(raw_rows, qid)
            print(json.dumps(detail, indent=2, default=str))
        return 0

    print("\n=== Step 1: raw schema inspection (required before trusting the filter) ===")
    schema_report = inspect_raw_schema(raw_rows)
    print(json.dumps(schema_report, indent=2, default=str))
    if not schema_report["assumed_schema_confirmed"]:
        print(
            "\nSTOP: assumed_schema_confirmed=False -- the expected types.semantic/"
            "types.structural fields were not found in a sample of the real raw rows. "
            "Update benchmarks/adapters/gqa_raw_schema.py's field-name constants to match "
            "before proceeding -- refusing to build a filter against fields that don't exist.",
            file=sys.stderr,
        )
        return 1

    print("\n=== Step 2: building the high-purity spatial/relational/mixed partition ===")
    spatial_ids, relational_ids, mixed_ids, stats = build_spatial_relational_filters(raw_rows)
    print(json.dumps(stats, indent=2))
    if not stats["sum_check_ok"]:
        print(
            "\nWARNING: sum_check_ok is False -- n_not_relation_type + n_pure_spatial + "
            "n_pure_relational + n_mixed_excluded + n_no_extractable_predicate does not equal "
            "n_total_rows. Investigate before trusting this run's counts.",
            file=sys.stderr,
        )

    if args.inspect_only:
        print("\n--inspect-only: not persisting any filter files.")
        return 0

    output_dir = Path(args.output_dir)
    spatial_path = output_dir / "gqa_spatial_ids.json"
    relational_path = output_dir / "gqa_relational_ids.json"
    mixed_path = output_dir / "gqa_mixed_ids.json"
    stats_path = output_dir / "gqa_spatial_relational_stats.json"

    persist_filter_ids(spatial_ids, relational_ids, mixed_ids, stats, spatial_path, relational_path, mixed_path, stats_path)

    print(f"\nWrote {spatial_path} ({len(spatial_ids)} IDs)")
    print(f"Wrote {relational_path} ({len(relational_ids)} IDs)")
    print(f"Wrote {mixed_path} ({len(mixed_ids)} IDs -- excluded from both capabilities, persisted for auditability)")
    print(f"Wrote {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

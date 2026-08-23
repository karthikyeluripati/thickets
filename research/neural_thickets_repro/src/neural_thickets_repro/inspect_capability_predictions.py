"""Lightweight CPU-side manual-audit utility for a Capability Benchmark Gate
predictions.jsonl (see benchmarks/runner.py:write_predictions_jsonl) -- prints a compact
per-example report for manual inspection BEFORE trusting an N=5/N=200 run's aggregate
metrics.

Every real bug this repair pass fixed (the grounding coordinate-contract mismatch, the GQA
`\\boxed{}` parser fabricating "step step", the RefCOCO question/answer field mixup, the
Visual Genome category-vs-value confusion) was found by manually reading raw_generation /
parsed_prediction / target side by side on a small sample -- NOT by any aggregate metric
(parser_failure_rate was reported as 0 throughout). This tool makes that manual read faster;
it does not compute anything new or replace it.

Reads one JSONL file, pure Python -- no GPU/ray/vllm import, no network access. Not a
visualization tool (no image rendering) -- see CAPABILITY_BENCHMARK_GATE.md for that as a
possible future addition.

Usage:
    python -m neural_thickets_repro.inspect_capability_predictions --predictions path/to/predictions.jsonl
    python -m neural_thickets_repro.inspect_capability_predictions --predictions ... --capability visual_grounding --format json
    python -m neural_thickets_repro.inspect_capability_predictions --predictions ... --limit 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# Capability-specific extra fields to surface, pulled from each prediction record's own
# "detail" (ExampleScore.detail) and "metadata" (Example.metadata) dicts -- see each
# adapter's score_example()/load_examples() for where these are actually populated. A key
# missing from a given row (e.g. an older predictions.jsonl written before a field existed)
# is silently skipped, never a hard error -- this is a read-only audit tool.
CAPABILITY_EXTRA_FIELDS: Dict[str, Dict[str, List[str]]] = {
    "visual_grounding": {
        "detail": ["iou", "raw_prediction_box", "canonical_prediction_box", "coordinate_mode"],
        "metadata": ["bbox_pixels_xywh", "image_width", "image_height", "all_referring_expressions"],
    },
    "spatial_reasoning": {"detail": ["extracted", "extraction_mode"], "metadata": []},
    "relational_reasoning": {"detail": ["extracted", "extraction_mode"], "metadata": []},
    "ocr_text_recognition": {"detail": ["vqa_soft_accuracy"], "metadata": ["ocr_tokens", "ocr_grounded"]},
    "ocr_text_recognition_grounded": {"detail": ["vqa_soft_accuracy"], "metadata": ["ocr_tokens", "ocr_grounded"]},
    "attribute_recognition": {
        "detail": ["matched_attribute", "valid_targets"],
        "metadata": ["bbox_xywh", "object_id", "image_id", "raw_positive_attributes", "flagged_state_action_attributes"],
    },
    "counting": {"detail": [], "metadata": []},
    "fine_grained_recognition": {"detail": [], "metadata": []},
    "object_recognition": {"detail": [], "metadata": []},
}


def load_predictions(path: "str | Path") -> List[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def build_report_rows(predictions: List[dict], capability: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    extra_spec = CAPABILITY_EXTRA_FIELDS.get(capability, {})
    selected = predictions[:limit] if limit is not None else predictions

    rows: List[Dict[str, Any]] = []
    for p in selected:
        detail = p.get("detail") or {}
        metadata = p.get("metadata") or {}
        row: Dict[str, Any] = {
            "example_id": p.get("example_id"),
            "query": p.get("query"),
            "target": p.get("target"),
            "raw_generation": p.get("raw_generation"),
            "parsed_prediction": p.get("parsed_prediction"),
            "score": p.get("per_example_score"),
            "correct": p.get("correct"),
        }
        for key in extra_spec.get("detail", []):
            if key in detail:
                row[key] = detail[key]
        for key in extra_spec.get("metadata", []):
            if key in metadata:
                row[key] = metadata[key]
        rows.append(row)
    return rows


def render_json(rows: List[Dict[str, Any]]) -> str:
    return json.dumps(rows, indent=2, default=str)


def render_table(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "(no predictions)"
    lines: List[str] = []
    for row in rows:
        lines.append(f"--- {row.get('example_id')} (score={row.get('score')}, correct={row.get('correct')}) ---")
        for key, value in row.items():
            if key in ("example_id", "score", "correct"):
                continue
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="path to a predictions.jsonl written by run_capability_benchmark_gate.py")
    parser.add_argument("--capability", default=None, help="capability name -- selects which capability-specific detail/metadata fields to surface; omit to show only the generic fields")
    parser.add_argument("--limit", type=int, default=None, help="only show the first N predictions")
    parser.add_argument("--format", choices=["table", "json"], default="table")
    args = parser.parse_args(argv)

    predictions = load_predictions(args.predictions)
    rows = build_report_rows(predictions, capability=args.capability, limit=args.limit)
    print(render_json(rows) if args.format == "json" else render_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

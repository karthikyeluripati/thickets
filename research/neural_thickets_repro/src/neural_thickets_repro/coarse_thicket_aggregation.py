"""Coarse-map aggregation: reads completed scoped-run thicket_metrics.json files and builds
one scope-comparison table. Hard-fails if the runs being compared aren't actually comparable
(different task/model/dataset subset/N/seed/radius/scoring/candidate-seed-sequence) --
prevents silently plotting an invalid apples-to-oranges comparison.

Pure Python, no GPU/ray/vllm dependency -- CPU-testable directly against synthetic
thicket_metrics-shaped dicts. Does NOT run anything, sweep anything, or plot anything -- see
analysis/aggregate_coarse_thicket.py for the thin CLI wrapper around this module.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

# Fields that must be IDENTICAL across every run being compared in one coarse-map table --
# "scope" is deliberately excluded, since varying scope is the entire point of this table.
_SCALAR_COMPARABLE_FIELDS = (
    "task",
    "model_name",
    "model_revision",
    "N",
    "global_seed",
    "requested_relative_l2",
    "scoring_protocol",
    "dataset_revision",
    "dataset_selection_split",
    "selection_set_size",
)
# Sequence fields, compared as ordered tuples (order matters -- "same subset" also means
# "loaded/sampled in the same order", not merely the same set).
_SEQUENCE_COMPARABLE_FIELDS = ("candidate_seed_sequence", "selection_example_ids")


class AggregationMismatchError(RuntimeError):
    """Runs given to the aggregator are not directly comparable -- refuses to build a table
    that would silently mix results from different tasks/datasets/radii/candidate streams.
    """


def load_thicket_metrics(path: "str | Path") -> Dict:
    """path may be a thicket_metrics.json file itself, or the run directory containing one."""
    path = Path(path)
    if path.is_dir():
        path = path / "thicket_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"No thicket_metrics.json found at {path}")
    return json.loads(path.read_text())


def assert_comparable(runs: List[Dict]) -> None:
    if len(runs) < 2:
        return

    for field in _SCALAR_COMPARABLE_FIELDS:
        values = {run.get(field) for run in runs}
        if len(values) > 1:
            mismatch = {run.get("scope", "<unknown scope>"): run.get(field) for run in runs}
            raise AggregationMismatchError(
                f"Runs differ in {field!r}, which must be identical across a coarse-map "
                f"comparison (only 'scope' is expected to vary): {mismatch}"
            )

    for field in _SEQUENCE_COMPARABLE_FIELDS:
        values = {tuple(run.get(field) or []) for run in runs}
        if len(values) > 1:
            mismatch = {run.get("scope", "<unknown scope>"): run.get(field) for run in runs}
            raise AggregationMismatchError(
                f"Runs differ in {field!r} -- comparing scopes is only valid when every run "
                f"used the identical ordered sequence: {mismatch}"
            )


def build_coarse_map_rows(runs: List[Dict]) -> List[Dict]:
    """One row per run, ready for tabular display. Does not itself enforce comparability --
    call assert_comparable() first.
    """
    rows = []
    for run in runs:
        rows.append({
            "scope": run["scope"],
            "r": run.get("requested_relative_l2"),
            "N": run["N"],
            "base": run["base_score"],
            "density": run["expert_density"],
            "ci_lower": run["expert_density_ci_95"][0],
            "ci_upper": run["expert_density_ci_95"][1],
            "mean_delta": run["mean_delta"],
            "median_delta": run["median_delta"],
            "best_delta": run["best_candidate_score"] - run["base_score"],
        })
    return rows


# Canonical display order -- falls back to encounter order for any scope not in this list,
# so an unrecognized/future scope name still renders rather than being dropped.
_SCOPE_DISPLAY_ORDER = ("full_lm", "vision_encoder", "vision_merger", "lm_early", "lm_middle", "lm_late", "full_vlm")


def format_table(rows: List[Dict]) -> str:
    ordered = sorted(
        rows,
        key=lambda r: (_SCOPE_DISPLAY_ORDER.index(r["scope"]) if r["scope"] in _SCOPE_DISPLAY_ORDER else len(_SCOPE_DISPLAY_ORDER), r["scope"]),
    )

    headers = ["scope", "r", "N", "base", "density", "CI", "mean_delta", "median_delta", "best_delta"]
    lines = []

    def _fmt_row(values: List[str]) -> str:
        return "  ".join(v.ljust(w) for v, w in zip(values, widths))

    formatted_rows = []
    for row in ordered:
        formatted_rows.append([
            row["scope"],
            f"{row['r']:g}" if row["r"] is not None else "-",
            str(row["N"]),
            f"{row['base']:.4f}",
            f"{row['density']:.4f}",
            f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]",
            f"{row['mean_delta']:+.4f}",
            f"{row['median_delta']:+.4f}",
            f"{row['best_delta']:+.4f}",
        ])

    widths = [max(len(headers[i]), *(len(r[i]) for r in formatted_rows)) if formatted_rows else len(headers[i]) for i in range(len(headers))]
    lines.append(_fmt_row(headers))
    for r in formatted_rows:
        lines.append(_fmt_row(r))
    return "\n".join(lines)

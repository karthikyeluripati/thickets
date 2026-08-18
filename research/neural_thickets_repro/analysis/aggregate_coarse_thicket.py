"""Thin CLI wrapper around neural_thickets_repro.coarse_thicket_aggregation -- reads
completed scoped-run thicket_metrics.json files and prints one scope-comparison table:

    scope            r       N    base    density    CI                mean_delta   median_delta   best_delta
    full_lm          0.01    20   0.5230  0.4500     [0.2400, 0.6800]  +0.0120      +0.0080         +0.0410
    vision_encoder   0.01    20   0.5230  0.1500     [0.0500, 0.3600]  -0.0050      -0.0020         +0.0210
    ...

Hard-fails (does not silently proceed) if the given runs differ in task, model revision, GQA
candidate subset, N, global seed, requested relative-L2, scoring protocol, or candidate seed
sequence -- see coarse_thicket_aggregation.assert_comparable. Not an orchestration system:
takes explicit run-directory paths, does nothing else (no discovery, no plotting).

Usage:
    python analysis/aggregate_coarse_thicket.py \
        results/scoped_randopt_N20_K5_full_lm_relative_l2_r0.01 \
        results/scoped_randopt_N20_K5_vision_encoder_relative_l2_r0.01 \
        ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from neural_thickets_repro.coarse_thicket_aggregation import (  # noqa: E402
    assert_comparable,
    build_coarse_map_rows,
    format_table,
    load_thicket_metrics,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_dirs", nargs="+",
        help="paths to scoped_randopt run output directories (or thicket_metrics.json files directly), one per scope",
    )
    parser.add_argument("--out", default=None, help="optional path to also write the table as plain text")
    args = parser.parse_args(argv)

    try:
        runs = [load_thicket_metrics(d) for d in args.run_dirs]
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        assert_comparable(runs)
    except Exception as exc:  # AggregationMismatchError
        print(f"Refusing to aggregate: {exc}", file=sys.stderr)
        return 1

    rows = build_coarse_map_rows(runs)
    table = format_table(rows)
    print(table)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(table)
        print(f"\nWrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

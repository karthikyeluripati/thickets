"""Builds the EXPERIMENTAL OCR-grounded subset of TextVQA validation (see
benchmarks/ocr_grounding.py) -- persists a fixed question-ID filter the SAME way
prepare_gqa_capability_filters.py does, so a later benchmark run reads a stable,
previously-computed ID set rather than recomputing (and potentially silently resampling) it
on every run.

NOT an official TextVQA category -- an explicit, documented, deterministic experimental
definition: "at least one reference answer is recoverable from the dataset's own OCR token
sequence" (see benchmarks/ocr_grounding.py). Uses target answers + the dataset's own provided
OCR tokens ONLY, never model predictions -- filter membership must never depend on how well
any model actually answers.

Usage:
    python -m neural_thickets_repro.prepare_textvqa_ocr_filter
    python -m neural_thickets_repro.prepare_textvqa_ocr_filter --sample-size 500
    python -m neural_thickets_repro.prepare_textvqa_ocr_filter --inspect-only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .benchmarks.ocr_grounding import is_ocr_grounded

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "benchmark_subsets"
DATASET_NAME = "lmms-lab-encoder/textvqa"
DEFAULT_SPLIT = "validation"


def load_textvqa_rows(split: str, sample_size: Optional[int] = None) -> List[dict]:
    """Loads the same lmms-lab-encoder/textvqa split the ocr_text_recognition adapter uses --
    a documented prefix slice when --sample-size is given, matching this project's existing
    "first N" convention elsewhere (prepare_gqa_data.py, prepare_visual_genome_data.py).
    """
    from datasets import load_dataset

    ds = load_dataset(DATASET_NAME, split=split)
    if sample_size is not None:
        ds = ds.select(range(min(sample_size, len(ds))))
    return list(ds)


def build_ocr_grounded_filter(rows: List[dict]) -> Tuple[List[str], Dict[str, Any]]:
    """Returns (retained_question_ids, stats). stats reports total/retained/rejected/percent
    so a fresh run's real numbers are visible before anything is persisted or trusted.
    """
    retained: List[str] = []
    for row in rows:
        if is_ocr_grounded(row.get("answers") or [], row.get("ocr_tokens") or []):
            retained.append(str(row["question_id"]))

    total = len(rows)
    stats = {
        "total_examples": total,
        "retained": len(retained),
        "rejected": total - len(retained),
        "percent_retained": (len(retained) / total * 100.0) if total else 0.0,
    }
    return retained, stats


def persist_ocr_grounded_ids(ids: List[str], stats: Dict[str, Any], ids_path: Path, stats_path: Path) -> None:
    for path in (ids_path, stats_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    ids_path.write_text(json.dumps(sorted(ids), indent=2))
    stats_path.write_text(json.dumps(stats, indent=2))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--sample-size", type=int, default=None, help="load only the first N rows (quick check); default loads the full split")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--inspect-only", action="store_true", help="print stats but do not persist the ID file")
    args = parser.parse_args(argv)

    print(f"Loading {DATASET_NAME} split={args.split!r} (sample_size={args.sample_size}) ...")
    rows = load_textvqa_rows(args.split, args.sample_size)
    print(f"  loaded {len(rows)} rows")

    retained, stats = build_ocr_grounded_filter(rows)
    print(json.dumps(stats, indent=2))

    if args.inspect_only:
        print("\n--inspect-only: not persisting any filter file.")
        return 0

    output_dir = Path(args.output_dir)
    ids_path = output_dir / "textvqa_ocr_grounded_ids.json"
    stats_path = output_dir / "textvqa_ocr_grounded_stats.json"
    persist_ocr_grounded_ids(retained, stats, ids_path, stats_path)

    print(f"\nWrote {ids_path} ({len(retained)} IDs)")
    print(f"Wrote {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Verifies that prepared Visual Genome attribute-recognition artifacts (see
prepare_visual_genome_data.py) are complete and readable before trusting them for the
Capability Benchmark Gate -- same motivation and structure as verify_gqa_data.py: don't trust
prepare_visual_genome_data.py's own exit status, re-derive every check from what actually
landed on disk.

This is defense-in-depth, not a re-implementation of prepare's filtering: the checks below
(invalid bbox / empty object name / empty attribute set / duplicate instance id) should all
read as zero on a correctly prepared artifact, since prepare_visual_genome_data.py already
excludes these at flatten time -- a non-zero count here means the parquet was hand-edited, or
prepare has a bug, either of which should block the gate.

Usage:
    python -m neural_thickets_repro.verify_visual_genome_data
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

from .prepare_visual_genome_data import DEFAULT_DATA_DIR, VG_DATASET_CONFIG, VG_DATASET_NAME, validate_bbox

REQUIRED_COLUMNS = {"image_id", "instance_id", "object_name", "positive_attributes", "bbox_x", "bbox_y", "bbox_w", "bbox_h"}
# Optional -- present in artifacts written by the current prepare_visual_genome_data.py, but
# not required for an artifact to be usable (bbox-vs-image-bounds re-validation and the
# prepare_stats cross-check are simply skipped if these aren't there).
OPTIONAL_COLUMNS = {"image_width", "image_height", "url"}


def verify_visual_genome_data(data_dir: Path) -> Dict:
    import pandas as pd
    from PIL import Image, UnidentifiedImageError

    parquet_path = data_dir / "vg_attributes.parquet"
    images_dir = data_dir / "images"
    stats_path = data_dir / "vg_prepare_stats.json"

    result: Dict = {
        "data_dir": str(data_dir),
        "dataset_source": f"{VG_DATASET_NAME} (config={VG_DATASET_CONFIG!r})",
        "parquet_path": str(parquet_path),
        "parquet_exists": parquet_path.exists(),
        "prepare_stats": None,
        "n_candidate_images": None,
        "n_flattened_examples": None,
        "n_selected": None,
        "row_count": None,
        "missing_columns": [],
        "duplicate_instance_ids": [],
        "rows_with_empty_positive_attributes": 0,
        "rows_with_empty_object_name": 0,
        "rows_with_invalid_bbox": 0,
        "attribute_cardinality_distribution": {},
        "examples_with_multiple_attributes": 0,
        "unique_image_ids": None,
        "images_present": None,
        "images_missing": [],
        "images_corrupt": [],
        "ok": False,
    }
    if stats_path.exists():
        result["prepare_stats"] = json.loads(stats_path.read_text())

    if not parquet_path.exists():
        return result

    df = pd.read_parquet(parquet_path)
    result["row_count"] = len(df)
    # "number selected" here means: examples that survived prepare_visual_genome_data.py's own
    # flatten-time filtering and are available for the (separate, later) fixed-N benchmark
    # subset selection -- NOT the final N=200 the gate itself draws from this pool.
    result["n_flattened_examples"] = len(df)
    result["n_selected"] = len(df)

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    result["missing_columns"] = sorted(missing_columns)
    if missing_columns:
        return result

    duplicate_ids = df["instance_id"][df["instance_id"].duplicated()].unique().tolist()
    result["duplicate_instance_ids"] = duplicate_ids

    attribute_lists = [json.loads(v) for v in df["positive_attributes"]]
    result["rows_with_empty_positive_attributes"] = sum(1 for a in attribute_lists if not a)
    result["rows_with_empty_object_name"] = int((df["object_name"].astype(str).str.strip() == "").sum())

    cardinality_dist: Dict[str, int] = {}
    for attrs in attribute_lists:
        key = str(len(attrs))
        cardinality_dist[key] = cardinality_dist.get(key, 0) + 1
    result["attribute_cardinality_distribution"] = dict(sorted(cardinality_dist.items(), key=lambda kv: int(kv[0])))
    result["examples_with_multiple_attributes"] = sum(1 for a in attribute_lists if len(a) > 1)

    if {"image_width", "image_height"} <= set(df.columns):
        invalid_bbox = 0
        for _, row in df.iterrows():
            if not validate_bbox(row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"], row["image_width"], row["image_height"]):
                invalid_bbox += 1
        result["rows_with_invalid_bbox"] = invalid_bbox

    image_ids = set(df["image_id"].astype(str))
    result["unique_image_ids"] = len(image_ids)
    result["n_candidate_images"] = len(image_ids)

    present = 0
    for img_id in sorted(image_ids):
        img_path = images_dir / f"{img_id}.jpg"
        if not img_path.exists():
            result["images_missing"].append(img_id)
            continue
        try:
            with Image.open(img_path) as im:
                im.verify()
            present += 1
        except (UnidentifiedImageError, OSError) as exc:
            result["images_corrupt"].append({"image_id": img_id, "error": str(exc)})
    result["images_present"] = present

    result["ok"] = (
        result["parquet_exists"]
        and not missing_columns
        and not duplicate_ids
        and result["rows_with_empty_positive_attributes"] == 0
        and result["rows_with_empty_object_name"] == 0
        and result["rows_with_invalid_bbox"] == 0
        and not result["images_missing"]
        and not result["images_corrupt"]
    )
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    args = parser.parse_args(argv)

    report = verify_visual_genome_data(Path(args.data_dir))
    print(json.dumps(report, indent=2, default=str))

    if not report["ok"]:
        print("\nVISUAL GENOME DATA VERIFICATION FAILED -- do not proceed until this is fixed.", file=sys.stderr)
        return 1
    print("\nVisual Genome data verification PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

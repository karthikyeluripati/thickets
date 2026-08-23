"""Verifies that prepared Visual Genome attribute-recognition artifacts (see
prepare_visual_genome_data.py) are complete and readable before trusting them for the
Capability Benchmark Gate -- same motivation and structure as verify_gqa_data.py.

Usage:
    python -m neural_thickets_repro.verify_visual_genome_data
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

from .prepare_visual_genome_data import DEFAULT_DATA_DIR, REQUIRED_VAW_COLUMNS

REQUIRED_COLUMNS = {"image_id", "instance_id", "object_name", "positive_attributes", "bbox_x", "bbox_y", "bbox_w", "bbox_h"}


def verify_visual_genome_data(data_dir: Path) -> Dict:
    import pandas as pd
    from PIL import Image, UnidentifiedImageError

    parquet_path = data_dir / "vg_attributes.parquet"
    images_dir = data_dir / "images"

    result: Dict = {
        "data_dir": str(data_dir),
        "parquet_path": str(parquet_path),
        "parquet_exists": parquet_path.exists(),
        "row_count": None,
        "missing_columns": [],
        "duplicate_instance_ids": [],
        "rows_with_empty_positive_attributes": 0,
        "unique_image_ids": None,
        "images_present": None,
        "images_missing": [],
        "images_corrupt": [],
        "ok": False,
    }
    if not parquet_path.exists():
        return result

    df = pd.read_parquet(parquet_path)
    result["row_count"] = len(df)

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    result["missing_columns"] = sorted(missing_columns)
    if missing_columns:
        return result

    duplicate_ids = df["instance_id"][df["instance_id"].duplicated()].unique().tolist()
    result["duplicate_instance_ids"] = duplicate_ids

    empty_attrs = sum(1 for v in df["positive_attributes"] if not json.loads(v))
    result["rows_with_empty_positive_attributes"] = empty_attrs

    image_ids = set(df["image_id"].astype(str))
    result["unique_image_ids"] = len(image_ids)

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
        and empty_attrs == 0
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

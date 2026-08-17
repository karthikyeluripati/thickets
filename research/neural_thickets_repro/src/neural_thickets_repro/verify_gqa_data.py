"""Verifies that prepared GQA parquet/image artifacts (see prepare_gqa_data.py) are
complete and readable before trusting them for Gate 1.

Motivation: a noisy/uncertain exit from prepare_gqa_data.py (e.g. a shutdown-time
interpreter abort happening AFTER the script's own "Done." print) does not by itself prove
the actual data writing failed -- Python interpreter teardown issues in third-party C
extensions can fire after all real work has completed. This tool checks the actual
artifacts on disk (row counts, required columns, every referenced image present and not
corrupt) instead of inferring anything from the prep script's exit behavior.

Usage:
    python -m neural_thickets_repro.verify_gqa_data --config configs/gqa_repro.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

from .config import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_COLUMNS = {"id", "imageId", "question", "answer", "fullAnswer"}


def _verify_split(parquet_path: Path, images_dir: Path, expected_rows: Optional[int]) -> Dict:
    import pandas as pd
    from PIL import Image, UnidentifiedImageError

    result: Dict = {
        "parquet_path": str(parquet_path),
        "parquet_exists": parquet_path.exists(),
        "row_count": None,
        "expected_row_count": expected_rows,
        "row_count_matches": None,
        "missing_columns": [],
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
    if expected_rows is not None:
        result["row_count_matches"] = len(df) == expected_rows

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    result["missing_columns"] = sorted(missing_columns)

    image_ids = set(df["imageId"].astype(str)) if "imageId" in df.columns else set()
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
            result["images_corrupt"].append({"imageId": img_id, "error": str(exc)})
    result["images_present"] = present

    result["ok"] = (
        result["parquet_exists"]
        and not missing_columns
        and result["row_count_matches"] is not False
        and not result["images_missing"]
        and not result["images_corrupt"]
    )
    return result


def verify_gqa_data(data_dir: Path, selection_set_size: int, test_set_size: Optional[int]) -> Dict:
    testdev = _verify_split(data_dir / "testdev.parquet", data_dir / "images", expected_rows=test_set_size)
    train_selection = _verify_split(data_dir / "train.parquet", data_dir / "images", expected_rows=selection_set_size)
    return {
        "data_dir": str(data_dir),
        "testdev": testdev,
        "train_selection": train_selection,
        "overall_ok": testdev["ok"] and train_selection["ok"],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "gqa_repro.yaml"))
    parser.add_argument(
        "--data-dir",
        default=str(REPO_ROOT / "external" / "RandOpt" / "data" / "gqa"),
        help="where prepare_gqa_data.py wrote {train,testdev}.parquet and images/",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    report = verify_gqa_data(Path(args.data_dir), cfg.dataset.selection_set_size, cfg.dataset.test_set_size)

    print(json.dumps(report, indent=2, default=str))

    if not report["overall_ok"]:
        print("\nGQA DATA VERIFICATION FAILED -- do not proceed to Gate 1 until this is fixed.", file=sys.stderr)
        return 1
    print("\nGQA data verification PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

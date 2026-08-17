"""Resolves REPRO_SPEC.md's "Dataset revision / exact subset" row: derives
data/gqa/{train,testdev}.parquet + data/gqa/images/{imageId}.jpg in the shape
external/RandOpt/data_handlers/gqa.py:GQAHandler.load_data expects.

REPRODUCTION ASSUMPTION (see REPRO_SPEC.md): upstream never showed how its own
train.parquet/testdev.parquet were produced, so this is a documented choice, not an
upstream-confirmed provenance:
  - test split      = testdev_balanced (12,578 questions over 398 images) -- the split GQA
    papers conventionally report a single headline accuracy number against, and the one
    upstream's own file name ("testdev.parquet") most plausibly refers to.
  - selection split = the first N rows (N = dataset.selection_set_size, default 200) of
    train_balanced (943,000 questions total), matching paper Appendix E.3 ("first 200
    samples from the training set").
Source: HF dataset `lmms-lab-encoder/GQA` (renamed from `lmms-lab/GQA` -- confirmed via HTTP
redirect from the old name during this session), configs testdev_balanced_instructions /
testdev_balanced_images / train_balanced_instructions / train_balanced_images.

If the paper/repo authors ever clarify the actual split used, REPRO_SPEC.md and this script
should be updated together -- do not silently change one without the other.

Usage:
    python -m neural_thickets_repro.prepare_gqa_data --config configs/gqa_repro.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Set

from .config import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]

DATASET_NAME = "lmms-lab-encoder/GQA"  # renamed from lmms-lab/GQA -- see REPRO_SPEC.md
TEST_INSTRUCTIONS_CONFIG = "testdev_balanced_instructions"
TEST_IMAGES_CONFIG = "testdev_balanced_images"
TRAIN_INSTRUCTIONS_CONFIG = "train_balanced_instructions"
TRAIN_IMAGES_CONFIG = "train_balanced_images"


def _split_name(config_name: str) -> str:
    from datasets import get_dataset_split_names

    return get_dataset_split_names(DATASET_NAME, config_name)[0]


def _write_instructions_parquet(rows: Iterable[dict], out_path: Path) -> List[str]:
    import pandas as pd

    records = [
        {
            "id": row.get("id"),
            "imageId": row.get("imageId"),
            "question": row.get("question"),
            "answer": row.get("answer"),
            "fullAnswer": row.get("fullAnswer", ""),
        }
        for row in rows
    ]
    df = pd.DataFrame.from_records(records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    return [r["imageId"] for r in records]


def _save_needed_images(images_config: str, needed_ids: Set[str], images_dir: Path) -> None:
    """Streams the (large) images config rather than downloading it in full, stopping as
    soon as every needed image has been found -- avoids pulling multi-GB image shards for
    a handful of unique images.
    """
    from datasets import load_dataset

    images_dir.mkdir(parents=True, exist_ok=True)
    remaining = set(needed_ids)
    if not remaining:
        return

    already_present = {p.stem for p in images_dir.glob("*.jpg")}
    remaining -= already_present
    if not remaining:
        print(f"  all {len(needed_ids)} images already present in {images_dir}")
        return

    split_name = _split_name(images_config)
    ds = load_dataset(DATASET_NAME, images_config, split=split_name, streaming=True)
    for row in ds:
        img_id = row["id"]
        if img_id in remaining:
            row["image"].convert("RGB").save(images_dir / f"{img_id}.jpg", "JPEG")
            remaining.discard(img_id)
            if not remaining:
                break
    if remaining:
        print(f"  WARNING: {len(remaining)} image id(s) not found in {images_config}: {sorted(remaining)[:10]}")


def prepare_testdev(data_dir: Path) -> None:
    from datasets import load_dataset

    split_name = _split_name(TEST_INSTRUCTIONS_CONFIG)
    rows = list(load_dataset(DATASET_NAME, TEST_INSTRUCTIONS_CONFIG, split=split_name))
    image_ids = _write_instructions_parquet(rows, data_dir / "testdev.parquet")
    print(f"testdev: {len(rows)} questions, {len(set(image_ids))} unique images")
    _save_needed_images(TEST_IMAGES_CONFIG, set(image_ids), data_dir / "images")


def prepare_train_selection(data_dir: Path, selection_set_size: int) -> None:
    from datasets import load_dataset

    split_name = _split_name(TRAIN_INSTRUCTIONS_CONFIG)
    full = load_dataset(DATASET_NAME, TRAIN_INSTRUCTIONS_CONFIG, split=split_name)
    rows = list(full.select(range(selection_set_size)))
    image_ids = _write_instructions_parquet(rows, data_dir / "train.parquet")
    print(f"train (selection set): {len(rows)} questions, {len(set(image_ids))} unique images")
    _save_needed_images(TRAIN_IMAGES_CONFIG, set(image_ids), data_dir / "images")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "gqa_repro.yaml"))
    parser.add_argument(
        "--data-dir",
        default=str(REPO_ROOT / "external" / "RandOpt" / "data" / "gqa"),
        help="where GQAHandler expects data/gqa/{train,testdev}.parquet and data/gqa/images/",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    data_dir = Path(args.data_dir)

    print(f"Preparing GQA data into {data_dir} (source: {DATASET_NAME}) ...")
    prepare_testdev(data_dir)
    prepare_train_selection(data_dir, cfg.dataset.selection_set_size)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

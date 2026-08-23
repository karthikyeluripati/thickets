"""Prepares attribute_recognition capability-benchmark data into
external/RandOpt/data/visual_genome/{vg_attributes.parquet,images/,vg_prepare_stats.json} --
mirrors prepare_gqa_data.py's own "prepare once offline, adapter reads the prepared local
artifact" pattern, for the same reason: bounding the per-run cost and keeping the evaluated
pool deterministic and inspectable on disk.

SOURCE HISTORY (see CAPABILITY_BENCHMARK_GATE.md for the full record):
  1. `ranjaykrishna/visual_genome` (config attributes_v1.2.0) -- the canonical VG-authors repo
     -- is script-based and FAILS on modern `datasets` with "Dataset scripts are no longer
     supported"; confirmed on a real RunPod, datasets==5.0.1.
  2. `mikewang/vaw` (VAW) was tried as a script-free replacement, but ALSO turned out to still
     ship a `vaw.py` loading script -- confirmed FAILING on a real RunPod with datasets==5.0.1
     for the same "Dataset scripts are no longer supported" reason. Never load this repo.
  3. `AnnaZ1103/visual_genome_revised` (config "attributes", split "train") -- CONFIRMED
     WORKING on a real RunPod with datasets==5.0.1, script-free Parquet. This is a community
     repackaging of Visual Genome's own attribute annotations, not the canonical upstream VG
     distribution -- documented here and in known_caveats(), never claimed otherwise.

Real observed schema (105414 rows), confirmed on a real RunPod this session:
    {
        "image_id": int64, "url": string, "width": int64, "height": int64,
        "coco_id": int64, "flickr_id": int64,
        "attributes": [
            {"attributes": [string], "h": int64, "names": [string], "object_id": int64,
             "synsets": [string], "w": int64, "x": int64, "y": int64}
        ],
    }
One row = one image; "attributes" is a list of annotated objects in that image, each with its
own bbox (x, y, w, h), a list of synonym names, and a list of positive visual attributes.

DESIGN CHANGE from the VAW-era version of this script: this dataset carries no separate image
column, but it DOES carry each image's own canonical `url` (Stanford-hosted originals) --
so images are fetched ONE AT A TIME, by URL, only for images actually referenced by the
flattened (image, object) examples this script selects. The previous "download the entire
VG_100K(_2).zip archive" design is intentionally removed: it is unnecessary now that a
per-image URL is available directly in the annotation rows, and downloading the full ~100k
image archive to evaluate ~hundreds of examples was always wasteful.

Object-level filtering (documented, deterministic, NOT model-performance-driven):
  - Objects with an empty `attributes` list carry no usable target -- excluded.
  - Objects with an empty `names` list have no queryable object name -- excluded.
  - Objects with a bbox that fails validate_bbox() (see below) -- excluded.
  No other semantic filtering is applied. In particular, VG's raw attribute vocabulary is NOT
  filtered for "visual appearance vs. action/state" (e.g. "walking" is a real observed VG
  attribute value that is arguably a state/action, not a visual appearance attribute) -- there
  is currently no existing, defensible criterion in this codebase for that distinction, and
  inventing one here would be silent ontology engineering, not a wiring repair. This is
  flagged explicitly in known_caveats() as a scientific-review item for the upcoming N=5
  manual inspection pass, not silently resolved.

`object_name` is taken as the FIRST entry of the object's `names` list -- VG objects can carry
multiple synonym names (e.g. ["man", "person"]); using only the first is a documented,
revisable simplification for building an unambiguous single-noun prompt, entirely separate
from (and not to be confused with) the multi-attribute TARGET set, which is always preserved
in full (see rows_with_positive_attributes-equivalent logic in flatten_attribute_examples).

Usage:
    python -m neural_thickets_repro.prepare_visual_genome_data
    python -m neural_thickets_repro.prepare_visual_genome_data --max-candidates 500 --split train
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "external" / "RandOpt" / "data" / "visual_genome"

VG_DATASET_NAME = "AnnaZ1103/visual_genome_revised"
VG_DATASET_CONFIG = "attributes"
DEFAULT_SPLIT = "train"
# Prefix-slice bound on raw image ROWS loaded from the dataset (each row can hold several
# annotated objects) -- bounds dataset-load size and the number of images downloaded. This is
# NOT the final benchmark subset size (that is the separate, later `shuffled_prefix` N=200
# selection the Capability Benchmark Gate applies on top of whatever this script produces) --
# it only needs to be large enough that flattening yields comfortably more than 200 examples.
DEFAULT_MAX_CANDIDATES = 500

REQUIRED_ROW_COLUMNS = {"image_id", "url", "width", "height", "attributes"}
REQUIRED_OBJECT_FIELDS = {"attributes", "h", "names", "object_id", "w", "x", "y"}

# VG's original crowd-sourced bbox annotations are a widely documented source of occasional
# off-by-a-pixel-or-two overflow past the image's own width/height (annotator/UI rounding) --
# a tiny tolerance is allowed ONLY on the two upper-bound checks below, never on the strictly-
# positive lower-bound checks (x >= 0, y >= 0, w > 0, h > 0), which a genuinely malformed
# annotation would violate outright.
BBOX_UPPER_BOUND_TOLERANCE_PX = 1


class VisualGenomeDataError(RuntimeError):
    """Raised for a malformed dataset row (missing required column/field) or a failed image
    download -- never silently skipped or guessed around.
    """


def validate_bbox(x: float, y: float, w: float, h: float, image_width: float, image_height: float) -> bool:
    """x, y, w, h are treated as absolute pixel coordinates: top-left corner (x, y), width w,
    height h. See BBOX_UPPER_BOUND_TOLERANCE_PX for the one documented tolerance.
    """
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        return False
    if x + w > image_width + BBOX_UPPER_BOUND_TOLERANCE_PX:
        return False
    if y + h > image_height + BBOX_UPPER_BOUND_TOLERANCE_PX:
        return False
    return True


def load_attribute_rows(split: str, max_candidates: "int | None") -> List[dict]:
    """Loads AnnaZ1103/visual_genome_revised (config "attributes", script-free Parquet) -- a
    documented prefix slice of `split`, same "first N rows" convention prepare_gqa_data.py
    already uses for its own selection split, bounding the number of images this preparation
    step can possibly need to fetch.
    """
    from datasets import load_dataset

    ds = load_dataset(VG_DATASET_NAME, VG_DATASET_CONFIG, split=split)
    missing = REQUIRED_ROW_COLUMNS - set(ds.column_names)
    if missing:
        raise VisualGenomeDataError(
            f"{VG_DATASET_NAME} (config={VG_DATASET_CONFIG!r}) split {split!r} is missing "
            f"expected column(s) {sorted(missing)} (found: {ds.column_names}) -- refusing to "
            f"guess a different schema."
        )
    if max_candidates is not None:
        ds = ds.select(range(min(max_candidates, len(ds))))
    return list(ds)


def flatten_attribute_examples(image_rows: Iterable[dict]) -> Tuple[List[dict], Dict[str, int]]:
    """Flattens each image row's nested `attributes` object list into one flat record per
    (image, object) pair, applying the three documented, deterministic exclusion rules from
    this module's docstring. Returns (kept_records, stats) -- stats always reports every
    skip-reason count, even when zero, so callers/tests can assert on it precisely rather than
    parsing print output.
    """
    kept: List[dict] = []
    stats = {"n_image_rows": 0, "n_objects_seen": 0, "skipped_no_attributes": 0, "skipped_empty_name": 0, "skipped_invalid_bbox": 0}

    for image_row in image_rows:
        stats["n_image_rows"] += 1
        objects = image_row.get("attributes") or []
        for obj in objects:
            stats["n_objects_seen"] += 1
            missing_fields = REQUIRED_OBJECT_FIELDS - set(obj.keys())
            if missing_fields:
                raise VisualGenomeDataError(
                    f"object_id={obj.get('object_id')!r} in image_id={image_row.get('image_id')!r} "
                    f"is missing expected field(s) {sorted(missing_fields)} -- refusing to guess a different schema."
                )

            positive_attributes = list(obj.get("attributes") or [])
            if not positive_attributes:
                stats["skipped_no_attributes"] += 1
                continue

            names = obj.get("names") or []
            if not names:
                stats["skipped_empty_name"] += 1
                continue

            x, y, w, h = obj["x"], obj["y"], obj["w"], obj["h"]
            if not validate_bbox(x, y, w, h, image_row["width"], image_row["height"]):
                stats["skipped_invalid_bbox"] += 1
                continue

            kept.append({
                "image_id": str(image_row["image_id"]),
                "url": image_row["url"],
                "image_width": image_row["width"],
                "image_height": image_row["height"],
                "instance_id": str(obj["object_id"]),
                "object_name": names[0],  # documented simplification -- see module docstring
                "positive_attributes": positive_attributes,  # FULL set preserved, never collapsed
                "bbox_x": x, "bbox_y": y, "bbox_w": w, "bbox_h": h,
            })

    return kept, stats


def needed_images_with_urls(rows: Iterable[dict]) -> Dict[str, str]:
    """Deduplicates (image_id -> url); several flattened rows commonly share the same image."""
    mapping: Dict[str, str] = {}
    for row in rows:
        mapping[row["image_id"]] = row["url"]
    return mapping


def write_attributes_parquet(rows: List[dict], out_path: Path) -> None:
    import pandas as pd

    records = [
        {
            "image_id": row["image_id"],
            "instance_id": row["instance_id"],
            "object_name": row["object_name"],
            "positive_attributes": json.dumps(row["positive_attributes"]),  # JSON-encoded list, decoded by the adapter
            "bbox_x": row["bbox_x"], "bbox_y": row["bbox_y"], "bbox_w": row["bbox_w"], "bbox_h": row["bbox_h"],
            "image_width": row["image_width"], "image_height": row["image_height"],
            "url": row["url"],  # kept for reproducibility -- lets a fresh image fetch be re-derived without re-downloading the dataset
        }
        for row in rows
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_parquet(out_path)


def write_prepare_stats(stats: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stats, indent=2))


def download_file(url: str, dest_path: Path) -> None:
    """Thin, deliberately untested-beyond-mocking network glue -- the actual HTTP call.
    Everything upstream/downstream of this (which images are needed, where they're cached) is
    pure logic covered by tests.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest_path)
    except Exception as exc:  # noqa: BLE001 -- re-raised with an actionable message below
        raise VisualGenomeDataError(
            f"Failed to download image from {url} (this URL comes directly from the "
            f"{VG_DATASET_NAME} dataset row's own `url` field, not a script default). "
            f"{type(exc).__name__}: {exc}. Re-running this script is safe/idempotent -- "
            f"already-downloaded images are skipped -- so a transient network failure or a "
            f"single stale/moved image URL can simply be retried."
        ) from exc


def fetch_needed_images(id_to_url: Dict[str, str], images_dir: Path) -> None:
    """Downloads only the images actually referenced by the flattened examples, one at a
    time by URL -- never the full VG image archive. Already-cached images (from a previous
    run of this script) are skipped, making re-runs cheap and idempotent.
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    already_present = {p.stem for p in images_dir.glob("*.jpg")}
    remaining = {image_id: url for image_id, url in id_to_url.items() if image_id not in already_present}
    if not remaining:
        print(f"  all {len(id_to_url)} needed images already present in {images_dir}")
        return

    print(f"  downloading {len(remaining)} image(s) (of {len(id_to_url)} needed; {len(already_present)} already cached) ...")
    failed: List[Tuple[str, str]] = []
    for image_id, url in remaining.items():
        try:
            download_file(url, images_dir / f"{image_id}.jpg")
        except VisualGenomeDataError as exc:
            failed.append((image_id, str(exc)))
    if failed:
        preview = failed[:5]
        raise VisualGenomeDataError(
            f"{len(failed)} image(s) failed to download: {preview}"
            f"{' ... (' + str(len(failed) - 5) + ' more)' if len(failed) > 5 else ''}"
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--split", default=DEFAULT_SPLIT, help=f"{VG_DATASET_NAME} split to draw the candidate pool from")
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES, help="prefix-slice bound on VG image ROWS loaded (each row may hold several annotated objects) -- bounds dataset-load size and image-download volume, NOT the final benchmark subset size")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)

    print(f"Loading {VG_DATASET_NAME} config={VG_DATASET_CONFIG!r} split={args.split!r} (max_candidates={args.max_candidates} image rows) ...")
    image_rows = load_attribute_rows(args.split, args.max_candidates)
    print(f"  {len(image_rows)} image rows loaded")

    flattened, stats = flatten_attribute_examples(image_rows)
    print(
        f"  flattened: kept {len(flattened)} (image, object) examples out of {stats['n_objects_seen']} objects seen "
        f"(skipped {stats['skipped_no_attributes']} no-attributes, {stats['skipped_empty_name']} empty-name, "
        f"{stats['skipped_invalid_bbox']} invalid-bbox)"
    )

    id_to_url = needed_images_with_urls(flattened)
    print(f"  {len(id_to_url)} unique images needed")
    fetch_needed_images(id_to_url, data_dir / "images")

    out_path = data_dir / "vg_attributes.parquet"
    write_attributes_parquet(flattened, out_path)
    print(f"Wrote {out_path} ({len(flattened)} rows)")

    stats_path = data_dir / "vg_prepare_stats.json"
    write_prepare_stats(
        {
            "dataset_source": VG_DATASET_NAME,
            "dataset_config": VG_DATASET_CONFIG,
            "split": args.split,
            "max_candidates": args.max_candidates,
            "n_candidate_image_rows": stats["n_image_rows"],
            "n_objects_seen": stats["n_objects_seen"],
            "n_flattened_examples": len(flattened),
            "skipped_no_attributes": stats["skipped_no_attributes"],
            "skipped_empty_name": stats["skipped_empty_name"],
            "skipped_invalid_bbox": stats["skipped_invalid_bbox"],
            "n_images_needed": len(id_to_url),
        },
        stats_path,
    )
    print(f"Wrote {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

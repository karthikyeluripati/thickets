"""Prepares attribute_recognition capability-benchmark data into
external/RandOpt/data/visual_genome/{vg_attributes.parquet,images/} -- mirrors
prepare_gqa_data.py's own "prepare once offline, adapter reads the prepared local artifact"
pattern, for the same reason: bounding the per-run cost and keeping the evaluated pool
deterministic and inspectable on disk.

REPLACES the original `ranjaykrishna/visual_genome` (config attributes_v1.2.0) source, which
is script-based and FAILS on modern `datasets` (>=4.0-style) with "Dataset scripts are no
longer supported, but found visual_genome.py" -- confirmed on a real RunPod this session,
datasets==5.0.1. Root cause: that repo requires trust_remote_code execution of a Python
loading script, which recent `datasets` versions refuse to run automatically; no auto-
converted-parquet mirror exists for it either (confirmed: the datasets-server parquet API
404s for this repo/config).

Two-source design (documented, not accidental):
  A. Annotations: `mikewang/vaw` (VAW -- "Visual Attributes in the Wild", Pham et al. CVPR
     2021) -- CONFIRMED live this session: script-free, auto-converted-to-Parquet, schema
     `image_id, instance_id, instance_bbox[x,y,w,h], instance_polygon, object_name,
     positive_attributes[...], negative_attributes[...]`. `positive_attributes` is preserved
     as the FULL target SET (never collapsed to one answer) -- this is VAW's own intended
     evaluation design, not a simplification we're introducing.
  B. Images: VAW has NO image column, only `image_id` referencing Visual Genome's own image
     archives -- so images are fetched separately from Visual Genome's official Stanford-
     hosted zip archives. **UNRESOLVED / pod-side-confirm**: the exact archive URLs below are
     the commonly-cited ones (cross-referenced across independent sources: GitHub download
     scripts, HF dataset discussions, academic-torrent listings) but were NOT independently
     verified by an actual successful download from this environment -- see
     CAPABILITY_BENCHMARK_GATE.md. `--images-zip-url-part1`/`--images-zip-url-part2` override
     these if they turn out stale; the script hard-fails with a clear, actionable message on
     a failed/404 download rather than silently producing an incomplete image set.

Usage:
    python -m neural_thickets_repro.prepare_visual_genome_data
    python -m neural_thickets_repro.prepare_visual_genome_data --max-candidates 1000 --split train
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import IO, Iterable, List, Set

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "external" / "RandOpt" / "data" / "visual_genome"

VAW_DATASET_NAME = "mikewang/vaw"

# See module docstring's "UNRESOLVED / pod-side-confirm" note -- commonly cited across
# independent sources, not independently verified by an actual download from this
# environment. Override with --images-zip-url-part1/2 if these are stale.
DEFAULT_IMAGES_ZIP_URL_PART1 = "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip"
DEFAULT_IMAGES_ZIP_URL_PART2 = "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip"

REQUIRED_VAW_COLUMNS = {"image_id", "instance_id", "instance_bbox", "object_name", "positive_attributes"}


class VisualGenomeDataError(RuntimeError):
    """Raised for a malformed VAW row (wrong bbox shape, missing required column) or a
    failed/incomplete image download -- never silently skipped or guessed around.
    """


def load_vaw_rows(split: str, max_candidates: "int | None") -> List[dict]:
    """Loads mikewang/vaw (script-free Parquet) -- a documented prefix slice of `split`,
    same "first N rows" convention prepare_gqa_data.py already uses for its own selection
    split, bounding the number of unique images this preparation step needs to fetch.
    """
    from datasets import load_dataset

    ds = load_dataset(VAW_DATASET_NAME, split=split)
    missing = REQUIRED_VAW_COLUMNS - set(ds.column_names)
    if missing:
        raise VisualGenomeDataError(
            f"{VAW_DATASET_NAME} split {split!r} is missing expected column(s) {sorted(missing)} "
            f"(found: {ds.column_names}) -- refusing to guess a different schema."
        )
    if max_candidates is not None:
        ds = ds.select(range(min(max_candidates, len(ds))))
    return list(ds)


def rows_with_positive_attributes(rows: Iterable[dict]) -> List[dict]:
    """Objects with zero positive attributes carry no usable target -- excluded here (not a
    parser failure downstream, a data-selection step, same as GQA's own "images_available"
    filter in GQAHandler).
    """
    kept = []
    for row in rows:
        bbox = row.get("instance_bbox")
        if bbox is None or len(bbox) != 4:
            raise VisualGenomeDataError(f"instance_id={row.get('instance_id')!r} has a malformed instance_bbox: {bbox!r}")
        if row.get("positive_attributes"):
            kept.append(row)
    return kept


def needed_image_ids(rows: Iterable[dict]) -> Set[str]:
    return {str(row["image_id"]) for row in rows}


def write_attributes_parquet(rows: List[dict], out_path: Path) -> None:
    import pandas as pd

    records = [
        {
            "image_id": str(row["image_id"]),
            "instance_id": str(row["instance_id"]),
            "object_name": row["object_name"],
            "positive_attributes": json.dumps(list(row["positive_attributes"])),  # JSON-encoded list, decoded by the adapter
            "bbox_x": row["instance_bbox"][0], "bbox_y": row["instance_bbox"][1],
            "bbox_w": row["instance_bbox"][2], "bbox_h": row["instance_bbox"][3],
        }
        for row in rows
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_parquet(out_path)


def extract_needed_images_from_zip(zip_file: "IO[bytes] | str | Path", needed_ids: Set[str], images_dir: Path) -> Set[str]:
    """Reads a zip file's member list and extracts ONLY members whose base filename (minus
    extension) is in needed_ids, into images_dir (flat, no subdirectories -- matches
    prepare_gqa_data.py's own images/ layout). Returns the set of ids actually found in THIS
    zip (so the caller can check the second archive for whatever's still missing). Pure
    zipfile logic -- fully unit-testable against a small synthetic in-memory zip, no network.
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    found: Set[str] = set()
    with zipfile.ZipFile(zip_file) as zf:
        for member in zf.namelist():
            stem = Path(member).stem
            if stem in needed_ids:
                with zf.open(member) as src, open(images_dir / f"{stem}.jpg", "wb") as dst:
                    dst.write(src.read())
                found.add(stem)
    return found


def download_file(url: str, dest_path: Path) -> None:
    """Thin, deliberately untested-beyond-mocking network glue -- the actual HTTP call.
    Everything upstream/downstream of this (which URLs to try, which zip members to extract)
    is pure logic covered by tests.
    """
    import urllib.request

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest_path)
    except Exception as exc:  # noqa: BLE001 -- re-raised with an actionable message below
        raise VisualGenomeDataError(
            f"Failed to download Visual Genome images archive from {url}: "
            f"{type(exc).__name__}: {exc}. This URL is a commonly-cited but NOT independently "
            f"verified default (see prepare_visual_genome_data.py's module docstring) -- if "
            f"it has moved, pass the correct URL via --images-zip-url-part1/--images-zip-url-part2."
        ) from exc


def fetch_needed_images(
    needed_ids: Set[str], images_dir: Path, cache_dir: Path, zip_url_part1: str, zip_url_part2: str,
) -> None:
    already_present = {p.stem for p in images_dir.glob("*.jpg")}
    remaining = needed_ids - already_present
    if not remaining:
        print(f"  all {len(needed_ids)} images already present in {images_dir}")
        return

    for label, url in (("part 1", zip_url_part1), ("part 2", zip_url_part2)):
        if not remaining:
            break
        zip_path = cache_dir / Path(url).name
        if not zip_path.exists():
            print(f"  downloading Visual Genome images {label} from {url} (one-time, may be several GB) ...")
            download_file(url, zip_path)
        print(f"  extracting needed images from {label} ...")
        found = extract_needed_images_from_zip(zip_path, remaining, images_dir)
        remaining -= found

    if remaining:
        raise VisualGenomeDataError(
            f"{len(remaining)} image id(s) not found in either archive: {sorted(remaining)[:10]}"
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--cache-dir", default=str(DEFAULT_DATA_DIR / "_zip_cache"))
    parser.add_argument("--split", default="train", help="mikewang/vaw split to draw the candidate pool from")
    parser.add_argument("--max-candidates", type=int, default=1000, help="prefix-slice bound on VAW rows loaded, before the positive-attributes filter -- bounds the number of unique images fetched")
    parser.add_argument("--images-zip-url-part1", default=DEFAULT_IMAGES_ZIP_URL_PART1)
    parser.add_argument("--images-zip-url-part2", default=DEFAULT_IMAGES_ZIP_URL_PART2)
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)

    print(f"Loading {VAW_DATASET_NAME} split={args.split!r} (max_candidates={args.max_candidates}) ...")
    rows = load_vaw_rows(args.split, args.max_candidates)
    rows = rows_with_positive_attributes(rows)
    print(f"  {len(rows)} (image, object) rows with at least one positive attribute")

    ids = needed_image_ids(rows)
    print(f"  {len(ids)} unique images needed")
    fetch_needed_images(ids, data_dir / "images", cache_dir, args.images_zip_url_part1, args.images_zip_url_part2)

    out_path = data_dir / "vg_attributes.parquet"
    write_attributes_parquet(rows, out_path)
    print(f"Wrote {out_path} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Visual Genome attributes (attribute_recognition) adapter.

Dataset source: a LOCALLY PREPARED artifact (see prepare_visual_genome_data.py), not a live
`datasets.load_dataset()` call. Source history (see CAPABILITY_BENCHMARK_GATE.md for the full
record): `ranjaykrishna/visual_genome` (original config `attributes`) and then `mikewang/vaw`
(VAW) were BOTH tried and BOTH fail on modern `datasets` with "Dataset scripts are no longer
supported" -- confirmed on a real RunPod, datasets==5.0.1, for each in turn. The operational
source is now `AnnaZ1103/visual_genome_revised` (config "attributes", split "train") --
CONFIRMED script-free Parquet on a real RunPod with datasets==5.0.1 -- a community
repackaging of Visual Genome's own attribute annotations (not the canonical upstream VG
distribution; documented, not claimed otherwise). Each row carries its own image `url`
(Stanford-hosted), so `prepare_visual_genome_data.py` fetches only the images actually needed
by the flattened examples -- never the full ~100k-image VG archive -- and writes
`<source>/vg_attributes.parquet` + `<source>/images/`. This adapter reads ONLY that prepared
local artifact, mirroring GQAHandler's own "prepare offline, adapter reads locally" split of
labor.

To keep scoring automatic and unambiguous without reducing this to unrestricted captioning:
one Example = one (image, object) pair. The queried object is made unambiguous by drawing a
deterministic bounding-box marker around it (prepare_image() below, PIL ImageDraw, on a COPY
of the image -- the original is preserved separately, never mutated, and the bbox itself is
recorded in Example.metadata) -- an outline only, never filled, so the object itself is not
obscured. VAW's `positive_attributes` list is preserved in FULL as Example.target (never
collapsed to one answer) -- a prediction is scored correct if it matches ANY of them (see
score_example), the same "preserve the valid target set" discipline TextVQA's 10-answer list
already uses in this package.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from ..base import CapabilityBenchmark, Example, ExampleScore, ParsedPrediction
from ..normalization import normalize_answer
from ..prompting import build_image_text_messages

INSTRUCTION = (
    "Look at the object outlined in red in the image. Name ONE visual attribute of that "
    "object (for example its color, material, size, or texture). Answer with a single word "
    "or short phrase -- do not describe the whole scene."
)
MARKER_COLOR = (255, 0, 0)
MARKER_WIDTH = 3
PREPARE_COMMAND = "python -m neural_thickets_repro.prepare_visual_genome_data"


class VisualGenomeSchemaError(RuntimeError):
    """The prepared local Visual Genome artifact is missing or doesn't match the expected
    schema (image_id/instance_id/object_name/positive_attributes/bbox_x/y/w/h) -- refuses to
    guess a different one.
    """


class VisualGenomeAttributeBenchmark(CapabilityBenchmark):
    capability = "attribute_recognition"
    name = "visual_genome_attributes"

    def dataset_source(self) -> str:
        return "AnnaZ1103/visual_genome_revised (attributes config, train split) + per-row image URLs, prepared locally by prepare_visual_genome_data.py"

    def known_caveats(self) -> List[str]:
        return [
            "Annotation source is AnnaZ1103/visual_genome_revised (config 'attributes'), a "
            "community repackaging of Visual Genome's own attribute annotations -- NOT the "
            "canonical ranjaykrishna/visual_genome upstream distribution. Migrated because "
            "BOTH ranjaykrishna/visual_genome AND a prior mikewang/vaw replacement fail on "
            "modern `datasets` with 'Dataset scripts are no longer supported' (each confirmed "
            "failing on a real RunPod, datasets==5.0.1); this source was confirmed working "
            "(script-free Parquet) on a real RunPod before being adopted.",
            "Images are fetched individually via each row's own `url` field (Stanford-hosted "
            "originals), only for images actually referenced by the flattened examples -- "
            "never the full ~100k-image Visual Genome archive.",
            "`object_name` is the FIRST entry of the object's `names` list -- VG objects can "
            "carry multiple synonym names; only the first is used as the single-noun prompt "
            "target. This is separate from the attribute TARGET set, which is always kept in "
            "full (see below).",
            "No appearance-vs-state semantic filtering is applied to VG's raw attribute "
            "vocabulary -- e.g. 'walking' is a real observed VG attribute value that arguably "
            "describes an action/state rather than a visual appearance attribute. There is no "
            "existing, defensible criterion in this codebase for that distinction yet, and one "
            "is deliberately NOT invented here. Flagged as a scientific-review item for the "
            "upcoming N=5 manual inspection pass, not silently resolved.",
            "A bounding-box marker overlay is drawn on every image as part of this benchmark's "
            "own protocol (to make the queried object unambiguous) -- it is not naturally "
            "occurring VG data. The shuffled-image sanity condition keeps the same fixed "
            "marker coordinates on a swapped photo; this is still a valid distractor for that "
            "check, not a bug.",
            "The full positive_attributes list is preserved as the target set; a prediction "
            "is scored correct if it matches ANY of them.",
        ]

    def load_examples(self, cfg: Any) -> List[Example]:
        import pandas as pd
        from PIL import Image

        data_dir = Path(cfg.dataset.source)
        parquet_path = data_dir / "vg_attributes.parquet"
        images_dir = data_dir / "images"
        if not parquet_path.exists():
            raise VisualGenomeSchemaError(
                f"No prepared Visual Genome attributes parquet found at {parquet_path}. "
                f"Generate it first with:\n    {PREPARE_COMMAND}"
            )

        df = pd.read_parquet(parquet_path)
        missing_columns = {"image_id", "instance_id", "object_name", "positive_attributes", "bbox_x", "bbox_y", "bbox_w", "bbox_h"} - set(df.columns)
        if missing_columns:
            raise VisualGenomeSchemaError(f"{parquet_path} is missing expected column(s) {sorted(missing_columns)} -- refusing to guess a different schema.")

        has_image_dims = {"image_width", "image_height"} <= set(df.columns)

        examples: List[Example] = []
        for _, row in df.iterrows():
            image_path = images_dir / f"{row['image_id']}.jpg"
            image = Image.open(image_path).convert("RGB") if image_path.exists() else None
            attributes = json.loads(row["positive_attributes"])
            metadata: Dict[str, Any] = {
                "bbox_xywh": [row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"]],
                "image_id": str(row["image_id"]),
            }
            if has_image_dims:
                metadata["image_width"] = row["image_width"]
                metadata["image_height"] = row["image_height"]
            examples.append(Example(
                example_id=str(row["instance_id"]),
                image=image,
                image_ref=str(image_path),
                prompt_input={"object_name": row["object_name"]},
                target=list(attributes),
                metadata=metadata,
            ))
        return examples

    def prepare_image(self, example: Example):
        if example.image is None:
            return None
        from PIL import ImageDraw

        marked = example.image.copy()  # never mutate the original -- preserved separately
        x, y, w, h = example.metadata["bbox_xywh"]
        ImageDraw.Draw(marked).rectangle([x, y, x + w, y + h], outline=MARKER_COLOR, width=MARKER_WIDTH)
        return marked

    def build_prompt(self, example: Example) -> List[dict]:
        return build_image_text_messages(INSTRUCTION)

    def parse_prediction(self, raw_generation: str, example: Example) -> ParsedPrediction:
        stripped = raw_generation.strip()
        if not stripped:
            return ParsedPrediction(parsed="", parse_ok=False, parse_error="empty generation")
        return ParsedPrediction(parsed=stripped, parse_ok=True)

    def score_example(self, parsed: ParsedPrediction, example: Example) -> ExampleScore:
        if not parsed.parse_ok:
            return ExampleScore(score=0.0, correct=False, detail={"reason": "parse_failure"})

        predicted_norm = normalize_answer(parsed.parsed)
        padded_prediction = f" {predicted_norm} "
        valid_targets_norm = [normalize_answer(a) for a in example.target]
        matched = next((a for a in valid_targets_norm if a and f" {a} " in padded_prediction), None)

        correct = matched is not None
        return ExampleScore(score=1.0 if correct else 0.0, correct=correct, detail={"matched_attribute": matched, "valid_targets": example.target})

    def aggregate_metrics(self, scores: List[ExampleScore]) -> Dict[str, float]:
        n = len(scores)
        if n == 0:
            return {"accuracy": 0.0, "primary_metric": 0.0, "parser_failure_rate": 0.0}
        parser_failures = sum(1 for s in scores if s.detail.get("reason") == "parse_failure")
        accuracy = sum(s.score for s in scores) / n
        return {"accuracy": accuracy, "primary_metric": accuracy, "parser_failure_rate": parser_failures / n}

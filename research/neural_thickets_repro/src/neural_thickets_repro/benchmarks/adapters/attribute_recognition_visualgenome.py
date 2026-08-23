"""Visual Genome attributes (attribute_recognition) adapter.

Dataset source: a LOCALLY PREPARED artifact (see prepare_visual_genome_data.py), not a live
`datasets.load_dataset()` call -- `ranjaykrishna/visual_genome` (the original config
`attributes`) is script-based and FAILS on modern `datasets` ("Dataset scripts are no longer
supported", confirmed on a real RunPod this session, datasets==5.0.1). Replaced with
`mikewang/vaw` (VAW -- "Visual Attributes in the Wild", CVPR 2021; script-free, confirmed
live this session) for annotations, joined with Visual Genome's own official images (fetched
separately, VAW carries no image data) by `prepare_visual_genome_data.py`, which writes
`<source>/vg_attributes.parquet` + `<source>/images/` -- this adapter reads ONLY that
prepared local artifact, mirroring GQAHandler's own "prepare offline, adapter reads locally"
split of labor. See CAPABILITY_BENCHMARK_GATE.md for the full source-migration record.

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
        return "mikewang/vaw (annotations) + Visual Genome official images, prepared locally by prepare_visual_genome_data.py"

    def known_caveats(self) -> List[str]:
        return [
            "Annotation source is mikewang/vaw (VAW), not the original ranjaykrishna/"
            "visual_genome 'attributes' config -- migrated because that repo's script-based "
            "loading fails on modern `datasets` versions. VAW is a peer-reviewed (CVPR 2021) "
            "derivative of Visual Genome specifically curated for attribute recognition, not "
            "a lower-quality substitute.",
            "Visual Genome's own image archive URLs (fetched by prepare_visual_genome_data.py) "
            "were not independently verified by an actual successful download from this "
            "environment -- see that script's module docstring and CAPABILITY_BENCHMARK_GATE.md.",
            "A bounding-box marker overlay is drawn on every image as part of this benchmark's "
            "own protocol (to make the queried object unambiguous) -- it is not naturally "
            "occurring VG/VAW data. The shuffled-image sanity condition keeps the same fixed "
            "marker coordinates on a swapped photo; this is still a valid distractor for that "
            "check, not a bug.",
            "VAW's full positive_attributes list is preserved as the target set; a prediction "
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

        examples: List[Example] = []
        for _, row in df.iterrows():
            image_path = images_dir / f"{row['image_id']}.jpg"
            image = Image.open(image_path).convert("RGB") if image_path.exists() else None
            attributes = json.loads(row["positive_attributes"])
            examples.append(Example(
                example_id=str(row["instance_id"]),
                image=image,
                image_ref=str(image_path),
                prompt_input={"object_name": row["object_name"]},
                target=list(attributes),
                metadata={
                    "bbox_xywh": [row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"]],
                    "image_id": str(row["image_id"]),
                },
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

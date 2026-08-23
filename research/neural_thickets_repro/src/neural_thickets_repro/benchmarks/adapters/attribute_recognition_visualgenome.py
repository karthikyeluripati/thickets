"""Visual Genome attributes (attribute_recognition) adapter.

Dataset source: HuggingFace `ranjaykrishna/visual_genome` (the canonical, original-authors
repo), config `attributes` -- exact version suffix (e.g. "attributes_v1.2.0") UNCONFIRMED,
see CAPABILITY_BENCHMARK_GATE.md; `config_name` is a constructor parameter, not hardcoded,
so it can be corrected without touching this file's logic.

To keep scoring automatic and unambiguous without reducing this to unrestricted captioning:
one Example = one (image, object) pair. The queried object is made unambiguous by drawing a
deterministic bounding-box marker around it (prepare_image() below, PIL ImageDraw, on a COPY
of the image -- the original is preserved separately, never mutated, and the bbox itself is
recorded in Example.metadata) -- an outline only, never filled, so the object itself is not
obscured. If VG annotates MULTIPLE valid attributes for an object, ALL of them are preserved
as Example.target (a list) rather than arbitrarily picking one -- a prediction is scored
correct if it matches ANY of them (see score_example), the same "preserve the valid target
set" discipline TextVQA's 10-answer list already uses in this package.
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..base import CapabilityBenchmark, Example, ExampleScore, ParsedPrediction
from ..normalization import normalize_answer
from ..prompting import build_image_text_messages

DEFAULT_CONFIG_NAME = "attributes_v1.2.0"  # UNCONFIRMED, see module docstring
INSTRUCTION = (
    "Look at the object outlined in red in the image. Name ONE visual attribute of that "
    "object (for example its color, material, size, or texture). Answer with a single word "
    "or short phrase -- do not describe the whole scene."
)
MARKER_COLOR = (255, 0, 0)
MARKER_WIDTH = 3


class VisualGenomeSchemaError(RuntimeError):
    """The loaded Visual Genome 'attributes' rows don't match the assumed per-object shape
    (image / attributes list of {object_id, attributes, x, y, w, h}) -- refuses to guess.
    """


class VisualGenomeAttributeBenchmark(CapabilityBenchmark):
    capability = "attribute_recognition"
    name = "visual_genome_attributes"

    def __init__(self, config_name: str = DEFAULT_CONFIG_NAME):
        self.config_name = config_name

    def dataset_source(self) -> str:
        return f"ranjaykrishna/visual_genome (config: {self.config_name})"

    def known_caveats(self) -> List[str]:
        return [
            f"Config name {self.config_name!r} is a documented, revisable choice -- the exact "
            f"version suffix was not independently confirmed, see CAPABILITY_BENCHMARK_GATE.md.",
            "A bounding-box marker overlay is drawn on every image as part of this benchmark's "
            "own protocol (to make the queried object unambiguous) -- it is not naturally "
            "occurring VG data. The shuffled-image sanity condition keeps the same fixed "
            "marker coordinates on a swapped photo; this is still a valid distractor for that "
            "check, not a bug.",
            "When VG annotates multiple valid attributes for an object, ALL are preserved as "
            "the target set; a prediction is scored correct if it matches ANY of them.",
        ]

    def load_examples(self, cfg: Any) -> List[Example]:
        from datasets import load_dataset

        hf_dataset = load_dataset(cfg.dataset.source, self.config_name, split=cfg.dataset.split, revision=cfg.dataset.revision)

        examples: List[Example] = []
        for row in hf_dataset:
            image = row.get("image")
            image_id = row.get("image_id")
            objects = row.get("attributes")
            if image is None or objects is None:
                raise VisualGenomeSchemaError(
                    f"Expected each row to have 'image' and 'attributes' fields, got keys "
                    f"{list(row.keys())}. Refusing to guess a different schema."
                )
            for obj in objects:
                attributes = obj.get("attributes") or []
                if not attributes:
                    continue  # object has no attribute annotation -- not a usable example, not an error
                for field in ("object_id", "x", "y", "w", "h"):
                    if field not in obj:
                        raise VisualGenomeSchemaError(
                            f"Expected each object to have {field!r}, got keys {list(obj.keys())}."
                        )
                examples.append(Example(
                    example_id=f"{image_id}_{obj['object_id']}",
                    image=image,
                    image_ref=f"vg_image_{image_id}",
                    prompt_input={"object_names": obj.get("names", [])},
                    target=list(attributes),
                    metadata={"bbox_xywh": [obj["x"], obj["y"], obj["w"], obj["h"]], "image_id": image_id, "object_id": obj["object_id"]},
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

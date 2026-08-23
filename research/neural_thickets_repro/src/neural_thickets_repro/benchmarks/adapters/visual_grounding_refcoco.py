"""RefCOCO / RefCOCO+ (visual_grounding) adapter.

Dataset source (RefCOCO confirmed live this session): HuggingFace `lmms-lab-encoder/RefCOCO`
(ungated, matches this project's established `lmms-lab-encoder` org convention). RefCOCO+'s
exact repo id is assumed analogous (`lmms-lab-encoder/RefCOCO+`) but NOT independently
confirmed -- see CAPABILITY_BENCHMARK_GATE.md, to be verified on the pod before that variant
is treated as ready. One class handles both via the `dataset_repo_id`/`variant_name`
constructor params, since RefCOCO and RefCOCO+ share an identical schema.

Confirmed schema: `image`, `question` (the referring expression text), `bbox` in
`[x_min, y_min, width, height]` COCO convention, `question_id`. Coordinate convention used
throughout this adapter (and by box_iou.py): [x1, y1, x2, y2], normalized to [0, 1] by image
width/height -- xywh_to_xyxy() converts the raw annotation, normalize_xyxy() divides by the
image's own actual pixel size (loaded once per example, not assumed).

Grounding has no well-defined text-only condition (the image IS the query target) --
supports_text_only_condition() is False, reported honestly rather than scoring a meaningless
condition.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from ..base import CapabilityBenchmark, Example, ExampleScore, ParsedPrediction
from ..box_iou import box_iou, normalize_xyxy, xywh_to_xyxy
from ..prompting import build_image_text_messages

IOU_THRESHOLD = 0.5
INSTRUCTION_TEMPLATE = (
    'Locate "{referring_expression}" in the image. Respond with ONLY its bounding box as '
    "[x1, y1, x2, y2], where each coordinate is a number between 0 and 1 representing the "
    "fraction of the image width (for x1/x2) or height (for y1/y2) -- x1,y1 is the top-left "
    "corner and x2,y2 is the bottom-right corner."
)
# Preferred: the exact "[x1, y1, x2, y2]" format the prompt asks for.
_BRACKET_BOX_PATTERN = re.compile(
    r"\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]"
)
# Fallback for a verbose/differently-formatted generation: any 4 standalone numbers, in
# order. Word-boundary-guarded (no adjacent letter/digit) so a labeled coordinate like
# "x1=0.1" doesn't spuriously extract the "1" from the "x1" label itself as a separate number.
_STANDALONE_NUMBER_PATTERN = re.compile(r"(?<![a-zA-Z0-9])[-+]?\d*\.?\d+(?![a-zA-Z0-9])")


def _extract_four_floats(text: str) -> Optional[Tuple[float, float, float, float]]:
    bracket_match = _BRACKET_BOX_PATTERN.search(text)
    if bracket_match:
        return tuple(float(g) for g in bracket_match.groups())  # type: ignore[return-value]

    numbers = _STANDALONE_NUMBER_PATTERN.findall(text)
    if len(numbers) < 4:
        return None
    try:
        return tuple(float(n) for n in numbers[:4])  # type: ignore[return-value]
    except ValueError:
        return None


class RefCOCOGroundingBenchmark(CapabilityBenchmark):
    capability = "visual_grounding"

    def __init__(self, dataset_repo_id: str = "lmms-lab-encoder/RefCOCO", variant_name: str = "refcoco_val"):
        self.dataset_repo_id = dataset_repo_id
        self.name = variant_name

    def dataset_source(self) -> str:
        return self.dataset_repo_id

    def supports_text_only_condition(self) -> bool:
        return False

    def text_only_unsupported_reason(self) -> Optional[str]:
        return "Visual grounding's query IS the image -- a text-only condition has no well-defined target box to predict against."

    def known_caveats(self) -> List[str]:
        caveats = []
        if "refcoco+" in self.dataset_repo_id.lower():
            caveats.append(f"{self.dataset_repo_id!r} is assumed analogous to the confirmed lmms-lab-encoder/RefCOCO repo id, not independently verified.")
        return caveats

    def load_examples(self, cfg: Any) -> List[Example]:
        from datasets import load_dataset

        hf_dataset = load_dataset(cfg.dataset.source, split=cfg.dataset.split, revision=cfg.dataset.revision)

        examples: List[Example] = []
        for row in hf_dataset:
            image = row["image"]
            width, height = image.size
            xyxy_pixels = xywh_to_xyxy(tuple(row["bbox"]))
            target_normalized = normalize_xyxy(xyxy_pixels, width, height)
            examples.append(Example(
                example_id=str(row["question_id"]),
                image=image,
                image_ref=str(row.get("file_name", row["question_id"])),
                prompt_input={"referring_expression": row["question"]},
                target=target_normalized,
                metadata={"bbox_pixels_xywh": list(row["bbox"]), "image_width": width, "image_height": height},
            ))
        return examples

    def build_prompt(self, example: Example) -> List[dict]:
        instruction = INSTRUCTION_TEMPLATE.format(referring_expression=example.prompt_input["referring_expression"])
        return build_image_text_messages(instruction)

    def parse_prediction(self, raw_generation: str, example: Example) -> ParsedPrediction:
        box = _extract_four_floats(raw_generation)
        if box is None:
            return ParsedPrediction(parsed=None, parse_ok=False, parse_error=f"could not find 4 coordinates in {raw_generation!r}")
        return ParsedPrediction(parsed=box, parse_ok=True)

    def score_example(self, parsed: ParsedPrediction, example: Example) -> ExampleScore:
        if not parsed.parse_ok:
            return ExampleScore(score=0.0, correct=False, detail={"reason": "parse_failure", "iou": 0.0})
        iou = box_iou(parsed.parsed, example.target)
        correct = iou >= IOU_THRESHOLD
        return ExampleScore(score=iou, correct=correct, detail={"iou": iou})

    def aggregate_metrics(self, scores: List[ExampleScore]) -> Dict[str, float]:
        n = len(scores)
        if n == 0:
            return {"accuracy_at_iou_0.5": 0.0, "mean_iou": 0.0, "primary_metric": 0.0, "parser_failure_rate": 0.0}

        parser_failures = sum(1 for s in scores if s.detail.get("reason") == "parse_failure")
        acc_at_iou = sum(1 for s in scores if s.correct) / n
        mean_iou_value = sum(s.detail.get("iou", 0.0) for s in scores) / n

        return {
            "accuracy_at_iou_0.5": acc_at_iou,
            "mean_iou": mean_iou_value,
            "primary_metric": acc_at_iou,
            "parser_failure_rate": parser_failures / n,
        }

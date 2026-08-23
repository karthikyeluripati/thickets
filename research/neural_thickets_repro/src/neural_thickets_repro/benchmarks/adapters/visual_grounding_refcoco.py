"""RefCOCO / RefCOCO+ (visual_grounding) adapter.

Dataset source (RefCOCO confirmed live this session): HuggingFace `lmms-lab-encoder/RefCOCO`
(ungated, matches this project's established `lmms-lab-encoder` org convention). RefCOCO+'s
exact repo id is assumed analogous (`lmms-lab-encoder/RefCOCO+`) but NOT independently
confirmed -- see CAPABILITY_BENCHMARK_GATE.md, to be verified on the pod before that variant
is treated as ready. One class handles both via the `dataset_repo_id`/`variant_name`
constructor params, since RefCOCO and RefCOCO+ share an identical schema.

REFERRING-EXPRESSION FIELD BUG (fixed this repair pass): a real N=5 smoke test showed
`row["question"]` was, in every example, the literal fixed string "Please carefully observe
the area circled in the image and come up with a caption for the area." -- a region-captioning
INSTRUCTION, not a referring expression. Confirmed via live schema re-inspection this session:
this HF repo repackages RefCOCO as an instruction-tuned region-captioning dataset. The real
schema is `image`, `question` (the fixed circling instruction -- NOT usable as a referring
expression), `answer` (a LIST of independent human-written descriptions of the circled
region -- these ARE RefCOCO's real referring expressions), `bbox` in
`[x_min, y_min, width, height]` COCO convention, `segmentation`, `iscrowd`, `file_name`,
`question_id`. This adapter now reads `row["answer"][0]` (the first of possibly several
annotations, a documented deterministic choice) as the referring expression -- see
`load_examples()` below and `known_caveats()`. All annotations are preserved in
`Example.metadata["all_referring_expressions"]` for audit; only the first is ever prompted.

EXPLICIT OUTPUT CONTRACT (this repair pass, round 2): the prompt now explicitly asks the model
to return PIXEL-space [x1,y1,x2,y2] and states the image's own width/height directly in the
prompt text (INSTRUCTION_TEMPLATE below) -- a model-agnostic, reproducible contract, rather
than relying primarily on post-hoc auto-detection to guess what a model actually returned.
`Example.target` is still stored normalized to [0, 1] internally (xywh_to_xyxy() converts the
raw annotation, normalize_xyxy() divides by the image's own actual pixel size) -- that's just
this adapter's own internal storage format; score_example() denormalizes it back to pixel
space (the canonical representation for IoU) before scoring. Auto-detection
(box_iou.detect_coordinate_mode/canonicalize_prediction_box) is kept as a documented,
backward-compatible FALLBACK for a model that doesn't comply exactly, and now also CLIPS a
prediction that slightly overshoots the image edge into bounds before IoU, rather than
rejecting it -- see score_example()'s own docstring for the real example this fixed.

Grounding has no well-defined text-only condition (the image IS the query target) --
supports_text_only_condition() is False, reported honestly rather than scoring a meaningless
condition.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from ..base import CapabilityBenchmark, Example, ExampleScore, ParsedPrediction
from ..box_iou import box_iou, canonicalize_prediction_box, denormalize_xyxy, normalize_xyxy, xywh_to_xyxy
from ..prompting import build_image_text_messages

IOU_THRESHOLD = 0.5
# EXPLICIT pixel-space output contract (this repair pass) -- states the image's own real
# dimensions directly in the prompt text so "pixel coordinates" is unambiguous per example,
# and is model-agnostic (never assumes/relies on Qwen-specific behavior).
INSTRUCTION_TEMPLATE = (
    'Locate "{referring_expression}" in the image. The image is {image_width} pixels wide '
    "and {image_height} pixels tall. Respond with ONLY its bounding box as PIXEL coordinates "
    "[x1, y1, x2, y2], where x1,y1 is the top-left corner and x2,y2 is the bottom-right "
    "corner, with x in [0, {image_width}] and y in [0, {image_height}]."
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


class RefCOCOSchemaError(RuntimeError):
    """A row's real schema doesn't match what this adapter expects (e.g. an empty `answer`
    list -- the referring expression has nowhere to come from) -- refuses to guess a
    substitute expression or silently fall back to the misleading `question` instruction
    field.
    """


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
        caveats = [
            "This HF repackaging exposes a region-captioning INSTRUCTION under the field "
            "name 'question' ('Please carefully observe the area circled in the image and "
            "come up with a caption for the area.') -- confirmed via live schema inspection "
            "this session NOT to be a referring expression. The real referring expression(s) "
            "are recovered from the 'answer' field (a list of independent human-written "
            "region descriptions); this adapter uses the FIRST one deterministically. All "
            "annotations are preserved in Example.metadata['all_referring_expressions'] for "
            "audit, never used for scoring.",
            "The prompt explicitly asks for PIXEL-space [x1,y1,x2,y2] coordinates and states "
            "the image's own real width/height -- an explicit, model-agnostic output "
            "contract, not something left to post-hoc auto-detection. A prediction that "
            "slightly overshoots the image edge is clipped into bounds (never rejected) "
            "before IoU -- see box_iou.clip_box_to_image().",
        ]
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
            answers = list(row.get("answer") or [])
            if not answers:
                raise RefCOCOSchemaError(
                    f"question_id={row.get('question_id')!r} has an empty 'answer' list -- "
                    f"this is where the real RefCOCO referring expression(s) live in this "
                    f"repackaged dataset (NOT the 'question' field, which is a fixed "
                    f"region-captioning instruction) -- refusing to guess a substitute."
                )
            referring_expression = answers[0]  # documented, deterministic: first of possibly several human annotations
            xyxy_pixels = xywh_to_xyxy(tuple(row["bbox"]))
            target_normalized = normalize_xyxy(xyxy_pixels, width, height)
            examples.append(Example(
                example_id=str(row["question_id"]),
                image=image,
                image_ref=str(row.get("file_name", row["question_id"])),
                prompt_input={"referring_expression": referring_expression},
                target=target_normalized,
                metadata={
                    "bbox_pixels_xywh": list(row["bbox"]), "image_width": width, "image_height": height,
                    "all_referring_expressions": answers,
                },
            ))
        return examples

    def build_prompt(self, example: Example) -> List[dict]:
        instruction = INSTRUCTION_TEMPLATE.format(
            referring_expression=example.prompt_input["referring_expression"],
            image_width=example.metadata["image_width"],
            image_height=example.metadata["image_height"],
        )
        return build_image_text_messages(instruction)

    def parse_prediction(self, raw_generation: str, example: Example) -> ParsedPrediction:
        box = _extract_four_floats(raw_generation)
        if box is None:
            return ParsedPrediction(parsed=None, parse_ok=False, parse_error=f"could not find 4 coordinates in {raw_generation!r}")
        return ParsedPrediction(parsed=box, parse_ok=True)

    def score_example(self, parsed: ParsedPrediction, example: Example) -> ExampleScore:
        """COORDINATE-CONTRACT FIX, round 1 (earlier repair pass): a real N=5 Qwen2.5-VL smoke
        test found the model reliably emitting PIXEL-space boxes despite the prompt asking for
        [0,1]-normalized coordinates. Both the prediction and the target are converted into
        ONE canonical representation (pixel-space xyxy, via box_iou.canonicalize_prediction_box
        / denormalize_xyxy) before computing IoU, using deterministic value-range + this
        example's real image-dimension rules -- never by checking which interpretation scores
        better, and never special-cased to Qwen.

        COORDINATE-CONTRACT FIX, round 2 (this repair pass): the corrected N=5 metric still had
        one real failure -- example_id=471277, a 500x375 image, prediction [386,0,504,364]
        (clearly pixel-space, overshooting the image width by only 4px) was misclassified as
        qwen_normalized_0_1000 by the old 2px tolerance, converting it to a tiny wrong box and
        scoring IoU=0. The prompt now explicitly asks for pixel coordinates with the image's
        own dimensions stated (see INSTRUCTION_TEMPLATE) -- the EXPLICIT contract, not
        auto-detection, is the primary mechanism. Auto-detection remains as a documented
        backward-compatible fallback with a wider, non-accuracy-tuned tolerance
        (box_iou._pixel_bound_tolerance), and canonicalize_prediction_box now CLIPS the result
        into the image's own bounds (box_iou.clip_box_to_image) rather than rejecting a small
        boundary overshoot.
        """
        if not parsed.parse_ok:
            return ExampleScore(score=0.0, correct=False, detail={
                "reason": "parse_failure", "iou": 0.0,
                "raw_prediction_box": None, "canonical_prediction_box": None, "coordinate_mode": None,
            })

        image_width = example.metadata["image_width"]
        image_height = example.metadata["image_height"]
        target_pixel_xyxy = denormalize_xyxy(example.target, image_width, image_height)

        raw_box = parsed.parsed
        canonical_box, mode = canonicalize_prediction_box(raw_box, image_width, image_height)
        if canonical_box is None:
            return ExampleScore(score=0.0, correct=False, detail={
                "reason": "unrecognized_coordinate_convention", "iou": 0.0,
                "raw_prediction_box": list(raw_box), "canonical_prediction_box": None, "coordinate_mode": mode,
            })

        iou = box_iou(canonical_box, target_pixel_xyxy)
        correct = iou >= IOU_THRESHOLD
        return ExampleScore(score=iou, correct=correct, detail={
            "iou": iou,
            "raw_prediction_box": list(raw_box),
            "canonical_prediction_box": list(canonical_box),
            "coordinate_mode": mode,
        })

    def aggregate_metrics(self, scores: List[ExampleScore]) -> Dict[str, float]:
        n = len(scores)
        if n == 0:
            return {"accuracy_at_iou_0.5": 0.0, "mean_iou": 0.0, "primary_metric": 0.0, "parser_failure_rate": 0.0}

        parser_failures = sum(1 for s in scores if s.detail.get("reason") in ("parse_failure", "unrecognized_coordinate_convention"))
        acc_at_iou = sum(1 for s in scores if s.correct) / n
        mean_iou_value = sum(s.detail.get("iou", 0.0) for s in scores) / n

        return {
            "accuracy_at_iou_0.5": acc_at_iou,
            "mean_iou": mean_iou_value,
            "primary_metric": acc_at_iou,
            "parser_failure_rate": parser_failures / n,
        }

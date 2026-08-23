"""ImageNet-1K (object_recognition) adapter.

Dataset source (confirmed live this session): HuggingFace `ILSVRC/imagenet-1k`, validation
split (50,000 images). CONFIRMED GATED -- requires a logged-in HF token that has accepted
ImageNet's license terms. Per explicit decision: build for gated access and hard-fail with a
clear, actionable message if access is denied, never silently substitute a different/ungated
dataset (that would be a real change to what's being measured, not a wiring detail).

Canonical label is `dataset.features["label"].int2str(idx)`, a COMMA-SEPARATED SYNONYM LIST
(e.g. "tench, Tinca tinca") -- scoring must not use fragile raw substring matching. Both
sides are normalized (normalization.normalize_answer) and a match is accepted if the FULL
multi-word phrase of any one synonym appears as a contiguous, whole-word sequence in the
normalized prediction (same word-boundary-padded contiguous-phrase check used by the CUB-200
adapter) -- never a bare single generic-word substring test.
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..base import CapabilityBenchmark, Example, ExampleScore, ParsedPrediction
from ..normalization import normalize_answer
from ..prompting import build_image_text_messages

INSTRUCTION = "What is the single main object or subject in this image? Answer with just its name."


class ImageNetGatedAccessError(RuntimeError):
    """ILSVRC/imagenet-1k is gated -- the configured HF credentials don't have accepted-
    license access. Never silently falls back to an alternate/ungated dataset.
    """


class ImageNetObjectRecognitionBenchmark(CapabilityBenchmark):
    capability = "object_recognition"
    name = "imagenet1k_val"

    def dataset_source(self) -> str:
        return "ILSVRC/imagenet-1k"

    def known_caveats(self) -> List[str]:
        return [
            "ILSVRC/imagenet-1k is gated -- requires an HF token with accepted ImageNet "
            "license terms, configured on the machine running load_examples().",
        ]

    def load_examples(self, cfg: Any) -> List[Example]:
        from datasets import load_dataset

        try:
            hf_dataset = load_dataset(cfg.dataset.source, split=cfg.dataset.split, revision=cfg.dataset.revision)
        except Exception as exc:  # noqa: BLE001 -- re-raised as a clear, specific error below
            raise ImageNetGatedAccessError(
                f"Failed to load {cfg.dataset.source!r} (split={cfg.dataset.split!r}): {type(exc).__name__}: {exc}. "
                f"This dataset is GATED -- confirm `huggingface-cli login` has been run with a "
                f"token belonging to an account that has accepted the ImageNet license terms "
                f"at https://huggingface.co/datasets/ILSVRC/imagenet-1k. Refusing to silently "
                f"substitute a different/ungated dataset."
            ) from exc

        label_feature = hf_dataset.features.get("label")
        int2str = getattr(label_feature, "int2str", None)
        if int2str is None:
            raise ImageNetGatedAccessError(
                f"{cfg.dataset.source}'s 'label' feature does not expose int2str() -- cannot "
                f"resolve canonical class names, refusing to guess."
            )

        examples: List[Example] = []
        for row_idx, row in enumerate(hf_dataset):
            if row["label"] < 0:
                continue  # test-split rows have label=-1 (no ground truth) -- excluded, not scored as a failure
            canonical_label = int2str(row["label"])
            examples.append(Example(
                example_id=str(row_idx),
                image=row["image"],
                image_ref=f"imagenet1k_val_row{row_idx}",
                prompt_input={},
                target=canonical_label,
                metadata={"label_index": row["label"]},
            ))
        return examples

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
        synonyms = [s.strip() for s in example.target.split(",")]
        synonyms_norm = [normalize_answer(s) for s in synonyms]

        padded_prediction = f" {predicted_norm} "
        matched_synonym = next((s for s in synonyms_norm if s and f" {s} " in padded_prediction), None)
        correct = matched_synonym is not None
        return ExampleScore(
            score=1.0 if correct else 0.0, correct=correct,
            detail={"predicted_norm": predicted_norm, "matched_synonym": matched_synonym, "synonyms": synonyms},
        )

    def aggregate_metrics(self, scores: List[ExampleScore]) -> Dict[str, float]:
        n = len(scores)
        if n == 0:
            return {"top1_accuracy": 0.0, "primary_metric": 0.0, "parser_failure_rate": 0.0}
        parser_failures = sum(1 for s in scores if s.detail.get("reason") == "parse_failure")
        top1_accuracy = sum(s.score for s in scores) / n
        return {
            "top1_accuracy": top1_accuracy,
            "primary_metric": top1_accuracy,
            "parser_failure_rate": parser_failures / n,
        }

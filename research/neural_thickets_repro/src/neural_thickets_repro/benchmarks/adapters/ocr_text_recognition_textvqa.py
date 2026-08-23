"""TextVQA (ocr_text_recognition) adapter.

Dataset source (confirmed live this session): HuggingFace `lmms-lab-encoder/textvqa`,
matching this project's own established `lmms-lab-encoder` org convention (already used for
GQA, per REPRO_SPEC.md). Validation split has 5k rows; each has `question_id`, `question`,
`image`, and CRITICALLY `answers`: a list of 10 human-annotated answer strings -- the
question's own accepted-answer protocol is NOT a single string, so scoring uses the standard
published VQA soft-accuracy metric (vqa_soft_accuracy.py) against the full 10-answer set,
never a reduction to one "the" answer.
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..base import CapabilityBenchmark, Example, ExampleScore, ParsedPrediction
from ..prompting import build_image_text_messages
from ..vqa_soft_accuracy import vqa_soft_accuracy

INSTRUCTION_SUFFIX = " Answer with a short phrase, reading any text visible in the image if relevant."
CORRECT_THRESHOLD = 0.5  # documented boundary for the boolean ExampleScore.correct field; the continuous VQA soft score is the scientific quantity of record


class TextVQAOCRBenchmark(CapabilityBenchmark):
    capability = "ocr_text_recognition"
    name = "textvqa_validation"

    def dataset_source(self) -> str:
        return "lmms-lab-encoder/textvqa"

    def load_examples(self, cfg: Any) -> List[Example]:
        from datasets import load_dataset

        hf_dataset = load_dataset(cfg.dataset.source, split=cfg.dataset.split, revision=cfg.dataset.revision)

        examples: List[Example] = []
        for row in hf_dataset:
            examples.append(Example(
                example_id=str(row["question_id"]),
                image=row["image"],
                image_ref=str(row.get("image_id", row["question_id"])),
                prompt_input={"question": row["question"]},
                target=list(row["answers"]),
                metadata={"ocr_tokens": row.get("ocr_tokens", [])},
            ))
        return examples

    def build_prompt(self, example: Example) -> List[dict]:
        return build_image_text_messages(example.prompt_input["question"] + INSTRUCTION_SUFFIX)

    def parse_prediction(self, raw_generation: str, example: Example) -> ParsedPrediction:
        stripped = raw_generation.strip()
        if not stripped:
            return ParsedPrediction(parsed="", parse_ok=False, parse_error="empty generation")
        return ParsedPrediction(parsed=stripped, parse_ok=True)

    def score_example(self, parsed: ParsedPrediction, example: Example) -> ExampleScore:
        if not parsed.parse_ok:
            return ExampleScore(score=0.0, correct=False, detail={"reason": "parse_failure"})
        soft_score = vqa_soft_accuracy(parsed.parsed, example.target)
        return ExampleScore(score=soft_score, correct=soft_score >= CORRECT_THRESHOLD, detail={"vqa_soft_accuracy": soft_score})

    def aggregate_metrics(self, scores: List[ExampleScore]) -> Dict[str, float]:
        n = len(scores)
        if n == 0:
            return {"vqa_soft_accuracy": 0.0, "primary_metric": 0.0, "parser_failure_rate": 0.0}
        parser_failures = sum(1 for s in scores if s.detail.get("reason") == "parse_failure")
        mean_soft_accuracy = sum(s.score for s in scores) / n
        return {
            "vqa_soft_accuracy": mean_soft_accuracy,
            "primary_metric": mean_soft_accuracy,
            "parser_failure_rate": parser_failures / n,
        }

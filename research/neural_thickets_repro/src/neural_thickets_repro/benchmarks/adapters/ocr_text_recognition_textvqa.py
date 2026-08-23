"""TextVQA (ocr_text_recognition) adapter.

Dataset source (confirmed live this session): HuggingFace `lmms-lab-encoder/textvqa`,
matching this project's own established `lmms-lab-encoder` org convention (already used for
GQA, per REPRO_SPEC.md). Validation split has 5k rows; each has `question_id`, `question`,
`image`, and CRITICALLY `answers`: a list of 10 human-annotated answer strings -- the
question's own accepted-answer protocol is NOT a single string, so scoring uses the standard
published VQA soft-accuracy metric (vqa_soft_accuracy.py) against the full 10-answer set,
never a reduction to one "the" answer.

CAPABILITY-LEAKAGE FINDING (this repair pass): a real N=5 manual inspection found TextVQA
questions ("how many wheels does this van have?" -> "4", "is this book material?" -> a
yes/no-style answer) that are not OCR reading questions at all -- TextVQA != pure OCR
capability. TextVQAOCRBenchmark (this class) is UNCHANGED: it remains the full, official
TextVQA validation set under the `ocr_text_recognition` capability name, per this project's
existing scope. TextVQAOCRGroundedBenchmark below is a NEW, separate, EXPERIMENTAL subclass
(capability `ocr_text_recognition_grounded`) narrowing to a deterministically OCR-grounded
subset -- see benchmarks/ocr_grounding.py and prepare_textvqa_ocr_filter.py. Every Example
from this class now also carries `metadata["ocr_grounded"]` (computed once, from target
answers + provided OCR tokens only, never model predictions) so it's visible for audit on
either variant without re-deriving it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from ..base import CapabilityBenchmark, Example, ExampleScore, ParsedPrediction
from ..ocr_grounding import is_ocr_grounded
from ..prompting import build_image_text_messages
from ..vqa_soft_accuracy import vqa_soft_accuracy

INSTRUCTION_SUFFIX = " Answer with a short phrase, reading any text visible in the image if relevant."
CORRECT_THRESHOLD = 0.5  # documented boundary for the boolean ExampleScore.correct field; the continuous VQA soft score is the scientific quantity of record

# src/neural_thickets_repro/benchmarks/adapters/ocr_text_recognition_textvqa.py -> research/neural_thickets_repro
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_FILTER_IDS_DIR = REPO_ROOT / "artifacts" / "benchmark_subsets"
DEFAULT_OCR_GROUNDED_FILTER_FILENAME = "textvqa_ocr_grounded_ids.json"
PREPARE_OCR_GROUNDED_FILTER_COMMAND = "python -m neural_thickets_repro.prepare_textvqa_ocr_filter"


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
            answers = list(row["answers"])
            ocr_tokens = row.get("ocr_tokens", [])
            examples.append(Example(
                example_id=str(row["question_id"]),
                image=row["image"],
                image_ref=str(row.get("image_id", row["question_id"])),
                prompt_input={"question": row["question"]},
                target=answers,
                metadata={"ocr_tokens": ocr_tokens, "ocr_grounded": is_ocr_grounded(answers, ocr_tokens)},
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


class TextVQAOCRGroundedBenchmark(TextVQAOCRBenchmark):
    """EXPERIMENTAL OCR-grounded subset of TextVQA (this repair pass) -- NOT an official
    TextVQA category. Reuses TextVQAOCRBenchmark's prompt/parser/scorer/aggregation entirely
    unchanged; only load_examples() is narrowed by a persisted question-ID filter (built by
    prepare_textvqa_ocr_filter.py, see benchmarks/ocr_grounding.py) -- same
    prepare-then-filter pattern as the two GQA capability adapters
    (adapters/_gqa_filtered_base.py).
    """
    capability = "ocr_text_recognition_grounded"
    name = "textvqa_validation_ocr_grounded"

    def __init__(self, filter_ids_path: "str | Path | None" = None):
        self._filter_ids_path: Path = Path(filter_ids_path) if filter_ids_path is not None else DEFAULT_FILTER_IDS_DIR / DEFAULT_OCR_GROUNDED_FILTER_FILENAME

    def dataset_source(self) -> str:
        return "lmms-lab-encoder/textvqa, filtered to the experimental OCR-grounded subset -- see benchmarks/ocr_grounding.py and prepare_textvqa_ocr_filter.py"

    def known_caveats(self) -> List[str]:
        return [
            "EXPERIMENTAL subset, NOT an official TextVQA category -- retained only when at "
            "least one reference answer is recoverable from the row's own OCR token sequence. "
            "A real N=5 manual inspection found plain-TextVQA questions (e.g. 'how many "
            "wheels does this van have?', a counting question with answer '4' unsupported by "
            "any OCR token; 'is this book material?', not an OCR reading question at all) "
            "that are not OCR reading questions despite living in the full TextVQA set.",
            "Filter membership is derived from target answers + the dataset's own provided "
            "OCR tokens ONLY, computed once by prepare_textvqa_ocr_filter.py and persisted -- "
            "never recomputed from, or influenced by, model predictions.",
        ]

    def load_examples(self, cfg: Any) -> List[Example]:
        if not self._filter_ids_path.exists():
            raise RuntimeError(
                f"{type(self).__name__}: no OCR-grounded filter IDs found at "
                f"{self._filter_ids_path}. Generate it first with:\n"
                f"    {PREPARE_OCR_GROUNDED_FILTER_COMMAND}"
            )
        allowed_ids = set(json.loads(self._filter_ids_path.read_text()))
        all_examples = super().load_examples(cfg)
        return [ex for ex in all_examples if ex.example_id in allowed_ids]

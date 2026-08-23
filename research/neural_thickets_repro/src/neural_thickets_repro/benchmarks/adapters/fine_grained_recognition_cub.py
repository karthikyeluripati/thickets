"""CUB-200-2011 (fine_grained_recognition) adapter.

Dataset source: `bentrevett/caltech-ucsd-birds-200-2011` -- one of several community HF
mirrors of this dataset, none singularly canonical (documented, revisable choice, see
CAPABILITY_BENCHMARK_GATE.md). Canonical species names are read from the dataset's own
`features["label"].names` (a HF `ClassLabel` feature) when exposed -- the same pattern
ImageNet-1K's `int2str` uses -- rather than a hardcoded 200-name list, since getting a
200-item ordered species list wrong by even one entry would silently mislabel every example.
If the chosen mirror does NOT expose a ClassLabel with `.names`, load_examples() hard-fails
with a clear message rather than guessing a name list.
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..base import CapabilityBenchmark, Example, ExampleScore, ParsedPrediction
from ..normalization import normalize_answer
from ..prompting import build_image_text_messages

INSTRUCTION = "What species of bird is shown in this image? Answer with just the species name."


class CUBSchemaError(RuntimeError):
    """The chosen CUB-200-2011 mirror doesn't expose a ClassLabel `label` feature with
    `.names` -- refuses to guess the 200-species name list rather than risk a silent
    off-by-one mislabeling across every example.
    """


def _canonical_species_name(raw_class_name: str) -> str:
    """Mirror class names are typically like "001.Black_footed_Albatross" -- strips any
    leading numeric index and replaces underscores with spaces for a human-readable,
    normalizable label.
    """
    name = raw_class_name.split(".", 1)[-1] if "." in raw_class_name else raw_class_name
    return name.replace("_", " ").strip()


class CUBFineGrainedBenchmark(CapabilityBenchmark):
    capability = "fine_grained_recognition"
    name = "cub200_2011_test"

    def dataset_source(self) -> str:
        return "bentrevett/caltech-ucsd-birds-200-2011"

    def known_caveats(self) -> List[str]:
        return [
            "Loaded from a community HF mirror, not an official CUB-200-2011 HF release -- "
            "documented, revisable dataset-source choice, see CAPABILITY_BENCHMARK_GATE.md.",
        ]

    def load_examples(self, cfg: Any) -> List[Example]:
        from datasets import load_dataset

        hf_dataset = load_dataset(cfg.dataset.source, split=cfg.dataset.split, revision=cfg.dataset.revision)

        label_feature = hf_dataset.features.get("label")
        names = getattr(label_feature, "names", None)
        if not names:
            raise CUBSchemaError(
                f"{cfg.dataset.source}'s 'label' feature does not expose class names via "
                f".names -- refusing to guess the 200-species mapping. Confirm the correct "
                f"mirror/field before proceeding."
            )

        examples: List[Example] = []
        for row_idx, row in enumerate(hf_dataset):
            species = _canonical_species_name(names[row["label"]])
            examples.append(Example(
                example_id=str(row_idx),
                image=row["image"],
                image_ref=f"cub200_row{row_idx}",
                prompt_input={},
                target=species,
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
        # Normalize both sides identically -- lets a verbose generation ("I think this is a
        # Black footed Albatross.") still match, WITHOUT falling back to fragile short-token
        # substring matching: the check requires the full multi-word canonical species phrase
        # to appear as a contiguous, whole-word sequence, padded with spaces so e.g. target
        # "tern" cannot spuriously match inside an unrelated word like "external".
        predicted_norm = normalize_answer(parsed.parsed)
        target_norm = normalize_answer(example.target)
        correct = f" {target_norm} " in f" {predicted_norm} "
        return ExampleScore(score=1.0 if correct else 0.0, correct=correct, detail={"target_norm": target_norm, "predicted_norm": predicted_norm})

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

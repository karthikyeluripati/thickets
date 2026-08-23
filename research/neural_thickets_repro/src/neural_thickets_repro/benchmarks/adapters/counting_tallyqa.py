"""TallyQA (counting) adapter.

Dataset source (confirmed live this session, not guessed): HuggingFace `HuggingFaceM4/the_cauldron`,
config "tallyqa" -- schema confirmed via the HF dataset viewer: each row has `images: [image]`
(one image) and `texts: [{"user": question, "assistant": answer, "source": "TallyQA"}, ...]`
(1-40 conversational turns per image). Answers are plain integers as strings with a trailing
period (e.g. "2."). Documented deviation: this repackaged corpus exposes only ONE split (no
separate held-out test split) -- the fixed N=200 subset is therefore drawn from that single
split, same "no silent protocol change, state the reason" discipline as every other
documented deviation in this project.

One TallyQA "row" (image + up to 40 Q&A turns) must be FLATTENED into one Example per
(image, user, assistant) triple before subset selection -- done once in load_examples().
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..base import CapabilityBenchmark, Example, ExampleScore, ParsedPrediction
from ..normalization import extract_integer
from ..prompting import build_image_text_messages

INSTRUCTION_SUFFIX = " Answer with a single number only."


class TallyQACountingBenchmark(CapabilityBenchmark):
    capability = "counting"
    name = "tallyqa_the_cauldron"

    def dataset_source(self) -> str:
        return "HuggingFaceM4/the_cauldron (config: tallyqa)"

    def subset_selection_rule(self) -> str:
        return "shuffled_prefix"

    def known_caveats(self) -> List[str]:
        return [
            "Repackaged via HuggingFaceM4/the_cauldron, not the original TallyQA release -- "
            "documented reproduction assumption, not confirmed identical to the paper's own split.",
            "Only one split is exposed by this repackaging (no separate held-out test split); "
            "the fixed subset is drawn from that single split.",
        ]

    def load_examples(self, cfg: Any) -> List[Example]:
        from datasets import load_dataset

        hf_dataset = load_dataset(cfg.dataset.source, "tallyqa", split=cfg.dataset.split, revision=cfg.dataset.revision)

        examples: List[Example] = []
        for row_idx, row in enumerate(hf_dataset):
            image = row["images"][0]
            for turn_idx, turn in enumerate(row["texts"]):
                target = extract_integer(turn["assistant"])
                examples.append(Example(
                    example_id=f"{row_idx}_{turn_idx}",
                    image=image,
                    image_ref=f"the_cauldron_tallyqa_row{row_idx}",
                    prompt_input={"question": turn["user"]},
                    target=target,  # None here means the ground-truth ITSELF was unparseable -- an integrity issue, not a model failure
                    metadata={"source": turn.get("source"), "raw_ground_truth": turn["assistant"]},
                ))
        return examples

    def build_prompt(self, example: Example) -> List[dict]:
        question = example.prompt_input["question"]
        return build_image_text_messages(question + INSTRUCTION_SUFFIX)

    def parse_prediction(self, raw_generation: str, example: Example) -> ParsedPrediction:
        value = extract_integer(raw_generation)
        if value is None:
            return ParsedPrediction(parsed=None, parse_ok=False, parse_error=f"no integer found in {raw_generation!r}")
        return ParsedPrediction(parsed=value, parse_ok=True)

    def score_example(self, parsed: ParsedPrediction, example: Example) -> ExampleScore:
        if not parsed.parse_ok:
            return ExampleScore(score=0.0, correct=False, detail={"reason": "parse_failure"})
        exact = parsed.parsed == example.target
        abs_error = abs(parsed.parsed - example.target)
        return ExampleScore(score=1.0 if exact else 0.0, correct=exact, detail={"abs_error": abs_error})

    def aggregate_metrics(self, scores: List[ExampleScore]) -> Dict[str, float]:
        n = len(scores)
        if n == 0:
            return {"exact_match_accuracy": 0.0, "mae": 0.0, "primary_metric": 0.0, "parser_failure_rate": 0.0}

        parser_failures = sum(1 for s in scores if s.detail.get("reason") == "parse_failure")
        exact_match_accuracy = sum(s.score for s in scores) / n

        parsed_errors = [s.detail["abs_error"] for s in scores if "abs_error" in s.detail]
        mae = sum(parsed_errors) / len(parsed_errors) if parsed_errors else float("nan")

        return {
            "exact_match_accuracy": exact_match_accuracy,
            "mae": mae,
            "primary_metric": exact_match_accuracy,
            "parser_failure_rate": parser_failures / n,
        }

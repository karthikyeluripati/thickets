"""Image-dependence sanity check -- a WIRING check, NOT yet a scientific result. Compares
correct-image vs. shuffled-image vs. (where supported) text-only conditions on a small,
deterministic subset, to catch the exact "image never reaches the model" failure class this
project has already hit once (GATE1_DIAGNOSIS.md: a benchmark score barely changing because
the image was silently never attached to the request).

PER-EXAMPLE PERSISTENCE (this repair pass): a real N=50 attribute_recognition sanity run
(correct=0.15, shuffled=0.10, text-only=0.15 -- i.e. IDENTICAL to correct-image) needed
per-example evidence, not just the three aggregate scores, to diagnose. ImageSanityResult now
also carries the full per-condition RunResult objects (correct_result/shuffled_result/
text_only_result) so write_image_sanity_predictions_jsonl() below can persist one line per
(example, condition) -- to_dict() itself stays unchanged/compact (aggregate scores only) so
image_sanity.json doesn't balloon in size.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .base import CapabilityBenchmark, Example
from .runner import RunResult, run_benchmark


class ImageSanityError(RuntimeError):
    """Cannot construct a valid sanity condition (e.g. a derangement is impossible at n<=1)."""


def _derangement(indices: List[int], rng: random.Random) -> List[int]:
    n = len(indices)
    perm = list(indices)
    # Fisher-Yates shuffle, retry until no fixed points. Fine for the small sanity-subset
    # sizes this is used for (default n=40) -- terminates with probability 1.
    while True:
        rng.shuffle(perm)
        if all(perm[i] != indices[i] for i in range(n)):
            return perm


def make_shuffled_variant(examples: List[Example], seed: int, benchmark: CapabilityBenchmark) -> List[Example]:
    """Returns a NEW list of Examples (same example_id/prompt_input/target order and values
    as the input) with each example's VISUAL INPUT swapped for a DIFFERENT example's -- a
    true derangement: no example receives its own original visual input. Deterministic given
    seed. Delegates the actual per-example construction to
    `benchmark.make_shuffled_image_variant()` (see CapabilityBenchmark's own docstring) --
    correct for the common "swap the whole image" case AND for a capability (e.g. Visual
    Genome attributes) whose visual input is a localized crop derived from BOTH the image and
    capability-specific metadata, which needs to swap in the SOURCE example's own metadata
    too, not just its image.
    """
    n = len(examples)
    if n <= 1:
        raise ImageSanityError(f"Cannot construct a derangement with n={n} examples (need n > 1)")

    rng = random.Random(seed)
    permutation = _derangement(list(range(n)), rng)
    return [benchmark.make_shuffled_image_variant(examples[i], examples[permutation[i]]) for i in range(n)]


def make_text_only_variant(examples: List[Example]) -> List[Example]:
    """Returns a NEW list of Examples with image=None -- prompt/target unchanged. Caller must
    check benchmark.supports_text_only_condition() first; this function has no opinion on
    that (single responsibility).
    """
    variants = []
    for ex in examples:
        new_metadata = dict(ex.metadata)
        new_metadata["sanity_condition"] = "text_only"
        variants.append(Example(
            example_id=ex.example_id, image=None, image_ref="text_only",
            prompt_input=ex.prompt_input, target=ex.target, metadata=new_metadata,
        ))
    return variants


@dataclass
class ImageSanityResult:
    n: int
    correct_image_primary_metric: float
    shuffled_image_primary_metric: float
    text_only_primary_metric: Optional[float]
    text_only_supported: bool
    text_only_unsupported_reason: Optional[str]
    correct_result: Optional[RunResult] = None
    shuffled_result: Optional[RunResult] = None
    text_only_result: Optional[RunResult] = None

    @property
    def correct_minus_shuffled(self) -> float:
        return self.correct_image_primary_metric - self.shuffled_image_primary_metric

    @property
    def correct_minus_text_only(self) -> Optional[float]:
        if self.text_only_primary_metric is None:
            return None
        return self.correct_image_primary_metric - self.text_only_primary_metric

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "correct_image_score": self.correct_image_primary_metric,
            "shuffled_image_score": self.shuffled_image_primary_metric,
            "text_only_score": self.text_only_primary_metric,
            "text_only_supported": self.text_only_supported,
            "text_only_unsupported_reason": self.text_only_unsupported_reason,
            "correct_minus_shuffled": self.correct_minus_shuffled,
            "correct_minus_text_only": self.correct_minus_text_only,
        }


def run_image_sanity_check(
    benchmark: CapabilityBenchmark, sanity_examples: List[Example], llm, tokenizer, sampling_params, seed: int,
) -> ImageSanityResult:
    correct_result = run_benchmark(benchmark, sanity_examples, llm, tokenizer, sampling_params)

    shuffled_examples = make_shuffled_variant(sanity_examples, seed, benchmark)
    shuffled_result = run_benchmark(benchmark, shuffled_examples, llm, tokenizer, sampling_params)

    text_only_supported = benchmark.supports_text_only_condition()
    text_only_metric = None
    text_only_result = None
    if text_only_supported:
        text_only_examples = make_text_only_variant(sanity_examples)
        text_only_result = run_benchmark(benchmark, text_only_examples, llm, tokenizer, sampling_params, allow_missing_image=True)
        text_only_metric = text_only_result.aggregate_metrics["primary_metric"]

    return ImageSanityResult(
        n=len(sanity_examples),
        correct_image_primary_metric=correct_result.aggregate_metrics["primary_metric"],
        shuffled_image_primary_metric=shuffled_result.aggregate_metrics["primary_metric"],
        text_only_primary_metric=text_only_metric,
        text_only_supported=text_only_supported,
        text_only_unsupported_reason=None if text_only_supported else benchmark.text_only_unsupported_reason(),
        correct_result=correct_result, shuffled_result=shuffled_result, text_only_result=text_only_result,
    )


def write_image_sanity_predictions_jsonl(result: ImageSanityResult, path: "str | Path") -> None:
    """Persists PER-EXAMPLE predictions for every sanity condition that actually ran (this
    repair pass) -- one line per (example, condition), so a real prediction change (or lack
    thereof) across conditions is directly auditable, not just inferred from the three
    aggregate scores. The text-only condition contributes zero lines when the capability
    doesn't support it (never a fabricated row for an untested condition).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for condition, run_result in (
            ("correct", result.correct_result), ("shuffled", result.shuffled_result), ("text_only", result.text_only_result),
        ):
            if run_result is None:
                continue
            for r in run_result.per_example:
                f.write(json.dumps({
                    "condition": condition,
                    "example_id": r.example_id,
                    "image_ref": r.image_ref,
                    "raw_generation": r.raw_generation,
                    "parsed_prediction": r.parsed.parsed,
                    "score": r.score.score,
                    "correct": r.score.correct,
                    "detail": r.score.detail,
                }, default=str) + "\n")

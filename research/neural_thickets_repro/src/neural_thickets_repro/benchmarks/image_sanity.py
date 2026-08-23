"""Image-dependence sanity check -- a WIRING check, NOT yet a scientific result. Compares
correct-image vs. shuffled-image vs. (where supported) text-only conditions on a small,
deterministic subset, to catch the exact "image never reaches the model" failure class this
project has already hit once (GATE1_DIAGNOSIS.md: a benchmark score barely changing because
the image was silently never attached to the request).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

from .base import CapabilityBenchmark, Example
from .runner import run_benchmark


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


def make_shuffled_variant(examples: List[Example], seed: int) -> List[Example]:
    """Returns a NEW list of Examples (same example_id/prompt_input/target order and values
    as the input) with each example's image swapped for a DIFFERENT example's image -- a true
    derangement: no example receives its own original image. Deterministic given seed; the
    resulting permutation is recorded per-example (metadata["sanity_shuffle_source_id"]) so
    the check is reproducible/auditable.
    """
    n = len(examples)
    if n <= 1:
        raise ImageSanityError(f"Cannot construct a derangement with n={n} examples (need n > 1)")

    rng = random.Random(seed)
    permutation = _derangement(list(range(n)), rng)

    shuffled = []
    for i, ex in enumerate(examples):
        source = examples[permutation[i]]
        new_metadata = dict(ex.metadata)
        new_metadata["sanity_shuffle_source_id"] = source.example_id
        shuffled.append(Example(
            example_id=ex.example_id, image=source.image, image_ref=f"shuffled_from:{source.image_ref}",
            prompt_input=ex.prompt_input, target=ex.target, metadata=new_metadata,
        ))
    return shuffled


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

    shuffled_examples = make_shuffled_variant(sanity_examples, seed)
    shuffled_result = run_benchmark(benchmark, shuffled_examples, llm, tokenizer, sampling_params)

    text_only_supported = benchmark.supports_text_only_condition()
    text_only_metric = None
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
    )

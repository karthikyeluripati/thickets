"""Shared interface for the Capability Benchmark Gate (base-model, zero-perturbation
evaluation across 8 visual capabilities). Pure Python, no GPU/ray/vllm/datasets import at
module scope -- CPU-testable against hand-built Example objects.

CapabilityBenchmark deliberately has no evaluate() method: the model-execution path lives in
runner.run_benchmark(benchmark, examples, llm, tokenizer, sampling_params), a free function
that takes an already-resolved example list and "whatever engine/model state currently
exists" -- this is what lets a future perturbation sweep do

    for seed: perturb(model, scope, radius, seed)
    for benchmark: run_benchmark(benchmark, fixed_examples, llm, ...)

without this milestone's code changing at all. The benchmark adapter owns dataset revision/
split, sample filtering, prompt, target representation, prediction parser, metric, and
capability metadata; the VLM adapter (vlm_adapter.py) owns image preprocessing, chat
template, generation, and device/model handling. prepare_image() below is the one place a
benchmark may do dataset-specific PIXEL preparation (e.g. drawing a bbox marker) -- the VLM
adapter itself never becomes dataset-aware.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional


@dataclass(frozen=True)
class Example:
    """example_id must be stable and unique within this benchmark's full candidate pool
    (post sample-filtering, pre subset-selection) -- subset persistence and the dataset-drift
    guard in subset_selection.py both key off it. `prompt_input`/`target` are adapter-owned,
    opaque payloads: runner.py and card.py never inspect their contents, only the owning
    adapter's build_prompt/parse_prediction/score_example do.
    """
    example_id: str
    image: Optional[Any] = None            # PIL.Image.Image, or None only for the text-only sanity condition
    image_ref: str = ""                    # human-readable id/path/url, recorded in predictions.jsonl
    prompt_input: Dict[str, Any] = field(default_factory=dict)
    target: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedPrediction:
    parsed: Any
    parse_ok: bool
    parse_error: Optional[str] = None


@dataclass
class ExampleScore:
    score: float                           # in [0, 1] for every adapter, so cross-capability aggregation is comparable
    correct: Optional[bool] = None         # boolean correctness where meaningful; None for pure partial-credit metrics
    detail: Dict[str, Any] = field(default_factory=dict)


class CapabilityBenchmark(ABC):
    capability: ClassVar[str]
    name: ClassVar[str]

    @abstractmethod
    def load_examples(self, cfg: Any) -> List[Example]:
        """Loads the full candidate pool (after any dataset-specific sample filter, e.g. the
        GQA spatial/relational ID filter or TallyQA's multi-turn flattening) -- deterministic
        given cfg, no internal randomness. Subset selection to a fixed N is NOT done here,
        see subset_selection.py. May hit network/dataset downloads; never touches GPU/vllm/
        ray.
        """

    @abstractmethod
    def build_prompt(self, example: Example) -> List[dict]:
        """-> chat "messages" list, the exact shape vlm_adapter.format_chat_prompt expects."""

    def prepare_image(self, example: Example):
        """Default: passthrough. Override for dataset-specific pixel preparation (e.g. a
        drawn bounding-box marker for Visual Genome attributes). Must never mutate
        example.image in place -- operate on a copy so the original is preserved separately
        (needed for the image-sanity shuffle condition and for auditability).
        """
        return example.image

    @abstractmethod
    def parse_prediction(self, raw_generation: str, example: Example) -> ParsedPrediction: ...

    @abstractmethod
    def score_example(self, parsed: ParsedPrediction, example: Example) -> ExampleScore: ...

    @abstractmethod
    def aggregate_metrics(self, scores: List[ExampleScore]) -> Dict[str, float]:
        """Must include a "primary_metric" key (float, the headline score card.py/summary.py
        read generically) and a "parser_failure_rate" key. Other capability-specific keys
        (e.g. "mae", "mean_iou", "vqa_soft_accuracy") are additive.
        """

    # --- card/runner metadata, overridable; defaults given where safe ---

    def default_subset_size(self) -> int:
        return 200

    def subset_selection_rule(self) -> str:
        """"prefix" or "shuffled_prefix" -- see subset_selection.py's module docstring for
        when each is appropriate. Default is "shuffled_prefix" since most new datasets here
        risk class/category-ordered rows; GQA-derived adapters override back to "prefix" to
        match the existing project convention (prepare_gqa_data.py's own prefix slice).
        """
        return "shuffled_prefix"

    def subset_selection_seed(self, global_seed: int) -> Optional[int]:
        return global_seed

    def supports_text_only_condition(self) -> bool:
        return True

    def text_only_unsupported_reason(self) -> Optional[str]:
        return None

    def dataset_source(self) -> str:
        raise NotImplementedError

    def dataset_revision(self) -> Optional[str]:
        return None

    def known_caveats(self) -> List[str]:
        return []

    def repeatability_verdict(self, base_result: Any, repeat_result: Any) -> "tuple[bool, Dict[str, Any]]":
        """Decides whether two identical-input runs (same fixed subset, same greedy
        SamplingParams) count as "repeatable" for THIS capability, and returns any
        capability-specific diagnostics to merge into repeatability.json (empty dict if
        none). `base_result`/`repeat_result` are `runner.RunResult` instances -- typed as
        `Any` here to avoid base.py importing runner.py (runner.py already imports base.py).

        DEFAULT (appropriate for a DISCRETE-answer capability -- a class label, yes/no, a
        count, a short phrase): every example's parsed prediction must be EXACTLY identical
        (repr-equal) between runs, AND the primary metric must match exactly. For a discrete
        answer, "the parser extracted a different token on an identical greedy run" IS a
        genuine instability worth failing on.

        Greedy decoding (temperature=0) guarantees deterministic SAMPLING, not necessarily
        bitwise-identical floating-point kernel output on real GPU hardware -- a capability
        whose parsed prediction is a CONTINUOUS measurement (e.g. a bounding box) should
        override this with a measurement-stability criterion instead of byte-identical-token
        equality, which is the wrong scientific question for a continuous value. See
        RefCOCOGroundingBenchmark.repeatability_verdict() for that override.
        """
        base_by_id = {r.example_id: r for r in base_result.per_example}
        repeat_by_id = {r.example_id: r for r in repeat_result.per_example}
        common_ids = set(base_by_id) & set(repeat_by_id)
        exact_matches = sum(1 for eid in common_ids if repr(base_by_id[eid].parsed.parsed) == repr(repeat_by_id[eid].parsed.parsed))
        metrics_match = base_result.aggregate_metrics.get("primary_metric") == repeat_result.aggregate_metrics.get("primary_metric")
        repeatable = metrics_match and bool(common_ids) and exact_matches == len(common_ids)
        return repeatable, {}

    def make_shuffled_image_variant(self, example: "Example", source_example: "Example") -> "Example":
        """Builds ONE shuffled-image-condition Example for the image-dependence sanity check:
        `example`'s own prompt/target, but the VISUAL INPUT drawn from `source_example`
        instead (image_sanity.make_shuffled_variant() calls this once per example, pairing
        each with a different example via a true derangement).

        DEFAULT: swaps in `source_example.image` wholesale, keeping `example`'s own metadata
        otherwise unchanged -- correct whenever prepare_image() operates on the whole
        attached image with no other per-example localization metadata involved (true for
        most capabilities).

        Override when prepare_image() derives the actual visual input from BOTH `image` AND
        capability-specific metadata (e.g. a bounding-box crop) -- the override must ALSO
        carry over whatever of `source_example`'s OWN metadata is needed to correctly
        reproduce THAT image's own localized region. Swapping only `image` while keeping
        `example`'s own localization metadata would silently apply one example's box to a
        DIFFERENT photo, producing a misaligned/meaningless crop rather than a genuine
        "different but valid" visual distractor -- see
        VisualGenomeAttributeBenchmark.make_shuffled_image_variant() for the real fix.
        """
        new_metadata = dict(example.metadata)
        new_metadata["sanity_shuffle_source_id"] = source_example.example_id
        return Example(
            example_id=example.example_id, image=source_example.image, image_ref=f"shuffled_from:{source_example.image_ref}",
            prompt_input=example.prompt_input, target=example.target, metadata=new_metadata,
        )

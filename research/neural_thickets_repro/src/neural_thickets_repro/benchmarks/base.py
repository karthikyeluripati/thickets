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

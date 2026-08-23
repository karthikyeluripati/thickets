"""Typed config loading for configs/gqa_repro.yaml.

Mirrors the schema described in REPRO_SPEC.md. Fields that are UNRESOLVED reproduction
assumptions are represented as None in the YAML and must be explicitly resolved (never
silently guessed) before Gate 1+ code paths use them for a real run -- see
ExperimentConfig.require_resolved().
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Optional

import yaml


class UnresolvedFieldError(RuntimeError):
    """A config field needed for a real run is still an unresolved reproduction assumption."""


@dataclasses.dataclass
class ModelConfig:
    name: str
    revision: Optional[str]
    architecture: Optional[str]
    precision: str


@dataclasses.dataclass
class DatasetConfig:
    name: str
    source: str
    revision: Optional[str]
    selection_split: Optional[str]
    test_split: Optional[str]
    selection_set_size: Optional[int]
    test_set_size: Optional[int]


@dataclasses.dataclass
class RandoptConfig:
    N: int
    K: int
    sigmas: Optional[list]
    sigma_candidates: dict
    perturbation_distribution: str
    perturbation_scope: str
    freeze_visual_encoder: bool
    visual_param_prefixes: list


@dataclasses.dataclass
class EvaluationConfig:
    answer_normalization: str
    decoding: str
    max_tokens: int
    voting: str
    tie_break: str


@dataclasses.dataclass
class ReproducibilityConfig:
    global_seed: int


@dataclasses.dataclass
class BaselineTolerance:
    proceed_at_most: float
    investigate_at_most: float


@dataclasses.dataclass
class GatesConfig:
    baseline_tolerance_pp: BaselineTolerance
    require_gate1_before_gate2: bool
    require_gate2_before_gate3: bool


@dataclasses.dataclass
class HardwareConfig:
    min_free_disk_gb: float


@dataclasses.dataclass
class ExperimentConfig:
    experiment: str
    model: ModelConfig
    dataset: DatasetConfig
    randopt: RandoptConfig
    evaluation: EvaluationConfig
    reproducibility: ReproducibilityConfig
    gates: GatesConfig
    hardware: HardwareConfig

    def require_resolved(self, *field_paths: str) -> None:
        """Raise UnresolvedFieldError if any dotted field path currently resolves to None.

        Call this at the top of any Gate-1+ code path about to run something real.
        Gate-0 code that only inspects/reports the config should not call this.
        """
        for path in field_paths:
            obj: Any = self
            for part in path.split("."):
                obj = getattr(obj, part)
            if obj is None:
                raise UnresolvedFieldError(
                    f"Config field '{path}' is an UNRESOLVED reproduction assumption "
                    f"(see REPRO_SPEC.md) and cannot be used for a real run without "
                    f"being explicitly resolved first."
                )


@dataclasses.dataclass
class GenerationConfig:
    decoding: str
    max_tokens: int


@dataclasses.dataclass
class CapabilityDatasetConfig:
    capability: str                  # must equal the adapter class's `capability` ClassVar
    adapter: str                     # dotted class path, e.g. "neural_thickets_repro.benchmarks.adapters.visual_grounding_refcoco.RefCOCOGroundingBenchmark"
    source: str                      # HF dataset repo id
    revision: Optional[str]
    split: str
    subset_size: int                 # default 200; a documented deviation (e.g. TallyQA's single-split situation) sets deviation_reason
    deviation_reason: Optional[str]  # required non-null iff subset_size != 200 or the split has no held-out test counterpart
    subset_selection_rule: str       # "prefix" | "shuffled_prefix"
    subset_seed: Optional[int]       # null iff rule == "prefix" (no RNG involved)


@dataclasses.dataclass
class BenchmarkGatesConfig:
    max_parser_failure_rate_pass: float
    max_parser_failure_rate_needs_review: float
    # image-sanity gap (correct - shuffled, or correct - text_only) thresholds: gap <= 0 is
    # always a forced FAIL (the image isn't detectably reaching the model) regardless of this
    # value; a gap in [0, image_sanity_min_gap_pass) is NEEDS_REVIEW (see card.py's
    # decide_status -- at a ~40-example sanity subset a small positive gap isn't yet
    # distinguishable from a proportion metric's own noise floor); only >= this value is clean.
    image_sanity_min_gap_pass: float
    image_sanity_subset_size: int
    floor_ceiling_low: float
    floor_ceiling_high: float


def _check_resolved(root: Any, field_paths) -> None:
    """Shared by CapabilityBenchmarkConfig.require_resolved() only -- ExperimentConfig's own
    require_resolved() above is left untouched (frozen Gate 0-2 dependency).
    """
    for path in field_paths:
        obj: Any = root
        for part in path.split("."):
            obj = getattr(obj, part)
        if obj is None:
            raise UnresolvedFieldError(
                f"Config field '{path}' is an UNRESOLVED value and cannot be used for a "
                f"real run without being explicitly resolved first."
            )


@dataclasses.dataclass
class CapabilityBenchmarkConfig:
    """One capability per config file (mirrors DatasetConfig's existing singular convention
    -- not a multi-benchmark registry). Deliberately does NOT reuse EvaluationConfig: its
    voting/tie_break fields are RandOpt-ensemble-specific and don't generalize to a
    single-pass, zero-perturbation evaluation; ModelConfig/ReproducibilityConfig/
    HardwareConfig ARE reused as-is since model/repro/hardware concerns are genuinely
    dataset-independent already.
    """
    experiment: str
    model: ModelConfig
    reproducibility: ReproducibilityConfig
    hardware: HardwareConfig
    generation: GenerationConfig
    dataset: CapabilityDatasetConfig
    gates: BenchmarkGatesConfig

    def require_resolved(self, *field_paths: str) -> None:
        _check_resolved(self, field_paths)


def load_capability_benchmark_config(path: "str | Path") -> CapabilityBenchmarkConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return CapabilityBenchmarkConfig(
        experiment=raw["experiment"],
        model=ModelConfig(**raw["model"]),
        reproducibility=ReproducibilityConfig(**raw["reproducibility"]),
        hardware=HardwareConfig(**raw["hardware"]),
        generation=GenerationConfig(**raw["generation"]),
        dataset=CapabilityDatasetConfig(**raw["dataset"]),
        gates=BenchmarkGatesConfig(**raw["gates"]),
    )


def load_config(path: "str | Path") -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text())

    gates_raw = raw["gates"]
    gates = GatesConfig(
        baseline_tolerance_pp=BaselineTolerance(**gates_raw["baseline_tolerance_pp"]),
        require_gate1_before_gate2=gates_raw["require_gate1_before_gate2"],
        require_gate2_before_gate3=gates_raw["require_gate2_before_gate3"],
    )

    return ExperimentConfig(
        experiment=raw["experiment"],
        model=ModelConfig(**raw["model"]),
        dataset=DatasetConfig(**raw["dataset"]),
        randopt=RandoptConfig(**raw["randopt"]),
        evaluation=EvaluationConfig(**raw["evaluation"]),
        reproducibility=ReproducibilityConfig(**raw["reproducibility"]),
        gates=gates,
        hardware=HardwareConfig(**raw["hardware"]),
    )

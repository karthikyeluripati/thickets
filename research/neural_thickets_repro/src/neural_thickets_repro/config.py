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

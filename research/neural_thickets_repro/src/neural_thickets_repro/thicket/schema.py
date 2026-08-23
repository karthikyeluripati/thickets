"""Experiment output record schema (spec section O) -- one record per EVALUATED perturbation
x capability pair. Deliberately references per-example results by path/hash rather than
embedding them, to avoid duplicating giant payloads across every record.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ExperimentResultRecord:
    experiment_id: str
    perturbation_id: str
    model_family: str
    model_scale: str
    model_revision: str

    perturbation_mode: str
    anatomy_region: Optional[str]
    radius: Optional[float]
    sigma: Optional[float]
    seed: int
    parameter_mask_hash: str

    capability: str
    dataset_role: str
    subset_hash: str

    base_score: float
    perturbed_score: float
    delta: float

    parser_failure_rate: Optional[float]

    per_example_result_path: Optional[str]
    per_example_result_hash: Optional[str]

    runtime_metadata: Dict[str, Any]

    def __post_init__(self) -> None:
        expected_delta = self.perturbed_score - self.base_score
        if abs(self.delta - expected_delta) > 1e-9:
            raise ValueError(f"delta ({self.delta}) does not equal perturbed_score - base_score ({expected_delta})")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentResultRecord":
        return cls(**d)

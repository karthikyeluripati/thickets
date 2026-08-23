"""CPU-only experiment-SIZE estimator (spec section K) -- NOT a dollar-cost estimator, does
not assume any billing model or GPU throughput. Purpose: catch an accidentally 10x/100x larger
sweep than intended before it is launched, by making the arithmetic explicit and inspectable.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentSizeInputs:
    n_models: int
    n_capabilities: int
    n_anatomy_regions: int
    n_radii: int
    n_perturbations_per_condition: int
    n_examples_per_capability: int
    n_repeats: int = 1
    n_sanity_runs: int = 0
    ensemble_k: int = 1

    def __post_init__(self) -> None:
        for field_name in (
            "n_models", "n_capabilities", "n_anatomy_regions", "n_radii",
            "n_perturbations_per_condition", "n_examples_per_capability", "n_repeats", "ensemble_k",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive, got {getattr(self, field_name)}")
        if self.n_sanity_runs < 0:
            raise ValueError(f"n_sanity_runs must be non-negative, got {self.n_sanity_runs}")


@dataclass(frozen=True)
class ExperimentSizeReport:
    unique_candidate_models: int
    conditions: int
    total_model_example_evaluations: int
    evaluations_per_capability: int
    evaluations_per_anatomy: int
    evaluations_per_radius: int
    baseline_evaluations: int
    sanity_evaluations: int
    multiplier_vs_one_baseline: float


def estimate_experiment_size(inputs: ExperimentSizeInputs) -> ExperimentSizeReport:
    """Definitions (all arithmetic, no dollar cost, no throughput assumption):

    unique_candidate_models = n_models * n_anatomy_regions * n_radii * n_perturbations_per_condition
        -- one distinct perturbed weight set per (model, region, radius, perturbation).

    conditions = n_capabilities * n_anatomy_regions * n_radii
        -- one (capability, region, radius) cell per candidate's evaluation sweep.

    total_model_example_evaluations = unique_candidate_models * n_capabilities
        * n_examples_per_capability * n_repeats * ensemble_k
        -- every candidate is evaluated on every capability's full example set, `n_repeats`
        times (e.g. for repeatability checks), with `ensemble_k` generations per example where
        an ensemble/voting scheme applies (ensemble_k=1 when not applicable).

    baseline_evaluations = n_models * n_capabilities * n_examples_per_capability
        -- ONE full baseline sweep (the unperturbed model evaluated once across every
        capability's examples) -- the unit "multiplier_vs_one_baseline" is relative to.

    sanity_evaluations = n_sanity_runs * n_models * n_capabilities * n_examples_per_capability
        -- image-/text-dependence sanity conditions run at the BASE-model level (correct/
        shuffled/text-only), not per perturbed candidate; 0 when n_sanity_runs=0.

    multiplier_vs_one_baseline = total_model_example_evaluations / baseline_evaluations
        -- how many baseline-equivalent sweeps this experiment costs; equals
        unique_candidate_models exactly when n_repeats=ensemble_k=1 (each perturbed candidate
        costs exactly one baseline-equivalent sweep).
    """
    unique_candidate_models = inputs.n_models * inputs.n_anatomy_regions * inputs.n_radii * inputs.n_perturbations_per_condition
    conditions = inputs.n_capabilities * inputs.n_anatomy_regions * inputs.n_radii
    total_model_example_evaluations = (
        unique_candidate_models * inputs.n_capabilities * inputs.n_examples_per_capability * inputs.n_repeats * inputs.ensemble_k
    )
    baseline_evaluations = inputs.n_models * inputs.n_capabilities * inputs.n_examples_per_capability
    sanity_evaluations = inputs.n_sanity_runs * inputs.n_models * inputs.n_capabilities * inputs.n_examples_per_capability

    return ExperimentSizeReport(
        unique_candidate_models=unique_candidate_models,
        conditions=conditions,
        total_model_example_evaluations=total_model_example_evaluations,
        evaluations_per_capability=total_model_example_evaluations // inputs.n_capabilities,
        evaluations_per_anatomy=total_model_example_evaluations // inputs.n_anatomy_regions,
        evaluations_per_radius=total_model_example_evaluations // inputs.n_radii,
        baseline_evaluations=baseline_evaluations,
        sanity_evaluations=sanity_evaluations,
        multiplier_vs_one_baseline=total_model_example_evaluations / baseline_evaluations,
    )

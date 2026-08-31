"""Paired-condition per-candidate evaluator -- extends the existing verified infrastructure
minimally: scoped_apply_perturbation (scoped_perturbation.py, unchanged), the upstream
reset_to_base_weights/restoration-verification RPC pattern (run_scoped_randopt.py's own
_collective_rpc_single_worker dispatch convention, reused BY PATTERN), and the already-
validated shuffled/text-only Example construction (image_sanity.py / shuffle_manifest.py).
Every dependency is injected (engine RPC dispatch, run_benchmark, restore/verify) so this
module is fully CPU-testable with fakes, matching this repository's established convention
throughout the Stage 6-11 lineage.

TRANSACTIONAL ORDERING (identical discipline to every other per-candidate evaluator in this
repo): rows are only constructed and returned AFTER perturb -> evaluate(every capability x
every condition) -> restore -> verify-restoration has fully succeeded. A failure at any point
resets to base and re-raises -- no row for that candidate is ever appended.

SCOPE ISOLATION: scoped_apply_perturbation's own code (see scoped_perturbation.py) can only
ever call `p.add_(delta)` for `name in manifest.selected_param_names` -- the scope's own
tensors -- this is a STRUCTURAL guarantee (by construction of that function, not by luck), and
diagnostics/scope_isolation_gpu_check.py is the ONE authorized GPU mechanical validation of it
(per SCOPED_PERTURBATION_DESIGN.md). Re-diffing every out-of-scope parameter on all 600
candidates x 5 capabilities would be prohibitively expensive and would not check anything the
structural guarantee + the one-time mechanical validation doesn't already establish -- so
`scope_isolation_verification_ok` on every row is the PRECONDITION result (was this exact
scope's mechanical isolation check confirmed PASS before this candidate ever ran?), never a
per-candidate re-diff. This is the same "verify once as a precondition, not per-candidate"
pattern the 32B G1-G8 readiness gates already established for this repository.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..benchmarks.base import CapabilityBenchmark, Example
from ..scopes import compute_relative_l2_sigma
from .candidates import PerturbationCandidate
from .schema import CausalDensityResultRow

NORM_VERIFICATION_RELATIVE_TOLERANCE = 1e-9  # bit-exact formula recomputation, not sampling variance


class NormVerificationFailedError(RuntimeError):
    """The applied per-tensor sigma does not match compute_relative_l2_sigma's own formula
    recomputed from the candidate's own returned scope statistics -- a real perturbation-
    magnitude bug, never sampling variance (the formula is deterministic given (base_l2_norm,
    param_count, r); this checks the MECHANISM, not where one particular Gaussian draw landed).
    """


class RestorationVerificationFailedError(RuntimeError):
    """Exact restoration to the stored base was not confirmed after a candidate -- hard stop,
    never proceeds to the next candidate with an unverified weight state.
    """


class ScopeIsolationPreconditionError(RuntimeError):
    """This scope's mechanical isolation precondition (diagnostics/scope_isolation_gpu_check.py)
    was never confirmed PASS -- refuses to evaluate any candidate in this scope.
    """


def verify_norm(perturb_result: Dict[str, Any], *, tolerance: float = NORM_VERIFICATION_RELATIVE_TOLERANCE) -> bool:
    if perturb_result.get("requested_relative_l2") is None:
        return False
    expected_sigma = compute_relative_l2_sigma(
        perturb_result["scope_base_l2_norm"], perturb_result["scope_param_count"], perturb_result["requested_relative_l2"],
    )
    actual_sigma = perturb_result["derived_sigma"]
    if expected_sigma == 0.0:
        return actual_sigma == 0.0
    return abs(actual_sigma - expected_sigma) / abs(expected_sigma) <= tolerance


@dataclass(frozen=True)
class CapabilityAuditData:
    """One capability's frozen audit-set data for all three conditions -- built ONCE
    (subsets.py + shuffle_manifest.py), reused for every one of the 600 candidates, never
    reconstructed per candidate ("never reshuffle between candidates").
    """
    capability: str
    benchmark: CapabilityBenchmark
    dataset_source: str
    correct_examples: Sequence[Example]
    shuffled_examples: Sequence[Example]
    text_only_examples: Optional[Sequence[Example]]  # None iff not benchmark.supports_text_only_condition()


def _rows_for_condition(
    *, capability: str, dataset_source: str, subset_role: str, condition: str,
    original_examples: Sequence[Example], evaluated_examples: Sequence[Example], run_result: Any,
    scope: Optional[str], radius: Optional[float], seed: Optional[int], candidate_id: Optional[str], is_base: bool,
    perturbation_norm: Optional[float], decoding_config: Dict[str, Any],
    source_commit: str, run_id: str, model_name: str, model_revision: str,
) -> List[CausalDensityResultRow]:
    aggregate_score = run_result.aggregate_metrics["primary_metric"]
    per_example_by_id = {r.example_id: r for r in run_result.per_example}
    evaluated_by_id = {e.example_id: e for e in evaluated_examples}
    rows = []
    for orig in original_examples:
        per_ex = per_example_by_id[orig.example_id]
        evaluated_ex = evaluated_by_id[orig.example_id]
        rows.append(CausalDensityResultRow(
            source_commit=source_commit, run_id=run_id, model_name=model_name, model_revision=model_revision,
            capability=capability, dataset_source=dataset_source, subset_role=subset_role, sample_id=orig.example_id,
            original_image_id=orig.image_ref, evaluated_image_id=evaluated_ex.image_ref, condition=condition,
            scope=scope, radius=radius, seed=seed, candidate_id=candidate_id, is_base=is_base,
            prediction=per_ex.raw_generation, normalized_prediction=repr(per_ex.parsed.parsed), target=orig.target,
            per_example_score=per_ex.score.score, aggregate_score=aggregate_score, perturbation_norm=perturbation_norm,
            norm_verification_ok=None, scope_isolation_verification_ok=None, restoration_verification_ok=None,
            decoding_config=decoding_config,
        ))
    return rows


def evaluate_one_candidate_all_capabilities(
    engine: Any, candidate: PerturbationCandidate, capability_data: Dict[str, CapabilityAuditData], tokenizer: Any, sampling_params: Any,
    *, run_benchmark: Callable, apply_perturbation: Callable, reset_to_base_weights: Callable,
    scope_requires_encoder_cache_reset: Callable, reset_vllm_encoder_cache_full: Callable, verify_restoration: Callable,
    scope_isolation_precondition_ok: bool, decoding_config: Dict[str, Any],
    source_commit: str, run_id: str, model_name: str, model_revision: str,
) -> List[CausalDensityResultRow]:
    if not scope_isolation_precondition_ok:
        raise ScopeIsolationPreconditionError(
            f"Scope {candidate.scope!r}'s mechanical isolation precondition is not confirmed PASS -- "
            f"refusing to evaluate candidate {candidate.candidate_id!r}."
        )

    pending_rows: List[CausalDensityResultRow] = []
    try:
        perturb_result = apply_perturbation(engine, candidate.seed, candidate.radius, candidate.scope)
        norm_ok = verify_norm(perturb_result)
        if not norm_ok:
            raise NormVerificationFailedError(
                f"Candidate {candidate.candidate_id!r}: derived_sigma {perturb_result.get('derived_sigma')} does not "
                f"match compute_relative_l2_sigma's own recomputation from this candidate's own scope statistics."
            )

        if scope_requires_encoder_cache_reset(candidate.scope):
            reset_vllm_encoder_cache_full(engine)

        for capability, data in capability_data.items():
            conditions = [
                ("correct_image", data.correct_examples),
                ("shuffled_image", data.shuffled_examples),
            ]
            if data.text_only_examples is not None:
                conditions.append(("text_only", data.text_only_examples))
            for condition, examples in conditions:
                result = run_benchmark(
                    data.benchmark, list(examples), engine, tokenizer, sampling_params,
                    allow_missing_image=(condition == "text_only"),
                )
                pending_rows.extend(_rows_for_condition(
                    capability=capability, dataset_source=data.dataset_source, subset_role="audit", condition=condition,
                    original_examples=data.correct_examples, evaluated_examples=examples, run_result=result,
                    scope=candidate.scope, radius=candidate.radius, seed=candidate.seed, candidate_id=candidate.candidate_id, is_base=False,
                    perturbation_norm=perturb_result["actual_perturbation_l2"], decoding_config=decoding_config,
                    source_commit=source_commit, run_id=run_id, model_name=model_name, model_revision=model_revision,
                ))
    except Exception:
        reset_to_base_weights(engine)
        raise

    reset_to_base_weights(engine)
    restoration_ok = verify_restoration(engine)
    if not restoration_ok:
        raise RestorationVerificationFailedError(
            f"Exact restoration not confirmed after candidate {candidate.candidate_id!r} (scope={candidate.scope!r}, "
            f"radius={candidate.radius!r}, seed={candidate.seed!r})."
        )

    from dataclasses import replace

    finalized_rows = [
        replace(r, norm_verification_ok=True, scope_isolation_verification_ok=scope_isolation_precondition_ok, restoration_verification_ok=True)
        for r in pending_rows
    ]
    return finalized_rows

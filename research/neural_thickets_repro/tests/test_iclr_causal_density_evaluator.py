"""Tests for iclr_causal_density.evaluator -- candidate identity preserved across conditions
(item 7), relative-L2 norm matching (item 10), scope isolation precondition (item 11), exact
restoration (item 12), cache reset (item 13). CPU-only -- engine/run_benchmark/apply_perturbation
are all injected fakes, matching this project's established convention.
"""
from __future__ import annotations

import pytest

from neural_thickets_repro.benchmarks.base import Example, ExampleScore, ParsedPrediction
from neural_thickets_repro.benchmarks.runner import PerExampleResult, RunResult
from neural_thickets_repro.iclr_causal_density.candidates import PerturbationCandidate
from neural_thickets_repro.iclr_causal_density.evaluator import (
    CapabilityAuditData,
    NormVerificationFailedError,
    RestorationVerificationFailedError,
    ScopeIsolationPreconditionError,
    evaluate_one_candidate_all_capabilities,
    verify_norm,
)


def _examples(n=4, prefix="ex"):
    return [Example(example_id=f"{prefix}_{i}", image=f"img_{i}", image_ref=f"img_{i}", prompt_input={}, target="cat") for i in range(n)]


def _fake_run_result(examples, score=0.5):
    per_example = [
        PerExampleResult(example_id=e.example_id, image_ref=e.image_ref, raw_generation="cat", parsed=ParsedPrediction(parsed="cat", parse_ok=True), score=ExampleScore(score=score))
        for e in examples
    ]
    return RunResult(per_example=per_example, aggregate_metrics={"primary_metric": score, "parser_failure_rate": 0.0})


def _capability_data(capability="visual_grounding", n=4, text_only=True):
    correct = _examples(n)
    shuffled = _examples(n)  # same ids, distinct object identity is enough for these tests
    text = _examples(n) if text_only else None
    return CapabilityAuditData(capability=capability, benchmark=object(), dataset_source="fake", correct_examples=correct, shuffled_examples=shuffled, text_only_examples=text)


def _candidate(scope="full_lm", radius=0.02, seed=42, candidate_id="full_lm_r020000_seed42"):
    return PerturbationCandidate(candidate_id=candidate_id, scope=scope, radius=radius, seed=seed, seed_index=0)


def _make_fake_apply_perturbation(*, requested_r=0.02, base_l2_norm=10.0, param_count=100, total_element_count=100_000, correct_sigma=True):
    """param_count (number of selected TENSORS) is deliberately different from
    total_element_count (total scalar ELEMENTS across those tensors) by default -- on any real
    model these are never equal, and a fixture that coincidentally set them equal is exactly
    what let scoped_apply_perturbation.py/evaluator.py's field-name mismatch (real 7B live-run
    bug, fixed alongside this test) go undetected by the CPU suite.
    """
    calls = []

    def _apply(engine, seed, r, scope):
        calls.append((seed, r, scope))
        from neural_thickets_repro.scopes import compute_relative_l2_sigma

        # Real scoped_apply_perturbation always derives sigma from total_element_count (the
        # formula's dimensionality d_m) -- mirrored here, never from param_count (tensor count).
        sigma = compute_relative_l2_sigma(base_l2_norm, total_element_count, r) if correct_sigma else 999.0
        return {
            "scope": scope, "seed": seed, "requested_relative_l2": r, "derived_sigma": sigma,
            "actual_perturbation_l2": r * base_l2_norm, "scope_param_count": param_count, "scope_total_element_count": total_element_count,
            "scope_base_l2_norm": base_l2_norm, "noise_semantics": "upstream_per_tensor_reseed",
        }
    return _apply, calls


def _run_benchmark_recording(call_log):
    def _run(benchmark, examples, engine, tokenizer, sampling_params, allow_missing_image=False):
        call_log.append((len(examples), allow_missing_image))
        return _fake_run_result(examples)
    return _run


def test_candidate_identity_preserved_across_all_conditions_and_capabilities():
    """Item 7: candidate_id, scope, radius, seed identical on EVERY row this candidate
    produces, across every capability and every condition.
    """
    apply_fn, _ = _make_fake_apply_perturbation()
    call_log = []
    data = {"visual_grounding": _capability_data("visual_grounding"), "counting": _capability_data("counting")}
    candidate = _candidate()

    rows = evaluate_one_candidate_all_capabilities(
        engine=object(), candidate=candidate, capability_data=data, tokenizer=None, sampling_params=None,
        run_benchmark=_run_benchmark_recording(call_log), apply_perturbation=apply_fn,
        reset_to_base_weights=lambda e: None, scope_requires_encoder_cache_reset=lambda s: False,
        reset_vllm_encoder_cache_full=lambda e: None, verify_restoration=lambda e: True,
        scope_isolation_precondition_ok=True, decoding_config={"temperature": 0.0},
        source_commit="abc", run_id="r1", model_name="Qwen/Qwen2.5-VL-7B-Instruct", model_revision="a" * 40,
    )
    assert len(rows) == 2 * 3 * 4  # 2 capabilities x 3 conditions x 4 examples
    assert all(r.candidate_id == candidate.candidate_id for r in rows)
    assert all(r.scope == candidate.scope for r in rows)
    assert all(r.radius == candidate.radius for r in rows)
    assert all(r.seed == candidate.seed for r in rows)
    assert {r.condition for r in rows} == {"correct_image", "shuffled_image", "text_only"}
    assert {r.capability for r in rows} == {"visual_grounding", "counting"}


def test_norm_verification_ok_when_sigma_matches_formula():
    apply_fn, _ = _make_fake_apply_perturbation(correct_sigma=True)
    result = apply_fn(object(), 42, 0.02, "full_lm")
    assert verify_norm(result) is True


def test_norm_verification_fails_when_sigma_does_not_match_formula():
    """Item 10: a genuine perturbation-magnitude bug (sigma applied doesn't match the formula
    recomputed from the candidate's own returned scope stats) is detected, never silently
    accepted.
    """
    apply_fn, _ = _make_fake_apply_perturbation(correct_sigma=False)
    result = apply_fn(object(), 42, 0.02, "full_lm")
    assert verify_norm(result) is False


def test_norm_verification_uses_total_element_count_not_tensor_count(monkeypatch):
    """Live regression (first-candidate failure on the real 7B run): verify_norm must recompute
    expected_sigma from scope_total_element_count (the formula's true d_m -- total scalar
    elements across the scope's selected tensors), never from scope_param_count (merely the
    number of selected tensors). A correct, real derived_sigma (computed from
    total_element_count, exactly as scoped_apply_perturbation.py does) must verify OK even
    when param_count (tensor count) and total_element_count are wildly different -- as they
    always are on a real model.
    """
    apply_fn, _ = _make_fake_apply_perturbation(correct_sigma=True, param_count=37, total_element_count=48_233_216)
    result = apply_fn(object(), 42, 0.02, "vision_encoder")
    assert result["scope_param_count"] != result["scope_total_element_count"]
    assert verify_norm(result) is True


def test_evaluator_raises_norm_verification_failed_and_restores(monkeypatch):
    apply_fn, _ = _make_fake_apply_perturbation(correct_sigma=False)
    restore_calls = []
    data = {"visual_grounding": _capability_data("visual_grounding")}
    with pytest.raises(NormVerificationFailedError):
        evaluate_one_candidate_all_capabilities(
            engine=object(), candidate=_candidate(), capability_data=data, tokenizer=None, sampling_params=None,
            run_benchmark=lambda *a, **k: pytest.fail("must not evaluate any capability after a norm failure"),
            apply_perturbation=apply_fn, reset_to_base_weights=lambda e: restore_calls.append(e),
            scope_requires_encoder_cache_reset=lambda s: False, reset_vllm_encoder_cache_full=lambda e: None,
            verify_restoration=lambda e: True, scope_isolation_precondition_ok=True, decoding_config={},
            source_commit="abc", run_id="r1", model_name="m", model_revision="a" * 40,
        )
    assert len(restore_calls) == 1  # restore-on-failure still ran


def test_scope_isolation_precondition_blocks_evaluation_before_any_perturbation():
    """Item 11: an unconfirmed scope-isolation precondition must refuse the candidate
    entirely, BEFORE any perturbation is even applied.
    """
    apply_calls = []

    def _should_never_be_called(engine, seed, r, scope):
        apply_calls.append(1)
        raise AssertionError("apply_perturbation must never be called when the isolation precondition is not confirmed")

    with pytest.raises(ScopeIsolationPreconditionError):
        evaluate_one_candidate_all_capabilities(
            engine=object(), candidate=_candidate(), capability_data={"visual_grounding": _capability_data()}, tokenizer=None, sampling_params=None,
            run_benchmark=lambda *a, **k: None, apply_perturbation=_should_never_be_called,
            reset_to_base_weights=lambda e: None, scope_requires_encoder_cache_reset=lambda s: False,
            reset_vllm_encoder_cache_full=lambda e: None, verify_restoration=lambda e: True,
            scope_isolation_precondition_ok=False, decoding_config={},
            source_commit="abc", run_id="r1", model_name="m", model_revision="a" * 40,
        )
    assert apply_calls == []


def test_scope_isolation_ok_propagates_to_every_row():
    apply_fn, _ = _make_fake_apply_perturbation()
    call_log = []
    rows = evaluate_one_candidate_all_capabilities(
        engine=object(), candidate=_candidate(), capability_data={"visual_grounding": _capability_data()}, tokenizer=None, sampling_params=None,
        run_benchmark=_run_benchmark_recording(call_log), apply_perturbation=apply_fn,
        reset_to_base_weights=lambda e: None, scope_requires_encoder_cache_reset=lambda s: False,
        reset_vllm_encoder_cache_full=lambda e: None, verify_restoration=lambda e: True,
        scope_isolation_precondition_ok=True, decoding_config={},
        source_commit="abc", run_id="r1", model_name="m", model_revision="a" * 40,
    )
    assert all(r.scope_isolation_verification_ok is True for r in rows)


def test_exact_restoration_failure_raises_and_no_rows_returned():
    """Item 12: restoration must be verified after every candidate; a failure hard-stops and
    never returns any row for that candidate (transactional -- nothing is appended on failure).
    """
    apply_fn, _ = _make_fake_apply_perturbation()
    with pytest.raises(RestorationVerificationFailedError):
        evaluate_one_candidate_all_capabilities(
            engine=object(), candidate=_candidate(), capability_data={"visual_grounding": _capability_data()}, tokenizer=None, sampling_params=None,
            run_benchmark=_run_benchmark_recording([]), apply_perturbation=apply_fn,
            reset_to_base_weights=lambda e: None, scope_requires_encoder_cache_reset=lambda s: False,
            reset_vllm_encoder_cache_full=lambda e: None, verify_restoration=lambda e: False,
            scope_isolation_precondition_ok=True, decoding_config={},
            source_commit="abc", run_id="r1", model_name="m", model_revision="a" * 40,
        )


def test_restoration_verification_ok_recorded_true_on_every_row_when_it_passes():
    apply_fn, _ = _make_fake_apply_perturbation()
    rows = evaluate_one_candidate_all_capabilities(
        engine=object(), candidate=_candidate(), capability_data={"visual_grounding": _capability_data()}, tokenizer=None, sampling_params=None,
        run_benchmark=_run_benchmark_recording([]), apply_perturbation=apply_fn,
        reset_to_base_weights=lambda e: None, scope_requires_encoder_cache_reset=lambda s: False,
        reset_vllm_encoder_cache_full=lambda e: None, verify_restoration=lambda e: True,
        scope_isolation_precondition_ok=True, decoding_config={},
        source_commit="abc", run_id="r1", model_name="m", model_revision="a" * 40,
    )
    assert all(r.restoration_verification_ok is True for r in rows)


def test_cache_reset_invoked_when_scope_requires_it():
    """Item 13: the full multimodal-encoder-cache reset must be called exactly once per
    candidate (before the correct/shuffled/text-only generation loop starts) whenever the
    perturbed scope can change vision-encoder output.
    """
    apply_fn, _ = _make_fake_apply_perturbation()
    cache_reset_calls = []
    evaluate_one_candidate_all_capabilities(
        engine=object(), candidate=_candidate(scope="vision_encoder"), capability_data={"visual_grounding": _capability_data()}, tokenizer=None, sampling_params=None,
        run_benchmark=_run_benchmark_recording([]), apply_perturbation=apply_fn,
        reset_to_base_weights=lambda e: None, scope_requires_encoder_cache_reset=lambda s: s == "vision_encoder",
        reset_vllm_encoder_cache_full=lambda e: cache_reset_calls.append(1), verify_restoration=lambda e: True,
        scope_isolation_precondition_ok=True, decoding_config={},
        source_commit="abc", run_id="r1", model_name="m", model_revision="a" * 40,
    )
    assert cache_reset_calls == [1]


def test_cache_reset_not_invoked_when_scope_does_not_require_it():
    apply_fn, _ = _make_fake_apply_perturbation()
    cache_reset_calls = []
    evaluate_one_candidate_all_capabilities(
        engine=object(), candidate=_candidate(scope="full_lm"), capability_data={"visual_grounding": _capability_data()}, tokenizer=None, sampling_params=None,
        run_benchmark=_run_benchmark_recording([]), apply_perturbation=apply_fn,
        reset_to_base_weights=lambda e: None, scope_requires_encoder_cache_reset=lambda s: s == "vision_encoder",
        reset_vllm_encoder_cache_full=lambda e: cache_reset_calls.append(1), verify_restoration=lambda e: True,
        scope_isolation_precondition_ok=True, decoding_config={},
        source_commit="abc", run_id="r1", model_name="m", model_revision="a" * 40,
    )
    assert cache_reset_calls == []


def test_text_only_condition_calls_run_benchmark_with_allow_missing_image_true():
    apply_fn, _ = _make_fake_apply_perturbation()
    call_log = []
    evaluate_one_candidate_all_capabilities(
        engine=object(), candidate=_candidate(), capability_data={"visual_grounding": _capability_data()}, tokenizer=None, sampling_params=None,
        run_benchmark=_run_benchmark_recording(call_log), apply_perturbation=apply_fn,
        reset_to_base_weights=lambda e: None, scope_requires_encoder_cache_reset=lambda s: False,
        reset_vllm_encoder_cache_full=lambda e: None, verify_restoration=lambda e: True,
        scope_isolation_precondition_ok=True, decoding_config={},
        source_commit="abc", run_id="r1", model_name="m", model_revision="a" * 40,
    )
    # 3 conditions were run: correct_image (False), shuffled_image (False), text_only (True)
    allow_missing_flags = [flag for _, flag in call_log]
    assert allow_missing_flags.count(True) == 1
    assert allow_missing_flags.count(False) == 2


def test_capability_without_text_only_support_skips_that_condition():
    apply_fn, _ = _make_fake_apply_perturbation()
    call_log = []
    data = {"visual_grounding": _capability_data("visual_grounding", text_only=False)}
    rows = evaluate_one_candidate_all_capabilities(
        engine=object(), candidate=_candidate(), capability_data=data, tokenizer=None, sampling_params=None,
        run_benchmark=_run_benchmark_recording(call_log), apply_perturbation=apply_fn,
        reset_to_base_weights=lambda e: None, scope_requires_encoder_cache_reset=lambda s: False,
        reset_vllm_encoder_cache_full=lambda e: None, verify_restoration=lambda e: True,
        scope_isolation_precondition_ok=True, decoding_config={},
        source_commit="abc", run_id="r1", model_name="m", model_revision="a" * 40,
    )
    assert {r.condition for r in rows} == {"correct_image", "shuffled_image"}
    assert len(call_log) == 2

"""Tests for run_iclr_causal_density_live.py -- item 23 continued: protection ensuring no 32B
(or 72B) command is ever dispatched by the LIVE execution script either. Also covers the pure-
logic base-control gate evaluator (evaluate_base_control_gate), which needs no GPU/ray/vllm.
"""
from __future__ import annotations

import ast
import inspect

import pytest

import neural_thickets_repro.run_iclr_causal_density_live as live_module


def test_module_does_not_import_any_32b_module_at_import_time():
    source = inspect.getsource(live_module)
    tree = ast.parse(source)
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_names.add(node.module)
    for name in imported_names:
        assert "32b" not in name.lower(), f"Unexpected 32B-related import: {name!r}"


def test_main_function_body_never_references_32b_or_72b():
    source = inspect.getsource(live_module.main)
    forbidden = (
        "run_stage11_coarse_anatomical_atlas_32b", "stage11_32b_s2_live_evidence",
        "stage11_32b_s2_live_v3_solver_probe", "stage11_32b_live_evidence", "stage11_32b_readiness",
        "run_stage11_whole_model_scaling", "run_stage11_visual_thicket_scaling",
        "--scale 32B", "--scale 72B", "--track anatomy",
    )
    for token in forbidden:
        assert token not in source, f"run_iclr_causal_density_live.py's main() must never reference {token!r}"


def test_runtime_guard_refuses_argv_containing_32b_or_72b_markers():
    with pytest.raises(ValueError, match="strictly 7B-only"):
        live_module.main(["--phase", "base_control", "--scale", "32B"])
    with pytest.raises(ValueError, match="strictly 7B-only"):
        live_module.main(["--phase", "decisive_pilot", "--scale", "72B"])


def test_model_name_bound_in_main_is_the_frozen_7b_design():
    from neural_thickets_repro.iclr_causal_density.design import FROZEN_DESIGN

    assert FROZEN_DESIGN.model_scale == "7B"
    assert "32B" not in FROZEN_DESIGN.model_name and "72B" not in FROZEN_DESIGN.model_name


def test_main_passes_max_pixels_via_mm_processor_kwargs_at_engine_launch():
    """Live regression (caught mid-run on the pod): a high-resolution TextVQA audit image
    produced a 16,215-token prompt against the frozen max_model_len=4096 budget. main() must cap
    Qwen2.5-VL's own vision-token count via mm_processor_kwargs={"max_pixels": ...} at the
    launch_stage6_engine call site, without changing max_model_len or any other frozen constant.
    """
    source = inspect.getsource(live_module.main)
    assert "mm_processor_kwargs" in source
    assert "QWEN2_5_VL_MAX_PIXELS" in source
    assert live_module.QWEN2_5_VL_MAX_PIXELS == 1024 * 28 * 28


def test_main_exposes_full_encoder_cache_reset_before_launching_the_engine():
    """Live regression (decisive-pilot candidate #1, vision_encoder scope): the Ray-wrapped
    RandOptNcclLLM actor only exposes 'reset_encoder_cache_full' if
    vlm_adapter.ensure_full_encoder_cache_reset_exposed() ran BEFORE launch_stage6_engine --
    otherwise every vision_encoder/full_vlm candidate hard-fails at
    reset_vllm_encoder_cache_full. main() must call it in that order.
    """
    source = inspect.getsource(live_module.main)
    assert "ensure_full_encoder_cache_reset_exposed" in source
    expose_pos = source.index("ensure_full_encoder_cache_reset_exposed(EXTERNAL_ROOT)")
    launch_pos = source.index("launch_stage6_engine(")
    assert expose_pos < launch_pos, "ensure_full_encoder_cache_reset_exposed must run BEFORE launch_stage6_engine"


def test_main_verifies_encoder_cache_reset_works_end_to_end_after_engine_launch():
    """Mirrors run_global_visual_thicket_pilot.py's own Stage 6 main() discipline: a hard,
    one-time precondition check that the cache-reset mechanism actually works against the live
    engine (not merely that it was exposed pre-launch), before any candidate evaluation starts.
    """
    source = inspect.getsource(live_module.main)
    assert "_real_reset_vllm_encoder_cache_full(engine)" in source
    assert "Refusing to start candidate evaluation without a proven-working cache" in source


# =================================================================================================
# evaluate_base_control_gate -- pure logic, no GPU
# =================================================================================================


def _cond(score, n=5):
    return {"aggregate_score": score, "parser_failure_rate": 0.0, "per_example_scores": {f"ex_{i}": score for i in range(n)}, "generation_hash": "h"}


def test_base_control_gate_passes_with_a_real_visual_advantage():
    report = {
        "visual_grounding": {
            "selection:correct_image": _cond(0.6), "selection:shuffled_image": _cond(0.2), "selection:text_only": _cond(0.1),
            "audit:correct_image": _cond(0.6), "audit:shuffled_image": _cond(0.2), "audit:text_only": _cond(0.1),
        },
    }
    gate = live_module.evaluate_base_control_gate(report)
    assert gate["pass"] is True
    assert gate["failures"] == []


def test_base_control_gate_fails_when_no_advantage_over_shuffled():
    report = {
        "counting": {
            "selection:correct_image": _cond(0.3), "selection:shuffled_image": _cond(0.3), "selection:text_only": _cond(0.1),
            "audit:correct_image": _cond(0.3), "audit:shuffled_image": _cond(0.3), "audit:text_only": _cond(0.1),
        },
    }
    gate = live_module.evaluate_base_control_gate(report)
    assert gate["pass"] is False
    assert any("shuffled_image" in f for f in gate["failures"])


def test_base_control_gate_fails_when_no_advantage_over_text_only():
    report = {
        "ocr_text_recognition": {
            "selection:correct_image": _cond(0.5), "selection:shuffled_image": _cond(0.1), "selection:text_only": _cond(0.5),
            "audit:correct_image": _cond(0.5), "audit:shuffled_image": _cond(0.1), "audit:text_only": _cond(0.5),
        },
    }
    gate = live_module.evaluate_base_control_gate(report)
    assert gate["pass"] is False
    assert any("text_only" in f for f in gate["failures"])


def test_base_control_gate_handles_capability_without_text_only_support():
    report = {
        "visual_grounding": {
            "selection:correct_image": _cond(0.6), "selection:shuffled_image": _cond(0.2), "selection:text_only": None,
            "audit:correct_image": _cond(0.6), "audit:shuffled_image": _cond(0.2), "audit:text_only": None,
        },
    }
    gate = live_module.evaluate_base_control_gate(report)
    assert gate["pass"] is True


def test_score_condition_passes_allow_missing_image_only_for_text_only():
    """Live regression (caught mid-run on the pod): _score_condition must pass
    allow_missing_image=True to run_benchmark ONLY for the text_only condition -- a text-only
    Example legitimately has image=None (benchmarks.image_sanity.make_text_only_variant), and
    run_benchmark hard-raises MissingImageError unless explicitly told to allow it.
    """
    calls = []

    def _fake_run_benchmark(benchmark, examples, llm, tokenizer, sampling_params, allow_missing_image=False, max_requests_per_generate=None):
        calls.append(allow_missing_image)

        class _FakeResult:
            aggregate_metrics = {"primary_metric": 0.5, "parser_failure_rate": 0.0}
            per_example = []

            def generation_hash(self):
                return "h"

        return _FakeResult()

    live_module._score_condition(_fake_run_benchmark, object(), [], object(), object(), object(), 10, allow_missing_image=False)
    live_module._score_condition(_fake_run_benchmark, object(), [], object(), object(), object(), 10, allow_missing_image=True)
    assert calls == [False, True]


def test_run_base_control_gate_calls_text_only_condition_with_allow_missing_image_true(monkeypatch):
    calls = []

    def _fake_run_benchmark(benchmark, examples, llm, tokenizer, sampling_params, allow_missing_image=False, max_requests_per_generate=None):
        calls.append((examples, allow_missing_image))

        class _FakeResult:
            aggregate_metrics = {"primary_metric": 0.5, "parser_failure_rate": 0.0}
            per_example = []

            def generation_hash(self):
                return "h"

        return _FakeResult()

    class _FakeAdapter:
        pass

    class _FakeRayEngineLLMAdapter:
        def __init__(self, engine):
            pass

    monkeypatch.setattr("neural_thickets_repro.run_global_visual_thicket_pilot.RayEngineLLMAdapter", _FakeRayEngineLLMAdapter)

    data = {
        "cap1": {
            "selection_correct": ["a"], "selection_shuffled": ["b"], "selection_text_only": ["c"],
            "audit_correct": ["d"], "audit_shuffled": ["e"], "audit_text_only": ["f"],
        }
    }
    live_module.run_base_control_gate(object(), data, {"cap1": _FakeAdapter()}, object(), object(), generation_batch_size=10, run_benchmark=_fake_run_benchmark)
    text_only_calls = [c for c in calls if c[0] in (["c"], ["f"])]
    assert len(text_only_calls) == 2
    assert all(allow_missing for _, allow_missing in text_only_calls)
    non_text_only_calls = [c for c in calls if c[0] not in (["c"], ["f"])]
    assert all(not allow_missing for _, allow_missing in non_text_only_calls)


def test_base_control_gate_fails_on_missing_result():
    report = {"counting": {"selection:correct_image": None, "selection:shuffled_image": _cond(0.2), "selection:text_only": None, "audit:correct_image": _cond(0.5), "audit:shuffled_image": _cond(0.1), "audit:text_only": None}}
    gate = live_module.evaluate_base_control_gate(report)
    assert gate["pass"] is False
    assert any("missing" in f for f in gate["failures"])

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


def test_base_control_gate_fails_on_missing_result():
    report = {"counting": {"selection:correct_image": None, "selection:shuffled_image": _cond(0.2), "selection:text_only": None, "audit:correct_image": _cond(0.5), "audit:shuffled_image": _cond(0.1), "audit:text_only": None}}
    gate = live_module.evaluate_base_control_gate(report)
    assert gate["pass"] is False
    assert any("missing" in f for f in gate["failures"])

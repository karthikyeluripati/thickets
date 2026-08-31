"""Tests for run_iclr_causal_density_pilot.py -- item 23: protection ensuring no 32B (or 72B)
command is ever dispatched by this runner. This pilot is strictly 7B-only; these tests prove it
structurally (source inspection + no accidental import) rather than merely by convention.
"""
from __future__ import annotations

import inspect

import pytest

import neural_thickets_repro.run_iclr_causal_density_pilot as pilot_module


def test_dry_run_never_imports_vllm_ray_torch(monkeypatch):
    """--dry-run must never import vllm/ray/torch -- structural proof it cannot possibly reach
    (and therefore cannot possibly dispatch to) any GPU-touching code path, 32B included.
    """
    import sys

    blocked = {"vllm", "ray", "torch"}
    original_import = __import__

    def _guarded_import(name, *args, **kwargs):
        top = name.split(".")[0]
        if top in blocked:
            raise AssertionError(f"--dry-run must never import {top!r}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _guarded_import)
    rc = pilot_module.main(["--dry-run"])
    assert rc == 0


def test_main_function_body_never_references_32b_or_72b():
    """Structural proof: main()'s own EXECUTABLE body (not the module's prose docstring, which
    legitimately explains what is deliberately NOT imported) never mentions a 32B module name,
    the S2 live-evidence module, the S2 solver probe, or a --scale 32B/72B flag construction --
    catches a future accidental import/dispatch immediately, at review time, not only at runtime.
    """
    source = inspect.getsource(pilot_module.main)
    forbidden = (
        "run_stage11_coarse_anatomical_atlas_32b", "stage11_32b_s2_live_evidence",
        "stage11_32b_s2_live_v3_solver_probe", "stage11_32b_live_evidence", "stage11_32b_readiness",
        "run_stage11_whole_model_scaling", "run_stage11_visual_thicket_scaling",
        "--scale 32B", "--scale 72B", "--track anatomy",
    )
    for token in forbidden:
        assert token not in source, f"run_iclr_causal_density_pilot.py's main() must never reference {token!r}"


def test_module_does_not_import_any_32b_module_at_import_time():
    """The module's own top-level import list must not name any 32B module -- checked against
    sys.modules-independent static source (never merely "it happened not to be imported this
    run").
    """
    import ast

    source = inspect.getsource(pilot_module)
    tree = ast.parse(source)
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_names.add(node.module)
    for name in imported_names:
        assert "32b" not in name.lower(), f"Unexpected 32B-related import: {name!r}"


def test_model_scale_in_frozen_design_is_7b_only():
    from neural_thickets_repro.iclr_causal_density.design import FROZEN_DESIGN

    assert FROZEN_DESIGN.model_scale == "7B"
    assert "32B" not in FROZEN_DESIGN.model_name
    assert "72B" not in FROZEN_DESIGN.model_name


def test_runtime_guard_refuses_argv_containing_32b_or_72b_markers():
    with pytest.raises(ValueError, match="strictly 7B-only"):
        pilot_module.main(["--scale", "32B"])
    with pytest.raises(ValueError, match="strictly 7B-only"):
        pilot_module.main(["--track", "anatomy", "--scale", "72B"])


def test_non_dry_run_path_also_never_touches_32b(monkeypatch, capsys):
    """The non-dry-run path (GPU execution blocked in this environment) must print the
    blocked-execution message and return cleanly, never attempt to import or dispatch to any
    32B module.
    """
    import sys

    def _guard(name, *args, **kwargs):
        if "32b" in name.lower():
            raise AssertionError(f"must never import {name!r}")
        return _real_import(name, *args, **kwargs)

    _real_import = __import__
    monkeypatch.setattr("builtins.__import__", _guard)
    rc = pilot_module.main([])
    assert rc == 0
    captured = capsys.readouterr()
    assert "not performed by this script" in captured.err

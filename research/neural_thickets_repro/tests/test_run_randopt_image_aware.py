"""Tests for the pure-logic pieces of run_randopt_image_aware.py that don't need
GPU/vllm/ray/the external clone: sample_candidates() and the spawn-fix regression check
(same pattern as test_eval_base_image_aware_spawn_fix.py).
"""
import importlib
import os

import pytest

import neural_thickets_repro.run_randopt_image_aware as m


def test_sample_candidates_deterministic_given_same_seed():
    a = m.sample_candidates(20, [0.001, 0.002], seed=42)
    b = m.sample_candidates(20, [0.001, 0.002], seed=42)
    assert a == b


def test_sample_candidates_different_seeds_differ():
    a = m.sample_candidates(20, [0.001, 0.002], seed=1)
    b = m.sample_candidates(20, [0.001, 0.002], seed=2)
    assert a != b


def test_sample_candidates_returns_n_candidates():
    candidates = m.sample_candidates(20, [0.001], seed=42)
    assert len(candidates) == 20


def test_sample_candidates_seeds_are_unique():
    candidates = m.sample_candidates(50, [0.001, 0.002, 0.005], seed=42)
    seeds = [c[0] for c in candidates]
    assert len(seeds) == len(set(seeds))


def test_sample_candidates_sigmas_drawn_only_from_given_values():
    sigma_values = [0.001, 0.002, 0.005]
    candidates = m.sample_candidates(50, sigma_values, seed=42)
    sigmas = {c[1] for c in candidates}
    assert sigmas <= set(sigma_values)


def test_sample_candidates_types():
    candidates = m.sample_candidates(5, [0.001], seed=42)
    for seed, sigma in candidates:
        assert isinstance(seed, int)
        assert isinstance(sigma, float)


# --- spawn-fix regression checks (same pattern as eval_base_image_aware.py's) ---


def test_importing_module_forces_spawn_when_unset(monkeypatch):
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    importlib.reload(m)
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"


def test_importing_module_forces_spawn_even_if_already_set_to_fork(monkeypatch):
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "fork")
    importlib.reload(m)
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"


def test_assert_spawn_configured_passes_after_module_import():
    importlib.reload(m)
    m._assert_spawn_configured()  # should not raise


def test_assert_spawn_configured_raises_if_tampered_with_after_import(monkeypatch):
    importlib.reload(m)
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "fork")
    with pytest.raises(RuntimeError, match="spawn"):
        m._assert_spawn_configured()

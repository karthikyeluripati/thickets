"""Regression tests for the VLLM_WORKER_MULTIPROC_METHOD=spawn runtime compatibility fix
(RuntimeError: Cannot re-initialize CUDA in forked subprocess). Pure environment/import
checks -- no GPU/vllm/torch import needed, so these run anywhere.
"""
import importlib
import os

import pytest

import neural_thickets_repro.eval_base_image_aware as m


def test_importing_module_forces_spawn_when_unset(monkeypatch):
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    importlib.reload(m)
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"


def test_importing_module_forces_spawn_even_if_already_set_to_fork(monkeypatch):
    """Forced, not setdefault: a stale "fork" (or anything else) already present in the
    environment must not silently defeat the fix.
    """
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "fork")
    importlib.reload(m)
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"


def test_assert_spawn_configured_passes_after_module_import():
    importlib.reload(m)
    m._assert_spawn_configured()  # should not raise -- import already forced it


def test_assert_spawn_configured_raises_if_tampered_with_after_import(monkeypatch):
    importlib.reload(m)
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "fork")  # simulate later tampering
    with pytest.raises(RuntimeError, match="spawn"):
        m._assert_spawn_configured()


def test_assert_spawn_configured_raises_when_unset(monkeypatch):
    importlib.reload(m)
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    with pytest.raises(RuntimeError, match="spawn"):
        m._assert_spawn_configured()

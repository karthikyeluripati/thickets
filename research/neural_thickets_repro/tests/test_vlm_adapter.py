"""Tests for vlm_adapter.py's Ray bootstrap logic. Needs the real `ray` package (not in
requirements-cpu.txt -- it's a GPU-stack dependency), so skipped locally; runs for real on
the pod, which has it installed.
"""
import pytest

ray = pytest.importorskip("ray", reason="ray is a GPU-stack dependency, not available locally")

from neural_thickets_repro.vlm_adapter import bootstrap_ray  # noqa: E402


@pytest.fixture(autouse=True)
def _ensure_ray_shutdown():
    """Every test starts and ends with no active Ray session, regardless of outcome."""
    if ray.is_initialized():
        ray.shutdown()
    yield
    if ray.is_initialized():
        ray.shutdown()


def test_bootstrap_ray_starts_a_session_when_none_running():
    assert ray.is_initialized() is False
    owned = bootstrap_ray()
    assert owned is True
    assert ray.is_initialized() is True


def test_bootstrap_ray_does_not_reinitialize_when_already_running():
    ray.init(address="local", ignore_reinit_error=True)
    assert ray.is_initialized() is True

    owned = bootstrap_ray()

    assert owned is False, "must report NOT owning a session that was already running"
    assert ray.is_initialized() is True


def test_bootstrap_ray_honors_ray_address_env_var(monkeypatch):
    """Mirrors upstream: RAY_ADDRESS set -> address='auto'; unset -> address='local'.
    Both branches must actually succeed in starting a local single-node session for this
    test to be meaningful (no external Ray cluster present here), so we only assert on the
    unset-env-var path, which is guaranteed to work in a bare test environment; the set
    path is exercised for coverage of the branch without asserting on connectivity to a
    cluster that doesn't exist in this environment.
    """
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    owned = bootstrap_ray()
    assert owned is True
    assert ray.is_initialized() is True

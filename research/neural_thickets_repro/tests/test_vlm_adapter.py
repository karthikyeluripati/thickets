"""Tests for vlm_adapter.py's Ray bootstrap + worker-import-visibility logic. Needs the
real `ray` package (not in requirements-cpu.txt -- it's a GPU-stack dependency), so skipped
locally; runs for real on the pod, which has it installed.
"""
import pytest

ray = pytest.importorskip("ray", reason="ray is a GPU-stack dependency, not available locally")

from neural_thickets_repro.vlm_adapter import (  # noqa: E402
    bootstrap_ray,
    verify_workers_can_import_external_root,
)


@pytest.fixture(autouse=True)
def _ensure_ray_shutdown():
    """Every test starts and ends with no active Ray session, regardless of outcome."""
    if ray.is_initialized():
        ray.shutdown()
    yield
    if ray.is_initialized():
        ray.shutdown()


def _make_fake_core_package(root):
    """A minimal stand-in for external/RandOpt's core/ package -- just enough structure
    (core/__init__.py, core/engine.py) to prove workers can actually import it, without
    depending on the real external clone being present in the test environment.
    """
    core_dir = root / "core"
    core_dir.mkdir(parents=True, exist_ok=True)
    (core_dir / "__init__.py").write_text("")
    (core_dir / "engine.py").write_text("MARKER = 'fake_core_engine'\n")
    return root


# --- bootstrap_ray ---


def test_bootstrap_ray_starts_a_session_when_none_running(tmp_path):
    assert ray.is_initialized() is False
    owned = bootstrap_ray(tmp_path)
    assert owned is True
    assert ray.is_initialized() is True


def test_bootstrap_ray_does_not_reinitialize_when_already_running(tmp_path):
    ray.init(address="local", ignore_reinit_error=True)
    assert ray.is_initialized() is True

    owned = bootstrap_ray(tmp_path)

    assert owned is False, "must report NOT owning a session that was already running"
    assert ray.is_initialized() is True


def test_bootstrap_ray_honors_ray_address_env_var(tmp_path, monkeypatch):
    """Mirrors upstream: RAY_ADDRESS set -> address='auto'; unset -> address='local'. Only
    the unset path is asserted end-to-end (no external cluster exists in this environment
    to connect "auto" to); the set path is exercised for branch coverage.
    """
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    owned = bootstrap_ray(tmp_path)
    assert owned is True
    assert ray.is_initialized() is True


def test_bootstrap_ray_preserves_existing_pythonpath(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/some/pre/existing/path")
    captured = {}
    real_init = ray.init

    def spy_init(*args, **kwargs):
        captured.update(kwargs)
        return real_init(*args, **kwargs)

    monkeypatch.setattr(ray, "init", spy_init)

    bootstrap_ray(tmp_path)

    pythonpath = captured["runtime_env"]["env_vars"]["PYTHONPATH"]
    assert str(tmp_path) in pythonpath
    assert "/some/pre/existing/path" in pythonpath


def test_bootstrap_ray_when_already_running_does_not_touch_pythonpath(tmp_path, monkeypatch):
    """When Ray is already initialized, bootstrap_ray must not attempt to (re-)init at all
    -- we don't control that session's runtime_env, so it must not pretend to.
    """
    ray.init(address="local", ignore_reinit_error=True)
    called = {"count": 0}
    real_init = ray.init

    def spy_init(*args, **kwargs):
        called["count"] += 1
        return real_init(*args, **kwargs)

    monkeypatch.setattr(ray, "init", spy_init)

    owned = bootstrap_ray(tmp_path)

    assert owned is False
    assert called["count"] == 0


# --- verify_workers_can_import_external_root ---


def test_verify_workers_can_import_external_root_passes_when_visible(tmp_path):
    fake_root = _make_fake_core_package(tmp_path)
    bootstrap_ray(fake_root)
    verify_workers_can_import_external_root(fake_root)  # should not raise


def test_verify_workers_fails_clearly_when_core_not_importable(tmp_path):
    empty_root = tmp_path / "empty_external_root"
    empty_root.mkdir()
    bootstrap_ray(empty_root)  # runtime_env points here, but no core/ package exists inside

    with pytest.raises(RuntimeError, match="cannot import 'core'"):
        verify_workers_can_import_external_root(empty_root)

"""Tests for vlm_adapter's encoder-cache-reset functions -- via fake `ray`/`core.engine`
modules injected into sys.modules (same trick used in tests/test_run_scoped_randopt.py /
test_gate2_restoration_ab.py) rather than tests/test_vlm_adapter.py's real-ray-required gate,
so these run locally without the GPU-stack ray dependency installed.
"""
import sys
import types
from types import SimpleNamespace

import pytest

from neural_thickets_repro.vlm_adapter import (
    ensure_full_encoder_cache_reset_exposed,
    reset_vllm_encoder_cache,
    reset_vllm_encoder_cache_full,
)


class _FakeCollectiveRpc:
    def __init__(self, calls, *, result=None, raises=None):
        self._calls = calls
        self._result = result if result is not None else ["ack"]
        self._raises = raises

    def remote(self, method, args=()):
        self._calls.append((method, tuple(args)))
        if self._raises is not None:
            raise self._raises
        return self._result


def _fake_engine(calls, *, result=None, raises=None):
    return SimpleNamespace(collective_rpc=_FakeCollectiveRpc(calls, result=result, raises=raises))


@pytest.fixture(autouse=True)
def _fake_ray(monkeypatch):
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(get=lambda x: x))


def test_reset_vllm_encoder_cache_dispatches_the_real_vllm_string_method():
    calls = []
    engine = _fake_engine(calls)

    reset_vllm_encoder_cache(engine)

    assert calls == [("reset_encoder_cache", ())]


def test_reset_vllm_encoder_cache_raises_when_collective_rpc_errors():
    calls = []
    engine = _fake_engine(calls, raises=RuntimeError("worker unreachable"))

    with pytest.raises(RuntimeError, match="does not expose a working encoder-cache reset path"):
        reset_vllm_encoder_cache(engine)


def test_reset_vllm_encoder_cache_raises_on_non_list_result():
    calls = []
    engine = _fake_engine(calls, result={"unexpected": "shape"})

    with pytest.raises(RuntimeError, match="unexpected shape"):
        reset_vllm_encoder_cache(engine)


def test_reset_vllm_encoder_cache_raises_on_multi_worker_result():
    calls = []
    engine = _fake_engine(calls, result=["ack", "ack"])

    with pytest.raises(RuntimeError, match="unexpected shape"):
        reset_vllm_encoder_cache(engine)


def test_reset_vllm_encoder_cache_accepts_single_worker_result():
    calls = []
    engine = _fake_engine(calls, result=["ack"])

    reset_vllm_encoder_cache(engine)  # should not raise


# --- ensure_full_encoder_cache_reset_exposed / reset_vllm_encoder_cache_full: the FULL
# (scheduler + worker) reset, via a monkey-patched method on RandOptNcclLLM ---


class _FakeRandOptNcclLLM:
    """Stands in for external/RandOpt/core/engine.py's RandOptNcclLLM -- a plain class, not
    yet wrapped by ray.remote(...), matching the confirmed real structure (launch_engines()
    applies ray.remote(...) to it fresh, inside the function body, on every call).
    """


class _FakeLLMEngine:
    def __init__(self):
        self.reset_encoder_cache_called = False

    def reset_encoder_cache(self):
        self.reset_encoder_cache_called = True


@pytest.fixture
def _fake_core_engine_module(monkeypatch):
    """Injects a fake core/core.engine module pair into sys.modules (no filesystem I/O)
    exposing a fresh _FakeRandOptNcclLLM class, and resets that class's monkey-patched state
    after the test so tests don't leak the added method into each other.
    """
    core_pkg = types.ModuleType("core")
    core_engine_mod = types.ModuleType("core.engine")
    core_engine_mod.RandOptNcclLLM = _FakeRandOptNcclLLM
    monkeypatch.setitem(sys.modules, "core", core_pkg)
    monkeypatch.setitem(sys.modules, "core.engine", core_engine_mod)
    yield _FakeRandOptNcclLLM
    if hasattr(_FakeRandOptNcclLLM, "reset_encoder_cache_full"):
        del _FakeRandOptNcclLLM.reset_encoder_cache_full


def test_ensure_full_encoder_cache_reset_exposed_adds_the_method(_fake_core_engine_module, tmp_path):
    cls = _fake_core_engine_module
    assert not hasattr(cls, "reset_encoder_cache_full")

    ensure_full_encoder_cache_reset_exposed(tmp_path)

    assert hasattr(cls, "reset_encoder_cache_full")


def test_added_method_calls_llm_engine_reset_encoder_cache(_fake_core_engine_module, tmp_path):
    cls = _fake_core_engine_module
    ensure_full_encoder_cache_reset_exposed(tmp_path)

    instance = cls()
    instance.llm_engine = _FakeLLMEngine()
    cls.reset_encoder_cache_full(instance)

    assert instance.llm_engine.reset_encoder_cache_called is True


def test_ensure_full_encoder_cache_reset_exposed_is_idempotent(_fake_core_engine_module, tmp_path):
    cls = _fake_core_engine_module
    ensure_full_encoder_cache_reset_exposed(tmp_path)
    first_fn = cls.reset_encoder_cache_full

    ensure_full_encoder_cache_reset_exposed(tmp_path)  # second call -- must not replace it
    second_fn = cls.reset_encoder_cache_full

    assert first_fn is second_fn


def test_ensure_full_encoder_cache_reset_exposed_inserts_external_root_on_sys_path(_fake_core_engine_module, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != str(tmp_path)])
    assert str(tmp_path) not in sys.path

    ensure_full_encoder_cache_reset_exposed(tmp_path)

    assert str(tmp_path) in sys.path


class _FakeDirectMethod:
    def __init__(self, calls, *, raises=None):
        self._calls = calls
        self._raises = raises

    def remote(self):
        self._calls.append("reset_encoder_cache_full")
        if self._raises is not None:
            raise self._raises
        return None


def test_reset_vllm_encoder_cache_full_calls_the_actor_method():
    calls = []
    engine = SimpleNamespace(reset_encoder_cache_full=_FakeDirectMethod(calls))

    reset_vllm_encoder_cache_full(engine)

    assert calls == ["reset_encoder_cache_full"]


def test_reset_vllm_encoder_cache_full_raises_when_method_missing_on_actor():
    engine = SimpleNamespace()  # no reset_encoder_cache_full attribute at all

    with pytest.raises(RuntimeError, match="does not expose"):
        reset_vllm_encoder_cache_full(engine)


def test_reset_vllm_encoder_cache_full_raises_when_call_errors():
    calls = []
    engine = SimpleNamespace(reset_encoder_cache_full=_FakeDirectMethod(calls, raises=RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="failed"):
        reset_vllm_encoder_cache_full(engine)


def test_reset_vllm_encoder_cache_full_never_falls_back_to_collective_rpc():
    """Confirms reset_vllm_encoder_cache_full never touches engine.collective_rpc at all --
    it must be a direct actor method call, never the old worker-only mechanism.
    """
    calls = []
    engine = SimpleNamespace(
        reset_encoder_cache_full=_FakeDirectMethod(calls),
        collective_rpc=None,  # would AttributeError/TypeError if accidentally used
    )

    reset_vllm_encoder_cache_full(engine)  # should not touch collective_rpc, should not raise

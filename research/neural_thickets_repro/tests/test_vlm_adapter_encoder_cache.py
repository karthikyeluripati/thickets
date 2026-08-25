"""Tests for vlm_adapter's encoder-cache-reset functions -- via fake `ray`/`core.engine`
modules injected into sys.modules (same trick used in tests/test_run_scoped_randopt.py /
test_gate2_restoration_ab.py) rather than tests/test_vlm_adapter.py's real-ray-required gate,
so these run locally without the GPU-stack ray dependency installed.

reset_vllm_encoder_cache_full now orchestrates TWO dispatches (this repair pass, pinned vLLM
0.11.0 compatibility -- see vlm_adapter.py's own docstring, divergence #9): the direct actor
method (layers A+B+C: frontend/receiver/scheduler, engine-process-side) AND collective_rpc on
every worker (layer D: the GPU worker's own physical embedding cache) -- collective_rpc is no
longer something this function avoids, it's a REQUIRED second half. Fakes below give every
engine a working collective_rpc returning a well-formed, all-clear worker-side result unless a
test is specifically about that layer.
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


@pytest.fixture(autouse=True)
def _fake_vllm_module_for_native_dispatch(monkeypatch):
    """_reset_encoder_cache_engine_side (the function ensure_full_encoder_cache_reset_exposed
    adds to RandOptNcclLLM) does `import vllm` to read __version__ for its return report --
    inject a minimal fake so tests in this file don't need the real (GPU-stack-only) package.
    """
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.__version__ = "0.11.0"
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)


def _fake_worker_collective_rpc(calls, *, entry_count_after=0):
    """A collective_rpc.remote stand-in returning ONE well-formed, all-clear
    _clear_worker_encoder_cache_v011-shaped result -- matches _validate_worker_encoder_cache_
    reset_results' expectations so tests not specifically about layer D can ignore it.
    """
    def _remote(method, args=()):
        calls.append((method, args))
        return [{"encoder_cache_entry_count_before": 1, "encoder_cache_entry_count_after": entry_count_after}]
    return SimpleNamespace(remote=_remote)


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
    """_FakeLLMEngine exposes a native reset_encoder_cache() -- the added method must take the
    NATIVE path (see _reset_encoder_cache_engine_side's own docstring) and call it directly,
    never falling through to the pinned-v0.11.0 compatibility reproduction.
    """
    cls = _fake_core_engine_module
    ensure_full_encoder_cache_reset_exposed(tmp_path)

    instance = cls()
    instance.llm_engine = _FakeLLMEngine()
    report = cls.reset_encoder_cache_full(instance)

    assert instance.llm_engine.reset_encoder_cache_called is True
    assert report["path"] == "native_reset_encoder_cache"
    assert report["vllm_version"] == "0.11.0"


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
    rpc_calls = []
    engine = SimpleNamespace(
        reset_encoder_cache_full=_FakeDirectMethod(calls),
        collective_rpc=_fake_worker_collective_rpc(rpc_calls),
    )

    report = reset_vllm_encoder_cache_full(engine)

    assert calls == ["reset_encoder_cache_full"]
    assert report["engine_side"] is None  # _FakeDirectMethod.remote() returns None
    assert report["worker_side"] == [{"encoder_cache_entry_count_before": 1, "encoder_cache_entry_count_after": 0}]


def test_reset_vllm_encoder_cache_full_raises_when_method_missing_on_actor():
    engine = SimpleNamespace()  # no reset_encoder_cache_full attribute at all

    with pytest.raises(RuntimeError, match="does not expose"):
        reset_vllm_encoder_cache_full(engine)


def test_reset_vllm_encoder_cache_full_raises_when_call_errors():
    calls = []
    engine = SimpleNamespace(reset_encoder_cache_full=_FakeDirectMethod(calls, raises=RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="failed"):
        reset_vllm_encoder_cache_full(engine)


def test_reset_vllm_encoder_cache_full_never_uses_collective_rpc_for_the_engine_side_layers():
    """The engine-side reset (layers A+B+C: frontend/receiver/scheduler) must be a direct actor
    method call, NOT collective_rpc -- collective_rpc only ever reaches worker processes.
    Confirmed here by giving collective_rpc a fake that would raise if ever asked to run the
    engine-side method name.
    """
    calls = []

    def _rpc_that_rejects_engine_side_calls(method, args=()):
        if method == "reset_encoder_cache_full":
            raise AssertionError("engine-side reset must never be dispatched via collective_rpc")
        calls.append((method, args))
        return [{"encoder_cache_entry_count_before": 1, "encoder_cache_entry_count_after": 0}]

    engine = SimpleNamespace(
        reset_encoder_cache_full=_FakeDirectMethod([]),
        collective_rpc=SimpleNamespace(remote=_rpc_that_rejects_engine_side_calls),
    )

    reset_vllm_encoder_cache_full(engine)  # should not raise -- the direct method call is separate

    assert len(calls) == 1  # exactly one collective_rpc dispatch -- the layer-D worker clear


def test_reset_vllm_encoder_cache_full_uses_collective_rpc_for_the_worker_side_layer():
    """Layer D (the GPU worker's physical embedding cache) MUST be dispatched via
    collective_rpc -- confirmed by checking the callable actually dispatched matches
    _clear_worker_encoder_cache_v011.
    """
    from neural_thickets_repro.vlm_adapter import _clear_worker_encoder_cache_v011

    rpc_calls = []
    engine = SimpleNamespace(
        reset_encoder_cache_full=_FakeDirectMethod([]),
        collective_rpc=_fake_worker_collective_rpc(rpc_calls),
    )

    reset_vllm_encoder_cache_full(engine)

    assert len(rpc_calls) == 1
    dispatched_method, dispatched_args = rpc_calls[0]
    assert dispatched_method is _clear_worker_encoder_cache_v011
    assert dispatched_args == ()


def test_reset_vllm_encoder_cache_full_raises_when_worker_still_populated_after_reset():
    rpc_calls = []
    engine = SimpleNamespace(
        reset_encoder_cache_full=_FakeDirectMethod([]),
        collective_rpc=_fake_worker_collective_rpc(rpc_calls, entry_count_after=3),
    )

    with pytest.raises(RuntimeError, match="NOT actually cleared"):
        reset_vllm_encoder_cache_full(engine)

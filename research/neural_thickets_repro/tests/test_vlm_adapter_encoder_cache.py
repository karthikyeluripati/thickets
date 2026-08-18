"""Tests for vlm_adapter.reset_vllm_encoder_cache -- via a fake `ray` module injected into
sys.modules (same trick used in tests/test_run_scoped_randopt.py / test_gate2_restoration_ab.py)
rather than tests/test_vlm_adapter.py's real-ray-required gate, so these run locally without
the GPU-stack ray dependency installed.
"""
import sys
from types import SimpleNamespace

import pytest

from neural_thickets_repro.vlm_adapter import reset_vllm_encoder_cache


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

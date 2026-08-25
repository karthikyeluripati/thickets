"""Tests for vlm_adapter.py's version-aware full multimodal-encoder-cache reset (pinned vLLM
0.11.0 compatibility path). Unlike test_vlm_adapter.py (entirely ray-gated, skipped locally),
NONE of these tests need the real `ray` or `vllm` packages installed -- every function under
test defers `import ray`/`import vllm` to inside reset_vllm_encoder_cache_full/
_reset_encoder_cache_engine_side specifically so the lower-level compatibility logic tested
here stays reachable without either GPU-stack dependency. A minimal fake `vllm` module tree is
injected into sys.modules (autouse fixture below) so `import vllm` / `from vllm.v1.core.
encoder_cache_manager import EncoderCacheManager` resolve to controllable fakes.

Root cause this fixes (live GPU failure, commit 74f273b's cache-safety smoke): the pinned
runtime is vLLM 0.11.0, whose LLMEngine has no reset_encoder_cache() at all
(AttributeError). Confirmed present on 0.11.0: reset_mm_cache() (frontend processor + engine
MM receiver cache together) and reset_prefix_cache() -- NEITHER touches the scheduler's own
EncoderCacheManager bookkeeping or the GPU worker's physical encoder_cache tensors, which is
why a full, version-aware reset must be reproduced explicitly (this module's divergence #9).
"""
from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from neural_thickets_repro.vlm_adapter import (
    EncoderCacheResetPreconditionError,
    EncoderCacheResetVerificationError,
    UnsupportedVLLMLayoutError,
    _clear_worker_encoder_cache_v011,
    _engine_num_unfinished_requests,
    _report_encoder_cache_manager_state,
    _report_encoder_cache_state_engine_side,
    _report_worker_encoder_cache_v011,
    _reset_encoder_cache_engine_side,
    _reset_encoder_cache_engine_side_v011_compat,
    _resolve_v011_scheduler,
    _validate_worker_encoder_cache_reset_results,
    _verify_encoder_cache_manager_reset,
)


class _FakeEncoderCacheManager:
    """Matches the interface _report_encoder_cache_manager_state/_verify_encoder_cache_manager_
    reset expect from a real vllm.v1.core.encoder_cache_manager.EncoderCacheManager instance --
    see this module's own docstring for the UNRESOLVED-against-real-vLLM caveat on these exact
    attribute names.
    """

    def __init__(self, cache_size: int):
        self.cache_size = cache_size
        self.cached: Dict[str, Any] = {}
        self.freeable: Dict[str, Any] = {}
        self.num_free_slots = cache_size

    def populate(self, key: str, slots_used: int) -> None:
        self.cached[key] = {1}
        self.num_free_slots -= slots_used


@pytest.fixture(autouse=True)
def _fake_vllm_module(monkeypatch):
    """Injects sys.modules["vllm"] / sys.modules["vllm.v1.core.encoder_cache_manager"] so
    `import vllm` and `from vllm.v1.core.encoder_cache_manager import EncoderCacheManager`
    resolve to controllable fakes without the real (GPU-stack-only) vllm package installed.
    Returns the fake vllm module so tests can set `.__version__` per-test.
    """
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.__version__ = "0.11.0"
    fake_v1 = types.ModuleType("vllm.v1")
    fake_core = types.ModuleType("vllm.v1.core")
    fake_ecm_module = types.ModuleType("vllm.v1.core.encoder_cache_manager")
    fake_ecm_module.EncoderCacheManager = _FakeEncoderCacheManager
    fake_vllm.v1 = fake_v1
    fake_v1.core = fake_core
    fake_core.encoder_cache_manager = fake_ecm_module

    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.v1", fake_v1)
    monkeypatch.setitem(sys.modules, "vllm.v1.core", fake_core)
    monkeypatch.setitem(sys.modules, "vllm.v1.core.encoder_cache_manager", fake_ecm_module)
    return fake_vllm


def _fake_llm_engine(*, cache_size=10, unfinished=0, native_reset=False):
    manager = _FakeEncoderCacheManager(cache_size=cache_size)
    scheduler = SimpleNamespace(encoder_cache_manager=manager)
    engine_core_inner = SimpleNamespace(scheduler=scheduler)
    engine_core = SimpleNamespace(engine_core=engine_core_inner)
    calls = []

    def _reset_mm_cache():
        calls.append("reset_mm_cache")

    engine = SimpleNamespace(
        engine_core=engine_core,
        get_num_unfinished_requests=lambda: unfinished,
        reset_mm_cache=_reset_mm_cache,
        calls=calls,
    )
    if native_reset:
        engine.reset_encoder_cache = lambda: calls.append("native_reset_encoder_cache")
    return engine, manager


# =================================================================================================
# _report_encoder_cache_manager_state / _verify_encoder_cache_manager_reset
# =================================================================================================


def test_report_encoder_cache_manager_state_fresh_manager():
    manager = _FakeEncoderCacheManager(cache_size=10)
    report = _report_encoder_cache_manager_state(manager)
    assert report == {"cached_entry_count": 0, "freeable_entry_count": 0, "cache_size": 10, "num_free_slots": 10}


def test_report_encoder_cache_manager_state_populated_manager():
    manager = _FakeEncoderCacheManager(cache_size=10)
    manager.populate("mm_hash_1", slots_used=3)
    report = _report_encoder_cache_manager_state(manager)
    assert report["cached_entry_count"] == 1
    assert report["num_free_slots"] == 7


def test_report_encoder_cache_manager_state_hard_fails_on_missing_attribute():
    class _Incomplete:
        cache_size = 10
        cached = {}
        # freeable and num_free_slots deliberately missing

    with pytest.raises(UnsupportedVLLMLayoutError, match="missing expected attribute"):
        _report_encoder_cache_manager_state(_Incomplete())


def test_verify_encoder_cache_manager_reset_passes_for_fresh_state():
    _verify_encoder_cache_manager_reset({"cached_entry_count": 0, "freeable_entry_count": 0, "cache_size": 10, "num_free_slots": 10})


@pytest.mark.parametrize("bad_after", [
    {"cached_entry_count": 1, "freeable_entry_count": 0, "cache_size": 10, "num_free_slots": 10},
    {"cached_entry_count": 0, "freeable_entry_count": 1, "cache_size": 10, "num_free_slots": 10},
    {"cached_entry_count": 0, "freeable_entry_count": 0, "cache_size": 10, "num_free_slots": 7},
])
def test_verify_encoder_cache_manager_reset_fails_when_not_fresh(bad_after):
    with pytest.raises(EncoderCacheResetVerificationError):
        _verify_encoder_cache_manager_reset(bad_after)


# =================================================================================================
# _engine_num_unfinished_requests
# =================================================================================================


def test_engine_num_unfinished_requests_uses_engine_level_method():
    engine = SimpleNamespace(get_num_unfinished_requests=lambda: 3)
    assert _engine_num_unfinished_requests(engine) == 3


def test_engine_num_unfinished_requests_falls_back_to_engine_core():
    engine = SimpleNamespace(engine_core=SimpleNamespace(get_num_unfinished_requests=lambda: 2))
    assert _engine_num_unfinished_requests(engine) == 2


def test_engine_num_unfinished_requests_hard_fails_when_neither_reachable():
    engine = SimpleNamespace(engine_core=SimpleNamespace())
    with pytest.raises(UnsupportedVLLMLayoutError, match="cannot verify"):
        _engine_num_unfinished_requests(engine)


# =================================================================================================
# _resolve_v011_scheduler
# =================================================================================================


def test_resolve_v011_scheduler_returns_expected_path():
    sentinel = object()
    engine = SimpleNamespace(engine_core=SimpleNamespace(engine_core=SimpleNamespace(scheduler=sentinel)))
    assert _resolve_v011_scheduler(engine) is sentinel


def test_resolve_v011_scheduler_hard_fails_on_missing_layout():
    engine = SimpleNamespace(engine_core=SimpleNamespace())  # no nested .engine_core.scheduler
    with pytest.raises(UnsupportedVLLMLayoutError, match="engine_core.engine_core.scheduler"):
        _resolve_v011_scheduler(engine)


# =================================================================================================
# _clear_worker_encoder_cache_v011 (layer D)
# =================================================================================================


def test_clear_worker_encoder_cache_v011_clears_and_reports():
    worker = SimpleNamespace(model_runner=SimpleNamespace(encoder_cache={"a": object(), "b": object()}))
    report = _clear_worker_encoder_cache_v011(worker)
    assert report == {"encoder_cache_entry_count_before": 2, "encoder_cache_entry_count_after": 0}
    assert worker.model_runner.encoder_cache == {}


def test_clear_worker_encoder_cache_v011_empty_to_start():
    worker = SimpleNamespace(model_runner=SimpleNamespace(encoder_cache={}))
    report = _clear_worker_encoder_cache_v011(worker)
    assert report == {"encoder_cache_entry_count_before": 0, "encoder_cache_entry_count_after": 0}


def test_clear_worker_encoder_cache_v011_hard_fails_when_missing():
    worker = SimpleNamespace(model_runner=SimpleNamespace())  # no .encoder_cache
    with pytest.raises(UnsupportedVLLMLayoutError, match="model_runner.encoder_cache"):
        _clear_worker_encoder_cache_v011(worker)


def test_clear_worker_encoder_cache_v011_is_empty_on_every_worker_when_dispatched_to_several():
    workers = [SimpleNamespace(model_runner=SimpleNamespace(encoder_cache={f"k{i}": object()})) for i in range(3)]
    reports = [_clear_worker_encoder_cache_v011(w) for w in workers]
    assert all(r["encoder_cache_entry_count_after"] == 0 for r in reports)
    assert all(w.model_runner.encoder_cache == {} for w in workers)


# =================================================================================================
# _validate_worker_encoder_cache_reset_results (layer D verification, no ray needed)
# =================================================================================================


def test_validate_worker_encoder_cache_reset_results_passes_when_all_clear():
    _validate_worker_encoder_cache_reset_results([
        {"encoder_cache_entry_count_before": 2, "encoder_cache_entry_count_after": 0},
        {"encoder_cache_entry_count_before": 1, "encoder_cache_entry_count_after": 0},
    ])


def test_validate_worker_encoder_cache_reset_results_fails_if_any_worker_still_populated():
    with pytest.raises(EncoderCacheResetVerificationError, match="worker 1"):
        _validate_worker_encoder_cache_reset_results([
            {"encoder_cache_entry_count_before": 2, "encoder_cache_entry_count_after": 0},
            {"encoder_cache_entry_count_before": 1, "encoder_cache_entry_count_after": 1},
        ])


def test_validate_worker_encoder_cache_reset_results_fails_on_empty_list():
    with pytest.raises(RuntimeError, match="unexpected shape"):
        _validate_worker_encoder_cache_reset_results([])


def test_validate_worker_encoder_cache_reset_results_fails_on_non_list():
    with pytest.raises(RuntimeError, match="unexpected shape"):
        _validate_worker_encoder_cache_reset_results({"not": "a list"})


# =================================================================================================
# _reset_encoder_cache_engine_side_v011_compat (layers A+B+C together)
# =================================================================================================


def test_v011_compat_full_flow_clears_scheduler_state_and_calls_reset_mm_cache():
    engine, original_manager = _fake_llm_engine(cache_size=8, unfinished=0)
    original_manager.populate("mm_hash_1", slots_used=3)  # simulate a cached image

    report = _reset_encoder_cache_engine_side_v011_compat(engine)

    assert report["mm_cache_reset_called"] is True
    assert "reset_mm_cache" in engine.calls
    assert report["scheduler_before"]["cached_entry_count"] == 1
    assert report["scheduler_after"] == {"cached_entry_count": 0, "freeable_entry_count": 0, "cache_size": 8, "num_free_slots": 8}
    # The manager was REBUILT fresh, not mutated in place.
    assert engine.engine_core.engine_core.scheduler.encoder_cache_manager is not original_manager
    assert engine.engine_core.engine_core.scheduler.encoder_cache_manager.cache_size == 8


def test_v011_compat_hard_fails_with_unfinished_requests_and_never_touches_the_cache():
    engine, manager = _fake_llm_engine(cache_size=8, unfinished=1)
    manager.populate("mm_hash_1", slots_used=2)

    with pytest.raises(EncoderCacheResetPreconditionError, match="unfinished/in-flight"):
        _reset_encoder_cache_engine_side_v011_compat(engine)

    assert "reset_mm_cache" not in engine.calls  # precondition blocked BEFORE any reset action
    assert manager.cached  # untouched -- still populated
    assert engine.engine_core.engine_core.scheduler.encoder_cache_manager is manager  # never replaced


def test_reset_mm_cache_alone_is_insufficient_to_clear_scheduler_state():
    """Direct demonstration (item 4 in the task): reset_mm_cache() alone does NOT touch the
    scheduler's EncoderCacheManager -- only the explicit rebuild (layer C) does.
    """
    engine, manager = _fake_llm_engine(cache_size=8, unfinished=0)
    manager.populate("mm_hash_1", slots_used=3)

    engine.reset_mm_cache()  # layer A+B only, called directly -- NOT the full compat path

    assert manager.cached, "reset_mm_cache() alone should NOT clear the scheduler's EncoderCacheManager"
    assert manager.num_free_slots == 5


def test_v011_compat_hard_fails_on_missing_scheduler_layout():
    engine = SimpleNamespace(
        engine_core=SimpleNamespace(engine_core=SimpleNamespace()),  # no .scheduler at all
        get_num_unfinished_requests=lambda: 0,
    )
    with pytest.raises(UnsupportedVLLMLayoutError):
        _reset_encoder_cache_engine_side_v011_compat(engine)


def test_v011_compat_hard_fails_on_missing_encoder_cache_manager():
    engine = SimpleNamespace(
        engine_core=SimpleNamespace(engine_core=SimpleNamespace(scheduler=SimpleNamespace())),  # scheduler has no encoder_cache_manager
        get_num_unfinished_requests=lambda: 0,
    )
    with pytest.raises(UnsupportedVLLMLayoutError, match="encoder_cache_manager not found"):
        _reset_encoder_cache_engine_side_v011_compat(engine)


# =================================================================================================
# _reset_encoder_cache_engine_side (native-vs-compat dispatch + strict version guard)
# =================================================================================================


def test_engine_side_uses_native_path_when_available(_fake_vllm_module):
    engine, _ = _fake_llm_engine(native_reset=True)
    self_ = SimpleNamespace(llm_engine=engine)

    report = _reset_encoder_cache_engine_side(self_)

    assert report["path"] == "native_reset_encoder_cache"
    assert report["vllm_version"] == "0.11.0"
    assert "native_reset_encoder_cache" in engine.calls
    assert "reset_mm_cache" not in engine.calls  # native path never falls through to compat


def test_engine_side_uses_v011_compat_when_native_absent_and_version_matches(_fake_vllm_module):
    engine, manager = _fake_llm_engine(cache_size=4, unfinished=0, native_reset=False)
    manager.populate("mm_hash_1", slots_used=1)
    self_ = SimpleNamespace(llm_engine=engine)

    report = _reset_encoder_cache_engine_side(self_)

    assert report["path"] == "v011_compat"
    assert report["vllm_version"] == "0.11.0"
    assert "reset_mm_cache" in engine.calls


def test_engine_side_hard_fails_on_unsupported_version_without_native_api(_fake_vllm_module):
    _fake_vllm_module.__version__ = "0.9.3"  # not the pinned 0.11.0, no native method either
    engine, _ = _fake_llm_engine(native_reset=False)
    self_ = SimpleNamespace(llm_engine=engine)

    with pytest.raises(UnsupportedVLLMLayoutError, match="0.9.3"):
        _reset_encoder_cache_engine_side(self_)


# =================================================================================================
# Read-only report counterparts (diagnostic use, never mutate anything)
# =================================================================================================


def test_report_encoder_cache_state_engine_side_reports_without_mutating(_fake_vllm_module):
    engine, manager = _fake_llm_engine(cache_size=6)
    manager.populate("mm_hash_1", slots_used=2)
    self_ = SimpleNamespace(llm_engine=engine)

    report = _report_encoder_cache_state_engine_side(self_)

    assert report["vllm_version"] == "0.11.0"
    assert report["scheduler_state"]["cached_entry_count"] == 1
    assert report["scheduler_state"]["num_free_slots"] == 4
    # Nothing was reset -- the manager instance is untouched.
    assert engine.engine_core.engine_core.scheduler.encoder_cache_manager is manager
    assert manager.cached  # still populated


def test_report_worker_encoder_cache_v011_reports_without_clearing():
    worker = SimpleNamespace(model_runner=SimpleNamespace(encoder_cache={"img_abc": object()}))
    report = _report_worker_encoder_cache_v011(worker)
    assert report == {"encoder_cache_entry_count": 1, "encoder_cache_keys": ["img_abc"]}
    assert worker.model_runner.encoder_cache  # untouched


def test_report_worker_encoder_cache_v011_hard_fails_when_missing():
    worker = SimpleNamespace(model_runner=SimpleNamespace())
    with pytest.raises(UnsupportedVLLMLayoutError, match="model_runner.encoder_cache"):
        _report_worker_encoder_cache_v011(worker)


def test_engine_side_v011_layout_hard_fail_still_matches_pinned_version_string(_fake_vllm_module):
    """Confirms the AttributeError this whole compatibility layer exists to fix is exactly
    what a naive `self.llm_engine.reset_encoder_cache()` call would raise on the pinned
    runtime -- the native-path branch here must NOT be silently taken when it's absent.
    """
    engine, _ = _fake_llm_engine(native_reset=False)
    assert not hasattr(engine, "reset_encoder_cache")
    assert _fake_vllm_module.__version__ == "0.11.0"

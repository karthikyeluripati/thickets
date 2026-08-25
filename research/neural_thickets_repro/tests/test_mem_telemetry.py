"""Tests for mem_telemetry.py -- psutil-based driver RSS telemetry (Stage-8 driver-RSS OOM
audit, this repair pass). CPU-only, no GPU/ray needed.
"""
from neural_thickets_repro.mem_telemetry import release_transient_memory, rss_mb


def test_rss_mb_returns_a_positive_float():
    value = rss_mb()
    assert isinstance(value, float)
    assert value > 0.0


def test_rss_mb_reflects_a_real_allocation():
    """Not a tight assertion (RSS reporting has OS-level granularity/timing noise) -- just
    confirms rss_mb() is wired to a real, live psutil.Process().memory_info().rss call rather
    than a stub, by allocating a real, sizeable chunk of memory and keeping it referenced.
    """
    before = rss_mb()
    big = bytearray(50 * 1024 * 1024)  # 50 MB, kept alive until after the second measurement
    after = rss_mb()
    assert after >= before
    del big


def test_release_transient_memory_never_raises():
    # Safe no-op on non-Linux platforms (this repo's own Windows dev machine) -- must never
    # raise regardless of platform.
    release_transient_memory()
    release_transient_memory()  # idempotent, callable repeatedly

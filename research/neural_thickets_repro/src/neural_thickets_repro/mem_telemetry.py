"""Driver-side RSS telemetry (Stage 8, this repair pass -- the 576-candidate full run OOM'd
during the N=50 baseline-repeatability preflight, node RAM at ~110.67/116.42 GB, driver Python
process RSS at ~100.86 GB while the RayWorkerWrapper/RandOptNcclLLM actors themselves were only
~2.18/1.88 GB -- i.e. primarily DRIVER/HOST RAM growth, not GPU VRAM). Uses psutil (real
process RSS from the OS), not tracemalloc, because the suspected growth is native/PIL/numpy/
Ray-serialization-backed memory that pure-Python allocation tracking would miss entirely.
"""
from __future__ import annotations

import ctypes
import gc
import platform

import psutil

_PROCESS = psutil.Process()


def rss_mb() -> float:
    """Current process resident set size, in MB (driver/host RAM, not GPU VRAM)."""
    return _PROCESS.memory_info().rss / (1024 * 1024)


def release_transient_memory() -> None:
    """Best-effort: collect cyclic garbage, then (glibc/Linux only) ask the C allocator to
    return freed arenas to the OS via malloc_trim(0).

    gc.collect() alone is NOT the primary fix for RSS staying elevated after a large
    transient allocation (e.g. benchmarks/*.py's load_examples() decoding an entire dataset's
    images before subset selection reduces it to N=50) -- ordinary refcounted objects are
    already freed the instant their last reference drops, with or without gc.collect(). What
    keeps RSS elevated afterward is glibc's own allocator retaining freed memory in its own
    arenas for reuse rather than returning it to the OS; malloc_trim(0) is the correct,
    targeted fix for THAT specific symptom. A safe no-op (never raises) on non-glibc platforms
    (this repo's own Windows dev machine, macOS, or any environment where libc.so.6 isn't
    resolvable) -- never load-bearing for correctness, only for keeping driver RSS from
    ratcheting upward across sequential large transient allocations.
    """
    gc.collect()
    if platform.system() == "Linux":
        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except (OSError, AttributeError):
            pass

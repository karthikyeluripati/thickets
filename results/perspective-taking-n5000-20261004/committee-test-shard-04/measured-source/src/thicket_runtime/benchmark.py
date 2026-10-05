"""Serial, synchronized lifecycle measurements; diagnostics are separate runs."""
from dataclasses import asdict
import math
import platform
import statistics
import time
from typing import Callable

import torch

from .candidate import CandidateSpec
from .execution import WeightState


def finite_score(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("nonfinite score; refusing to publish a successful measurement")
    return value


def timed(state: WeightState, label: str, fn: Callable,
          *, cuda_events: bool = True):
    """Events measure a device timeline interval, not a sum of kernel runtimes.

    Stage wall timings include synchronization. Never add event milliseconds
    to wall milliseconds: these are different views of overlapping work.
    """
    state.synchronize()
    events = state.device.type == "cuda" and cuda_events
    start_event = torch.cuda.Event(enable_timing=True) if events else None
    end_event = torch.cuda.Event(enable_timing=True) if events else None
    wall_start = time.perf_counter()
    if events:
        start_event.record(torch.cuda.current_stream(state.device))
    with state.region(label):
        value = fn()
    if events:
        end_event.record(torch.cuda.current_stream(state.device))
    state.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1e3
    device_ms = start_event.elapsed_time(end_event) if events else None
    return value, {"wall_ms": wall_ms, "cuda_interval_ms": device_ms}


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("cannot summarize an empty run")
    totals = [r["total_wall_ms"] for r in rows]
    total = sum(totals)
    state_ms = sum(r["apply"]["wall_ms"] + r["restore"]["wall_ms"] for r in rows)
    fraction = state_ms / total
    return {"candidate_evaluations": len(rows),
            "candidate_latency_median_ms": statistics.median(totals),
            "candidate_latency_min_ms": min(totals),
            "candidate_latency_max_ms": max(totals),
            "measured_loop_candidates_per_second": 1000 * len(rows) / total,
            "state_wall_fraction": fraction,
            "ideal_state_elimination_speedup_bound": 1 / (1 - fraction) if fraction < 1 else None,
            "phase_wall_median_ms": {
                phase: statistics.median(r[phase]["wall_ms"] for r in rows)
                for phase in ("apply", "inference", "restore", "score")}}


def environment(device: torch.device) -> dict:
    result = {"python": platform.python_version(), "torch": torch.__version__,
              "cuda_runtime": torch.version.cuda, "device": str(device),
              "cpu_threads": torch.get_num_threads(),
              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
              "float32_matmul_precision": torch.get_float32_matmul_precision()}
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        result.update({"gpu": p.name, "gpu_total_bytes": p.total_memory,
                       "compute_capability": [p.major, p.minor],
                       "cudnn": torch.backends.cudnn.version()})
    return result


def run_trace(state: WeightState, candidates: list[CandidateSpec], infer: Callable,
              score: Callable, *, repeats: int = 3, warmup: int = 2,
              cuda_events: bool = True) -> dict:
    if not candidates or repeats < 1 or warmup < 0:
        raise ValueError("nonempty candidates, repeats>=1 and warmup>=0 required")
    if any(c.base_id != state.base_id for c in candidates):
        raise ValueError("trace/base identity mismatch")
    rows, drift = [], []
    try:
        for i in range(warmup):
            state.rebase()
            with state.using(candidates[i % len(candidates)]):
                output = infer()
                finite_score(score(output))
            del output
        state.rebase()
        for repetition in range(repeats):
            # Start each repeated trace from the exact base, NOT each candidate.
            # Thus sequential drift in the legacy path is not concealed.
            state.rebase()
            if state.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(state.device)
            initial_allocated = torch.cuda.memory_allocated(state.device) if state.device.type == "cuda" else None
            for index, candidate in enumerate(candidates):
                state.synchronize()
                t0 = time.perf_counter()
                _, apply = timed(state, "apply", lambda: state.apply(candidate), cuda_events=cuda_events)
                try:
                    output, inference = timed(state, "inference", infer, cuda_events=cuda_events)
                finally:
                    _, restore = timed(state, "restore", state.reset, cuda_events=cuda_events)
                try:
                    reward, score_timing = timed(state, "score", lambda: finite_score(score(output)),
                                                 cuda_events=False)
                    total_wall_ms = (time.perf_counter() - t0) * 1e3
                    rows.append({"candidate_id": candidate.candidate_id,
                                 "repeat": repetition, "index": index, "reward": reward,
                                 "apply": apply, "inference": inference,
                                 "restore": restore, "score": score_timing,
                                 "total_wall_ms": total_wall_ms})
                finally:
                    del output
            # Capture memory BEFORE the expensive correctness audit.
            memory = {"baseline_allocated_bytes": initial_allocated,
                      "peak_allocated_bytes": torch.cuda.max_memory_allocated(state.device) if state.device.type == "cuda" else None,
                      "peak_reserved_bytes": torch.cuda.max_memory_reserved(state.device) if state.device.type == "cuda" else None}
            drift.append({"repeat": repetition, "memory": memory, "post_trace_drift": state.drift()})
        return {"strategy": state.strategy, "base_id": state.base_id,
                "rng_contract": sorted({c.rng for c in candidates}),
                "parameter_bytes": state.parameter_bytes,
                "snapshot_parameter_bytes": state.parameter_bytes,
                "cleanup_cache": state.cleanup_cache, "cuda_events": cuda_events,
                "measurement_mode": "diagnostic" if state.detail else "coarse",
                "gpu_performance_evidence": state.device.type == "cuda" and not state.detail,
                "rows": rows, "repeats": drift, "summary": summarize(rows)}
    finally:
        # An exception cannot silently leave the reusable model perturbed.
        state.rebase()


@torch.inference_mode()
def audit_trace(state: WeightState, candidates: list[CandidateSpec], infer: Callable,
                score: Callable) -> dict:
    """Sequential execution vs independent, snapshot-anchored candidates.

    This is deliberately OUTSIDE timed measurements. Exact output equality is
    reported, not promised. For generated integer token IDs, equality means
    token agreement, not that hidden logits were bitwise equal.
    """
    original_strategy = state.strategy
    observed, rows = [], []
    try:
        state.rebase()
        for candidate in candidates:
            with state.using(candidate):
                output = infer().detach().cpu().clone()
                reward = finite_score(score(output))
            observed.append((output, reward, state.drift()))
        state.rebase()
        state.strategy = "snapshot-copy"
        for candidate, (actual, actual_reward, drift) in zip(candidates, observed):
            with state.using(candidate):
                expected = infer().detach().cpu()
                expected_reward = finite_score(score(expected))
            same_shape = actual.shape == expected.shape
            equal = same_shape and torch.equal(actual, expected)
            max_abs = (float((actual.double() - expected.double()).abs().max().item())
                       if same_shape and actual.numel() else (0.0 if same_shape else None))
            if max_abs is not None and not math.isfinite(max_abs):
                max_abs = None
            rows.append({"candidate_id": candidate.candidate_id,
                         "output_exact": equal, "output_max_abs": max_abs,
                         "reward_exact": actual_reward == expected_reward,
                         "actual_reward": actual_reward, "reference_reward": expected_reward,
                         "post_candidate_drift": drift})
        return {"reference": "independent-snapshot-anchored-candidates",
                "all_outputs_exact": all(r["output_exact"] for r in rows),
                "all_rewards_exact": all(r["reward_exact"] for r in rows),
                "all_resets_exact": all(r["post_candidate_drift"]["exact_base"] for r in rows),
                "rows": rows}
    finally:
        state.strategy = original_strategy
        state.rebase()

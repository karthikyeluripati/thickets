"""Synchronous RPC lifecycle timing; audits never execute in timed spans."""
from dataclasses import asdict
import hashlib
import json
import time

from .benchmark import finite_score, summarize
from .candidate import CandidateSpec


def bind_recipes(recipes, native_base_id):
    """Retain source identities but never assign an HF ID to a native vLLM state."""
    result = []
    for recipe in recipes:
        source = CandidateSpec(**{k: recipe[k] for k in
                                 ("base_id", "seed", "sigma", "rng", "sign")})
        if source.candidate_id != recipe["candidate_id"]:
            raise ValueError("source candidate identity mismatch")
        if source.rng != "randopt-per-tensor-v1" or source.sign != 1:
            raise ValueError("this upstream snapshot comparison requires original RNG and positive sign")
        native = CandidateSpec(native_base_id, source.seed, source.sigma, source.rng, source.sign)
        result.append(asdict(native) | {"candidate_id": native.candidate_id,
                      "source_candidate_id": source.candidate_id,
                      "source_base_id": source.base_id})
    if not result or len({r["candidate_id"] for r in result}) != len(result):
        raise ValueError("a nonempty, unique candidate trace is required")
    return result


def output_record(outputs):
    """Inputs are plain per-request text, token IDs, and stopping metadata."""
    tokens = [r["token_ids"] for r in outputs]
    return {"generated_token_counts": [len(t) for t in tokens],
            "generated_tokens": sum(map(len, tokens)),
            "finish_reasons": [r["finish_reason"] for r in outputs],
            "token_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest()}


def measure(fn):
    start = time.perf_counter()
    result = fn()
    return result, {"wall_ms": (time.perf_counter() - start) * 1000,
                    "cuda_interval_ms": None}


def run_remote_trace(api, candidates, strategy, *, repeats=3, warmup=2):
    if not candidates or repeats < 1 or warmup < 0:
        raise ValueError("invalid trace bounds")
    rows, repetitions = [], []
    try:
        for i in range(warmup):
            api.rebase()
            api.apply(candidates[i % len(candidates)], strategy)
            try:
                output = api.infer()
            finally:
                api.restore(candidates[i % len(candidates)], strategy)
            finite_score(api.score(output))
        for repeat in range(repeats):
            api.rebase()
            initial = api.memory(reset=True)
            for index, candidate in enumerate(candidates):
                start = time.perf_counter()
                _, apply = measure(lambda: api.apply(candidate, strategy))
                try:
                    output, inference = measure(api.infer)
                finally:
                    _, restore = measure(lambda: api.restore(candidate, strategy))
                reward, score = measure(lambda: finite_score(api.score(output)))
                total = (time.perf_counter() - start) * 1000
                rows.append({"repeat": repeat, "index": index,
                             "candidate_id": candidate["candidate_id"],
                             "source_candidate_id": candidate["source_candidate_id"],
                             "apply": apply, "inference": inference, "restore": restore,
                             "score": score, "reward": reward, "total_wall_ms": total,
                             **output_record(output)})
            # This must precede the full-state audit, which allocates scratch space.
            memory = api.memory()
            repetitions.append({"repeat": repeat, "initial_memory": initial,
                                "memory": memory, "post_trace_drift": api.drift()})
        summary = summarize(rows)
        total_ms = sum(r["total_wall_ms"] for r in rows)
        summary["inference_wall_fraction"] = sum(r["inference"]["wall_ms"] for r in rows) / total_ms
        summary["phase_wall_mean_ms"] = {p: sum(r[p]["wall_ms"] for r in rows) / len(rows)
                                          for p in ("apply", "inference", "restore", "score")}
        summary["candidate_latency_mean_ms"] = total_ms / len(rows)
        summary["within_process_repeats"] = [summarize([r for r in rows if r["repeat"] == i])
                                              for i in range(repeats)]
        return {"strategy": strategy, "rows": rows, "repeats": repetitions,
                "summary": summary, "timing": "driver wall time including Ray and worker RPC",
                "measurement_mode": "coarse", "gpu_performance_evidence": True}
    finally:
        api.rebase()


def audit_remote_trace(api, candidates, strategy):
    observed, rows = [], []
    try:
        api.rebase()
        for candidate in candidates:
            api.apply(candidate, strategy)
            # Full parameter/buffer hashes and output collection are audit-only.
            candidate_hash = api.fingerprint()
            try:
                output = api.infer()
            finally:
                api.restore(candidate, strategy)
            observed.append((output, finite_score(api.score(output)), candidate_hash, api.drift()))
        for candidate, (actual, reward, candidate_hash, drift) in zip(candidates, observed):
            api.rebase()
            api.apply(candidate, "snapshot-copy")
            expected_hash = api.fingerprint()
            try:
                expected = api.infer()
            finally:
                api.restore(candidate, "snapshot-copy")
            reference_reward = finite_score(api.score(expected))
            rows.append({"candidate_id": candidate["candidate_id"],
                         "source_candidate_id": candidate["source_candidate_id"],
                         "candidate_state_exact": candidate_hash == expected_hash,
                         "actual_state_sha256": candidate_hash,
                         "reference_state_sha256": expected_hash,
                         "output_exact": [r["token_ids"] for r in actual] ==
                                         [r["token_ids"] for r in expected],
                         "reward_exact": reward == reference_reward,
                         "actual_reward": reward, "reference_reward": reference_reward,
                         "actual_outputs": actual, "reference_outputs": expected,
                         "post_candidate_drift": drift,
                         "reference_post_candidate_drift": api.drift()})
        result = {"reference": "independent original-worker snapshot-anchored native candidates",
                  "rows": rows,
                  "all_candidate_states_exact": all(r["candidate_state_exact"] for r in rows),
                  "all_outputs_exact": all(r["output_exact"] for r in rows),
                  "all_rewards_exact": all(r["reward_exact"] for r in rows),
                  "all_resets_exact": all(r["post_candidate_drift"]["exact_base"] for r in rows),
                  "all_reference_resets_exact": all(r["reference_post_candidate_drift"]["exact_base"] for r in rows)}
        result["exact_semantics_gate"] = all(v for k, v in result.items() if k.startswith("all_"))
        return result
    finally:
        api.rebase()

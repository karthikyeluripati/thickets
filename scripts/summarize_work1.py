"""Verify and summarize the bounded original-worker and vLLM/Ray closure artifacts."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def hf_validation(root, baseline):
    cases = []
    for tokens in (32, 128):
        original = root / f"original-hf-fixed-{tokens}"
        reference = baseline / f"fixed-{tokens}-b4"
        assert read(original / "manifest.json")["status"] == "complete"
        assert read(original / "candidates.json") == read(reference / "candidates.json")
        actual = read(original / "legacy-add-subtract.json")
        expected = read(reference / "legacy-add-subtract.json")
        assert actual["weight_operator_implementation"] == {
            "kind": "original-pinned-RandOpt-worker",
            "blob": "3b672d6636364845474be02e01cce764b6e18af3"}
        audit = actual["correctness"]
        cases.append({"tokens": tokens, "candidate_recipes_identical": True,
                      "audit_identical_to_reference_implementation": audit == expected["correctness"],
                      "exact_semantics_gate": actual["exact_semantics_gate"],
                      "output_matches": sum(r["output_exact"] for r in audit["rows"]),
                      "score_matches": sum(r["reward_exact"] for r in audit["rows"]),
                      "audited_candidates": len(audit["rows"]),
                      "post_trace_drift": actual["repeats"][0]["post_trace_drift"],
                      "original_sha256": hashlib.sha256((original / "legacy-add-subtract.json").read_bytes()).hexdigest(),
                      "reference_sha256": hashlib.sha256((reference / "legacy-add-subtract.json").read_bytes()).hexdigest()})
    return cases


def analyze(root, baseline):
    groups = {}
    native_traces, source_traces = set(), set()
    for path in sorted((root / "vllm-matrix").glob("*/manifest.json")):
        manifest = read(path)
        if manifest["status"] != "complete":
            raise ValueError(f"incomplete run: {path.parent}")
        recipes = read(path.parent / "candidates.json")
        ids = [r["candidate_id"] for r in recipes]
        native_traces.add(tuple(ids))
        source_traces.add(tuple(r["source_candidate_id"] for r in recipes))
        run = {"run": path.parent.name, "order": manifest["configuration"]["strategy_order"],
               "native_base_id": manifest["native_base_id"],
               "cross_backend_candidate_identity_equal": manifest["cross_backend_candidate_identity_equal"],
               "noise_layout_probe": read(path.parent / "noise-layout-probe.json"),
               "strategies": {}}
        for strategy in ("legacy-add-subtract", "snapshot-copy"):
            report = read(path.parent / (strategy + ".json"))
            rows, audit = report["rows"], report["correctness"]
            assert len(rows) == 12
            for repeat in range(3):
                assert [r["candidate_id"] for r in rows if r["repeat"] == repeat] == ids
            total = sum(r["total_wall_ms"] for r in rows)
            state_fraction = sum(r["apply"]["wall_ms"] + r["restore"]["wall_ms"] for r in rows) / total
            assert abs(state_fraction - report["summary"]["state_wall_fraction"]) < 1e-12
            lengths = [n for r in rows for n in r["generated_token_counts"]]
            finishes = [reason for r in rows for reason in r["finish_reasons"]]
            if manifest["configuration"]["fixed_length"]:
                assert set(lengths) == {manifest["configuration"]["max_new_tokens"]}
            run["strategies"][strategy] = {
                "candidate_evaluations": len(rows),
                "candidate_ms": total / len(rows),
                "candidates_per_second": len(rows) * 1000 / total,
                "state_fraction": state_fraction,
                "inference_fraction": sum(r["inference"]["wall_ms"] for r in rows) / total,
                "phase_mean_ms": {p: sum(r[p]["wall_ms"] for r in rows) / len(rows)
                                  for p in ("apply", "inference", "restore", "score")},
                "generated_length_mean": sum(lengths) / len(lengths),
                "generated_length_min": min(lengths), "generated_length_max": max(lengths),
                "length_capped_outputs": finishes.count("length"), "total_outputs": len(finishes),
                "timed_rewards": [r["reward"] for r in rows],
                "exact_semantics_gate": report["exact_semantics_gate"],
                "candidate_state_matches": sum(r["candidate_state_exact"] for r in audit["rows"]),
                "output_matches": sum(r["output_exact"] for r in audit["rows"]),
                "score_matches": sum(r["reward_exact"] for r in audit["rows"]),
                "audited_candidates": len(audit["rows"]),
                "max_peak_allocated_bytes": max(r["memory"]["peak_allocated_bytes"] for r in report["repeats"]),
                "max_peak_reserved_bytes": max(r["memory"]["peak_reserved_bytes"] for r in report["repeats"]),
                "post_trace_drift": [r["post_trace_drift"] for r in report["repeats"]]}
        group = path.parent.name.rsplit("-process-", 1)[0]
        groups.setdefault(group, []).append(run)
    if len(groups) != 3 or any(len(runs) != 2 for runs in groups.values()):
        raise ValueError("need all six independent processes before producing closure analysis")
    for runs in groups.values():
        assert {r["order"] for r in runs} == {"legacy-first", "snapshot-first"}
    assert len(native_traces) == len(source_traces) == 1
    assert next(iter(source_traces)) == tuple(r["candidate_id"] for r in read(baseline / "fixed-32-b4/candidates.json"))
    return {"scope": "one H100 / Qwen2.5-0.5B / original RandOpt worker and launcher / vLLM eager / Ray",
            "speedup_claim": False,
            "uncertainty": "two counterbalanced process invocations per workload; within-process repeats descriptive",
            "same_native_candidates_across_processes": True,
            "same_source_recipes_as_HF_pilot": True,
            "original_worker_HF_validation": hf_validation(root, baseline),
            "workloads": groups}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--baseline", type=Path, default=Path("results/llm-bounded-20261003"))
    args = p.parse_args()
    print(json.dumps(analyze(args.root, args.baseline), indent=2, allow_nan=False))

"""Summarize completed HF runs without treating candidate rows as replicates.

Usage: python scripts/summarize_llm_pilot.py runs/llm-pilot
Writes JSON to stdout; no PyTorch or model loading is needed.
"""
import argparse
import json
from pathlib import Path


def aggregate(rows):
    total = sum(row["total_wall_ms"] for row in rows)
    phases = {phase: sum(row[phase]["wall_ms"] for row in rows)
              for phase in ("apply", "inference", "restore", "score")}
    fraction = (phases["apply"] + phases["restore"]) / total
    return {
        "candidate_evaluations": len(rows),
        "mean_candidate_ms": total / len(rows),
        "candidates_per_second": 1000 * len(rows) / total,
        "phase_mean_ms": {phase: value / len(rows) for phase, value in phases.items()},
        "state_wall_fraction": fraction,
        "ideal_state_elimination_speedup_bound": 1 / (1 - fraction),
    }


def summarize(root):
    runs = []
    for path in sorted(root.glob("*/manifest.json")):
        manifest = json.loads(path.read_text())
        if manifest["status"] != "complete":
            raise ValueError(f"Incomplete run: {path.parent}")
        if manifest["configuration"]["workload"] != "hf":
            continue
        config = manifest["configuration"]
        workload = manifest["workload"]
        run = {
            "run": path.parent.name,
            "model": workload["model"], "revision": workload["revision"],
            "source": manifest["source"],
            "data_sha256": workload["data_sha256"],
            "batch": workload["examples"],
            "generation": workload["generation"],
            "attention": workload["attention"],
            "fixed_length_microbenchmark": workload["fixed_length_microbenchmark"],
            "cuda_events": not config["no_cuda_events"],
            "diagnostic": config["trace"],
            "strategies": {},
        }
        recipes = json.loads((path.parent / "candidates.json").read_text())
        run["candidate_ids"] = [recipe["candidate_id"] for recipe in recipes]
        for name in ("legacy-add-subtract", "snapshot-copy"):
            report_path = path.parent / (name + ".json")
            if not report_path.exists():
                if config["strategy"] in ("both", name):
                    raise ValueError(f"Missing strategy report: {report_path}")
                continue
            report = json.loads(report_path.read_text())
            rows = report["rows"]
            repeat_ids = sorted({row["repeat"] for row in rows})
            if repeat_ids != list(range(config["repeats"])):
                raise ValueError(f"Missing repeats: {report_path}")
            repeated = []
            for index in repeat_ids:
                selected = [row for row in rows if row["repeat"] == index]
                if [row["candidate_id"] for row in selected] != run["candidate_ids"]:
                    raise ValueError(f"Candidate trace mismatch: {report_path}")
                repeated.append({"repeat": index, **aggregate(selected)})
            audit = report["correctness"]
            run["strategies"][name] = {
                **aggregate(rows),
                "within_process_repeats": repeated,
                "exact_semantics_gate": report["exact_semantics_gate"],
                "output_exact_candidates": sum(row["output_exact"] for row in audit["rows"]),
                "reward_exact_candidates": sum(row["reward_exact"] for row in audit["rows"]),
                "audited_candidates": len(audit["rows"]),
                "all_resets_exact": audit["all_resets_exact"],
                "parameter_bytes": report["parameter_bytes"],
                "snapshot_parameter_bytes": report["snapshot_parameter_bytes"],
                "max_peak_allocated_bytes": max(r["memory"]["peak_allocated_bytes"]
                                                for r in report["repeats"]),
                "max_peak_reserved_bytes": max(r["memory"]["peak_reserved_bytes"]
                                               for r in report["repeats"]),
                "max_post_trace_changed_fraction": max(r["post_trace_drift"]["changed_fraction"]
                                                       for r in report["repeats"]),
                "max_post_trace_abs_drift": max(r["post_trace_drift"]["max_abs"]
                                                for r in report["repeats"]),
            }
        runs.append(run)
    if not runs:
        raise ValueError(f"No completed HF runs found in {root}")
    return {
        "scope": "Fixed-work HF lifecycle characterization; no kernel improvement claim",
        "uncertainty": "Repeat ranges are descriptive within-process variation, not independent-run confidence intervals",
        "runs": runs,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.root), indent=2, allow_nan=False))

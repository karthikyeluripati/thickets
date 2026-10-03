"""Descriptive report tables after locked validation; never selects a policy."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from thicket_runtime.agreement import distribution
from thicket_runtime.cli import revision_info, write_json


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def population_tables(matrix, summary, diagnostics, records, locked, ensemble=None):
    n = matrix["selection_count"]
    rewards = np.array([row["rewards"][:n] for row in matrix["rows"]])
    tokens = np.array([row["tokens"][:n] for row in matrix["rows"]])
    scores = rewards.mean(axis=1)
    truth = diagnostics["full_ranking"]
    ids = [row["candidate"]["candidate_id"] for row in matrix["rows"]]
    prompts = diagnostics["prompt_heterogeneity"]
    full = {"candidates": len(ids), "selection_pairs": int(rewards.size),
            "selection_tokens": int(tokens.sum()),
            "selection_capped_fraction": float(np.mean([row["capped"][:n] for row in matrix["rows"]])),
            "heldout_capped_fraction": float(np.mean([row["capped"][n:] for row in matrix["rows"]])),
            "distinct_reward_rows": len({tuple(row) for row in rewards}),
            "score_distribution": diagnostics["full_score_distribution"],
            "best_score": float(scores[truth[0]]), "best_ties": int(np.sum(scores == scores.max())),
            "top10_cutoff": diagnostics["tie_cutoff"],
            "top10_boundary_ties": diagnostics["top10_boundary_ties"],
            "top10_mean": float(scores[truth[:10]].mean()),
            "top10_sigma_counts": dict(Counter(str(matrix["rows"][i]["candidate"]["sigma"]) for i in truth[:10])),
            "per_sigma_scores": {}}
    for sigma in sorted({row["candidate"]["sigma"] for row in matrix["rows"]}):
        subset = [i for i, row in enumerate(matrix["rows"]) if row["candidate"]["sigma"] == sigma]
        full["per_sigma_scores"][str(sigma)] = distribution(scores[subset].tolist())
    heterogeneity = {"constant_prompts": sum(p["variance"] == 0 for p in prompts),
        "never_correct_prompts": sum(p["mean_reward"] == 0 for p in prompts),
        "always_correct_prompts": sum(p["mean_reward"] == 1 for p in prompts),
        "rare_correct_prompts": sum(0 < p["mean_reward"] <= .1 for p in prompts),
        "base_accuracy": float(np.mean([p["base_correct"] for p in prompts])),
        **{key: distribution(p[key] for p in prompts) for key in
           ("variance", "correlation_full", "correlation_leave_one_prompt_out", "top10_enrichment")},
        "highest_variance_prompt_ids": [p["prompt_id"] for p in sorted(prompts, key=lambda p: (-p["variance"], p["prompt_id"]))[:10]]}
    specialists = []
    for spec in diagnostics["top10_specialists"]:
        index = ids.index(spec["candidate_id"])
        output = {**spec, "candidate": matrix["rows"][index]["candidate"], "misses": {}}
        for policy in sorted({"subset-032", "subset-080", locked}):
            trials = [r for r in records if r["policy_id"] == policy]
            output["misses"][policy] = {
                "not_selected_fraction": float(np.mean([index not in r["result"]["ranking"][:10] for r in trials])),
                "outside_top50_fraction": float(np.mean([index not in r["result"]["ranking"][:50] for r in trials])),
                "eliminated_fraction": float(np.mean([index in r["result"]["eliminated"] for r in trials]))}
        specialists.append(output)
    flips = {}
    for a, b in sorted({(r["from"], r["to"]) for r in diagnostics["ranking_flips"]}):
        rows = [r for r in diagnostics["ranking_flips"] if (r["from"], r["to"]) == (a, b)]
        flips[f"{a}-{b}"] = {key: distribution(r[key] for r in rows)
                              for key in ("strict_pair_reversal_fraction", "top10_churn")}
    compact = {}
    for policy, item in summary.items():
        compact[policy] = {"pair_fraction": item["pair_fraction"]["mean"],
            "token_fraction": item["token_fraction"]["mean"],
            "joint_success_fraction": item["joint_success_fraction"],
            "spearman": item["spearman"].get("mean"), "kendall_tau_b": item["kendall_tau_b"].get("mean"),
            "top1_recovery": item["top1_recovery_fraction"],
            "mean_best_regret": item["best_regret"]["mean"], "max_best_regret": item["best_regret"]["max"],
            "top10_recall": item["topk"]["10"]["recall"]["mean"],
            "top10_tie_aware_recall": item["topk"]["10"]["tie_aware_recall"]["mean"],
            "selected_top10_mean": item["topk"]["10"]["selected_topk_mean"]["mean"],
            "top10_false_eliminations": item["topk"]["10"]["false_eliminations"]["mean"],
            "top10_miss_outside_top20": item["topk"]["10"]["miss_outside_top2k"]["mean"],
            "top10_miss_outside_top50": item["topk"]["10"]["miss_outside_top5k"]["mean"]}
    output = {"full_reference": full, "policies": compact, "prompt_heterogeneity": heterogeneity,
              "top10_specialists": specialists, "ranking_flips": flips}
    if ensemble:
        output["ensemble"] = {}
        for k, reference in ensemble["full_selected"].items():
            trials = [row["topk"][k] for row in ensemble["trials"]]
            output["ensemble"][k] = {"full_selected_accuracy": reference["accuracy"],
                "partial_selected_accuracy": distribution(r["accuracy"] for r in trials),
                "partial_minus_full_accuracy": distribution(r["accuracy"] - reference["accuracy"] for r in trials),
                "same_correctness_vector_fraction": float(np.mean([r["correct"] == reference["correct"] for r in trials]))}
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--development", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    gate = read(args.validation / "gate.json")  # Require completed locked validation first.
    lock = read(args.development / "policy-lock.json")
    if gate["lock_sha256"] != hashlib.sha256((args.development / "policy-lock.json").read_bytes()).hexdigest():
        raise ValueError("validation/lock mismatch")
    result = {"source": revision_info(), "command": [sys.executable, *sys.argv], "gate": gate,
              "scope": "descriptive only, after locked validation; no policy reselection", "populations": {}}
    costs, memories = [], []
    for population in ("development", "validation_a", "validation_b"):
        matrix = read(args.run / ("matrix-" + population + ".json"))
        folder = args.development if population == "development" else args.validation
        prefix = "" if population == "development" else population + "."
        summary = read(folder / (prefix + "summary.json"))
        diag = read(folder / (prefix + "diagnostics.json"))
        records = read(folder / (prefix + "trials.json.gz"))
        ensemble = None if population == "development" else read(folder / (prefix + "ensemble.json"))
        result["populations"][population] = population_tables(matrix, summary, diag, records, lock["chosen_policy_id"], ensemble)
        for row in matrix["rows"]:
            raw = read(args.run / population / (row["candidate"]["candidate_id"] + ".json.gz"))
            if not raw["restoration"]["exact_base"]:
                raise ValueError("inexact exhaustive reference")
            c = raw["descriptive_costs"]
            values = {"apply_seconds": c["apply_seconds"], "restore_seconds": c["restore_seconds"],
                      **c["workloads"]["matrix240"]}
            values["lifecycle_seconds"] = sum(values.values())
            values["state_fraction"] = (values["apply_seconds"] + values["restore_seconds"]) / values["lifecycle_seconds"]
            costs.append(values)
            memories.append(raw["memory"])
    result["collection_costs"] = {key: distribution(c[key] for c in costs) for key in costs[0]}
    result["collection_costs"]["note"] = "240-prompt batches including heldout40; within-process descriptive timings, excludes audits/setup"
    result["collection_memory_bytes"] = {key: distribution(m[key] for m in memories) for key in memories[0]}
    write_json(args.out, result)
    print(json.dumps({"report_tables": str(args.out), "gate_pass": gate["pass"]}))


if __name__ == "__main__":
    main()

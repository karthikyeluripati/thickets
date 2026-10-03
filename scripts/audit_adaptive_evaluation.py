"""Independently reconcile saved policy traces, budgets, and joint gate outcomes."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from thicket_runtime.cli import revision_info, write_json


def read(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    root = args.root
    lock = read(root / "development/policy-lock.json")
    gate = read(root / "validation/gate.json")
    assert gate["lock_sha256"] == hashlib.sha256((root / "development/policy-lock.json").read_bytes()).hexdigest()
    output = {"source": revision_info(), "command": [sys.executable, *sys.argv],
              "scope": "post-validation accounting audit; quality-only diagnostics do not change the gate", "populations": {}}
    for pop in ("development", "validation_a", "validation_b"):
        folder, prefix = (root / "development", "") if pop == "development" else (root / "validation", pop + ".")
        trials = read(folder / (prefix + "trials.json.gz"))
        matrix = read(root / "matrix-01" / ("matrix-" + pop + ".json"))
        n = matrix["selection_count"]
        tokens = np.array([row["tokens"][:n] for row in matrix["rows"]])
        scores = np.array([sum(row["rewards"][:n]) / n for row in matrix["rows"]])
        keys = [hashlib.sha256(("rank-tie-v1:" + row["candidate"]["candidate_id"]).encode()).hexdigest() for row in matrix["rows"]]
        gold = sorted(range(len(scores)), key=lambda i: (-scores[i], keys[i]))[:10]
        components = {}
        for trial in trials:
            r, m = trial["result"], trial["metrics"]
            counts = np.array(r["counts"])
            assert np.all((counts > 0) & (counts <= n))
            assert int(counts.sum()) == m["evaluations"]
            assert sorted(r["ranking"]) == list(range(len(scores)))
            # The saved per-candidate counts and shared prompt order encode the
            # complete observation mask without storing a redundant dense array.
            exact_tokens = sum(int(tokens[i, trial["prompt_order"][:count]].sum()) for i, count in enumerate(counts))
            assert exact_tokens == m["generated_tokens"]
            cost = int(counts.sum()) / tokens.size
            regret = float(scores.max() - scores[r["ranking"][0]])
            recall = len(set(gold) & set(r["ranking"][:10])) / 10
            assert abs(cost - m["pair_fraction"]) < 1e-12
            assert abs(exact_tokens / tokens.sum() - m["token_fraction"]) < 1e-12
            assert abs(regret - m["best_regret"]) < 1e-12
            assert recall == m["topk"]["10"]["recall"]
            checks = {"cost_pass": cost <= .4 + 1e-12, "recall_pass": recall >= .9 - 1e-12,
                      "regret_pass": regret <= .01 + 1e-12}
            checks["quality_only_pass"] = checks["recall_pass"] and checks["regret_pass"]
            checks["joint_pass"] = checks["cost_pass"] and checks["quality_only_pass"]
            assert checks["joint_pass"] == trial["joint_gate_success"]
            components.setdefault(trial["policy_id"], []).append(checks)
        stats = {policy: {key + "_fraction": sum(row[key] for row in rows) / len(rows) for key in rows[0]}
                 for policy, rows in components.items()}
        output["populations"][pop] = {"audited_trials": len(trials), "policy_gate_components": stats}
        if pop != "development":
            assert stats[lock["chosen_policy_id"]]["joint_pass_fraction"] == gate["populations"][pop]["joint_success_fraction"]
    output["all_accounting_checks_pass"] = True
    write_json(args.out, output)
    print(json.dumps({"audited_trials": sum(p["audited_trials"] for p in output["populations"].values()), "all_checks_pass": True}))


if __name__ == "__main__":
    main()

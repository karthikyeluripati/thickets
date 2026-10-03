"""Development/locked validation of simple policies on the exhaustive matrix."""
import argparse
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from thicket_runtime.agreement import distribution
from thicket_runtime.candidate import CandidateSpec
from thicket_runtime.cli import revision_info, write_json
from thicket_runtime.evaluation_matrix import reward_and_extractor, write_gzip
from thicket_runtime.partial_evaluation import (MatrixOracle, correlation, evaluate, joint_success,
                                               order_scores, run_policy, tie_keys, vote_answers)


def read(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_population(root, population, reward, extract):
    """Only the named population is parsed, including in development mode."""
    checksums = read(root / "sha256.json")
    file = root / ("matrix-" + population + ".json")
    if digest(file) != checksums[file.name]:
        raise ValueError("matrix checksum mismatch")
    matrix = read(file)
    protocol = read(root / "protocol.json")
    expected_count = read(root / "manifest.json")["candidates_per_population"][population]
    if len(matrix["rows"]) != expected_count or expected_count % len(protocol["sigmas"]):
        raise ValueError("matrix population is not a complete balanced seed sweep")
    expected_pairs = [(protocol["populations"][population] + i, sigma)
                      for i in range(expected_count // len(protocol["sigmas"])) for sigma in protocol["sigmas"]]
    if [(row["candidate"]["seed"], row["candidate"]["sigma"]) for row in matrix["rows"]] != expected_pairs:
        raise ValueError("matrix differs from predeclared candidate split/order")
    for row in matrix["rows"]:
        c = row["candidate"]
        spec = CandidateSpec(**{key: c[key] for key in ("base_id", "seed", "sigma", "rng", "sign")})
        if spec.candidate_id != c["candidate_id"] or c["population"] != population or c["base_id"] != matrix["base_id"]:
            raise ValueError("candidate identity/split mismatch")
        raw_path = root / population / (c["candidate_id"] + ".json.gz")
        if digest(raw_path) != checksums[raw_path.relative_to(root).as_posix()]:
            raise ValueError("raw trace checksum mismatch")
        raw = read(raw_path)
        if raw["candidate"] != c or raw["candidate_state_id"] != row["state_id"] or not raw["restoration"]["exact_base"]:
            raise ValueError("raw state audit mismatch")
        outputs = raw["outputs"]
        if len(outputs) != len(matrix["prompt_ids"]):
            raise ValueError("incomplete reference row")
        if [o["prompt_id"] for o in outputs] != matrix["prompt_ids"]:
            raise ValueError("prompt identity mismatch")
        if ([o["reward"] for o in outputs] != row["rewards"]
                or [len(o["token_ids"]) for o in outputs] != row["tokens"]
                or [o["voting_answer"] for o in outputs] != row["voting_answers"]
                or [o["finish_reason"] == "length" for o in outputs] != row["capped"]):
            raise ValueError("raw trace/matrix mismatch")
        for i, output in enumerate(outputs):
            if reward(output["text"], matrix["answers"][i]) != output["reward"] or extract(output["text"]) != output["voting_answer"]:
                raise ValueError("pinned reward/extraction mismatch")
    n = matrix["selection_count"]
    r = np.array([row["rewards"][:n] for row in matrix["rows"]])
    if n != protocol["selection_prompts"] or r.shape != (expected_count, n) or not np.isin(r, [0, 1]).all():
        raise ValueError("expected complete binary GSM8K selection matrix")
    return matrix, r, np.array([row["tokens"][:n] for row in matrix["rows"]])


def registry(protocol, wilson=False):
    policies = {}
    for m in protocol["prompt_budgets"]:
        policies[f"subset-{m:03d}"] = {"family": "subset", "prompts": m}
        policies[f"prefix-{m:03d}"] = {"family": "prefix", "prompts": m}
    for m in protocol["baseline_policies"]["successive_halving_initial"]:
        policies[f"halving-{m:03d}"] = {"family": "halving", "initial": m}
    policies["hoeffding-0.05"] = {"family": "hoeffding", "parameter": .05}
    if wilson:
        for z in protocol["exploratory_wilson_z"]:
            policies[f"wilson-{z:g}"] = {"family": "wilson", "parameter": z}
    return policies


def simulate(matrix, rewards, tokens, policies, order_seeds, protocol):
    ids = [r["candidate"]["candidate_id"] for r in matrix["rows"]]
    records = []
    for name, policy in policies.items():
        seeds = [None] if policy["family"] == "prefix" else order_seeds
        for seed in seeds:
            order = (list(range(rewards.shape[1])) if seed is None else
                     np.random.default_rng(seed).permutation(rewards.shape[1]).tolist())
            oracle = MatrixOracle(rewards, tokens)
            result = run_policy(oracle.observe, rewards.shape, ids, order, policy, k=protocol["primary_k"],
                                race_stages=protocol["racing_stages"])
            metrics = evaluate(result, oracle, rewards, ids, protocol["report_k"])
            records.append({"policy_id": name, "policy": policy, "order_seed": seed, "prompt_order": order,
                            "result": result, "metrics": metrics,
                            "joint_gate_success": joint_success(metrics, protocol["feasibility_gate"])})
    return records


def summarize(records):
    output = {}
    for name in sorted({r["policy_id"] for r in records}):
        group = [r for r in records if r["policy_id"] == name]
        metric_names = ("pair_fraction", "token_fraction", "best_regret", "selected_best_full_reward",
                        "spearman", "kendall_tau_b", "score_rmse", "mean_absolute_score_error")
        output[name] = {"orders": len(group), "joint_success_fraction": float(np.mean([r["joint_gate_success"] for r in group])),
                        "top1_recovery_fraction": float(np.mean([r["metrics"]["top1_recovery"] for r in group])),
                        **{key: distribution(r["metrics"][key] for r in group) for key in metric_names}, "topk": {}}
        for k in group[0]["metrics"]["topk"]:
            output[name]["topk"][k] = {key: distribution(r["metrics"]["topk"][k][key] for r in group)
                                       for key in group[0]["metrics"]["topk"][k]}
    return output


def phase4_signal(summary, protocol):
    gate = protocol["phase4_signal_gate"]
    checks = []
    for m in protocol["prompt_budgets"]:
        if m > gate["maximum_prompts"]:
            continue
        s = summary[f"subset-{m:03d}"]
        correlation_mean = s["spearman"].get("mean")
        miss = s["topk"]["10"]["miss_outside_top5k"]["mean"]
        checks.append({"prompts": m, "spearman": correlation_mean, "top10_miss_outside_top50": miss,
                       "pass": correlation_mean is not None and correlation_mean >= gate["minimum_mean_spearman"]
                               and miss <= gate["maximum_mean_top10_miss_outside_top50"]})
    return {"pass": any(c["pass"] for c in checks), "checks": checks}


def choose_policy(summary, protocol):
    threshold = protocol["feasibility_gate"]["minimum_joint_order_success_fraction"]
    qualified = [name for name, s in summary.items() if not name.startswith("prefix-")
                 and s["joint_success_fraction"] + 1e-12 >= threshold]
    if qualified:
        name = min(qualified, key=lambda n: (summary[n]["pair_fraction"]["mean"],
                    -summary[n]["topk"]["10"]["recall"]["mean"], summary[n]["best_regret"]["mean"], n))
    else:
        allowed = [name for name, s in summary.items() if not name.startswith("prefix-")
                   and s["pair_fraction"]["mean"] <= protocol["feasibility_gate"]["maximum_pair_fraction"] + 1e-12]
        name = min(allowed, key=lambda n: (-summary[n]["joint_success_fraction"],
                    -summary[n]["topk"]["10"]["recall"]["mean"], summary[n]["best_regret"]["mean"],
                    summary[n]["pair_fraction"]["mean"], n))
    return name, bool(qualified), qualified


def diagnostics(matrix, rewards, records, base):
    ids = [r["candidate"]["candidate_id"] for r in matrix["rows"]]
    keys = tie_keys(ids)
    full = rewards.mean(axis=1)
    truth = order_scores(full, keys)
    top = truth[:10]
    n, p = rewards.shape
    prevalence = rewards.mean(axis=0)
    prompts = []
    for j in range(p):
        other_scores = (rewards.sum(axis=1) - rewards[:, j]) / (p - 1)
        prompts.append({"prompt_id": matrix["prompt_ids"][j], "column": j,
                        "mean_reward": float(prevalence[j]), "variance": float(rewards[:, j].var()),
                        "correlation_full": correlation(rewards[:, j], full),
                        "correlation_leave_one_prompt_out": correlation(rewards[:, j], other_scores),
                        "top10_enrichment": float(rewards[top, j].mean() - np.delete(rewards[:, j], top).mean()),
                        "base_correct": base[j]["reward"]})
    subsets = [r for r in records if r["policy"]["family"] == "subset"]
    by_budget = {}
    for m in sorted({r["policy"]["prompts"] for r in subsets}):
        rows = [r for r in subsets if r["policy"]["prompts"] == m]
        observed = np.array([r["result"]["observed_means"] for r in rows])
        by_budget[str(m)] = {"empirical_candidate_score_sd": distribution(observed.std(axis=0, ddof=1).tolist()),
            "finite_set_sampling_se": distribution(np.sqrt(full * (1 - full) / m * (p - m) / (p - 1)).tolist()),
            "absolute_score_error": distribution(np.abs(observed - full).ravel().tolist()),
            "top10_candidates": [{"candidate_id": ids[i],
                "inclusion_probability": float(np.mean([i in r["result"]["ranking"][:10] for r in rows])),
                "outside_top20_probability": float(np.mean([i not in r["result"]["ranking"][:20] for r in rows])),
                "outside_top50_probability": float(np.mean([i not in r["result"]["ranking"][:50] for r in rows]))} for i in top]}
    flips = []
    i, j = np.triu_indices(n, 1)
    for seed in sorted({r["order_seed"] for r in subsets}):
        sequence = sorted((r for r in subsets if r["order_seed"] == seed), key=lambda r: r["policy"]["prompts"])
        for a, b in zip(sequence, sequence[1:]):
            x, y = np.array(a["result"]["observed_means"]), np.array(b["result"]["observed_means"])
            sx, sy = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
            comparable = np.count_nonzero(sx * sy)
            flips.append({"order_seed": seed, "from": a["policy"]["prompts"], "to": b["policy"]["prompts"],
                          "strict_pair_reversal_fraction": float(np.count_nonzero(sx * sy < 0) / comparable) if comparable else None,
                          "top10_churn": 1 - len(set(a["result"]["ranking"][:10]) & set(b["result"]["ranking"][:10])) / 10})
    specialists = []
    for index in top:
        correct = rewards[index] > 0
        rare = np.flatnonzero(correct & (prevalence <= .1))
        unique = np.flatnonzero(correct & (rewards.sum(axis=0) == 1))
        specialists.append({"candidate_id": ids[index], "full_score": float(full[index]),
            "rare_correct_prompt_ids": [matrix["prompt_ids"][j] for j in rare],
            "unique_correct_prompt_ids": [matrix["prompt_ids"][j] for j in unique],
            "late_quartile_correct_count": int(correct[3*p//4:].sum()),
            "rare_in_last_quartile": int(np.count_nonzero(rare >= 3*p//4)),
            "correct_count": int(correct.sum())})
    return {"prompt_heterogeneity": prompts, "score_uncertainty": by_budget, "ranking_flips": flips,
            "top10_specialists": specialists, "full_ranking": truth,
            "full_score_distribution": distribution(full.tolist()),
            "tie_cutoff": float(full[truth[9]]), "top10_boundary_ties": int(np.sum(full == full[truth[9]]))}


def ensemble(matrix, records, reward, ks):
    n = matrix["selection_count"]
    r = np.array([row["rewards"][:n] for row in matrix["rows"]])
    ids = [row["candidate"]["candidate_id"] for row in matrix["rows"]]
    truth = order_scores(r.mean(axis=1), tie_keys(ids))
    answers = [row["voting_answers"][n:] for row in matrix["rows"]]
    gold = matrix["answers"][n:]
    def score(ranking, k):
        votes = vote_answers(answers, ranking, k)
        correctness = [bool(a and reward("#### " + a, expected) > 0) for a, expected in zip(votes, gold)]
        return {"accuracy": float(np.mean(correctness)), "votes": votes, "correct": correctness}
    full = {str(k): score(truth, k) for k in ks}
    trials = [{"policy_id": rec["policy_id"], "order_seed": rec["order_seed"],
               "topk": {str(k): score(rec["result"]["ranking"], k) for k in ks}} for rec in records]
    return {"prompts": len(gold), "full_selected": full, "trials": trials,
            "scope": "disjoint test prompts; majority vote after frozen candidate selection; no quality tuning"}


def require_committed_lock(path):
    relative = path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    committed = subprocess.check_output(["git", "show", "HEAD:" + relative])
    if committed != path.read_bytes():
        raise ValueError("policy lock must be committed byte-for-byte before validation")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["develop", "validate"])
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--upstream-root", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--lock", type=Path)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    checksums = read(args.run / "sha256.json")
    for name in ("protocol.json", "manifest.json", "controls.json", "base.json.gz", "provenance.json"):
        if digest(args.run / name) != checksums[name]:
            raise ValueError("reference metadata checksum mismatch: " + name)
    protocol = read(args.run / "protocol.json")
    manifest = read(args.run / "manifest.json")
    controls = read(args.run / "controls.json")
    if manifest["status"] != "complete" or not all(controls.values()):
        raise ValueError("complete and correct exhaustive reference required")
    reward, extract = reward_and_extractor(args.upstream_root)
    base = read(args.run / "base.json.gz")["outputs"]
    args.out.mkdir(parents=True)
    write_json(args.out / "analysis-manifest.json", {"phase": args.phase, "source": revision_info(),
        "command": [sys.executable, *sys.argv], "run_manifest_sha256": digest(args.run / "manifest.json"),
        "protocol_sha256": digest(args.run / "protocol.json"), "note": "offline budget reductions only",
        "versions": {"python": sys.version, "numpy": np.__version__,
                     "torch": importlib.metadata.version("torch"),
                     "matplotlib": importlib.metadata.version("matplotlib")}})
    if args.phase == "develop":
        matrix, r, t = load_population(args.run, "development", reward, extract)
        policies = registry(protocol)
        records = simulate(matrix, r, t, policies, protocol["development_order_seeds"], protocol)
        signal = phase4_signal(summarize(records), protocol)
        if signal["pass"]:
            additional = {k: v for k, v in registry(protocol, True).items() if k not in policies}
            records += simulate(matrix, r, t, additional, protocol["development_order_seeds"], protocol)
            policies.update(additional)
        summary = summarize(records)
        chosen, qualified, qualifiers = choose_policy(summary, protocol)
        lock = {"schema": "development-policy-lock-v1", "chosen_policy_id": chosen, "chosen_policy": policies[chosen],
                "development_qualified": qualified, "development_qualifying_policies": qualifiers,
                "signal_gate": signal, "policy_registry": policies,
                "protocol_sha256": digest(args.run / "protocol.json"),
                "development_matrix_sha256": digest(args.run / "matrix-development.json"),
                "development_candidate_ids": [row["candidate"]["candidate_id"] for row in matrix["rows"]],
                "selection_rule": protocol["development_selection"], "source": revision_info(),
                "validation_inspected": False}
        write_json(args.out / "policy-lock.json", lock)
        write_gzip(args.out / "trials.json.gz", records)
        write_json(args.out / "summary.json", summary)
        write_json(args.out / "diagnostics.json", diagnostics(matrix, r, records, base))
        # Within-sigma diagnostics expose how much pooled ranking is just scale separation.
        within = {}
        for sigma in protocol["sigmas"]:
            indices = [i for i, row in enumerate(matrix["rows"]) if row["candidate"]["sigma"] == sigma]
            sub = dict(matrix, rows=[matrix["rows"][i] for i in indices])
            subset_policies = {k: v for k, v in policies.items() if v["family"] == "subset"}
            within[str(sigma)] = summarize(simulate(sub, r[indices], t[indices], subset_policies,
                                                  protocol["development_order_seeds"], protocol))
        write_json(args.out / "within-sigma.json", within)
        print(json.dumps({"locked_policy": chosen, "development_qualified": qualified, "phase4_signal": signal["pass"]}))
    else:
        if args.lock is None:
            p.error("validation requires --lock")
        require_committed_lock(args.lock)
        lock = read(args.lock)
        if lock["protocol_sha256"] != digest(args.run / "protocol.json") or lock["development_matrix_sha256"] != digest(args.run / "matrix-development.json"):
            raise ValueError("lock/reference mismatch")
        seeds = list(range(protocol["validation_order_seed_start"], protocol["validation_order_seed_start"] + protocol["validation_order_count"]))
        gate = {"development_qualified": lock["development_qualified"], "locked_policy": lock["chosen_policy_id"],
                "lock_sha256": digest(args.lock), "populations": {}, "actual_speedup_measured": False}
        for population in ("validation_a", "validation_b"):
            matrix, r, t = load_population(args.run, population, reward, extract)
            if set(lock["development_candidate_ids"]) & {row["candidate"]["candidate_id"] for row in matrix["rows"]}:
                raise ValueError("development/validation leakage")
            records = simulate(matrix, r, t, lock["policy_registry"], seeds, protocol)
            summary = summarize(records)
            selected = [rec for rec in records if rec["policy_id"] == lock["chosen_policy_id"]]
            write_gzip(args.out / (population + ".trials.json.gz"), records)
            write_json(args.out / (population + ".summary.json"), summary)
            write_json(args.out / (population + ".diagnostics.json"), diagnostics(matrix, r, records, base))
            write_json(args.out / (population + ".ensemble.json"), ensemble(matrix, selected, reward, protocol["report_k"]))
            locked_summary = summary[lock["chosen_policy_id"]]
            gate["populations"][population] = {"joint_success_fraction": locked_summary["joint_success_fraction"],
                "pass": locked_summary["joint_success_fraction"] + 1e-12 >= protocol["feasibility_gate"]["minimum_joint_order_success_fraction"],
                "locked_policy_summary": locked_summary}
        gate["pass"] = bool(gate["development_qualified"] and all(x["pass"] for x in gate["populations"].values()))
        gate["decision"] = "headroom_for_further_research" if gate["pass"] else "stop_no_novel_method"
        write_json(args.out / "gate.json", gate)
        print(json.dumps({"gate_pass": gate["pass"], "decision": gate["decision"]}))


if __name__ == "__main__":
    main()

"""Recompute agreement/simulation from immutable raw tokens; write a new audit.

No GPU or model is used. This refuses existing output and checks every captured
checksum, identity, per-prompt metric, aggregate and frozen gate against raw data.
"""
import argparse
import hashlib
import json
from pathlib import Path

from thicket_runtime.agreement import feasibility_gate, summarize_records
from thicket_runtime.candidate import CandidateSpec
from thicket_runtime.cli import write_json
from thicket_runtime.shared_speculative import load_reward, make_pairs, natural_reward


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate(root, upstream_root):
    checksums = read(root / "sha256.json")
    for name, expected in checksums.items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"artifact checksum mismatch: {name}")
    protocol, manifest = read(root / "protocol.json"), read(root / "manifest.json")
    controls, base = read(root / "controls.json"), read(root / "base.json")["outputs"]
    if manifest["status"] != "complete" or not all(controls.values()):
        raise ValueError("run is incomplete or has invalid correctness controls")
    planned = read(root / "planned-candidates.json")
    count = manifest["completed_candidates"]
    if count % len(protocol["sigmas"]):
        raise ValueError("unbalanced seed/sigma sweep")
    candidate_files = list((root / "candidates").glob("*.json"))
    if len(candidate_files) != count:
        raise ValueError("candidate count does not match manifest")
    inputs = {s["name"]: [json.loads(line) for line in
                          (root / (s["name"] + ".inputs.jsonl")).read_text(encoding="utf-8").splitlines()]
              for s in protocol["workloads"]}
    gsm_reward = load_reward(upstream_root)
    scorers = {s["name"]: gsm_reward if s["scorer"].startswith("pinned-randopt") else natural_reward
               for s in protocol["workloads"]}
    for name, outputs in base.items():
        for row, item in zip(outputs, inputs[name]):
            if row["reward"] != scorers[name](row["text"], str(item["answer"])):
                raise ValueError("base reward differs from rescored raw output")
    records = []
    for expected in planned[:count]:
        record = read(root / "candidates" / (expected["candidate_id"] + ".json"))
        c = record["candidate"]
        spec = CandidateSpec(**{k: c[k] for k in ("base_id", "seed", "sigma", "rng", "sign")})
        if c != expected or spec.candidate_id != c["candidate_id"] or c["base_id"] != manifest["native_base_id"]:
            raise ValueError("candidate identity mismatch")
        if not record["restoration"]["exact_base"]:
            raise ValueError("candidate failed restoration")
        for name, rows in record["workloads"].items():
            outputs = [r["candidate_output"] for r in rows]
            for row, item in zip(outputs, inputs[name]):
                if row["reward"] != scorers[name](row["text"], str(item["answer"])):
                    raise ValueError("candidate reward differs from rescored raw output")
            if make_pairs(base[name], outputs, inputs[name]) != rows:
                raise ValueError("derived per-prompt agreement/simulation differs from tokens")
        records.append(record)
    summary = summarize_records(records)
    if summary != read(root / "summary.json"):
        raise ValueError("derived aggregate differs from raw-token summary")
    gate = feasibility_gate(summary, protocol, controls_passed=True, complete=True)
    if gate != read(root / "gate.json"):
        raise ValueError("derived gate differs from archived gate")
    return protocol, manifest, summary, gate, records, len(checksums)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--upstream-root", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    protocol, manifest, summary, gate, records, checked = validate(args.run, args.upstream_root)
    args.out.mkdir(parents=True)
    details = {"run": str(args.run), "checksummed_files_verified": checked,
               "raw_token_metrics_reproduced": True, "candidate_identities_valid": True,
               "all_base_and_candidate_rewards_reproduced_from_text": True,
               "aggregate_and_gate_reproduced": True, "gate": gate,
               "source_commit": manifest["source"]["commit"],
               "candidates": len(records), "seeds_per_sigma": manifest["completed_seeds_per_sigma"],
               "peak_allocated_bytes": max(r["memory"]["peak_allocated_bytes"] for r in records),
               "peak_reserved_bytes": max(r["memory"]["peak_reserved_bytes"] for r in records),
               "workloads": {}}
    old_path = Path("results/work1-closure-20261003/vllm-matrix/natural-128-b8-process-1/snapshot-copy.json")
    if old_path.exists():
        by_id = {r["candidate"]["candidate_id"]: r for r in records}
        comparisons = []
        for old in read(old_path)["correctness"]["rows"]:
            new = by_id[old["candidate_id"]]
            comparisons.append({"candidate_id": old["candidate_id"],
                                "same_state": new["candidate_state_id"] == old["reference_state_sha256"],
                                "same_natural_tokens": [r["candidate_output"]["token_ids"] for r in new["workloads"]["natural8"]]
                                    == [r["token_ids"] for r in old["reference_outputs"]]})
        details["work1_cross_run_controls"] = comparisons
    lines = ["# Independent raw-token reanalysis", "",
             f"Validated {checked} checksummed files and all {len(records)} candidate identities.",
             "Recomputed all per-prompt metrics, simulations, aggregates and the frozen gate.", "",
             "Quantiles below are pooled descriptive prompt/candidate distributions, not iid estimates.", "",
             "| Workload | Sigma | LCP p10 / median / p90 | Mean LCP | Mean LCP fraction | Exact observed outputs | Immediate divergence | Candidate capped |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    def pct(value):
        return "n/a" if value is None else f"{100 * value:.2f}%"
    for name, scales in summary.items():
        details["workloads"][name] = {}
        for sigma, s in scales.items():
            q = s["common_prefix_tokens"]
            lines.append(f"| {name} | {sigma} | {q['p10']:g} / {q['median']:g} / {q['p90']:g} | {q['mean']:.2f} | {pct(s['common_prefix_fraction']['mean'])} | {pct(s['observed_output_equality_fraction'])} | {pct(s['immediate_divergence_fraction'])} | {pct(s['candidate_cap_fraction'])} |")
            details["workloads"][name][sigma] = {
                "base_reward": s["base_reward"]["mean"],
                "candidate_reward": s["candidate_reward"]["mean"],
                "reward_prefix_spearman": s["reward_prefix_spearman"],
                "above_base_candidates": s["above_base_candidates"],
                "top_quartile_candidates_with_ties": s["high_reward_candidates"],
                "top_quartile_prefix_mean": s["high_reward"]["common_prefix_tokens"]["mean"],
                "top_quartile_prefix_median": s["high_reward"]["common_prefix_tokens"]["median"],
                "above_base_prefix_mean": s["above_base"].get("common_prefix_tokens", {}).get("mean"),
                "above_base_prefix_median": s["above_base"].get("common_prefix_tokens", {}).get("median")}
    lines += ["", "| Workload | Sigma | Initial k=2 / 4 / 8 / 16 match (all pairs) | Ideal round reduction k=2 / 4 / 8 / 16 |",
              "|---|---:|---:|---:|"]
    for name, scales in summary.items():
        for sigma, s in scales.items():
            acceptance = " / ".join(pct(s["first_k"][str(k)]["fraction_all_pairs"]) for k in (2, 4, 8, 16))
            reduction = " / ".join(pct(s["simulation"][str(k)]["token_weighted_round_reduction"]) for k in (2, 4, 8, 16))
            lines.append(f"| {name} | {sigma} | {acceptance} | {reduction} |")
    lines += ["", "Initial acceptance is distinct from conditional later blocks attempted before rejection.",
              "The complete distributions, eligible denominators, conditional block acceptance and wasted",
              "verification positions remain in the immutable summary and per-candidate raw artifacts.",
              "No rejoining after rejection is assumed. These simulated reductions are not measured speedups.",
              "", f"Frozen gate: **{gate['status']}**.", ""]
    write_json(args.out / "audit.json", details)
    (args.out / "overview.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"verified_files": checked, "candidates": len(records), "gate": gate["status"]}))


if __name__ == "__main__":
    main()

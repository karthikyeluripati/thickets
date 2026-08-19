"""Paired seed-level analysis for the 2-cell vision_late split experiment (vision_late_a vs
vision_late_b, both r=0.04, N=100), reusing the exact same statistical methodology as
paired_seed_comparison.py (imported directly, not reimplemented) -- exact two-sided McNemar,
95% percentile bootstrap CI on the paired mean delta difference (fixed seed, 10000 resamples),
and Spearman rank correlation of per-seed deltas.

Refuses to run unless validate_vision_late_split_2.py passes first. Analysis only -- reads
existing thicket_metrics.json candidate_records, runs no inference, applies no new
perturbation, modifies no scientific/experiment code.

Also shows the already-validated parent vision_late@r=.04 cell (density=.93, from the earlier
vision-localization sweep) alongside vision_late_a/b for CONTEXT ONLY -- loaded from disk if
present, never combined mathematically with the children's own numbers, reported as MISSING
(not fabricated) if not found.

Usage:
    python analysis/paired_vision_late_split_analysis.py [--results-dir results] [--repo-root .]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ANALYSIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ANALYSIS_DIR))

from paired_seed_comparison import (  # noqa: E402  (reused, not reimplemented -- see module docstring)
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_SEED,
    N_BOOTSTRAP,
    bootstrap_paired_mean_diff_ci,
    mcnemar_exact_pvalue,
    spearman_correlation,
)

SCOPE_A = "vision_late_a"
SCOPE_B = "vision_late_b"
PARENT_SCOPE = "vision_late"
RADIUS = 0.04
EXPECTED_N = 100

VALIDATE_SCRIPT = ANALYSIS_DIR / "validate_vision_late_split_2.py"


def _cell_dir(results_dir: Path, scope: str) -> Path:
    return results_dir / f"scoped_randopt_N{EXPECTED_N}_K1_{scope}_relative_l2_r{RADIUS}"


def _run_validation(results_dir: Path, repo_root: Path) -> bool:
    result = subprocess.run(
        [sys.executable, str(VALIDATE_SCRIPT), "--results-dir", str(results_dir), "--repo-root", str(repo_root)],
        capture_output=True, text=True,
    )
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return result.returncode == 0


def _load_cell(results_dir: Path, scope: str):
    path = _cell_dir(results_dir, scope) / "thicket_metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _seed_indexed_records(metrics: dict) -> dict:
    return {rec["seed"]: rec for rec in metrics["candidate_records"] if rec.get("status") == "done"}


def _metrics_table(metrics_a: dict, metrics_b: dict) -> str:
    lines = [f"{'':<16}{'density':<10}{'95% CI':<20}{'experts':<9}{'ties':<7}{'regress':<9}{'mean_d':<10}{'median_d':<10}{'best_d':<10}{'best_score':<10}"]
    for label, m in (("vision_late_a", metrics_a), ("vision_late_b", metrics_b)):
        ci_lo, ci_hi = m["expert_density_ci_95"]
        best_delta = m["best_candidate_score"] - m["base_score"]
        lines.append(
            f"{label:<16}{m['expert_density']:<10.4f}[{ci_lo:.3f},{ci_hi:.3f}]".ljust(46)
            + f"{m['expert_count']:<9}{m['tie_count']:<7}{m['regression_count']:<9}"
            + f"{m['mean_delta']:<+10.4f}{m['median_delta']:<+10.4f}{best_delta:<+10.4f}{m['best_candidate_score']:<10.4f}"
        )
    return "\n".join(lines)


def _paired_test_results(records_a: dict, records_b: dict, seed_order: list) -> tuple:
    is_expert_a = np.array([bool(records_a[s]["is_expert"]) for s in seed_order])
    is_expert_b = np.array([bool(records_b[s]["is_expert"]) for s in seed_order])
    delta_a = np.array([records_a[s]["delta_score"] for s in seed_order], dtype=float)
    delta_b = np.array([records_b[s]["delta_score"] for s in seed_order], dtype=float)

    a = int(np.sum(is_expert_a & is_expert_b))
    b = int(np.sum(is_expert_a & ~is_expert_b))
    c = int(np.sum(~is_expert_a & is_expert_b))
    d = int(np.sum(~is_expert_a & ~is_expert_b))
    mcnemar_p = mcnemar_exact_pvalue(b, c)

    diffs = delta_a - delta_b
    mean_diff, ci_lo, ci_hi = bootstrap_paired_mean_diff_ci(diffs)
    rho = spearman_correlation(delta_a, delta_b)

    result = {
        "n": len(seed_order),
        "a_both_expert": a, "b_a_only_expert": b, "c_b_only_expert": c, "d_neither_expert": d,
        "mcnemar_exact_p": mcnemar_p,
        "paired_mean_delta_diff_a_minus_b": mean_diff, "ci_lower": ci_lo, "ci_upper": ci_hi,
        "spearman_rho_delta": rho,
    }

    text = (
        f"vision_late_a vs vision_late_b (r={RADIUS}, n={len(seed_order)} paired seeds)\n\n"
        f"Paired 2x2 (is_expert): a(both)={a}  b(A only)={b}  c(B only)={c}  d(neither)={d}\n"
        f"Exact two-sided McNemar p = {mcnemar_p:.4f}\n\n"
        f"Paired mean delta_score difference (A - B) = {mean_diff:+.4f}\n"
        f"95% bootstrap CI (seed={BOOTSTRAP_SEED}, n_resamples={N_BOOTSTRAP}) = [{ci_lo:+.4f}, {ci_hi:+.4f}]\n\n"
        f"Spearman rho of per-seed delta_score (A vs B) = {rho:.4f}\n"
    )
    return result, text


def _parent_vs_halves_context(metrics_a: dict, metrics_b: dict, parent) -> str:
    lines = [
        "Parent vs. halves -- CONTEXT ONLY. vision_late (parent) and vision_late_a/b are never",
        "combined mathematically; shown side by side purely for visual comparison. A child",
        "having lower density than the parent is not itself evidence against localization --",
        "it may mean the useful neighborhood is distributed across the broader region.",
        "",
        f"{'scope':<18}{'density':<10}{'block_range':<14}",
    ]
    parent_density = f"{parent['expert_density']:.4f}" if parent is not None else "MISSING"
    lines.append(f"{'vision_late':<18}{parent_density:<10}{'22-31':<14}")
    lines.append("")
    lines.append(f"{'vision_late_a':<18}{metrics_a['expert_density']:<10.4f}{'22-26':<14}")
    lines.append(f"{'vision_late_b':<18}{metrics_b['expert_density']:<10.4f}{'27-31':<14}")
    if parent is None:
        lines.append("")
        lines.append("NOTE: parent vision_late@r=.04 cell not found on disk -- reported as MISSING, not fabricated.")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    repo_root = Path(args.repo_root)

    print("=== Step 1: validation (reuses validate_vision_late_split_2.py) ===")
    if not _run_validation(results_dir, repo_root):
        print("\nSTOP: validation failed, refusing to analyze. See report above.", file=sys.stderr)
        return 1

    metrics_a = _load_cell(results_dir, SCOPE_A)
    metrics_b = _load_cell(results_dir, SCOPE_B)
    parent = _load_cell(results_dir, PARENT_SCOPE)

    seq_a = metrics_a["candidate_seed_sequence"]
    seq_b = metrics_b["candidate_seed_sequence"]
    if seq_a != seq_b:
        print("STOP: candidate_seed_sequence differs between vision_late_a and vision_late_b -- refusing to pair by seed.", file=sys.stderr)
        return 1
    seed_order = seq_a
    print(f"Verified identical {len(seed_order)}-seed ordering between vision_late_a and vision_late_b.\n")

    out_dir = results_dir / "vision_late_split_experiment" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Step 2: metrics table (density + Wilson CI, counts, deltas, best score) ===")
    metrics_text = _metrics_table(metrics_a, metrics_b)
    print(metrics_text)
    (out_dir / "metrics_table.txt").write_text(metrics_text)

    print("\n=== Step 3: paired test results ===")
    records_a = _seed_indexed_records(metrics_a)
    records_b = _seed_indexed_records(metrics_b)
    paired_result, paired_text = _paired_test_results(records_a, records_b, seed_order)
    print(paired_text)
    (out_dir / "paired_test_results.txt").write_text(paired_text)

    print("=== Step 4: parent vs. halves context ===")
    context_text = _parent_vs_halves_context(metrics_a, metrics_b, parent)
    print(context_text)
    (out_dir / "parent_vs_halves_context.txt").write_text(context_text)

    summary_json = {
        "radius": RADIUS, "n_seeds": len(seed_order),
        "bootstrap_seed": BOOTSTRAP_SEED, "n_bootstrap": N_BOOTSTRAP, "bootstrap_confidence": BOOTSTRAP_CONFIDENCE,
        "vision_late_a": {
            "density": metrics_a["expert_density"], "ci_95": metrics_a["expert_density_ci_95"],
            "expert_count": metrics_a["expert_count"], "tie_count": metrics_a["tie_count"], "regression_count": metrics_a["regression_count"],
            "mean_delta": metrics_a["mean_delta"], "median_delta": metrics_a["median_delta"],
            "best_delta": metrics_a["best_candidate_score"] - metrics_a["base_score"], "best_score": metrics_a["best_candidate_score"],
        },
        "vision_late_b": {
            "density": metrics_b["expert_density"], "ci_95": metrics_b["expert_density_ci_95"],
            "expert_count": metrics_b["expert_count"], "tie_count": metrics_b["tie_count"], "regression_count": metrics_b["regression_count"],
            "mean_delta": metrics_b["mean_delta"], "median_delta": metrics_b["median_delta"],
            "best_delta": metrics_b["best_candidate_score"] - metrics_b["base_score"], "best_score": metrics_b["best_candidate_score"],
        },
        "parent_vision_late": {"density": parent["expert_density"]} if parent is not None else None,
        "paired_comparison": paired_result,
    }
    (out_dir / "analysis_summary.json").write_text(json.dumps(summary_json, indent=2))

    print("\n=== DONE. Artifacts written under: ===")
    print(f"  {out_dir}/metrics_table.txt")
    print(f"  {out_dir}/paired_test_results.txt")
    print(f"  {out_dir}/parent_vs_halves_context.txt")
    print(f"  {out_dir}/analysis_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

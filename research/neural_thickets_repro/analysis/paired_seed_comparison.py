"""Paired seed-level statistical comparisons between vision_early/vision_middle/vision_late
at each radius (0.04, 0.07), exploiting that all six cells share the identical 100 candidate
seeds. Refuses to run unless validate_vision_localization_6.py passes first. Analysis only --
reads existing thicket_metrics.json candidate_records, runs no inference, applies no new
perturbation, modifies no scientific/experiment code.

For each radius, for each of the three scope pairs (early-middle, early-late, middle-late):
  - exact two-sided McNemar test on the paired is_expert (binary) outcomes
  - paired mean difference of candidate delta_score, with a 95% percentile bootstrap CI
    (fixed deterministic seed -- BOOTSTRAP_SEED/N_BOOTSTRAP below, identical every run)
  - Spearman rank correlation of per-seed delta_score between the two scopes

Also reports, per scope, the purely descriptive depth x radius interaction:
    density(r=.07) - density(r=.04)
    mean_delta(r=.07) - mean_delta(r=.04)
No inferential test is attached to this interaction figure -- descriptive only, as requested.

No scipy dependency (matches this project's existing convention -- see thicket_metrics.py's
own hand-rolled Wilson interval): McNemar's exact p-value is computed directly from the
binomial(n, 0.5) tail via math.comb (arbitrary-precision, exact), Spearman via rank-
transformed Pearson, bootstrap via numpy's Generator seeded once, deterministically.

Usage:
    python analysis/paired_seed_comparison.py [--results-dir results] [--repo-root .]
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

SCOPES = ("vision_early", "vision_middle", "vision_late")
RADII = (0.04, 0.07)
SCOPE_PAIRS = (
    ("vision_early", "vision_middle"),
    ("vision_early", "vision_late"),
    ("vision_middle", "vision_late"),
)
EXPECTED_N = 100

ANALYSIS_DIR = Path(__file__).resolve().parent
VALIDATE_SCRIPT = ANALYSIS_DIR / "validate_vision_localization_6.py"

N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 1_234_567  # fixed, documented -- identical resample draws every run
BOOTSTRAP_CONFIDENCE = 0.95


def _cell_dir(results_dir: Path, scope: str, r: float) -> Path:
    return results_dir / f"scoped_randopt_N{EXPECTED_N}_K1_{scope}_relative_l2_r{r}"


def _run_validation(results_dir: Path, repo_root: Path) -> bool:
    result = subprocess.run(
        [sys.executable, str(VALIDATE_SCRIPT), "--results-dir", str(results_dir), "--repo-root", str(repo_root)],
        capture_output=True, text=True,
    )
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return result.returncode == 0


def _load_cell(results_dir: Path, scope: str, r: float) -> dict:
    return json.loads((_cell_dir(results_dir, scope, r) / "thicket_metrics.json").read_text())


def _seed_indexed_records(metrics: dict) -> dict:
    return {rec["seed"]: rec for rec in metrics["candidate_records"] if rec.get("status") == "done"}


def mcnemar_exact_pvalue(b: int, c: int) -> float:
    """Exact two-sided McNemar test p-value from the two discordant-pair counts, via the
    binomial(n=b+c, p=0.5) tail -- math.comb keeps this exact (arbitrary-precision integer
    arithmetic all the way through; Python's int/int true division is correctly rounded even
    for huge operands), no scipy needed.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1))
    p_one_sided = tail / (2 ** n)
    return min(1.0, 2 * p_one_sided)


def _rank(values: np.ndarray) -> np.ndarray:
    """Average ranks (1-indexed); tied values get the mean of the ranks they'd jointly occupy."""
    order = np.argsort(values, kind="mergesort")
    n = len(values)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    return ranks


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = _rank(x), _rank(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def bootstrap_paired_mean_diff_ci(diffs: np.ndarray) -> tuple:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(diffs)
    boot_means = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = diffs[idx].mean()
    alpha = 1 - BOOTSTRAP_CONFIDENCE
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return float(diffs.mean()), lower, upper


def _compare_pair(records_a: dict, records_b: dict, seed_order: list) -> dict:
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

    return {
        "n": len(seed_order), "a_both_expert": a, "b_a_only": b, "c_b_only": c, "d_neither": d,
        "mcnemar_exact_p": mcnemar_p,
        "paired_mean_delta_diff": mean_diff, "ci_lower": ci_lo, "ci_upper": ci_hi,
        "spearman_rho_delta": rho,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    repo_root = Path(args.repo_root)

    print("=== Step 1: validation (reuses validate_vision_localization_6.py) ===")
    if not _run_validation(results_dir, repo_root):
        print("\nSTOP: validation failed, refusing to analyze. See report above.", file=sys.stderr)
        return 1

    cells = {(scope, r): _load_cell(results_dir, scope, r) for scope in SCOPES for r in RADII}

    # Requirement #1: verify identical seed ordering across ALL SIX cells before any pairing
    # -- re-checked independently here (not merely trusting a prior validation run) since
    # correctness of every paired comparison below depends on it.
    reference_seq = cells[(SCOPES[0], RADII[0])]["candidate_seed_sequence"]
    for (scope, r), metrics in cells.items():
        seq = metrics["candidate_seed_sequence"]
        if seq != reference_seq:
            print(
                f"STOP: candidate_seed_sequence for {scope} r={r} does not match the "
                f"reference ({SCOPES[0]} r={RADII[0]}) -- refusing to pair by seed.",
                file=sys.stderr,
            )
            return 1
    seed_order = reference_seq
    print(f"Verified identical {len(seed_order)}-seed ordering across all 6 cells.\n")

    out_dir = results_dir / "vision_localization_experiment" / "paired_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for r in RADII:
        print(f"=== Paired comparisons at r={r} ===")
        records_by_scope = {scope: _seed_indexed_records(cells[(scope, r)]) for scope in SCOPES}
        lines = [
            f"{'pair':<30}{'a':<5}{'b':<5}{'c':<5}{'d':<5}{'mcnemar_p':<12}{'mean_diff':<12}{'95% CI':<24}{'spearman_rho':<12}",
        ]
        radius_results = {}
        for scope_a, scope_b in SCOPE_PAIRS:
            result = _compare_pair(records_by_scope[scope_a], records_by_scope[scope_b], seed_order)
            radius_results[f"{scope_a}_vs_{scope_b}"] = result
            pair_label = f"{scope_a} vs {scope_b}"
            ci_str = f"[{result['ci_lower']:+.4f},{result['ci_upper']:+.4f}]"
            lines.append(
                f"{pair_label:<30}{result['a_both_expert']:<5}{result['b_a_only']:<5}"
                f"{result['c_b_only']:<5}{result['d_neither']:<5}{result['mcnemar_exact_p']:<12.4f}"
                f"{result['paired_mean_delta_diff']:<+12.4f}{ci_str:<24}{result['spearman_rho_delta']:<12.4f}"
            )
        table_text = "\n".join(lines)
        print(table_text)
        (out_dir / f"paired_comparison_r{r}.txt").write_text(table_text)
        all_results[r] = radius_results
        print()

    print("=== Depth x radius interaction (descriptive only, no test attached) ===")
    interaction_lines = [f"{'scope':<16}{'density(.07)-density(.04)':<28}{'mean_delta(.07)-mean_delta(.04)':<32}"]
    interaction_data = {}
    for scope in SCOPES:
        m04 = cells[(scope, 0.04)]
        m07 = cells[(scope, 0.07)]
        density_diff = m07["expert_density"] - m04["expert_density"]
        mean_delta_diff = m07["mean_delta"] - m04["mean_delta"]
        interaction_data[scope] = {"density_diff_r07_minus_r04": density_diff, "mean_delta_diff_r07_minus_r04": mean_delta_diff}
        interaction_lines.append(f"{scope:<16}{density_diff:<+28.4f}{mean_delta_diff:<+32.4f}")
    interaction_text = "\n".join(interaction_lines)
    print(interaction_text)
    (out_dir / "depth_radius_interaction.txt").write_text(interaction_text)

    summary_json = {
        "bootstrap_seed": BOOTSTRAP_SEED, "n_bootstrap": N_BOOTSTRAP, "bootstrap_confidence": BOOTSTRAP_CONFIDENCE,
        "seed_order_verified_identical": True, "n_seeds": len(seed_order),
        "paired_comparisons_by_radius": all_results,
        "depth_radius_interaction": interaction_data,
    }
    (out_dir / "paired_analysis_summary.json").write_text(json.dumps(summary_json, indent=2))

    print("\n=== DONE. Artifacts written under: ===")
    print(f"  {out_dir}/paired_comparison_r0.04.txt")
    print(f"  {out_dir}/paired_comparison_r0.07.txt")
    print(f"  {out_dir}/depth_radius_interaction.txt")
    print(f"  {out_dir}/paired_analysis_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

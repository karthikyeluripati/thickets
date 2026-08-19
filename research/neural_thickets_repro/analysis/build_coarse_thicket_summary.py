"""Aggregation-only step for the completed 28-cell coarse neural-thicket sweep. Refuses to
run unless validate_coarse_thicket_28.py passes first. Runs no inference, perturbs no
weights, modifies no scientific/experiment code -- reads existing thicket_metrics.json files
and the existing, unmodified analysis/aggregate_coarse_thicket.py only.

Produces, under <results-dir>/coarse_thicket_experiment/aggregate/:
    table_r<r>.txt              -- one per radius, via the unmodified aggregator CLI
    expert_density_matrix.txt   -- 7 (scope) x 4 (radius) expert_density matrix
    summary_table.txt           -- rows=scope, one column-group per radius: expert_density,
                                    Wilson 95% CI, tie_rate, regression_rate, mean_delta,
                                    median_delta, best_delta, best_score

Usage:
    python analysis/build_coarse_thicket_summary.py [--results-dir results] [--repo-root .]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCOPES = ("full_lm", "vision_encoder", "vision_merger", "lm_early", "lm_middle", "lm_late", "full_vlm")
RADII = (0.005, 0.02, 0.04, 0.07)
EXPECTED_N = 100

ANALYSIS_DIR = Path(__file__).resolve().parent
AGGREGATOR_SCRIPT = ANALYSIS_DIR / "aggregate_coarse_thicket.py"


def _cell_dir(results_dir: Path, scope: str, r: float) -> Path:
    return results_dir / f"scoped_randopt_N{EXPECTED_N}_K1_{scope}_relative_l2_r{r}"


def _run_validation(results_dir: Path, repo_root: Path) -> bool:
    result = subprocess.run(
        [sys.executable, str(ANALYSIS_DIR / "validate_coarse_thicket_28.py"),
         "--results-dir", str(results_dir), "--repo-root", str(repo_root)],
        capture_output=True, text=True,
    )
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return result.returncode == 0


def _run_per_radius_aggregator(results_dir: Path, out_dir: Path) -> None:
    for r in RADII:
        dirs = [str(_cell_dir(results_dir, scope, r)) for scope in SCOPES]
        out_path = out_dir / f"table_r{r}.txt"
        print(f"\n--- Aggregate table for r={r} (via unmodified {AGGREGATOR_SCRIPT.name}) ---")
        subprocess.run(
            [sys.executable, str(AGGREGATOR_SCRIPT), *dirs, "--out", str(out_path)],
            check=True,
        )


def _load_all_cells(results_dir: Path) -> dict:
    cells = {}
    for scope in SCOPES:
        for r in RADII:
            path = _cell_dir(results_dir, scope, r) / "thicket_metrics.json"
            cells[(scope, r)] = json.loads(path.read_text())
    return cells


def _build_density_matrix(cells: dict) -> str:
    header = "scope".ljust(16) + "".join(f"r={r:<9}" for r in RADII)
    lines = [header]
    for scope in SCOPES:
        row = scope.ljust(16) + "".join(f"{cells[(scope, r)]['expert_density']:<11.4f}" for r in RADII)
        lines.append(row)
    return "\n".join(lines)


def _build_summary_table(cells: dict) -> str:
    col_fields = ("density", "CI", "tie_rate", "regress", "mean_d", "median_d", "best_d", "best_score")
    lines = []
    header_top = "scope".ljust(16) + "".join(f"r={r}".center(11 * len(col_fields) - 1) + " | " for r in RADII)
    lines.append(header_top)
    header_sub = " " * 16 + "".join((" ".join(f"{c:<10}" for c in col_fields)) + " | " for _ in RADII)
    lines.append(header_sub)

    for scope in SCOPES:
        row = scope.ljust(16)
        for r in RADII:
            m = cells[(scope, r)]
            n = m["N"]
            tie_rate = m["tie_count"] / n
            regression_rate = m["regression_count"] / n
            best_delta = m["best_candidate_score"] - m["base_score"]
            ci_lo, ci_hi = m["expert_density_ci_95"]
            values = [
                f"{m['expert_density']:.4f}",
                f"[{ci_lo:.3f},{ci_hi:.3f}]",
                f"{tie_rate:.4f}",
                f"{regression_rate:.4f}",
                f"{m['mean_delta']:+.4f}",
                f"{m['median_delta']:+.4f}",
                f"{best_delta:+.4f}",
                f"{m['best_candidate_score']:.4f}",
            ]
            row += " ".join(f"{v:<10}" for v in values) + " | "
        lines.append(row)
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    repo_root = Path(args.repo_root)

    print("=== Step 1: validation ===")
    if not _run_validation(results_dir, repo_root):
        print("\nSTOP: validation failed, refusing to aggregate. See report above.", file=sys.stderr)
        return 1

    out_dir = results_dir / "coarse_thicket_experiment" / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Step 2: per-radius aggregator (existing, unmodified analysis/aggregate_coarse_thicket.py) ===")
    _run_per_radius_aggregator(results_dir, out_dir)

    cells = _load_all_cells(results_dir)

    print("\n=== Step 3: 7x4 expert-density matrix ===")
    matrix_text = _build_density_matrix(cells)
    print(matrix_text)
    (out_dir / "expert_density_matrix.txt").write_text(matrix_text)

    print("\n=== Step 4: full summary table (density, CI, tie_rate, regression_rate, mean/median/best delta, best_score) ===")
    summary_text = _build_summary_table(cells)
    print(summary_text)
    (out_dir / "summary_table.txt").write_text(summary_text)

    print("\n=== DONE. Artifacts written under: ===")
    print(f"  {out_dir}/table_r<radius>.txt   (4 files)")
    print(f"  {out_dir}/expert_density_matrix.txt")
    print(f"  {out_dir}/summary_table.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

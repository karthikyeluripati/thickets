"""Aggregation-only step for the completed 6-cell vision-encoder localization sweep
(vision_early/vision_middle/vision_late x {r=0.04, r=0.07}). Refuses to run unless
validate_vision_localization_6.py passes first. Runs no inference, perturbs no weights,
modifies no scientific/experiment code -- reads existing thicket_metrics.json files and the
existing, unmodified analysis/aggregate_coarse_thicket.py only.

Produces, under <results-dir>/vision_localization_experiment/aggregate/:
    table_r<r>.txt              -- one per radius (0.04, 0.07), via the unmodified
                                    aggregate_coarse_thicket.py CLI
    density_matrix.txt          -- 3 (scope) x 2 (radius) expert_density matrix
    summary_table.txt           -- rows=scope, one column-group per radius: expert_density,
                                    Wilson 95% CI, expert/tie/regression counts, mean_delta,
                                    median_delta, best_delta, best_score
    parent_vs_thirds_context.txt -- vision_encoder (parent, already-validated coarse-map
                                    cells) shown alongside the three thirds for CONTEXT ONLY,
                                    never combined mathematically

Usage:
    python analysis/build_vision_localization_summary.py [--results-dir results] [--repo-root .]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCOPES = ("vision_early", "vision_middle", "vision_late")
RADII = (0.04, 0.07)
PARENT_SCOPE = "vision_encoder"
EXPECTED_N = 100

ANALYSIS_DIR = Path(__file__).resolve().parent
AGGREGATOR_SCRIPT = ANALYSIS_DIR / "aggregate_coarse_thicket.py"


def _cell_dir(results_dir: Path, scope: str, r: float) -> Path:
    return results_dir / f"scoped_randopt_N{EXPECTED_N}_K1_{scope}_relative_l2_r{r}"


def _run_validation(results_dir: Path, repo_root: Path) -> bool:
    result = subprocess.run(
        [sys.executable, str(ANALYSIS_DIR / "validate_vision_localization_6.py"),
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


def _load_parent_cells(results_dir: Path) -> dict:
    """Best-effort load of the already-validated coarse-map vision_encoder cells, for context
    only -- never required for this milestone's own validation to pass. Missing files are
    reported, not fabricated.
    """
    parent = {}
    for r in RADII:
        path = _cell_dir(results_dir, PARENT_SCOPE, r) / "thicket_metrics.json"
        if path.exists():
            parent[r] = json.loads(path.read_text())
        else:
            parent[r] = None
    return parent


def _build_density_matrix(cells: dict) -> str:
    header = "scope".ljust(16) + "".join(f"r={r:<9}" for r in RADII)
    lines = [header]
    for scope in SCOPES:
        row = scope.ljust(16) + "".join(f"{cells[(scope, r)]['expert_density']:<11.4f}" for r in RADII)
        lines.append(row)
    return "\n".join(lines)


def _build_summary_table(cells: dict) -> str:
    col_fields = ("density", "CI", "experts", "ties", "regress", "mean_d", "median_d", "best_d", "best_score")
    lines = []
    header_top = "scope".ljust(16) + "".join(f"r={r}".center(11 * len(col_fields) - 1) + " | " for r in RADII)
    lines.append(header_top)
    header_sub = " " * 16 + "".join((" ".join(f"{c:<10}" for c in col_fields)) + " | " for _ in RADII)
    lines.append(header_sub)

    for scope in SCOPES:
        row = scope.ljust(16)
        for r in RADII:
            m = cells[(scope, r)]
            best_delta = m["best_candidate_score"] - m["base_score"]
            ci_lo, ci_hi = m["expert_density_ci_95"]
            values = [
                f"{m['expert_density']:.4f}",
                f"[{ci_lo:.3f},{ci_hi:.3f}]",
                f"{m['expert_count']}",
                f"{m['tie_count']}",
                f"{m['regression_count']}",
                f"{m['mean_delta']:+.4f}",
                f"{m['median_delta']:+.4f}",
                f"{best_delta:+.4f}",
                f"{m['best_candidate_score']:.4f}",
            ]
            row += " ".join(f"{v:<10}" for v in values) + " | "
        lines.append(row)
    return "\n".join(lines)


def _build_parent_vs_thirds_context(cells: dict, parent: dict) -> str:
    lines = [
        "Parent vs. thirds -- CONTEXT ONLY. Parent (vision_encoder) and the three thirds are",
        "never combined mathematically; shown side by side purely for visual comparison.",
        "",
        f"{'scope':<16}{'r=0.04':<14}{'r=0.07':<14}",
    ]

    parent_row = f"{PARENT_SCOPE:<16}"
    for r in RADII:
        m = parent.get(r)
        parent_row += (f"{m['expert_density']:.4f}".ljust(14) if m is not None else "MISSING".ljust(14))
    lines.append(parent_row)
    lines.append("")

    for scope in SCOPES:
        row = f"{scope:<16}"
        for r in RADII:
            row += f"{cells[(scope, r)]['expert_density']:.4f}".ljust(14)
        lines.append(row)

    missing = [r for r in RADII if parent.get(r) is None]
    if missing:
        lines.append("")
        lines.append(f"NOTE: parent vision_encoder cell(s) not found on disk for r={missing} -- reported as MISSING, not fabricated.")

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

    out_dir = results_dir / "vision_localization_experiment" / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Step 2: per-radius aggregator (existing, unmodified analysis/aggregate_coarse_thicket.py) ===")
    _run_per_radius_aggregator(results_dir, out_dir)

    cells = _load_all_cells(results_dir)
    parent = _load_parent_cells(results_dir)

    print("\n=== Step 3: 3x2 expert-density matrix ===")
    matrix_text = _build_density_matrix(cells)
    print(matrix_text)
    (out_dir / "density_matrix.txt").write_text(matrix_text)

    print("\n=== Step 4: full summary table (density, CI, expert/tie/regression counts, mean/median/best delta, best_score) ===")
    summary_text = _build_summary_table(cells)
    print(summary_text)
    (out_dir / "summary_table.txt").write_text(summary_text)

    print("\n=== Step 5: parent (vision_encoder) vs. thirds -- context only, not combined ===")
    context_text = _build_parent_vs_thirds_context(cells, parent)
    print(context_text)
    (out_dir / "parent_vs_thirds_context.txt").write_text(context_text)

    print("\n=== DONE. Artifacts written under: ===")
    print(f"  {out_dir}/table_r<radius>.txt   (2 files: r=0.04, r=0.07)")
    print(f"  {out_dir}/density_matrix.txt")
    print(f"  {out_dir}/summary_table.txt")
    print(f"  {out_dir}/parent_vs_thirds_context.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

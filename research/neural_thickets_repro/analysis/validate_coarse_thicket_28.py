"""Validates the completed 28-cell coarse neural-thicket sweep (7 scopes x 4 relative-L2
radii, N=100 each) BEFORE any aggregation. Reads only existing thicket_metrics.json /
run_metadata.json files under results/ -- runs no inference, perturbs no weights, calls no
scientific/experiment code.

Hard-checks, per cell and across all 28 cells: exactly 28 cells present; N=100; same
task/model/dataset/base-score/selection-subset/candidate-seed-sequence across all 28;
restoration_mode=fixed_base; noise_semantics=upstream_per_tensor_reseed; 100 candidate
records with expert+tie+regression==100; no NaNs; for visual scopes (vision_encoder,
vision_merger, full_vlm), the run's recorded git commit contains the full engine-level
encoder-cache-reset fix (0706a03) -- rejecting any pre-fix / forensic artifact.

On ANY failure, prints the exact cell + reason and exits 1. Never repairs, skips, or
silently drops a bad cell -- that is a deliberate choice, not an oversight: the caller
(build_coarse_thicket_summary.py) refuses to aggregate unless this exits 0.

Usage:
    python analysis/validate_coarse_thicket_28.py [--results-dir results] [--repo-root .]
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

SCOPES = ("full_lm", "vision_encoder", "vision_merger", "lm_early", "lm_middle", "lm_late", "full_vlm")
RADII = (0.005, 0.02, 0.04, 0.07)
VISUAL_SCOPES = ("vision_encoder", "vision_merger", "full_vlm")
EXPECTED_N = 100

# Commit that introduced the FULL (scheduler + worker) encoder-cache reset, superseding the
# worker-only reset confirmed insufficient on GPU (RuntimeError: Encoder cache miss). Any
# visual-scope cell recorded before this commit is a pre-fix / forensic artifact and must
# never be aggregated into the coarse map.
FULL_RESET_FIX_COMMIT = "0706a03"

# Must be identical across ALL 28 cells, not just same-radius (the aggregator's
# assert_comparable only ever compares cells sharing one radius).
_CROSS_CELL_SCALAR_FIELDS = (
    "task", "model_name", "model_revision", "global_seed", "restoration_mode",
    "noise_semantics", "dataset_revision", "dataset_selection_split", "scoring_protocol",
    "selection_set_size", "base_score",
)
_CROSS_CELL_SEQUENCE_FIELDS = ("selection_example_ids", "candidate_seed_sequence")

_NUMERIC_METRIC_FIELDS = (
    "base_score", "expert_density", "mean_score", "std_score", "mean_delta",
    "median_delta", "min_delta", "max_delta", "best_candidate_score",
)


def _cell_dir(results_dir: Path, scope: str, r: float) -> Path:
    return results_dir / f"scoped_randopt_N{EXPECTED_N}_K1_{scope}_relative_l2_r{r}"


def _load_json(path: Path, name: str, errors: list, cell: str):
    if not path.exists():
        errors.append((cell, f"missing {name}: {path}"))
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        errors.append((cell, f"malformed {name} ({path}): {exc}"))
        return None


def _check_visual_scope_fix_ancestry(repo_root: Path, commit, errors: list, cell: str) -> None:
    if not commit:
        errors.append((cell, "run_metadata.our_repo_git_commit missing -- cannot confirm full-reset fix is present"))
        return
    try:
        subprocess.run(
            ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", FULL_RESET_FIX_COMMIT, commit],
            check=True, capture_output=True,
        )
    except FileNotFoundError:
        errors.append((cell, "git not found on PATH -- cannot verify full-reset fix commit ancestry"))
    except subprocess.CalledProcessError:
        errors.append((
            cell,
            f"our_repo_git_commit={commit} does NOT contain the full encoder-cache-reset fix "
            f"({FULL_RESET_FIX_COMMIT}) -- this is a pre-fix / forensic artifact, must not be aggregated",
        ))
    # If ancestry holds: 100/100 candidate records with noise_semantics=upstream_per_tensor_reseed
    # under a post-fix commit IS the evidence the full reset succeeded every time --
    # run_scoped_randopt.py hard-fails the whole run (raises, never writes thicket_metrics.json)
    # on any reset failure, so a complete, validation-passing file for a visual scope cannot
    # exist any other way. No separate log-scraping needed.


def _check_no_nans(metrics: dict, errors: list, cell: str) -> None:
    for field in _NUMERIC_METRIC_FIELDS:
        value = metrics.get(field)
        if isinstance(value, float) and math.isnan(value):
            errors.append((cell, f"NaN in {field}"))
    for rec in metrics.get("candidate_records") or []:
        for field in ("selection_score", "delta_score"):
            value = rec.get(field)
            if isinstance(value, float) and math.isnan(value):
                errors.append((cell, f"NaN in candidate seed={rec.get('seed')} field {field}"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    repo_root = Path(args.repo_root)

    errors: list = []
    cells: dict = {}

    for scope in SCOPES:
        for r in RADII:
            cell = f"{scope} r={r}"
            cell_dir = _cell_dir(results_dir, scope, r)
            metrics = _load_json(cell_dir / "thicket_metrics.json", "thicket_metrics.json", errors, cell)
            metadata = _load_json(cell_dir / "run_metadata.json", "run_metadata.json", errors, cell)
            if metrics is None:
                continue

            if metrics.get("scope") != scope:
                errors.append((cell, f"scope mismatch: expected {scope}, got {metrics.get('scope')!r}"))
            if metrics.get("requested_relative_l2") != r:
                errors.append((cell, f"requested_relative_l2 mismatch: expected {r}, got {metrics.get('requested_relative_l2')!r}"))
            if metrics.get("N") != EXPECTED_N:
                errors.append((cell, f"N != {EXPECTED_N}: got {metrics.get('N')!r}"))
            if metrics.get("restoration_mode") != "fixed_base":
                errors.append((cell, f"restoration_mode != fixed_base: got {metrics.get('restoration_mode')!r}"))
            if metrics.get("noise_semantics") != "upstream_per_tensor_reseed":
                errors.append((cell, f"noise_semantics != upstream_per_tensor_reseed: got {metrics.get('noise_semantics')!r}"))
            if metrics.get("base_score") is None:
                errors.append((cell, "base_score missing"))

            records = metrics.get("candidate_records") or []
            if len(records) != EXPECTED_N:
                errors.append((cell, f"expected {EXPECTED_N} candidate records, got {len(records)}"))
            counts = (metrics.get("expert_count") or 0) + (metrics.get("tie_count") or 0) + (metrics.get("regression_count") or 0)
            if counts != EXPECTED_N:
                errors.append((cell, f"expert+tie+regression != {EXPECTED_N}: got {counts}"))

            _check_no_nans(metrics, errors, cell)

            if scope in VISUAL_SCOPES:
                _check_visual_scope_fix_ancestry(repo_root, (metadata or {}).get("our_repo_git_commit"), errors, cell)

            cells[(scope, r)] = metrics

    if len(cells) == 28:
        reference = cells[(SCOPES[0], RADII[0])]
        ref_label = f"{SCOPES[0]} r={RADII[0]}"
        for (scope, r), metrics in cells.items():
            cell = f"{scope} r={r}"
            for field in _CROSS_CELL_SCALAR_FIELDS:
                if metrics.get(field) != reference.get(field):
                    errors.append((
                        cell,
                        f"{field} differs from reference cell ({ref_label}): "
                        f"{metrics.get(field)!r} != {reference.get(field)!r}",
                    ))
            for field in _CROSS_CELL_SEQUENCE_FIELDS:
                if metrics.get(field) != reference.get(field):
                    errors.append((cell, f"{field} differs from reference cell ({ref_label}) -- ordering not identical"))
    else:
        errors.append(("<sweep>", f"expected 28 loadable cells, got {len(cells)} -- see missing/malformed-file errors above"))

    print(f"=== Coarse thicket sweep validation: {len(cells)}/28 cells loaded ===\n")
    if errors:
        print(f"VALIDATION FAILED -- {len(errors)} issue(s):\n")
        for cell, reason in errors:
            print(f"  [{cell}] {reason}")
        print("\nSTOP: not aggregating. Report the exact cell/reason above -- do not repair data automatically.")
        return 1

    print("ALL 28 CELLS PASSED\n")
    for scope in SCOPES:
        for r in RADII:
            m = cells[(scope, r)]
            print(
                f"  {scope:<15} r={r:<6} base={m['base_score']:.4f} "
                f"density={m['expert_density']:.4f} "
                f"(experts={m['expert_count']} ties={m['tie_count']} regressions={m['regression_count']})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

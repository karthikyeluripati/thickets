"""Validates the completed 2-cell vision_late split experiment (vision_late_a / vision_late_b,
both at relative-L2 r=0.04, N=100) BEFORE any paired analysis. Reads only existing
thicket_metrics.json / run_metadata.json files under results/ -- runs no inference, perturbs
no weights, modifies no scientific/experiment code.

Hard-checks, per cell and across both cells: exactly 2 cells present; N=100; requested
radius == 0.04; base_score == 0.5600 (within float tolerance); restoration_mode=fixed_base;
perturbation_scale_mode=relative_l2; same model/task/dataset/prompt-scorer identity and same
100 candidate seeds across both cells; 100 candidate records with expert+tie+regression==100;
no NaNs; the run's recorded git commit contains the full engine-level encoder-cache-reset fix
(0706a03) -- both scopes are visual-affecting.

On ANY failure, prints the exact cell + reason and exits 1. Never repairs or skips a bad
cell -- the caller (paired_vision_late_split_analysis.py) refuses to analyze unless this
exits 0.

Usage:
    python analysis/validate_vision_late_split_2.py [--results-dir results] [--repo-root .]
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

SCOPES = ("vision_late_a", "vision_late_b")
RADIUS = 0.04
EXPECTED_N = 100
EXPECTED_BASE_SCORE = 0.56
BASE_SCORE_TOLERANCE = 1e-6

FULL_RESET_FIX_COMMIT = "0706a03"

_CROSS_CELL_SCALAR_FIELDS = (
    "task", "model_name", "model_revision", "global_seed", "restoration_mode",
    "noise_semantics", "dataset_revision", "dataset_selection_split", "scoring_protocol",
    "selection_set_size",
)
_CROSS_CELL_SEQUENCE_FIELDS = ("selection_example_ids", "candidate_seed_sequence")

_NUMERIC_METRIC_FIELDS = (
    "base_score", "expert_density", "mean_score", "std_score", "mean_delta",
    "median_delta", "min_delta", "max_delta", "best_candidate_score",
)


def _cell_dir(results_dir: Path, scope: str) -> Path:
    return results_dir / f"scoped_randopt_N{EXPECTED_N}_K1_{scope}_relative_l2_r{RADIUS}"


def _load_json(path: Path, name: str, errors: list, cell: str):
    if not path.exists():
        errors.append((cell, f"missing {name}: {path}"))
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        errors.append((cell, f"malformed {name} ({path}): {exc}"))
        return None


def _check_full_reset_ancestry(repo_root: Path, commit, errors: list, cell: str) -> None:
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
            f"({FULL_RESET_FIX_COMMIT}) -- this cell was not produced with a full encoder-cache reset",
        ))


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
        cell = f"{scope} r={RADIUS}"
        cell_dir = _cell_dir(results_dir, scope)
        metrics = _load_json(cell_dir / "thicket_metrics.json", "thicket_metrics.json", errors, cell)
        metadata = _load_json(cell_dir / "run_metadata.json", "run_metadata.json", errors, cell)
        if metrics is None:
            continue

        if metrics.get("scope") != scope:
            errors.append((cell, f"scope mismatch: expected {scope}, got {metrics.get('scope')!r}"))
        if metrics.get("requested_relative_l2") != RADIUS:
            errors.append((cell, f"requested_relative_l2 mismatch: expected {RADIUS}, got {metrics.get('requested_relative_l2')!r}"))
        if metrics.get("N") != EXPECTED_N:
            errors.append((cell, f"N != {EXPECTED_N}: got {metrics.get('N')!r}"))
        if metrics.get("restoration_mode") != "fixed_base":
            errors.append((cell, f"restoration_mode != fixed_base: got {metrics.get('restoration_mode')!r}"))
        if metrics.get("perturbation_scale_mode") != "relative_l2":
            errors.append((cell, f"perturbation_scale_mode != relative_l2: got {metrics.get('perturbation_scale_mode')!r}"))
        if metrics.get("noise_semantics") != "upstream_per_tensor_reseed":
            errors.append((cell, f"noise_semantics != upstream_per_tensor_reseed: got {metrics.get('noise_semantics')!r}"))

        base_score = metrics.get("base_score")
        if base_score is None:
            errors.append((cell, "base_score missing"))
        elif abs(base_score - EXPECTED_BASE_SCORE) > BASE_SCORE_TOLERANCE:
            errors.append((cell, f"base_score={base_score} != expected {EXPECTED_BASE_SCORE}"))

        records = metrics.get("candidate_records") or []
        if len(records) != EXPECTED_N:
            errors.append((cell, f"expected {EXPECTED_N} candidate records, got {len(records)}"))
        counts = (metrics.get("expert_count") or 0) + (metrics.get("tie_count") or 0) + (metrics.get("regression_count") or 0)
        if counts != EXPECTED_N:
            errors.append((cell, f"expert+tie+regression != {EXPECTED_N}: got {counts}"))

        _check_no_nans(metrics, errors, cell)
        _check_full_reset_ancestry(repo_root, (metadata or {}).get("our_repo_git_commit"), errors, cell)

        cells[scope] = metrics

    if len(cells) == 2:
        reference = cells[SCOPES[0]]
        ref_label = f"{SCOPES[0]} r={RADIUS}"
        for scope, metrics in cells.items():
            cell = f"{scope} r={RADIUS}"
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
        errors.append(("<sweep>", f"expected 2 loadable cells, got {len(cells)} -- see missing/malformed-file errors above"))

    print(f"=== vision_late split validation: {len(cells)}/2 cells loaded ===\n")
    if errors:
        print(f"VALIDATION FAILED -- {len(errors)} issue(s):\n")
        for cell, reason in errors:
            print(f"  [{cell}] {reason}")
        print("\nSTOP: not analyzing. Report the exact cell/reason above -- do not repair data automatically.")
        return 1

    print("ALL 2 CELLS PASSED\n")
    for scope in SCOPES:
        m = cells[scope]
        print(
            f"  {scope:<16} r={RADIUS:<6} base={m['base_score']:.4f} "
            f"density={m['expert_density']:.4f} "
            f"(experts={m['expert_count']} ties={m['tie_count']} regressions={m['regression_count']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

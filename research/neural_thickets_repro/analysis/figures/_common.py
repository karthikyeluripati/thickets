"""Shared, minimal data-loading helpers for the paper-figure scripts in this directory.
Reads only existing thicket_metrics.json files under results/ -- no inference, no
computation of new scientific quantities beyond simple field lookups/derived deltas already
present in those files. Hard-fails (raises FigureDataError) rather than plotting a
placeholder/zero value for a missing cell -- a published figure must never silently show
fabricated data.

Not a script itself -- imported by figure_a_coarse_map.py / figure_b_hierarchical_
localization.py / figure_c_density_vs_radius.py.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_FIGURES_DIR = REPO_ROOT / "results" / "paper_figures"

EXPECTED_N = 100


class FigureDataError(RuntimeError):
    """A required thicket_metrics.json cell is missing, unreadable, or missing an expected
    field -- never silently substituted with a placeholder value in a published figure.
    """


def cell_dir(results_dir: Path, scope: str, r: float) -> Path:
    return results_dir / f"scoped_randopt_N{EXPECTED_N}_K1_{scope}_relative_l2_r{r}"


def load_metrics(results_dir: Path, scope: str, r: float) -> dict:
    path = cell_dir(results_dir, scope, r) / "thicket_metrics.json"
    if not path.exists():
        raise FigureDataError(f"Missing thicket_metrics.json for scope={scope!r} r={r} at {path}")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise FigureDataError(f"Malformed thicket_metrics.json for scope={scope!r} r={r} at {path}: {exc}") from exc
    if data.get("expert_density") is None:
        raise FigureDataError(f"thicket_metrics.json for scope={scope!r} r={r} has no expert_density field: {path}")
    return data


def load_json_file(path: Path) -> dict:
    if not path.exists():
        raise FigureDataError(f"Missing required result file: {path}")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise FigureDataError(f"Malformed JSON at {path}: {exc}") from exc


def save_figure(fig, figures_dir: Path, name: str) -> list:
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext, kwargs in (("pdf", {}), ("png", {"dpi": 300})):
        p = figures_dir / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight", **kwargs)
        paths.append(p)
    return paths

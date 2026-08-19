"""Figure A -- coarse thicket map: 7 scopes x 4 relative-L2 radii, expert density heatmap.
Every value is read directly from the validated coarse-sweep thicket_metrics.json files
(results/scoped_randopt_N100_K1_<scope>_relative_l2_r<r>/) -- nothing is hardcoded, nothing
is interpolated across scopes (rows/columns are discrete, unordered categories, not a
continuous surface). Hard-fails if any of the 28 cells is missing rather than plotting a
partial/placeholder heatmap.

Analysis/visualization only -- runs no inference, no perturbations, reads existing results.

Usage:
    python analysis/figures/figure_a_coarse_map.py [--results-dir results] [--figures-dir results/paper_figures]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DEFAULT_FIGURES_DIR, DEFAULT_RESULTS_DIR, FigureDataError, load_metrics, save_figure  # noqa: E402

SCOPES = ("full_lm", "vision_encoder", "vision_merger", "lm_early", "lm_middle", "lm_late", "full_vlm")
RADII = (0.005, 0.02, 0.04, 0.07)


def build_density_matrix(results_dir: Path) -> np.ndarray:
    matrix = np.full((len(SCOPES), len(RADII)), np.nan)
    for i, scope in enumerate(SCOPES):
        for j, r in enumerate(RADII):
            metrics = load_metrics(results_dir, scope, r)
            matrix[i, j] = metrics["expert_density"]
    return matrix


def plot_coarse_map(matrix: np.ndarray):
    fig, ax = plt.subplots(figsize=(6.0, 5.5))
    # interpolation="nearest": each cell is an independent measured value, never blended
    # with its neighbors -- rows (scopes) are discrete unordered categories, so implying any
    # continuity across them (via smoothing/interpolation) would misrepresent the data.
    im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto", interpolation="nearest")

    ax.set_xticks(range(len(RADII)))
    ax.set_xticklabels([f"r={r:g}" for r in RADII])
    ax.set_yticks(range(len(SCOPES)))
    ax.set_yticklabels(SCOPES)
    ax.set_xlabel("relative-L2 radius")
    ax.set_title("Coarse thicket map -- expert density $\\rho_{m,t}(r)$")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text_color = "white" if value < 0.55 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("expert density")
    fig.tight_layout()
    return fig


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES_DIR))
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)

    try:
        matrix = build_density_matrix(results_dir)
    except FigureDataError as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 1

    fig = plot_coarse_map(matrix)
    paths = save_figure(fig, figures_dir, "figure_a_coarse_map")
    plt.close(fig)

    print("Figure A written:")
    for p in paths:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Figure C -- expert density vs. relative-L2 radius for the anatomically meaningful,
non-overlapping/core scopes (vision_encoder, vision_merger, lm_early, lm_middle, lm_late).
Excludes full_lm/full_vlm (overlap with every other scope, not "core" anatomical units) and
the vision_early/middle/late/vision_late_a/b sub-scopes (only measured at r in {.04, .07},
not the full 4-radius grid this figure plots).

One line per scope, connecting exactly the four measured radii -- no smoothing/interpolation
beyond that straight-line connection. Error bars are the Wilson 95% CI already computed and
stored in each cell's thicket_metrics.json (expert_density_ci_95), not recomputed here.

Analysis/visualization only -- runs no inference, no perturbations, reads existing results.

Usage:
    python analysis/figures/figure_c_density_vs_radius.py [--results-dir results] [--figures-dir results/paper_figures]
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

SCOPES = ("vision_encoder", "vision_merger", "lm_early", "lm_middle", "lm_late")
RADII = (0.005, 0.02, 0.04, 0.07)

_MARKERS = {"vision_encoder": "o", "vision_merger": "s", "lm_early": "^", "lm_middle": "D", "lm_late": "v"}


def load_all_data(results_dir: Path) -> dict:
    data = {}
    for scope in SCOPES:
        densities, ci_lo, ci_hi = [], [], []
        for r in RADII:
            metrics = load_metrics(results_dir, scope, r)
            density = metrics["expert_density"]
            lo, hi = metrics["expert_density_ci_95"]
            densities.append(density)
            ci_lo.append(density - lo)
            ci_hi.append(hi - density)
        data[scope] = {
            "density": np.array(densities), "ci_lo": np.array(ci_lo), "ci_hi": np.array(ci_hi),
        }
    return data


def plot_density_vs_radius(data: dict):
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    x = np.array(RADII)

    for scope in SCOPES:
        d = data[scope]
        ax.errorbar(
            x, d["density"], yerr=[d["ci_lo"], d["ci_hi"]],
            marker=_MARKERS.get(scope, "o"), markersize=6, linewidth=1.4, capsize=3,
            label=scope,
        )

    ax.set_xlabel("relative-L2 radius $r$")
    ax.set_ylabel("expert density $\\rho_{m,t}(r)$")
    ax.set_title("Expert density vs. radius (core anatomical scopes, error bars = Wilson 95% CI)")
    ax.set_xticks(list(RADII))
    ax.set_xticklabels([f"{r:g}" for r in RADII])
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
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
        data = load_all_data(results_dir)
    except FigureDataError as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 1

    fig = plot_density_vs_radius(data)
    paths = save_figure(fig, figures_dir, "figure_c_density_vs_radius")
    plt.close(fig)

    print("Figure C written:")
    for p in paths:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

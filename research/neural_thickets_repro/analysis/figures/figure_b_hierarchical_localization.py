"""Figure B -- hierarchical visual localization tree: vision_encoder (context) -> {early,
middle, late} @ r=.04 -> {late_a, late_b} @ r=.04 beneath late. Every density value and the
vision_late_a-vs-vision_late_b McNemar p-value are read directly from the validated result
JSONs -- nothing is hardcoded.

Visual encoding is deliberately asymmetric, matching the actual statistical support:
  - vision_encoder is drawn muted/grey -- shown for context only, not a re-tested node here.
  - vision_late is emphasized (bold, highlighted) -- the supported localization level.
  - vision_late_a / vision_late_b are drawn identically to each other and NOT emphasized,
    with the null paired-comparison result annotated directly on the figure -- the paired
    McNemar test found no significant difference between them, so this figure must not
    visually suggest either half is a further hotspot.

Analysis/visualization only -- runs no inference, no perturbations, reads existing results.

Usage:
    python analysis/figures/figure_b_hierarchical_localization.py [--results-dir results] [--figures-dir results/paper_figures]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DEFAULT_FIGURES_DIR, DEFAULT_RESULTS_DIR, FigureDataError, load_json_file, load_metrics, save_figure  # noqa: E402

RADIUS = 0.04

MUTED = dict(facecolor="#e8e8e8", edgecolor="#888888", linewidth=1.0, linestyle="--")
NEUTRAL = dict(facecolor="#cfe3f7", edgecolor="#3b6ea5", linewidth=1.2, linestyle="-")
EMPHASIS = dict(facecolor="#ffd9a0", edgecolor="#c9660d", linewidth=2.6, linestyle="-")
NULL_RESULT = dict(facecolor="#eeeeee", edgecolor="#999999", linewidth=1.0, linestyle=":")


def _draw_box(ax, cx, cy, w, h, title, subtitle, style):
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor=style["facecolor"], edgecolor=style["edgecolor"],
        linewidth=style["linewidth"], linestyle=style["linestyle"],
    )
    ax.add_patch(box)
    ax.text(cx, cy + h * 0.14, title, ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(cx, cy - h * 0.20, subtitle, ha="center", va="center", fontsize=9)


def _connect(ax, x1, y1, x2, y2):
    ax.plot([x1, x2], [y1, y2], color="#666666", linewidth=1.0, zorder=0)


def load_all_data(results_dir: Path) -> dict:
    vision_encoder = load_metrics(results_dir, "vision_encoder", RADIUS)
    vision_early = load_metrics(results_dir, "vision_early", RADIUS)
    vision_middle = load_metrics(results_dir, "vision_middle", RADIUS)
    vision_late = load_metrics(results_dir, "vision_late", RADIUS)
    vision_late_a = load_metrics(results_dir, "vision_late_a", RADIUS)
    vision_late_b = load_metrics(results_dir, "vision_late_b", RADIUS)
    paired_summary = load_json_file(
        results_dir / "vision_late_split_experiment" / "analysis" / "analysis_summary.json"
    )
    mcnemar_p = paired_summary["paired_comparison"]["mcnemar_exact_p"]
    return {
        "vision_encoder": vision_encoder, "vision_early": vision_early, "vision_middle": vision_middle,
        "vision_late": vision_late, "vision_late_a": vision_late_a, "vision_late_b": vision_late_b,
        "mcnemar_p": mcnemar_p,
    }


def plot_hierarchy(data: dict):
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(f"Hierarchical visual-encoder localization (r={RADIUS:g})", fontsize=12, pad=14)

    root_y, mid_y, leaf_y = 0.88, 0.55, 0.18
    early_x, middle_x, late_x = 0.18, 0.5, 0.82
    late_a_x, late_b_x = 0.70, 0.94

    box_w, box_h = 0.24, 0.16
    leaf_w, leaf_h = 0.20, 0.14

    _draw_box(
        ax, 0.5, root_y, 0.30, box_h,
        "vision_encoder (context)", f"blocks 0-31, density={data['vision_encoder']['expert_density']:.2f}",
        MUTED,
    )

    _connect(ax, 0.5, root_y - box_h / 2, early_x, mid_y + box_h / 2)
    _connect(ax, 0.5, root_y - box_h / 2, middle_x, mid_y + box_h / 2)
    _connect(ax, 0.5, root_y - box_h / 2, late_x, mid_y + box_h / 2)

    _draw_box(ax, early_x, mid_y, box_w, box_h, "vision_early", f"blocks 0-10, density={data['vision_early']['expert_density']:.2f}", NEUTRAL)
    _draw_box(ax, middle_x, mid_y, box_w, box_h, "vision_middle", f"blocks 11-21, density={data['vision_middle']['expert_density']:.2f}", NEUTRAL)
    _draw_box(
        ax, late_x, mid_y, box_w, box_h,
        "vision_late ★ SUPPORTED", f"blocks 22-31, density={data['vision_late']['expert_density']:.2f}",
        EMPHASIS,
    )

    _connect(ax, late_x, mid_y - box_h / 2, late_a_x, leaf_y + leaf_h / 2)
    _connect(ax, late_x, mid_y - box_h / 2, late_b_x, leaf_y + leaf_h / 2)

    _draw_box(ax, late_a_x, leaf_y, leaf_w, leaf_h, "vision_late_a", f"blocks 22-26, density={data['vision_late_a']['expert_density']:.2f}", NULL_RESULT)
    _draw_box(ax, late_b_x, leaf_y, leaf_w, leaf_h, "vision_late_b", f"blocks 27-31, density={data['vision_late_b']['expert_density']:.2f}", NULL_RESULT)

    ax.text(
        (late_a_x + late_b_x) / 2, leaf_y - leaf_h * 0.85,
        f"paired McNemar p={data['mcnemar_p']:.3f} (n.s.) -- not further localized",
        ha="center", va="center", fontsize=8.5, style="italic", color="#555555",
    )

    legend_handles = [
        FancyBboxPatch((0, 0), 1, 1, facecolor=EMPHASIS["facecolor"], edgecolor=EMPHASIS["edgecolor"], linewidth=EMPHASIS["linewidth"]),
        FancyBboxPatch((0, 0), 1, 1, facecolor=NEUTRAL["facecolor"], edgecolor=NEUTRAL["edgecolor"], linewidth=NEUTRAL["linewidth"]),
        FancyBboxPatch((0, 0), 1, 1, facecolor=NULL_RESULT["facecolor"], edgecolor=NULL_RESULT["edgecolor"], linewidth=NULL_RESULT["linewidth"]),
        FancyBboxPatch((0, 0), 1, 1, facecolor=MUTED["facecolor"], edgecolor=MUTED["edgecolor"], linewidth=MUTED["linewidth"]),
    ]
    legend_labels = ["Supported localization level", "Measured (not the localization target)", "Null paired comparison (not further localized)", "Parent scope, context only"]
    ax.legend(legend_handles, legend_labels, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=1, fontsize=8, frameon=False)

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

    fig = plot_hierarchy(data)
    paths = save_figure(fig, figures_dir, "figure_b_hierarchical_localization")
    plt.close(fig)

    print("Figure B written:")
    for p in paths:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

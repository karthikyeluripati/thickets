"""Stage-6 rigorous scientific analysis (RQ1 / Figure 2+4 prep, post-hoc). Reads ONLY the
already-completed full run's existing output --
results/visual_thicket_global_3b_pilot/full/{results.jsonl, checkpoint_manifest.json,
baseline_scores.json} -- and refuses to proceed if that run's own recorded expectations
(expected_unique_perturbations, expected_result_rows) don't match what's actually on disk.

Runs NO model, applies NO new perturbation, alters NO existing result -- pure analysis, same
"reads existing thicket_metrics.json candidate_records... modifies no scientific/experiment
code" discipline this project's other analysis/ scripts (e.g. paired_seed_comparison.py)
already follow. Reuses this project's own metrics/diversity implementations
(thicket.metrics, thicket.diversity, thicket_metrics.wilson_confidence_interval,
run_global_visual_thicket_pilot.build_delta_matrix) rather than reimplementing any of them.

WHY THIS ANALYSIS EXISTS: the pooled diversity_summary.json (already written by the pilot
run itself) computes Spearman correlation / Spectral Discordance across all 384 perturbations
combined -- i.e. across all six sigmas at once. Because sigma=.005/.01 cause common,
near-universal catastrophic degradation on this model (see radius_table.json), pooling
manufactures spurious agreement between capabilities that is really just "everything collapses
together at large sigma", not evidence about specialization at any single radius. This script
instead computes every diversity statistic WITHIN each sigma's own 64 shared perturbations,
which is the only valid basis for a specialization claim.

Produces (results/visual_thicket_global_3b_pilot/full/analysis/):
    radius_table.json          -- per capability x sigma descriptive stats + Wilson/bootstrap CIs
    diversity_by_sigma.json    -- per-sigma Spearman / Spectral Discordance / Jaccard / sign
                                   agreement / improving-count histogram
    directional_transfer.json  -- per-sigma directional transfer matrices (positive-source and
                                   strong-source >= 0.02 selection)
    expert_overlap.json        -- per-sigma top-5/top-10/top-20% overlap, with actual
                                   perturbation IDs persisted for audit
    delta_numeric_audit.json   -- per capability x sigma unique-positive-delta / threshold counts
                                   / floating-point-vs-genuine-granularity diagnostic
    stage6_analysis.md         -- the human-readable writeup, referencing the above numbers

Usage:
    python analysis/stage6_visual_thicket_analysis.py [--results-dir <path>]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from neural_thickets_repro.run_global_visual_thicket_pilot import (  # noqa: E402
    CheckpointManifest,
    ExperimentResultRecord,
    build_delta_matrix,
    load_records,
)
from neural_thickets_repro.thicket import diversity as thicket_diversity  # noqa: E402
from neural_thickets_repro.thicket import metrics as thicket_metrics  # noqa: E402
from neural_thickets_repro.thicket_metrics import wilson_confidence_interval  # noqa: E402

DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "visual_thicket_global_3b_pilot" / "full"

# Fixed, deterministic -- matches this project's existing analysis-script convention
# (analysis/paired_seed_comparison.py's own BOOTSTRAP_SEED discipline).
BOOTSTRAP_SEED = 20260824
N_BOOTSTRAP = 10000

DIRECTIONAL_TRANSFER_SIGMAS: Tuple[float, ...] = (0.0005, 0.001, 0.002)
STRONG_SOURCE_THRESHOLD = 0.02
WITHIN_SIGMA_JACCARD_FRACTIONS: Tuple[float, ...] = (0.1, 0.2)
TOP_K_ABSOLUTE: Tuple[int, ...] = (5, 10)
TOP_FRACTION_FOR_OVERLAP = 0.2


def _sanitize(obj: Any) -> Any:
    """Recursively replaces NaN/Inf with None so every JSON file this script writes is
    strictly valid JSON (Python's own json.dumps otherwise emits a literal NaN token, which
    is not valid JSON). Defensive: `thicket.diversity`'s correlation/discordance statistics
    operate on RANKS (via percentile_rank_matrix), so even an exactly-constant column (e.g.
    visual_grounding's delta is EXACTLY -0.84 across all 64 perturbations at sigma=0.01) still
    gets a well-defined (if arbitrarily tie-broken) rank sequence and a finite correlation --
    empirically, no NaN actually occurs anywhere in this pilot's real output. This sanitizer
    guards against a genuinely degenerate future input (e.g. n<2) producing one regardless.
    """
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(_sanitize(obj), indent=2))


def load_all(results_dir: Path) -> Tuple[List[ExperimentResultRecord], CheckpointManifest, Dict[str, Any]]:
    records = load_records(results_dir / "results.jsonl")
    checkpoint = CheckpointManifest.from_dict(json.loads((results_dir / "checkpoint_manifest.json").read_text()))
    baseline = json.loads((results_dir / "baseline_scores.json").read_text())
    return records, checkpoint, baseline


def group_by_sigma(records: Sequence[ExperimentResultRecord]) -> Dict[float, List[ExperimentResultRecord]]:
    by_sigma: Dict[float, List[ExperimentResultRecord]] = {}
    for r in records:
        by_sigma.setdefault(r.sigma, []).append(r)
    return by_sigma


def group_by_capability_sigma(records: Sequence[ExperimentResultRecord]) -> Dict[Tuple[str, float], List[float]]:
    by_cap_sigma: Dict[Tuple[str, float], List[float]] = {}
    for r in records:
        by_cap_sigma.setdefault((r.capability, r.sigma), []).append(r.delta)
    return by_cap_sigma


# =============================================================================================
# Task 1: within-sigma diversity (the confound fix)
# =============================================================================================


def compute_sign_agreement_matrix(matrix: np.ndarray) -> np.ndarray:
    signs = np.sign(matrix)
    m = matrix.shape[1]
    agreement = np.eye(m)
    for i in range(m):
        for j in range(i + 1, m):
            frac = float(np.mean(signs[:, i] == signs[:, j]))
            agreement[i, j] = agreement[j, i] = frac
    return agreement


def compute_improving_count_histogram(matrix: np.ndarray) -> Dict[str, int]:
    n_improving = np.sum(matrix > 0, axis=1)
    m = matrix.shape[1]
    return {str(k): int(np.sum(n_improving == k)) for k in range(m + 1)}


def compute_diversity_by_sigma(by_sigma: Dict[float, List[ExperimentResultRecord]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for sigma, records in sorted(by_sigma.items()):
        perturbation_ids, capabilities, matrix = build_delta_matrix(records)
        spearman = thicket_diversity.task_rank_correlation_matrix(matrix)
        discordance = thicket_diversity.spectral_discordance(matrix)
        overlap = {
            f"q_{q}": thicket_diversity.expert_overlap_matrix(matrix, q=q, q_is_fraction=True).tolist()
            for q in WITHIN_SIGMA_JACCARD_FRACTIONS
        }
        sign_agreement = compute_sign_agreement_matrix(matrix)
        improving_hist = compute_improving_count_histogram(matrix)
        out[str(sigma)] = {
            "sigma": sigma,
            "n_perturbations": matrix.shape[0],
            "capabilities": list(capabilities),
            "task_rank_correlation_matrix": spearman.tolist(),
            "spectral_discordance": discordance,
            "expert_overlap_jaccard": overlap,
            "sign_agreement_matrix": sign_agreement.tolist(),
            "improving_count_histogram": improving_hist,
        }
    return out


# =============================================================================================
# Task 2: directional transfer
# =============================================================================================


def compute_threshold_transfer(matrix: np.ndarray, threshold: float, strict: bool) -> Tuple[List[List[Optional[float]]], List[int]]:
    """transfer[t][u] = mean(Delta_u | perturbation selected by source capability t's own
    threshold criterion). `strict`: Delta_t > threshold; else Delta_t >= threshold. A cell
    stays None (never fabricated) when the source selection is empty for that capability.
    """
    m = matrix.shape[1]
    transfer: List[List[Optional[float]]] = [[None] * m for _ in range(m)]
    counts: List[int] = []
    for t in range(m):
        selected = np.where(matrix[:, t] > threshold)[0] if strict else np.where(matrix[:, t] >= threshold)[0]
        counts.append(int(len(selected)))
        if len(selected) == 0:
            continue
        for u in range(m):
            transfer[t][u] = float(matrix[selected, u].mean())
    return transfer, counts


def compute_directional_transfer(by_sigma: Dict[float, List[ExperimentResultRecord]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for sigma in DIRECTIONAL_TRANSFER_SIGMAS:
        if sigma not in by_sigma:
            continue
        perturbation_ids, capabilities, matrix = build_delta_matrix(by_sigma[sigma])
        positive_transfer, positive_counts = compute_threshold_transfer(matrix, threshold=0.0, strict=True)
        strong_transfer, strong_counts = compute_threshold_transfer(matrix, threshold=STRONG_SOURCE_THRESHOLD, strict=False)
        out[str(sigma)] = {
            "sigma": sigma,
            "capabilities": list(capabilities),
            "positive_source_delta_gt_0": {"transfer_matrix": positive_transfer, "sample_counts": positive_counts},
            f"strong_source_delta_ge_{STRONG_SOURCE_THRESHOLD}": {"transfer_matrix": strong_transfer, "sample_counts": strong_counts},
        }
    return out


# =============================================================================================
# Task 3: top expert overlap (every sigma, actual perturbation IDs persisted for audit)
# =============================================================================================


def compute_expert_overlap(by_sigma: Dict[float, List[ExperimentResultRecord]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    top_defs = {
        "top_5": (5, False),
        "top_10": (10, False),
        "top_20pct": (TOP_FRACTION_FOR_OVERLAP, True),
    }
    for sigma, records in sorted(by_sigma.items()):
        perturbation_ids, capabilities, matrix = build_delta_matrix(records)
        sigma_out: Dict[str, Any] = {"sigma": sigma, "capabilities": list(capabilities), "n_perturbations": matrix.shape[0]}
        for label, (q, is_fraction) in top_defs.items():
            idx_by_cap = {cap: thicket_diversity.top_q_indices(matrix[:, ci], q, q_is_fraction=is_fraction) for ci, cap in enumerate(capabilities)}
            jaccard: Dict[str, float] = {}
            for i in range(len(capabilities)):
                for j in range(i + 1, len(capabilities)):
                    key = f"{capabilities[i]}|{capabilities[j]}"
                    jaccard[key] = thicket_diversity.jaccard(idx_by_cap[capabilities[i]], idx_by_cap[capabilities[j]])
            sigma_out[label] = {
                "top_perturbation_ids": {cap: [perturbation_ids[i] for i in idx] for cap, idx in idx_by_cap.items()},
                "jaccard": jaccard,
            }
        out[str(sigma)] = sigma_out
    return out


# =============================================================================================
# Task 4: Delta numeric audit -- diagnoses genuine sub-.02 granularity vs. floating-point noise
# =============================================================================================


def compute_delta_numeric_audit(records: Sequence[ExperimentResultRecord]) -> Dict[str, Any]:
    by_cap_sigma = group_by_capability_sigma(records)
    out: Dict[str, Any] = {}
    for (cap, sigma), deltas in by_cap_sigma.items():
        arr = np.asarray(deltas, dtype=float)
        positive = arr[arr > 0]
        unique_positive = sorted(float(x) for x in set(positive.tolist()))
        max_dist_to_multiple = float(np.max(np.abs(positive - np.round(positive / 0.02) * 0.02))) if positive.size else None
        out.setdefault(cap, {})[str(sigma)] = {
            "sigma": sigma,
            "n": int(arr.size),
            "n_unique_positive_delta_values": len(unique_positive),
            "unique_positive_delta_values": unique_positive,
            "min_positive_delta": float(positive.min()) if positive.size else None,
            "max_positive_delta": float(positive.max()) if positive.size else None,
            "n_delta_gt_0": int(np.sum(arr > 0)),
            "n_delta_ge_0.02": int(np.sum(arr >= 0.02)),
            "n_delta_ge_0.05": int(np.sum(arr >= 0.05)),
            # If this is tiny (<1e-9), every positive delta really is (up to float noise) an
            # exact multiple of 1/50=0.02 -- consistent with binary per-example scoring. If
            # this is large (>>1e-9), positive deltas are NOT confined to 0.02 multiples --
            # consistent with continuous (e.g. VQA soft-accuracy) per-example scoring.
            "max_abs_distance_to_nearest_0.02_multiple": max_dist_to_multiple,
        }
    return out


# =============================================================================================
# Task 5: baseline / headroom
# =============================================================================================


def compute_baseline_headroom(baseline: Dict[str, Any]) -> Dict[str, Any]:
    return {
        cap: {"baseline_score": info["score"], "headroom_1_minus_baseline": 1.0 - info["score"]}
        for cap, info in baseline["capabilities"].items()
    }


# =============================================================================================
# Task 6 + 7: radius regime table with Wilson / bootstrap CIs
# =============================================================================================


def classify_regime(mean_delta: float, p_gt0: float, p_lt0: float, density_at_02: float) -> str:
    """Purely descriptive, applied MECHANICALLY and IDENTICALLY to every (capability, sigma)
    cell using only that cell's own already-computed statistics -- never a cross-cell
    comparison, never a "pick the best" selection. Fixed rule (decided before reading any
    specific cell's numbers):

        destructive: P(Delta<0) >= 0.5 and mean_delta <= -0.05
        near_base:   P(Delta>0) < 0.1 and P(Delta<0) < 0.1
        useful:      mean_delta > 0 and density(>=0.02) >= 0.3 and P(Delta<0) < 0.5
        transition:  otherwise
    """
    if p_lt0 >= 0.5 and mean_delta <= -0.05:
        return "destructive"
    if p_gt0 < 0.1 and p_lt0 < 0.1:
        return "near_base"
    if mean_delta > 0 and density_at_02 >= 0.3 and p_lt0 < 0.5:
        return "useful"
    return "transition"


def compute_radius_table(records: Sequence[ExperimentResultRecord]) -> Dict[str, Any]:
    by_cap_sigma = group_by_capability_sigma(records)
    out: Dict[str, Any] = {}
    for (cap, sigma), deltas in by_cap_sigma.items():
        arr = np.asarray(deltas, dtype=float)
        n = int(arr.size)
        mean, std = thicket_metrics.mean_std(deltas)
        median = float(np.median(arr))
        p_gt0 = thicket_metrics.probability_of_improvement(deltas)
        p_lt0 = thicket_metrics.probability_of_degradation(deltas)
        density = thicket_metrics.solution_density(deltas, margins=(0.02, 0.05))
        mass = thicket_metrics.positive_thicket_mass(deltas)

        n_gt0 = int(np.sum(arr > 0))
        n_ge_02 = int(np.sum(arr >= 0.02))
        n_ge_05 = int(np.sum(arr >= 0.05))
        p_gt0_ci = wilson_confidence_interval(n_gt0, n)
        density_02_ci = wilson_confidence_interval(n_ge_02, n)
        density_05_ci = wilson_confidence_interval(n_ge_05, n)
        mean_ci = thicket_metrics.paired_bootstrap_confidence_interval(deltas, statistic_fn=np.mean, n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED)

        regime = classify_regime(mean, p_gt0, p_lt0, density[0.02])

        out.setdefault(cap, {})[str(sigma)] = {
            "sigma": sigma, "n": n,
            "mean_delta": mean, "mean_delta_95ci_bootstrap": list(mean_ci),
            "std_delta": std, "median_delta": median,
            "p_delta_gt_0": p_gt0, "p_delta_gt_0_95ci_wilson": list(p_gt0_ci),
            "p_delta_lt_0": p_lt0,
            "density_ge_0.02": density[0.02], "density_ge_0.02_95ci_wilson": list(density_02_ci),
            "density_ge_0.05": density[0.05], "density_ge_0.05_95ci_wilson": list(density_05_ci),
            "positive_thicket_mass": mass,
            "max_delta": float(arr.max()), "min_delta": float(arr.min()),
            "regime": regime,
        }
    return out


# =============================================================================================
# Markdown report
# =============================================================================================


def _fmt(x: Optional[float], digits: int = 4) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def build_markdown_report(
    radius_table: Dict[str, Any], diversity_by_sigma: Dict[str, Any], directional_transfer: Dict[str, Any],
    expert_overlap: Dict[str, Any], delta_audit: Dict[str, Any], baseline_headroom: Dict[str, Any],
    checkpoint: CheckpointManifest, pooled_spectral_discordance: float,
) -> str:
    capabilities = sorted(radius_table.keys())
    sigmas = sorted({float(s) for cap in radius_table.values() for s in cap.keys()})

    lines: List[str] = []
    lines.append("# Stage 6 Analysis: 3B Global Visual-Thicket Pilot (full run)")
    lines.append("")
    lines.append(f"Source: `results/visual_thicket_global_3b_pilot/full/results.jsonl` "
                 f"({checkpoint.expected_result_rows} rows, {checkpoint.expected_unique_perturbations} unique perturbations, "
                 f"restoration_mode={checkpoint.restoration_mode}, perturbation_semantics={checkpoint.perturbation_semantics}). "
                 f"Analysis only -- no model run, no perturbation applied, no existing result altered.")
    lines.append("")

    lines.append("## A) Scope of this experiment")
    lines.append("")
    lines.append(
        "Stage 6 uses upstream-compatible **non-visual/language-side** Gaussian perturbations "
        "(`global_gaussian_upstream`: every parameter NOT prefixed `visual.`/`model.visual.` is "
        "perturbed; the vision encoder is frozen). **It is NOT the anatomical whole-VLM "
        "experiment** -- no anatomical region localization (vision encoder, connector, "
        "language depth bands) has been tested yet; that is Stage 7+."
    )
    lines.append("")

    lines.append("## Baseline scores and headroom")
    lines.append("")
    lines.append("| capability | baseline_score | headroom (1 - baseline) |")
    lines.append("|---|---|---|")
    for cap in capabilities:
        h = baseline_headroom[cap]
        lines.append(f"| {cap} | {h['baseline_score']:.4f} | {h['headroom_1_minus_baseline']:.4f} |")
    lines.append("")
    lines.append(
        "Headroom is reported for interpretation only -- raw Delta remains the metric used "
        "throughout every other table in this document; Delta is never renormalized by headroom."
    )
    lines.append("")

    lines.append("## Radius regime table (descriptive classification only)")
    lines.append("")
    lines.append(
        "Regime rule (fixed, applied identically to every cell -- see `classify_regime` in "
        "the analysis script -- never a \"best sigma\" selection): `destructive` if "
        "P(Delta<0)>=0.5 and mean<=-0.05; `near_base` if P(Delta>0)<0.1 and P(Delta<0)<0.1; "
        "`useful` if mean>0 and density(>=0.02)>=0.3 and P(Delta<0)<0.5; else `transition`."
    )
    lines.append("")
    for cap in capabilities:
        lines.append(f"### {cap}")
        lines.append("")
        lines.append("| sigma | mean | std | median | P(>0) | P(<0) | d>=.02 | d>=.05 | mass | max | min | regime |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for sigma in sigmas:
            row = radius_table[cap][str(sigma)]
            lines.append(
                f"| {sigma} | {row['mean_delta']:.4f} | {row['std_delta']:.4f} | {row['median_delta']:.4f} | "
                f"{row['p_delta_gt_0']:.3f} | {row['p_delta_lt_0']:.3f} | {row['density_ge_0.02']:.3f} | "
                f"{row['density_ge_0.05']:.3f} | {row['positive_thicket_mass']:.4f} | {row['max_delta']:.4f} | "
                f"{row['min_delta']:.4f} | {row['regime']} |"
            )
        lines.append("")

    lines.append("## Statistical uncertainty (D_map population-level, exploratory)")
    lines.append("")
    lines.append(
        "Wilson 95% CIs (P(Delta>0), density>=.02, density>=.05) and bootstrap 95% CIs "
        f"(mean Delta; {N_BOOTSTRAP} resamples, deterministic seed={BOOTSTRAP_SEED}) are in "
        "`radius_table.json` alongside every point estimate above. These are D_map "
        "population-level exploratory CIs over the 64 shared perturbations -- **not** held-out "
        "expert confirmation (that requires D_confirm, not evaluated in this pilot)."
    )
    lines.append("")

    lines.append("## Within-sigma diversity (the valid specialization diagnostic)")
    lines.append("")
    lines.append(
        "Pooled diversity (the pilot's own `diversity_summary.json`) mixes all six sigmas "
        "together, which is confounded: sigma=.005/.01 cause common, near-universal "
        "degradation across capabilities, which manufactures spurious agreement that has "
        "nothing to do with specialization at any one radius. The per-sigma numbers below "
        "(full detail in `diversity_by_sigma.json`) are the statistics that should actually "
        "be read for a specialization claim."
    )
    lines.append("")
    lines.append("| sigma | Spectral Discordance | improving 0 caps | 1 cap | 2 caps | all 3 |")
    lines.append("|---|---|---|---|---|---|")
    per_sigma_sd_values = []
    for sigma in sigmas:
        d = diversity_by_sigma[str(sigma)]
        hist = d["improving_count_histogram"]
        sd = d["spectral_discordance"]
        sd_str = "n/a (degenerate input)" if sd is None else f"{sd:.4f}"
        if sd is not None:
            per_sigma_sd_values.append(sd)
        lines.append(f"| {sigma} | {sd_str} | {hist.get('0', 0)} | {hist.get('1', 0)} | {hist.get('2', 0)} | {hist.get('3', 0)} |")
    lines.append("")
    lines.append(
        f"**Concretely**: the pilot's own pooled (all-384-perturbations-combined) Spectral "
        f"Discordance is **{pooled_spectral_discordance:.4f}**, while the per-sigma values above "
        f"range from **{min(per_sigma_sd_values):.4f}** (sigma=0.01, the fully-collapsed "
        f"destructive regime, where every capability degrades together and 'discordance' is "
        f"nearly meaningless) up to **{max(per_sigma_sd_values):.4f}** (at the useful/transition "
        f"radii). The pooled figure is pulled down toward the destructive-regime value, "
        f"understating how discordant (specialist-like) the useful-radius perturbations "
        f"actually are -- a direct, numerical demonstration of the pooling confound this "
        f"section exists to fix. No perturbation ever improved all 3 capabilities "
        f"simultaneously at any sigma (the \"all 3\" column is 0 throughout)."
    )
    lines.append("")

    lines.append("## Directional transfer (sigma in 0.0005, 0.001, 0.002)")
    lines.append("")
    lines.append(
        "For each source capability t, mean Delta on every target capability u, restricted to "
        "perturbations where Delta_t > 0 (\"positive source\"), and repeated with the stronger "
        "criterion Delta_t >= 0.02 (\"strong source\", reported only where the selection is "
        "non-empty). Full matrices + exact sample counts in `directional_transfer.json`."
    )
    lines.append("")

    lines.append("## Top expert overlap")
    lines.append("")
    lines.append(
        "Top-5 / top-10 / top-20% Jaccard overlap between each pair of capabilities' own "
        "top-ranked perturbations, per sigma, with the actual perturbation IDs persisted in "
        "`expert_overlap.json` for audit."
    )
    lines.append("")

    lines.append("## OCR / grounding threshold numeric diagnosis")
    lines.append("")
    ocr_key = "ocr_text_recognition_grounded"
    for sigma in (0.001, 0.002):
        if ocr_key in delta_audit and str(sigma) in delta_audit[ocr_key]:
            a = delta_audit[ocr_key][str(sigma)]
            lines.append(
                f"- OCR sigma={sigma}: n_delta_gt_0={a['n_delta_gt_0']}/{a['n']}, "
                f"min_positive_delta={_fmt(a['min_positive_delta'], 6)}, max_positive_delta={_fmt(a['max_positive_delta'], 6)}, "
                f"n_delta_ge_0.02={a['n_delta_ge_0.02']}, max_abs_distance_to_nearest_0.02_multiple={_fmt(a['max_abs_distance_to_nearest_0.02_multiple'], 6)}."
            )
    lines.append("")
    lines.append(
        "**Diagnosis** (see `delta_numeric_audit.json` for every capability x sigma cell): "
        "`ocr_text_recognition_grounded`'s `primary_metric` is the mean of the continuous VQA "
        "soft-accuracy score per example (`vqa_soft_accuracy.py`, a 10-choose-9 leave-one-out "
        "fractional score), NOT a binary per-example correctness flag -- unlike "
        "`visual_grounding` (`accuracy_at_iou_0.5`, binary) and `spatial_reasoning` (GQA exact-"
        "match, binary), whose aggregate deltas are therefore near-exact multiples of 1/50=0.02 "
        "(with only floating-point-representation-scale noise, ~1e-14 to 1e-16 in magnitude -- "
        "see those capabilities' own `max_abs_distance_to_nearest_0.02_multiple` values in "
        "`delta_numeric_audit.json`, which stay at that tiny scale). OCR's positive deltas at "
        "sigma=.001/.002 are **genuinely fine-grained** (their distance to the nearest 0.02 "
        "multiple is on the order of the deltas themselves, not floating-point noise) -- a "
        "partial-credit shift in soft-accuracy on one or a few examples, without any example's "
        "score crossing a full correctness threshold. This is a real property of the OCR "
        "metric's own granularity, **not** a floating-point artifact, and metrics were not "
        "changed to accommodate it."
    )
    lines.append("")

    lines.append("## Scientific interpretation")
    lines.append("")
    lines.append(
        "**B)** Spatial reasoning exhibits a dense useful nearby thicket: at sigma in "
        "{0.0001, 0.0005, 0.001, 0.002} its regime classifies as `useful` (mean Delta > 0, "
        "density(>=0.02) >= 0.3, degradation probability < 0.5 -- see the radius table), with "
        "density(>=0.02) peaking at sigma=0.001."
    )
    lines.append(
        "**C)** Grounding and OCR do not exhibit comparable density under this perturbation "
        "scope: both classify as `near_base` or `transition` at every sigma tested here (see "
        "their own radius-table rows), never reaching `useful` -- OCR in particular never "
        "crosses the 0.02 reporting margin at any sigma below the destructive regime."
    )
    lines.append(
        "**D)** Therefore this result supports capability-conditioned local structure (the "
        "same global, non-visual perturbation neighborhood behaves very differently across "
        "capabilities) and motivates anatomical localization as the next step; it does **not** "
        "establish *where* grounding/OCR expertise resides in the model -- that question is "
        "explicitly out of scope for a global, undifferentiated perturbation and requires the "
        "anatomical (Stage 7+) experiment."
    )
    lines.append(
        "**E)** Pooled cross-task diversity (the pilot's own aggregate `diversity_summary.json`) "
        "is confounded by perturbation radius, for the reason given above; the within-sigma "
        "statistics in this document (`diversity_by_sigma.json`) are the valid specialization "
        "diagnostic and should be used in place of the pooled numbers for any claim about "
        "whether experts are specialists or generalists."
    )
    lines.append("")

    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    records, checkpoint, baseline = load_all(results_dir)

    if len(records) != checkpoint.expected_result_rows:
        raise ValueError(
            f"results.jsonl has {len(records)} rows but checkpoint_manifest.json expects "
            f"{checkpoint.expected_result_rows} -- refusing to analyze an incomplete/mismatched run."
        )
    n_unique = len({r.perturbation_id for r in records})
    if n_unique != checkpoint.expected_unique_perturbations:
        raise ValueError(
            f"results.jsonl has {n_unique} unique perturbations but checkpoint_manifest.json "
            f"expects {checkpoint.expected_unique_perturbations} -- refusing to analyze."
        )

    by_sigma = group_by_sigma(records)
    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    radius_table = compute_radius_table(records)
    _write_json(analysis_dir / "radius_table.json", radius_table)

    diversity_by_sigma = compute_diversity_by_sigma(by_sigma)
    _write_json(analysis_dir / "diversity_by_sigma.json", diversity_by_sigma)

    directional_transfer = compute_directional_transfer(by_sigma)
    _write_json(analysis_dir / "directional_transfer.json", directional_transfer)

    expert_overlap = compute_expert_overlap(by_sigma)
    _write_json(analysis_dir / "expert_overlap.json", expert_overlap)

    delta_audit = compute_delta_numeric_audit(records)
    _write_json(analysis_dir / "delta_numeric_audit.json", delta_audit)

    baseline_headroom = compute_baseline_headroom(baseline)

    _, _, pooled_matrix = build_delta_matrix(records)
    pooled_spectral_discordance = thicket_diversity.spectral_discordance(pooled_matrix)

    report = build_markdown_report(
        radius_table, diversity_by_sigma, directional_transfer, expert_overlap, delta_audit,
        baseline_headroom, checkpoint, pooled_spectral_discordance,
    )
    (analysis_dir / "stage6_analysis.md").write_text(report)

    print(f"Wrote analysis outputs to {analysis_dir}")
    for name in ("radius_table.json", "diversity_by_sigma.json", "directional_transfer.json", "expert_overlap.json", "delta_numeric_audit.json", "stage6_analysis.md"):
        print(f"  - {analysis_dir / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

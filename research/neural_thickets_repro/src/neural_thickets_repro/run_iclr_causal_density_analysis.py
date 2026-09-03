"""Step 12: the REAL Phase 7-10 analysis driver for the isolated 7B causal-density pilot.
Branch `iclr-causal-density-pilot` ONLY; never merged into main by this script or any other.
STRICTLY 7B-only, CPU-only -- no vllm/ray/torch/GPU import anywhere in this module.

Reads the real, already-collected, already-completeness-validated result artifacts
(base_control_report.json, results.jsonl[.gz], results_selection.jsonl[.gz]) and runs the
frozen, UNMODIFIED iclr_causal_density.metrics/search_budget/grounded_selection/decision_gate
modules against them, plus the frozen, separately-committed capability_divergence_aggregation
module (see that module's own docstring for what gap it fills and when it was authorized).
This script contains ZERO new scientific logic -- only I/O, orchestration, and JSON output.

visual_grounding is excluded from Phase 7-9: RefCOCO grounding has no meaningful text_only
condition, and metrics.ConditionScores requires all three conditions for every capability by
construction -- a structural incompatibility with the frozen Phase 7 formula, not a data gap
(see reports/iclr_causal_density/preregistration_amendment_2026-09-03.md for the full record).
Because decision_gate.evaluate_decision_gate (unmodified) requires exactly 5 capabilities with
valid results, this capability's absence deterministically routes the decision through that
module's own pre-existing precedence rule 2 ("not exactly 5 capabilities -> INCONCLUSIVE").

Usage:
    python -m neural_thickets_repro.run_iclr_causal_density_analysis \\
        --data-dir /path/to/results_backup --output-dir reports/iclr_causal_density
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

_FORBIDDEN_32B_72B_SUBSTRINGS = (
    "stage11_coarse_anatomical_atlas_32b", "stage11_32b_s2_live_evidence", "stage11_32b_s2_live_v3_solver_probe",
    "--scale 32B", "--scale 72B", "32B", "72B",
)


def _ensure_no_32b_72b_in_argv(argv: Optional[Sequence[str]]) -> None:
    if not argv:
        return
    joined = " ".join(argv)
    for token in _FORBIDDEN_32B_72B_SUBSTRINGS:
        if token in joined:
            raise ValueError(f"run_iclr_causal_density_analysis.py refuses argv containing {token!r} -- this pilot is strictly 7B-only.")


def open_maybe_gz(path_no_ext: Path):
    """Prefers a `.gz` sibling of `path_no_ext` (streamed, never decompressed to disk) if
    present, else opens the plain file directly. Used for the two large per-example result
    files, which are committed to persistent storage compressed (see the artifact-provenance
    report) rather than checked into git.
    """
    gz_path = path_no_ext.with_suffix(path_no_ext.suffix + ".gz")
    if gz_path.exists():
        return gzip.open(gz_path, "rt")
    return open(path_no_ext)


def load_base_condition_scores(base_report: Dict[str, Any], capability: str, subset: str):
    """Returns (sorted_sample_ids, ConditionScores, aggregate_scores_dict) for the base model's
    `subset` ("audit" or "selection") per-example scores for `capability`.
    """
    from .iclr_causal_density.metrics import ConditionScores

    def scores(cond):
        return base_report[capability][f"{subset}:{cond}"]["per_example_scores"]

    real, text, shuffle = scores("correct_image"), scores("text_only"), scores("shuffled_image")
    sample_ids = sorted(real.keys())
    if not (set(text.keys()) == set(real.keys()) == set(shuffle.keys())):
        raise ValueError(f"{capability}/{subset}: base condition sample_id sets don't match across conditions.")
    import numpy as np

    cs = ConditionScores(
        real=np.array([real[s] for s in sample_ids]), text=np.array([text[s] for s in sample_ids]), shuffle=np.array([shuffle[s] for s in sample_ids]),
    )
    aggregates = {"correct_image": float(np.mean(list(real.values()))), "text_only": float(np.mean(list(text.values()))), "shuffled_image": float(np.mean(list(shuffle.values())))}
    return sample_ids, cs, aggregates


def load_rows_by_candidate_capability_condition(path: Path) -> Tuple[Dict[Tuple[str, str, str], Dict[str, float]], Dict[str, Tuple[str, float, int]]]:
    """Streams a results(.jsonl or .jsonl.gz) file and returns:
      - rows_by_ccc: (candidate_id, capability, condition) -> {sample_id: per_example_score}
      - candidate_meta: candidate_id -> (scope, radius, seed)
    Never loads the whole file into one giant list -- only the two derived, much smaller
    aggregations are retained.
    """
    rows_by_ccc: Dict[Tuple[str, str, str], Dict[str, float]] = defaultdict(dict)
    candidate_meta: Dict[str, Tuple[str, float, int]] = {}
    with open_maybe_gz(path) as f:
        for line in f:
            r = json.loads(line)
            cid = r["candidate_id"]
            if cid is None:
                continue
            rows_by_ccc[(cid, r["capability"], r["condition"])][r["sample_id"]] = r["per_example_score"]
            candidate_meta[cid] = (r["scope"], r["radius"], r["seed"])
    return rows_by_ccc, candidate_meta


def run_phase7(eligible_caps: Sequence[str], base_audit: Dict[str, Any], audit_rows: Dict, candidate_meta: Dict) -> Dict[str, Any]:
    from .iclr_causal_density.metrics import build_resample_index_matrix, classify_candidate, compute_density_ratio, bootstrap_density_ratio_ci, ConditionScores
    import numpy as np

    # ONE shared paired-bootstrap resample matrix across every capability, per the frozen
    # design's own PREREGISTERED_BOOTSTRAP_METHOD_NOTE -- n_examples is derived from the actual
    # loaded audit-set size (200 in the real frozen design, guaranteed uniform across
    # capabilities by AUDIT_SET_SIZE) rather than hardcoded, so this function is correct for
    # whatever audit-set size the caller's data actually has.
    n_examples = len(base_audit[eligible_caps[0]][0])
    resample_matrix = build_resample_index_matrix(n_examples)
    result = {}
    for cap in eligible_caps:
        sample_ids, base_scores, _ = base_audit[cap]
        if len(sample_ids) != n_examples:
            raise ValueError(f"{cap}: audit-set size {len(sample_ids)} != the shared resample matrix's {n_examples} -- capabilities must share one audit-set size.")
        classifications = []
        cand_scores_by_id = {}
        for cid in candidate_meta:
            real, text, shuffle = audit_rows[(cid, cap, "correct_image")], audit_rows[(cid, cap, "text_only")], audit_rows[(cid, cap, "shuffled_image")]
            cs = ConditionScores(real=np.array([real[s] for s in sample_ids]), text=np.array([text[s] for s in sample_ids]), shuffle=np.array([shuffle[s] for s in sample_ids]))
            cand_scores_by_id[cid] = cs
            classifications.append(classify_candidate(cid, cs, base_scores, resample_matrix))
        classifications_by_id = {c.candidate_id: c for c in classifications}
        density = compute_density_ratio(classifications)
        ci_low, ci_high, d_dist = bootstrap_density_ratio_ci(cand_scores_by_id, base_scores, resample_matrix)
        result[cap] = {"classifications_by_id": classifications_by_id, "density": density, "D_ci_low": ci_low, "D_ci_high": ci_high, "n_valid_resamples": len(d_dist)}
    return result


def run_phase8(eligible_caps: Sequence[str], scopes: Sequence[str], radii: Sequence[float], selection_rows: Dict, candidates_by_cell: Dict, phase7: Dict) -> Dict[str, Any]:
    from .iclr_causal_density.search_budget import CandidatePoolEntry, monte_carlo_search_budget_analysis, check_registered_divergence, InsufficientPoolSizeError
    from .iclr_causal_density.capability_divergence_aggregation import capability_search_budget_divergence_confirmed
    import numpy as np

    result: Dict[str, Any] = {"per_cell": {}, "per_capability_divergence_confirmed": {}}
    for cap in eligible_caps:
        cell_divergence = {}
        for scope in scopes:
            for radius in radii:
                pool = []
                for cid in candidates_by_cell[(scope, radius)]:
                    real_score = float(np.mean(list(selection_rows[(cid, cap, "correct_image")].values())))
                    pool.append(CandidatePoolEntry(candidate_id=cid, selection_real_score=real_score, audit=phase7[cap]["classifications_by_id"][cid]))
                try:
                    budget_results = monte_carlo_search_budget_analysis(pool)
                except InsufficientPoolSizeError as exc:
                    result["per_cell"][(cap, scope, radius)] = {"error": str(exc)}
                    cell_divergence[(scope, radius)] = False
                    continue
                divergence = check_registered_divergence(budget_results)
                cell_divergence[(scope, radius)] = divergence["divergence_confirmed"]
                result["per_cell"][(cap, scope, radius)] = {"budget_points": {N: p.to_dict() for N, p in budget_results.items()}, "divergence": divergence}
        result["per_capability_divergence_confirmed"][cap] = capability_search_budget_divergence_confirmed(cell_divergence)
    return result


def run_phase9(eligible_caps: Sequence[str], base_selection: Dict[str, Any], selection_rows: Dict, candidate_meta: Dict, phase7: Dict) -> Dict[str, Any]:
    from .iclr_causal_density.grounded_selection import CandidateSelectionData, compare_standard_vs_grounded
    import numpy as np

    result = {}
    for cap in eligible_caps:
        pool = []
        for cid in candidate_meta:
            real = float(np.mean(list(selection_rows[(cid, cap, "correct_image")].values())))
            text = float(np.mean(list(selection_rows[(cid, cap, "text_only")].values())))
            shuffle = float(np.mean(list(selection_rows[(cid, cap, "shuffled_image")].values())))
            pool.append(CandidateSelectionData(candidate_id=cid, selection_real=real, selection_text=text, selection_shuffle=shuffle))
        _, _, base_agg = base_selection[cap]
        comparison = compare_standard_vs_grounded(
            pool, phase7[cap]["classifications_by_id"],
            base_selection_real=base_agg["correct_image"], base_selection_text=base_agg["text_only"], base_selection_shuffle=base_agg["shuffled_image"],
        )
        result[cap] = comparison
    return result


def run_phase10(eligible_caps: Sequence[str], phase7: Dict, phase8: Dict, phase9: Dict) -> Any:
    from .iclr_causal_density.decision_gate import CapabilityGateInputs, evaluate_decision_gate

    inputs = []
    for cap in eligible_caps:
        inputs.append(CapabilityGateInputs(
            capability=cap, D=phase7[cap]["density"].D, D_ci_low=phase7[cap]["D_ci_low"], D_ci_high=phase7[cap]["D_ci_high"],
            search_budget_divergence_confirmed=phase8["per_capability_divergence_confirmed"][cap],
            grounded_retention_top10=phase9[cap].top10_retention, grounded_g_improved_top10=phase9[cap].top10_g_materially_improved,
        ))
    return evaluate_decision_gate(inputs, integrity_ok=True)


def _cell_key_to_str(k):
    return "|".join(str(x) for x in k)


def main(argv: Optional[Sequence[str]] = None) -> int:
    _ensure_no_32b_72b_in_argv(argv)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True, help="Directory containing base_control_report.json, results.jsonl[.gz], results_selection.jsonl[.gz]")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "reports" / "iclr_causal_density"))
    args = parser.parse_args(argv)

    from .iclr_causal_density.design import CAPABILITIES, SCOPES, RADII

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eligible_caps = [c for c in CAPABILITIES if c != "visual_grounding"]

    with open(data_dir / "base_control_report.json") as f:
        base_report = json.load(f)

    base_audit = {cap: load_base_condition_scores(base_report, cap, "audit") for cap in eligible_caps}
    base_selection = {cap: load_base_condition_scores(base_report, cap, "selection") for cap in eligible_caps}

    audit_rows, candidate_meta = load_rows_by_candidate_capability_condition(data_dir / "results.jsonl")
    selection_rows, selection_candidate_meta = load_rows_by_candidate_capability_condition(data_dir / "results_selection.jsonl")
    if set(candidate_meta) != set(selection_candidate_meta):
        raise ValueError("Audit-pass and selection-pass candidate_id sets differ -- refusing to analyze mismatched data.")

    candidates_by_cell: Dict[Tuple[str, float], List[str]] = defaultdict(list)
    for cid, (scope, radius, _seed) in candidate_meta.items():
        candidates_by_cell[(scope, radius)].append(cid)

    phase7 = run_phase7(eligible_caps, base_audit, audit_rows, candidate_meta)
    phase8 = run_phase8(eligible_caps, SCOPES, RADII, selection_rows, candidates_by_cell, phase7)
    phase9 = run_phase9(eligible_caps, base_selection, selection_rows, candidate_meta, phase7)
    gate_result = run_phase10(eligible_caps, phase7, phase8, phase9)

    decision = {
        "decision": gate_result.decision,
        "reasons": gate_result.reasons,
        "criteria": gate_result.criteria,
        "eligible_capabilities": eligible_caps,
        "excluded_capabilities": ["visual_grounding"],
        "exclusion_reason": (
            "RefCOCO grounding has no meaningful text_only condition; metrics.ConditionScores requires all three "
            "conditions for every capability by construction -- structurally impossible to classify under the "
            "frozen, unmodified Phase 7 formula. Per decision_gate.py's own precedence rule 2 ('not exactly 5 "
            "capabilities with valid results -> INCONCLUSIVE'), this capability's absence is handled by the gate's "
            "own pre-existing logic, not a new rule."
        ),
        "per_capability": {
            cap: {
                "D": phase7[cap]["density"].D, "D_ci_low": phase7[cap]["D_ci_low"], "D_ci_high": phase7[cap]["D_ci_high"],
                "n_valid_resamples": phase7[cap]["n_valid_resamples"],
                "rho_standard": phase7[cap]["density"].rho_standard, "rho_visual": phase7[cap]["density"].rho_visual,
                "n_conventional": phase7[cap]["density"].n_conventional, "n_causally_visual": phase7[cap]["density"].n_causally_visual,
                "search_budget_divergence_confirmed": phase8["per_capability_divergence_confirmed"][cap],
                "grounded_retention_top10": phase9[cap].top10_retention,
                "grounded_g_improved_top10": phase9[cap].top10_g_materially_improved,
            }
            for cap in eligible_caps
        },
    }
    with (output_dir / "decision.json").open("w") as f:
        json.dump(decision, f, indent=2, default=str)

    full_output = {
        "phase8_per_cell": {_cell_key_to_str(k): v for k, v in phase8["per_cell"].items()},
        "phase9_per_capability": {cap: phase9[cap].to_dict() for cap in eligible_caps},
    }
    with (output_dir / "analysis_full_output.json").open("w") as f:
        json.dump(full_output, f, indent=2, default=str)

    print(f"DECISION: {gate_result.decision}")
    print(f"Reasons: {gate_result.reasons}")
    print(f"Wrote {output_dir / 'decision.json'}")
    print(f"Wrote {output_dir / 'analysis_full_output.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

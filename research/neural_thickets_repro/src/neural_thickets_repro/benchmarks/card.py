"""Benchmark card generation (Markdown + JSON) and the PASS/FAIL/NEEDS_REVIEW Status
decision. Thresholds live in config.BenchmarkGatesConfig, never hardcoded here -- see
CAPABILITY_BENCHMARK_GATE.md for the documented rationale behind each one.

Status decision, first matching rule wins (see decide_status()):
  1. repeatability == "FAIL"                                              -> FAIL
  2. parser_failure_rate > gates.max_parser_failure_rate_needs_review     -> FAIL
  3. any image-sanity gap (correct - shuffled, or correct - text_only)
     <= 0                                                                  -> FAIL
  4. parser_failure_rate in (max_pass, max_needs_review], OR any
     image-sanity gap in [0, min_gap_pass), OR repeatability == "NOT_RUN",
     OR no image-sanity check performed, OR primary metric at floor/ceiling -> NEEDS_REVIEW
  5. else                                                                   -> PASS

Every card exposes the RAW measurements (base score, repeat score, generation/parsed-
prediction hash equality, correct/shuffled/text-only scores and both gaps, parser failure
rate, integrity counts) regardless of the assigned Status -- the raw evidence matters more
than the label, per explicit instruction.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .image_sanity import ImageSanityResult
from .integrity import IntegrityReport

REQUIRED_CARD_FIELDS = (
    "Dataset", "Capability", "Dataset revision", "Split", "Subset size", "Subset seed",
    "Subset IDs file", "Image integrity", "Prompt template", "Generation configuration",
    "Prediction parser", "Metric", "Base-model score", "Repeat run score", "Repeatability",
    "Image sanity — correct", "Image sanity — shuffled", "Image sanity — text-only",
    "Parser failure rate", "Known caveats", "Status",
    # Baseline-characterization enrichment (this session) -- makes each card a
    # self-sufficient S_t(theta_0) record for later perturbation comparisons, without
    # needing to cross-reference run_metadata.json/repeatability.json separately.
    "Model", "Dataset source", "Candidate pool size", "Repeat delta", "Prediction disagreement rate",
)


@dataclass
class BenchmarkCardData:
    dataset: str
    capability: str
    dataset_revision: Optional[str]
    split: str
    subset_size: int
    subset_seed: Optional[int]
    subset_ids_path: str
    integrity: IntegrityReport
    prompt_template: str
    generation_config: Dict[str, Any]
    prediction_parser: str
    metric_description: str
    base_metrics: Dict[str, float]
    repeat_metrics: Optional[Dict[str, float]] = None
    repeatability_status: str = "NOT_RUN"  # "PASS" | "FAIL" | "NOT_RUN"
    generation_hash_match: Optional[bool] = None
    parsed_prediction_hash_match: Optional[bool] = None
    image_sanity: Optional[ImageSanityResult] = None
    known_caveats: List[str] = field(default_factory=list)
    # --- Baseline-characterization enrichment (this session), all additive/optional so
    # existing callers (and tests) that don't pass them keep working unchanged. ---
    model_name: str = ""
    model_revision: Optional[str] = None
    dataset_source: str = ""
    candidate_pool_size: Optional[int] = None
    subset_ids_hash: Optional[str] = None
    prompt_config_hash: Optional[str] = None
    repeat_absolute_difference: Optional[float] = None
    prediction_disagreement_rate: Optional[float] = None


def decide_status(card: BenchmarkCardData, gates: Any) -> "tuple[str, List[str]]":
    reasons: List[str] = []
    parser_failure_rate = card.base_metrics.get("parser_failure_rate", 0.0)

    if card.repeatability_status == "FAIL":
        return "FAIL", ["repeatability check failed (metrics or generation hash differed across two identical runs)"]

    if parser_failure_rate > gates.max_parser_failure_rate_needs_review:
        return "FAIL", [f"parser failure rate {parser_failure_rate:.1%} exceeds the NEEDS_REVIEW ceiling {gates.max_parser_failure_rate_needs_review:.1%}"]

    sanity = card.image_sanity
    if sanity is None:
        reasons.append("no image-sanity check performed")
    else:
        gap_shuffled = sanity.correct_minus_shuffled
        if gap_shuffled <= 0:
            return "FAIL", [f"image-sanity gap (correct - shuffled) = {gap_shuffled:+.4f} is non-positive -- the image is not detectably reaching the model"]
        if gap_shuffled < gates.image_sanity_min_gap_pass:
            reasons.append(f"correct-vs-shuffled image-sanity gap {gap_shuffled:+.4f} is below the pass threshold {gates.image_sanity_min_gap_pass:.4f} (may be noise at this subset size)")

        if sanity.text_only_supported and sanity.correct_minus_text_only is not None:
            gap_text_only = sanity.correct_minus_text_only
            if gap_text_only <= 0:
                return "FAIL", [f"image-sanity gap (correct - text_only) = {gap_text_only:+.4f} is non-positive -- the image is not detectably reaching the model"]
            if gap_text_only < gates.image_sanity_min_gap_pass:
                reasons.append(f"correct-vs-text-only image-sanity gap {gap_text_only:+.4f} is below the pass threshold {gates.image_sanity_min_gap_pass:.4f}")

    if card.repeatability_status == "NOT_RUN":
        reasons.append("no repeat run performed yet -- Status capped below PASS until repeatability is confirmed")

    if gates.max_parser_failure_rate_pass < parser_failure_rate <= gates.max_parser_failure_rate_needs_review:
        reasons.append(f"parser failure rate {parser_failure_rate:.1%} is above the pass threshold {gates.max_parser_failure_rate_pass:.1%}")

    primary = card.base_metrics.get("primary_metric")
    if primary is not None and (primary <= gates.floor_ceiling_low or primary >= gates.floor_ceiling_high):
        reasons.append(f"primary metric {primary:.4f} is at a floor/ceiling (<= {gates.floor_ceiling_low:g} or >= {gates.floor_ceiling_high:g}) -- scientifically informative, not necessarily a defect")

    if reasons:
        return "NEEDS_REVIEW", reasons
    return "PASS", []


def _fmt(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_markdown(card: BenchmarkCardData, status: str, reasons: List[str]) -> str:
    sanity = card.image_sanity
    lines = [
        f"Dataset: {card.dataset}",
        f"Capability: {card.capability}",
        f"Dataset revision: {_fmt(card.dataset_revision)}",
        f"Split: {card.split}",
        f"Subset size: {card.subset_size}",
        f"Subset seed: {_fmt(card.subset_seed)}",
        f"Subset IDs file: {card.subset_ids_path}",
        f"Image integrity: {card.integrity.n_valid_images}/{card.integrity.n_loaded} valid ({card.integrity.n_duplicate_ids} duplicate IDs, {card.integrity.n_missing_targets} missing targets)",
        f"Prompt template: {card.prompt_template}",
        f"Generation configuration: {card.generation_config}",
        f"Prediction parser: {card.prediction_parser}",
        f"Metric: {card.metric_description}",
        f"Base-model score: {_fmt(card.base_metrics.get('primary_metric'))}",
        f"Repeat run score: {_fmt(card.repeat_metrics.get('primary_metric')) if card.repeat_metrics else 'N/A'}",
        f"Repeatability: {card.repeatability_status} (generation_hash_match={card.generation_hash_match}, parsed_prediction_hash_match={card.parsed_prediction_hash_match})",
        f"Repeat delta: {_fmt(card.repeat_absolute_difference)}",
        f"Prediction disagreement rate: {_fmt(card.prediction_disagreement_rate)}",
        f"Image sanity — correct: {_fmt(sanity.correct_image_primary_metric) if sanity else 'N/A'}",
        f"Image sanity — shuffled: {_fmt(sanity.shuffled_image_primary_metric) if sanity else 'N/A'}",
        f"Image sanity — text-only: {_fmt(sanity.text_only_primary_metric) if sanity else 'N/A'}"
        + (f" (NOT_SUPPORTED: {sanity.text_only_unsupported_reason})" if sanity and not sanity.text_only_supported else ""),
        f"Parser failure rate: {_fmt(card.base_metrics.get('parser_failure_rate'))}",
        f"Model: {card.model_name}@{_fmt(card.model_revision)}",
        f"Dataset source: {card.dataset_source or 'N/A'}",
        f"Candidate pool size: {_fmt(card.candidate_pool_size)}",
        f"Subset IDs hash: {_fmt(card.subset_ids_hash)}",
        f"Prompt/config hash: {_fmt(card.prompt_config_hash)}",
        f"Known caveats: {'; '.join(card.known_caveats) if card.known_caveats else 'none'}",
        f"Status: {status}",
    ]
    if reasons:
        lines.append("")
        lines.append("Status reasons:")
        lines.extend(f"  - {r}" for r in reasons)
    return "\n".join(lines)


def render_json(card: BenchmarkCardData, status: str, reasons: List[str]) -> dict:
    return {
        "dataset": card.dataset,
        "capability": card.capability,
        "dataset_revision": card.dataset_revision,
        "split": card.split,
        "subset_size": card.subset_size,
        "subset_seed": card.subset_seed,
        "subset_ids_path": card.subset_ids_path,
        "integrity": card.integrity.to_dict(),
        "prompt_template": card.prompt_template,
        "generation_config": card.generation_config,
        "prediction_parser": card.prediction_parser,
        "metric_description": card.metric_description,
        "base_metrics": card.base_metrics,
        "repeat_metrics": card.repeat_metrics,
        "repeatability_status": card.repeatability_status,
        "generation_hash_match": card.generation_hash_match,
        "parsed_prediction_hash_match": card.parsed_prediction_hash_match,
        "repeat_absolute_difference": card.repeat_absolute_difference,
        "prediction_disagreement_rate": card.prediction_disagreement_rate,
        "image_sanity": card.image_sanity.to_dict() if card.image_sanity else None,
        "known_caveats": card.known_caveats,
        "model_name": card.model_name,
        "model_revision": card.model_revision,
        "dataset_source": card.dataset_source,
        "candidate_pool_size": card.candidate_pool_size,
        "subset_ids_hash": card.subset_ids_hash,
        "prompt_config_hash": card.prompt_config_hash,
        "status": status,
        "status_reasons": reasons,
    }


def write_card(card: BenchmarkCardData, gates: Any, out_dir: "str | Path") -> "tuple[str, List[str]]":
    status, reasons = decide_status(card, gates)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "card.md").write_text(render_markdown(card, status, reasons))
    (out_dir / "card.json").write_text(json.dumps(render_json(card, status, reasons), indent=2, default=str))
    return status, reasons

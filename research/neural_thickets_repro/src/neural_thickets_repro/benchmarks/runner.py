"""Generic single-engine, zero-perturbation, multi-example evaluation loop for the
Capability Benchmark Gate -- generalizes eval_base_image_aware.py's generate/parse/score/
aggregate flow (a single local `vllm.LLM`, no Ray actors, no perturbation) across any
CapabilityBenchmark.

run_benchmark() is a free function, not a CapabilityBenchmark method: it takes an
already-resolved `examples` list plus an `llm`-shaped object exposing
`.generate(requests, sampling_params, use_tqdm=...) -> List[output]` (matching vllm.LLM's own
interface exactly -- real on the pod, a SimpleNamespace fake in tests). This is what lets a
future perturbation sweep reuse this exact function, unmodified, as
`for seed: perturb(...); for benchmark: run_benchmark(benchmark, fixed_examples, llm, ...)`.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from .base import CapabilityBenchmark, Example, ExampleScore, ParsedPrediction
from .box_iou import box_iou
from ..vlm_adapter import build_multimodal_requests


class MissingImageError(RuntimeError):
    """An example has no image and allow_missing_image was not explicitly set -- refuses to
    silently continue. This is the exact "image never reaches the model" failure class this
    project has already hit once (GATE1_DIAGNOSIS.md) and must not repeat here.
    """


@dataclass
class PerExampleResult:
    example_id: str
    image_ref: str
    raw_generation: str
    parsed: ParsedPrediction
    score: ExampleScore


@dataclass
class RunResult:
    per_example: List[PerExampleResult]
    aggregate_metrics: Dict[str, float]

    def generation_hash(self) -> str:
        """sha256 over sorted (example_id, raw_generation) pairs. Used by the repeatability
        check to compare RAW generations across two runs.
        """
        payload = json.dumps(sorted((r.example_id, r.raw_generation) for r in self.per_example), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def parsed_prediction_hash(self) -> str:
        """sha256 over sorted (example_id, repr(parsed)) pairs. Compared SEPARATELY from
        generation_hash() so a repeat run can distinguish "raw wording changed but the
        parsed/scored answer didn't" from a genuine behavior change, rather than collapsing
        both into one pass/fail bit.
        """
        payload = json.dumps(sorted((r.example_id, repr(r.parsed.parsed)) for r in self.per_example), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prediction_disagreement_rate(result_a: "RunResult", result_b: "RunResult") -> float:
    """Fraction of examples (matched by example_id) whose parsed prediction differs between
    two RunResults -- a finer-grained MAGNITUDE companion to parsed_prediction_hash()'s
    whole-run boolean equality check, for baseline-characterization repeatability reporting
    (a hash mismatch alone doesn't say whether 1 example changed or all of them did).
    """
    a_by_id = {r.example_id: r.parsed.parsed for r in result_a.per_example}
    b_by_id = {r.example_id: r.parsed.parsed for r in result_b.per_example}
    common_ids = set(a_by_id) & set(b_by_id)
    if not common_ids:
        raise ValueError("prediction_disagreement_rate: the two RunResults share no example_ids")
    disagreements = sum(1 for eid in common_ids if repr(a_by_id[eid]) != repr(b_by_id[eid]))
    return disagreements / len(common_ids)


def write_repeat_comparison_jsonl(result_a: "RunResult", result_b: "RunResult", path: "str | Path") -> None:
    """Persists enough PER-EXAMPLE evidence to distinguish meaningless jitter from genuine
    instability between two identical-input runs (baseline-characterization repeatability,
    this repair pass) -- one line per example (matched by ID), with raw generation, parsed
    prediction, and score from BOTH runs, plus each run's own capability-specific
    ExampleScore.detail (e.g. grounding's canonical_prediction_box/iou) so nothing
    capability-specific needs to be special-cased here. Adds "box_to_box_iou" (the
    prediction-vs-itself agreement across runs, distinct from either run's own
    prediction-vs-target IoU) whenever both sides carry a "canonical_prediction_box" in
    their detail dict -- None otherwise, never fabricated for a non-grounding capability.
    """
    a_by_id = {r.example_id: r for r in result_a.per_example}
    b_by_id = {r.example_id: r for r in result_b.per_example}
    common_ids = sorted(set(a_by_id) & set(b_by_id))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for eid in common_ids:
            a, b = a_by_id[eid], b_by_id[eid]
            base_box = a.score.detail.get("canonical_prediction_box")
            repeat_box = b.score.detail.get("canonical_prediction_box")
            box_to_box_iou = box_iou(tuple(base_box), tuple(repeat_box)) if base_box is not None and repeat_box is not None else None
            f.write(json.dumps({
                "example_id": eid,
                "base_raw_generation": a.raw_generation,
                "repeat_raw_generation": b.raw_generation,
                "base_parsed_prediction": a.parsed.parsed,
                "repeat_parsed_prediction": b.parsed.parsed,
                "base_score": a.score.score,
                "repeat_score": b.score.score,
                "base_detail": a.score.detail,
                "repeat_detail": b.score.detail,
                "box_to_box_iou": box_to_box_iou,
                "parsed_prediction_agrees": repr(a.parsed.parsed) == repr(b.parsed.parsed),
            }, default=str) + "\n")


def bounded_disagreement_examples(result_a: "RunResult", result_b: "RunResult", limit: int = 10) -> List[Dict[str, Any]]:
    """Returns up to `limit` examples (matched by ID) whose parsed prediction differs between
    two runs -- a compact, bounded sample for embedding directly in repeatability.json so it
    stays a quick-glance file even when many examples disagree; the FULL per-example
    comparison always lives separately (see write_repeat_comparison_jsonl), never only here.
    """
    a_by_id = {r.example_id: r for r in result_a.per_example}
    b_by_id = {r.example_id: r for r in result_b.per_example}
    examples: List[Dict[str, Any]] = []
    for eid in sorted(set(a_by_id) & set(b_by_id)):
        a, b = a_by_id[eid], b_by_id[eid]
        if repr(a.parsed.parsed) != repr(b.parsed.parsed):
            examples.append({
                "example_id": eid,
                "base_parsed_prediction": a.parsed.parsed,
                "repeat_parsed_prediction": b.parsed.parsed,
                "base_score": a.score.score,
                "repeat_score": b.score.score,
            })
        if len(examples) >= limit:
            break
    return examples


def run_benchmark(
    benchmark: CapabilityBenchmark,
    examples: List[Example],
    llm: Any,
    tokenizer: Any,
    sampling_params: Any,
    allow_missing_image: bool = False,
) -> RunResult:
    prepared_images = [benchmark.prepare_image(ex) for ex in examples]

    if not allow_missing_image:
        missing = [ex.example_id for ex, img in zip(examples, prepared_images) if img is None]
        if missing:
            raise MissingImageError(
                f"{len(missing)} example(s) have no image and allow_missing_image=False: "
                f"{missing[:10]}. Pass allow_missing_image=True only for an explicit "
                f"text-only image-dependence-sanity condition."
            )

    prompts_and_images = [(benchmark.build_prompt(ex), img) for ex, img in zip(examples, prepared_images)]
    requests = build_multimodal_requests(prompts_and_images, tokenizer)
    outputs = llm.generate(requests, sampling_params, use_tqdm=True)
    raw_generations = [o.outputs[0].text for o in outputs]

    per_example: List[PerExampleResult] = []
    parser_failures = 0
    for ex, raw in zip(examples, raw_generations):
        parsed = benchmark.parse_prediction(raw, ex)
        if not parsed.parse_ok:
            parser_failures += 1
        score = benchmark.score_example(parsed, ex)
        per_example.append(PerExampleResult(
            example_id=ex.example_id, image_ref=ex.image_ref,
            raw_generation=raw, parsed=parsed, score=score,
        ))

    metrics = dict(benchmark.aggregate_metrics([r.score for r in per_example]))
    if "primary_metric" not in metrics:
        raise ValueError(f"{type(benchmark).__name__}.aggregate_metrics() must include a 'primary_metric' key")
    metrics.setdefault("parser_failure_rate", parser_failures / len(per_example) if per_example else 0.0)

    return RunResult(per_example=per_example, aggregate_metrics=metrics)


def write_predictions_jsonl(run_result: RunResult, examples: List[Example], path: "str | Path") -> None:
    by_id = {ex.example_id: ex for ex in examples}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in run_result.per_example:
            ex = by_id[r.example_id]
            f.write(json.dumps({
                "example_id": r.example_id,
                "image_ref": r.image_ref,
                "query": ex.prompt_input,
                "target": ex.target,
                "raw_generation": r.raw_generation,
                "parsed_prediction": r.parsed.parsed,
                "parse_ok": r.parsed.parse_ok,
                "parse_error": r.parsed.parse_error,
                "per_example_score": r.score.score,
                "correct": r.score.correct,
                "detail": r.score.detail,
                "metadata": ex.metadata,
            }, default=str) + "\n")

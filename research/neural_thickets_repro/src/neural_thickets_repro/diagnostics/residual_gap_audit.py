"""Residual Gate 1 gap diagnostic: image-aware baseline = 54.19% (march-era scoring) vs
published 56.60% = -2.41pp, inside the pre-agreed 1-3pp "investigate" band.

Reuses the ALREADY-GENERATED results/base_image_aware/predictions.jsonl from the full
12,578-example run -- no new GPU generation needed, just classification of existing
predictions. Diagnosis only: does not regenerate, does not change scoring/extraction, does
not tune anything toward 56.6%.

For a seeded sample of examples scored incorrect under march-era (paper-era) scoring,
classifies each into:
  - extraction_or_scoring_failure: GT answer appears (whole-word, incl. synonym/singular
    forms) somewhere in the raw response, but scoring didn't credit it -- likely NOT a
    model-content miss.
  - model_wrong_yes_no: both GT and the extracted answer are yes/no and they disagree --
    GQA's most common single failure mode, worth reporting separately since it dominates
    typical VQA error analyses.
  - empty_or_degenerate_response: raw response is empty, or extremely short/repetitive
    (possible generation/decoding artifact rather than a content miss).
  - model_wrong_other: genuine content disagreement, none of the above.

Usage (pod, reuses existing predictions -- fast, no GPU needed for this step):
    python -m neural_thickets_repro.diagnostics.residual_gap_audit \
        --predictions results/base_image_aware/predictions.jsonl --sample-size 200
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List

from ..vlm_adapter import load_gqa_handler

REPO_ROOT = Path(__file__).resolve().parents[3]

_YES_NO = {"yes", "no"}


def _classify(handler, rec: Dict) -> str:
    raw = rec["raw_prediction"]
    gt = rec["reference_answer"]
    extracted = rec.get("normalized_prediction", "")

    if not raw.strip():
        return "empty_or_degenerate_response"

    words = raw.split()
    if len(words) >= 6 and len(set(words)) <= 2:
        return "empty_or_degenerate_response"  # degenerate repetition

    gt_norm = handler._normalize_answer(gt)
    if handler._whole_word_search(raw.lower(), gt_norm):
        return "extraction_or_scoring_failure"

    extracted_norm = handler._normalize_answer(extracted) if extracted else ""
    if gt_norm in _YES_NO and extracted_norm in _YES_NO and gt_norm != extracted_norm:
        return "model_wrong_yes_no"

    return "model_wrong_other"


def audit(predictions_path: Path, sample_size: int, seed: int, scoring_key: str) -> Dict:
    records = [json.loads(line) for line in predictions_path.open()]
    handler = load_gqa_handler()

    incorrect = [r for r in records if not r.get(scoring_key, False)]
    rng = random.Random(seed)
    sample = rng.sample(incorrect, min(sample_size, len(incorrect)))

    counts: Counter = Counter()
    examples_by_class: Dict[str, List[Dict]] = {}
    for rec in sample:
        cls = _classify(handler, rec)
        counts[cls] += 1
        examples_by_class.setdefault(cls, []).append({
            "example_id": rec["example_id"],
            "question": rec["question"],
            "reference_answer": rec["reference_answer"],
            "raw_prediction": rec["raw_prediction"][:200],
            "normalized_prediction": rec.get("normalized_prediction", ""),
        })

    n_total = len(records)
    n_incorrect_total = len(incorrect)
    n_sampled = len(sample)

    return {
        "predictions_path": str(predictions_path),
        "scoring_key": scoring_key,
        "n_total_examples": n_total,
        "n_incorrect_total": n_incorrect_total,
        "overall_accuracy": (n_total - n_incorrect_total) / n_total if n_total else None,
        "sample_size": n_sampled,
        "seed": seed,
        "classification_counts": dict(counts),
        "classification_fractions_of_sample": {k: v / n_sampled for k, v in counts.items()} if n_sampled else {},
        "example_per_class": {k: v[:5] for k, v in examples_by_class.items()},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default=str(REPO_ROOT / "results" / "base_image_aware" / "predictions.jsonl"))
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scoring", choices=["head", "march"], default="march",
        help="which scoring's incorrect examples to sample from -- march is the fair comparison to 56.6%",
    )
    parser.add_argument("--out", default=str(REPO_ROOT / "results" / "gate1_diagnosis" / "residual_gap_audit.json"))
    args = parser.parse_args(argv)

    scoring_key = "correct_head_scoring" if args.scoring == "head" else "correct_march_scoring"
    report = audit(Path(args.predictions), args.sample_size, args.seed, scoring_key)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print(f"Overall accuracy ({args.scoring}-scoring): {report['overall_accuracy']:.4f} "
          f"({report['n_total_examples'] - report['n_incorrect_total']}/{report['n_total_examples']})")
    print(f"Sampled {report['sample_size']} of {report['n_incorrect_total']} incorrect examples (seed={args.seed})")
    print("Classification of sampled failures:")
    for cls, count in sorted(report["classification_counts"].items(), key=lambda kv: -kv[1]):
        frac = report["classification_fractions_of_sample"][cls]
        print(f"  {cls:<32} {count:>4}  ({frac:.1%})")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

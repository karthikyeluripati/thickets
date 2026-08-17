"""Pre-version-control fidelity gate.

generate_predictions.py is a standalone harness (deliberately not importing our package,
so it can run inside the isolated vLLM 0.11.0 Docker container). Before trusting its vLLM
0.27.1 output as one arm of the version comparison, this proves the harness actually
reproduces the ALREADY-VALIDATED eval_base_image_aware.py run (vlm_adapter.
generate_with_images, which completed successfully over all 12,578 testdev examples) on
the same fixed 200 IDs. If it doesn't, the harness itself is broken (as it was: a bare-
string chat message vs. the required content-list-with-image-block shape) and the version
comparison must not proceed to vLLM 0.11.0 -- a harness bug would be indistinguishable from
a real vLLM-version effect.

Environment-agnostic (no GPU/vLLM needed): compares two already-written prediction files.

Usage:
    python -m neural_thickets_repro.diagnostics.vllm_version_control.fidelity_gate \
        --fixed-sample results/gate1_diagnosis/vllm_version_control/fixed_200.json \
        --new-predictions results/gate1_diagnosis/vllm_version_control/predictions_vllm0271.jsonl \
        --baseline-predictions results/base_image_aware/predictions.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from ...vlm_adapter import load_gqa_handler

REPO_ROOT = Path(__file__).resolve().parents[4]

# Raw-text exact-match rate below which the gate fails outright, regardless of correctness
# agreement -- greedy (temperature=0) decoding on identical model/precision/prompt/image/seed
# within the SAME vLLM version should be deterministic, so anything short of ~exact match
# means the harness (not vLLM) is doing something different, not sampling noise.
MIN_EXACT_MATCH_RATE_FOR_PASS = 0.98
MIN_CORRECTNESS_AGREEMENT_RATE_FOR_PASS = 0.98


def _load_by_id(path: Path, id_field: str) -> Dict[str, Dict]:
    return {json.loads(line)[id_field]: json.loads(line) for line in Path(path).open()}


def run_fidelity_gate(fixed_sample_path: Path, new_predictions_path: Path, baseline_predictions_path: Path) -> Dict:
    fixed = json.loads(fixed_sample_path.read_text())
    fixed_ids = [ex["question_id"] for ex in fixed["examples"]]

    new_by_id = _load_by_id(new_predictions_path, "question_id")
    # eval_base_image_aware.py's predictions.jsonl uses "example_id" for the same concept.
    baseline_by_id = _load_by_id(baseline_predictions_path, "example_id")

    handler = load_gqa_handler()

    matched_ids = [qid for qid in fixed_ids if qid in new_by_id and qid in baseline_by_id]
    missing_from_new = [qid for qid in fixed_ids if qid not in new_by_id]
    missing_from_baseline = [qid for qid in fixed_ids if qid not in baseline_by_id]

    n_exact_match = 0
    n_correctness_agree = 0
    disagreements = []
    new_correct_count = 0
    baseline_correct_count = 0

    for qid in matched_ids:
        new_rec = new_by_id[qid]
        base_rec = baseline_by_id[qid]
        gt = new_rec["reference_answer"]
        assert gt == base_rec["reference_answer"], f"reference_answer mismatch for {qid}"

        new_raw = new_rec["raw_prediction"]
        base_raw = base_rec["raw_prediction"]
        exact_match = new_raw.strip() == base_raw.strip()
        n_exact_match += int(exact_match)

        new_correct = handler.compute_reward(new_raw, {"answer": gt}) > 0
        # Reuse the ALREADY-STORED march-era correctness from the validated full-baseline
        # run rather than recomputing -- that's the number that "counts" for that run.
        base_correct = bool(base_rec.get("correct_march_scoring"))

        new_correct_count += int(new_correct)
        baseline_correct_count += int(base_correct)

        agree = new_correct == base_correct
        n_correctness_agree += int(agree)
        if not agree or not exact_match:
            disagreements.append({
                "question_id": qid,
                "reference_answer": gt,
                "exact_text_match": exact_match,
                "new_raw_prediction": new_raw[:200],
                "baseline_raw_prediction": base_raw[:200],
                "new_correct_march_scoring": new_correct,
                "baseline_correct_march_scoring": base_correct,
                "correctness_agrees": agree,
            })

    n = len(matched_ids)
    exact_match_rate = n_exact_match / n if n else 0.0
    correctness_agreement_rate = n_correctness_agree / n if n else 0.0
    new_accuracy = new_correct_count / n if n else 0.0
    baseline_accuracy = baseline_correct_count / n if n else 0.0

    passed = (
        n > 0
        and not missing_from_new
        and not missing_from_baseline
        and exact_match_rate >= MIN_EXACT_MATCH_RATE_FOR_PASS
        and correctness_agreement_rate >= MIN_CORRECTNESS_AGREEMENT_RATE_FOR_PASS
    )

    return {
        "fixed_sample_size": len(fixed_ids),
        "ids_matched": n,
        "ids_missing_from_new_helper": missing_from_new,
        "ids_missing_from_baseline_records": missing_from_baseline,
        "raw_predictions_identical_count": n_exact_match,
        "raw_predictions_identical_rate": exact_match_rate,
        "march_score_accuracy_old_full_baseline_records": baseline_accuracy,
        "march_score_accuracy_new_helper": new_accuracy,
        "correctness_agreement_count": n_correctness_agree,
        "correctness_agreement_rate": correctness_agreement_rate,
        "n_disagreements": len(disagreements),
        "disagreements": disagreements[:20],
        "thresholds": {
            "min_exact_match_rate_for_pass": MIN_EXACT_MATCH_RATE_FOR_PASS,
            "min_correctness_agreement_rate_for_pass": MIN_CORRECTNESS_AGREEMENT_RATE_FOR_PASS,
        },
        "gate_result": "PASS" if passed else "FAIL",
        "instruction": (
            "PASS: proceed to vLLM 0.11.0 Docker generation. "
            "FAIL: do NOT proceed -- the harness does not yet reproduce the validated "
            "0.27.1 baseline on this sample; fix generate_predictions.py further before "
            "trusting any vLLM-version comparison built on top of it."
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-sample", default=str(REPO_ROOT / "results" / "gate1_diagnosis" / "vllm_version_control" / "fixed_200.json"))
    parser.add_argument("--new-predictions", default=str(REPO_ROOT / "results" / "gate1_diagnosis" / "vllm_version_control" / "predictions_vllm0271.jsonl"))
    parser.add_argument("--baseline-predictions", default=str(REPO_ROOT / "results" / "base_image_aware" / "predictions.jsonl"))
    parser.add_argument("--out", default=str(REPO_ROOT / "results" / "gate1_diagnosis" / "vllm_version_control" / "fidelity_gate_report.json"))
    args = parser.parse_args(argv)

    report = run_fidelity_gate(Path(args.fixed_sample), Path(args.new_predictions), Path(args.baseline_predictions))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print(f"IDs matched: {report['ids_matched']}/{report['fixed_sample_size']}")
    if report["ids_missing_from_new_helper"]:
        print(f"  MISSING from new helper output: {len(report['ids_missing_from_new_helper'])}")
    if report["ids_missing_from_baseline_records"]:
        print(f"  MISSING from baseline records: {len(report['ids_missing_from_baseline_records'])}")
    print(f"Raw predictions identical: {report['raw_predictions_identical_count']}/{report['ids_matched']} "
          f"({report['raw_predictions_identical_rate']:.1%})")
    print(f"March-score accuracy (old full-baseline records, this 200-subset): {report['march_score_accuracy_old_full_baseline_records']:.4f}")
    print(f"March-score accuracy (new helper, same 200):                      {report['march_score_accuracy_new_helper']:.4f}")
    print(f"Correctness agreement: {report['correctness_agreement_count']}/{report['ids_matched']} ({report['correctness_agreement_rate']:.1%})")
    print(f"\nGATE RESULT: {report['gate_result']}")
    print(report["instruction"])
    print(f"\nWrote {out_path}")
    return 0 if report["gate_result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

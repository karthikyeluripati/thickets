"""Environment-agnostic scoring + comparison (no vLLM/GPU needed -- run this in the normal
environment, after generate_predictions.py has produced both prediction files). Scores
both vLLM versions' outputs with march-era scoring (paper-comparable) on the identical
fixed 200-example sample, and applies the pre-agreed decision rule using an exact paired
significance test (McNemar / sign test on discordant pairs), not an arbitrary percentage
threshold.

Decision rule (stated before seeing any numbers): vLLM 0.11.0 counts as a "meaningful
improvement toward 56.6%" only if it is BOTH directionally better AND the paired difference
is statistically significant at p<0.05 (McNemar exact test on the 200-example paired
sample). At n=200, small raw percentage-point deltas are not reliably distinguishable from
sampling noise (a naive threshold would be arbitrary); this is the honest way to decide.

Usage:
    python -m neural_thickets_repro.diagnostics.vllm_version_control.compare_results \
        --predictions-a results/gate1_diagnosis/vllm_version_control/predictions_vllm0271.jsonl \
        --predictions-b results/gate1_diagnosis/vllm_version_control/predictions_vllm0110.jsonl \
        --label-a vllm0271 --label-b vllm0110
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

from ...vlm_adapter import load_gqa_handler

REPO_ROOT = Path(__file__).resolve().parents[4]
PUBLISHED_BASE_ACCURACY = 0.566


def _load_jsonl(path: Path) -> Dict[str, Dict]:
    records = [json.loads(line) for line in Path(path).open()]
    return {r["question_id"]: r for r in records}


def _binomial_cdf(k: int, n: int, p: float = 0.5) -> float:
    return sum(math.comb(n, i) * (p**i) * ((1 - p) ** (n - i)) for i in range(0, k + 1))


def mcnemar_exact_pvalue(b: int, c: int) -> float:
    """Exact two-sided sign test on discordant pairs b (A-wrong/B-correct) vs c (A-correct/B-wrong)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * _binomial_cdf(k, n))


def compare(predictions_a: Path, predictions_b: Path, label_a: str, label_b: str) -> Dict:
    handler = load_gqa_handler()
    a_by_id = _load_jsonl(predictions_a)
    b_by_id = _load_jsonl(predictions_b)

    shared_ids = sorted(set(a_by_id) & set(b_by_id))
    if set(a_by_id) != set(b_by_id):
        missing_in_b = set(a_by_id) - set(b_by_id)
        missing_in_a = set(b_by_id) - set(a_by_id)
        raise ValueError(
            f"Prediction sets cover different question_ids -- not a valid paired comparison. "
            f"{len(missing_in_b)} in {label_a} but not {label_b}; {len(missing_in_a)} the reverse."
        )

    a_correct = b_correct = 0
    both_correct = both_wrong = a_wrong_b_correct = a_correct_b_wrong = 0
    n_changed = 0
    per_example = []

    for qid in shared_ids:
        rec_a, rec_b = a_by_id[qid], b_by_id[qid]
        gt = rec_a["reference_answer"]
        assert gt == rec_b["reference_answer"], f"reference_answer mismatch for {qid}"

        correct_a = handler.compute_reward(rec_a["raw_prediction"], {"answer": gt}) > 0
        correct_b = handler.compute_reward(rec_b["raw_prediction"], {"answer": gt}) > 0
        a_correct += int(correct_a)
        b_correct += int(correct_b)

        changed = rec_a["raw_prediction"].strip() != rec_b["raw_prediction"].strip()
        n_changed += int(changed)

        if correct_a and correct_b:
            both_correct += 1
        elif not correct_a and not correct_b:
            both_wrong += 1
        elif not correct_a and correct_b:
            a_wrong_b_correct += 1
        else:
            a_correct_b_wrong += 1

        per_example.append({
            "question_id": qid, "reference_answer": gt,
            f"correct_{label_a}": correct_a, f"correct_{label_b}": correct_b, "prediction_changed": changed,
        })

    n = len(shared_ids)
    acc_a, acc_b = a_correct / n, b_correct / n
    p_value = mcnemar_exact_pvalue(a_wrong_b_correct, a_correct_b_wrong)
    b_significantly_better = (p_value < 0.05) and (a_wrong_b_correct > a_correct_b_wrong)

    recommendation = (
        "RUN_FULL_BASELINE_UNDER_B"
        if b_significantly_better
        else "STOP_AND_ACCEPT_GATE1_AS_PAPER_FAITHFUL_WITH_UNRECOVERABLE_RUNTIME_DISCREPANCY"
    )

    return {
        "n": n,
        f"accuracy_{label_a}": acc_a,
        f"accuracy_{label_b}": acc_b,
        "delta_b_minus_a_pp": (acc_b - acc_a) * 100,
        "n_predictions_changed": n_changed,
        f"{label_a}_wrong_{label_b}_correct": a_wrong_b_correct,
        f"{label_a}_correct_{label_b}_wrong": a_correct_b_wrong,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "mcnemar_exact_pvalue": p_value,
        "b_significantly_better_at_p05": b_significantly_better,
        "published_base_accuracy": PUBLISHED_BASE_ACCURACY,
        f"{label_a}_diff_vs_published_pp": (acc_a - PUBLISHED_BASE_ACCURACY) * 100,
        f"{label_b}_diff_vs_published_pp": (acc_b - PUBLISHED_BASE_ACCURACY) * 100,
        "decision_rule": (
            "B counts as a 'meaningful improvement' only if directionally better AND "
            "McNemar exact p<0.05 on the paired 200-example sample -- not a raw "
            "percentage-point threshold, which would be arbitrary at this sample size."
        ),
        "recommendation": recommendation,
        "per_example": per_example,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-a", required=True, help="e.g. predictions_vllm0271.jsonl (current)")
    parser.add_argument("--predictions-b", required=True, help="e.g. predictions_vllm0110.jsonl (paper-era)")
    parser.add_argument("--label-a", default="vllm0271")
    parser.add_argument("--label-b", default="vllm0110")
    parser.add_argument("--out", default=str(REPO_ROOT / "results" / "gate1_diagnosis" / "vllm_version_control" / "comparison_report.json"))
    args = parser.parse_args(argv)

    report = compare(Path(args.predictions_a), Path(args.predictions_b), args.label_a, args.label_b)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print(f"n = {report['n']}")
    print(f"{args.label_a} accuracy: {report[f'accuracy_{args.label_a}']:.4f}  ({report[f'{args.label_a}_diff_vs_published_pp']:+.2f}pp vs published)")
    print(f"{args.label_b} accuracy: {report[f'accuracy_{args.label_b}']:.4f}  ({report[f'{args.label_b}_diff_vs_published_pp']:+.2f}pp vs published)")
    print(f"delta ({args.label_b} - {args.label_a}): {report['delta_b_minus_a_pp']:+.2f}pp")
    print(f"predictions changed: {report['n_predictions_changed']}/{report['n']}")
    print(f"{args.label_a}-wrong/{args.label_b}-correct: {report[f'{args.label_a}_wrong_{args.label_b}_correct']}")
    print(f"{args.label_a}-correct/{args.label_b}-wrong: {report[f'{args.label_a}_correct_{args.label_b}_wrong']}")
    print(f"McNemar exact p-value: {report['mcnemar_exact_pvalue']:.4f}")
    print(f"\nRECOMMENDATION: {report['recommendation']}")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

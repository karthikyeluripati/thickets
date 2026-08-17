"""Top-K candidate selection and majority-vote ensembling, reimplemented from our own
understanding of REPRO_SPEC.md's "Top-K selection rule" / "Voting procedure" /
"Tie-breaking rule" rows.

Not copied from upstream: separate from the licensing reason (external/EXTERNAL_COMMIT.txt),
upstream's randopt.py imports `ray` and `from vllm import SamplingParams` at module level,
so nothing in it is importable in a CPU-only environment without those installed anyway.

SCAFFOLD / UNIT-TEST ONLY: exercised against synthetic scores/predictions in
tests/test_topk_voting.py, not against real model outputs.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, Hashable, List, Sequence, Tuple

Candidate = Tuple[int, float]  # (seed, sigma)


def select_top_k(scores: "Dict[Candidate, float]", k: int) -> List[Candidate]:
    """Sort candidates by score descending, return the first k.

    Ties are broken by the stability of Python's sort, which preserves `scores`'
    insertion order for equal scores -- mirrors REPRO_SPEC.md's tie-breaking note.
    """
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [candidate for candidate, _ in ranked[:k]]


def majority_vote(predictions_by_model: "Sequence[Sequence[Hashable]]", example_idx: int) -> Hashable:
    """predictions_by_model[m][example_idx] is model m's (possibly empty-string) predicted
    answer for the given example, with models ordered highest-selection-score first.

    Returns the most-common non-empty prediction among the given models; ties favor the
    earliest (= higher-scoring) model, per collections.Counter.most_common's
    insertion-order tie-break. Returns "" if every model's prediction is empty.
    """
    answers = [preds[example_idx] for preds in predictions_by_model if preds[example_idx]]
    if not answers:
        return ""
    return Counter(answers).most_common(1)[0][0]

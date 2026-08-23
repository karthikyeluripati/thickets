"""The standard, published VQA v2 soft-accuracy metric (Antol et al. 2015 / Goyal et al.
2017's well-known public formula) -- reimplemented directly from the public metric
definition, not transcribed from any dataset's own evaluation code.

For a prediction and a list of (per the TextVQA protocol, 10) human-annotated answers: for
each of the 10 leave-one-out draws of 9 answers, score = min(#exact-matches-of-prediction-
among-the-9 / 3, 1), averaged over the 10 draws. Answers are compared after normalization
(normalization.normalize_answer) -- do not substitute plain single-string exact match, which
is a materially different (harsher, order-of-magnitude-different) metric on this dataset.
"""
from __future__ import annotations

from typing import List

from .normalization import normalize_answer


def vqa_soft_accuracy(prediction: str, answers: List[str]) -> float:
    if not answers:
        raise ValueError("vqa_soft_accuracy requires at least one ground-truth answer")

    normalized_pred = normalize_answer(prediction)
    normalized_answers = [normalize_answer(a) for a in answers]

    n = len(normalized_answers)
    if n == 1:
        # The standard protocol assumes exactly 10 annotators; a single-answer fallback
        # (e.g. a malformed row) still yields a well-defined score rather than raising.
        return 1.0 if normalized_pred == normalized_answers[0] else 0.0

    scores = []
    for leave_out_idx in range(n):
        others = normalized_answers[:leave_out_idx] + normalized_answers[leave_out_idx + 1:]
        matches = sum(1 for a in others if a == normalized_pred)
        scores.append(min(matches / 3.0, 1.0))
    return sum(scores) / len(scores)

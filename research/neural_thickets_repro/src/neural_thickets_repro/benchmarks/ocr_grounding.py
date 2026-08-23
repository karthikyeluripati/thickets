"""Deterministic OCR-groundedness check, used to build an EXPERIMENTAL OCR-grounded subset of
TextVQA (see prepare_textvqa_ocr_filter.py and CAPABILITY_BENCHMARK_GATE.md) -- NOT an
official TextVQA category. A real N=5 manual inspection found several TextVQA questions
("how many wheels does this van have?", "is this book material?") that are not OCR reading
questions at all, despite living in the ocr_text_recognition capability's data source; the
paper's OCR/text-recognition capability claim needs a subset that is actually grounded in the
image's visible text.

An example is retained iff at least one normalized reference answer's WORD SEQUENCE appears
as a contiguous run of normalized OCR tokens -- this supports multi-token answers ("macbook
air" from OCR tokens ["MacBook", "Air"], "chicken noodle" from ["Chicken", "Noodle"]) as well
as single-token ones ("lithia" from ["LITHIA"]). Uses ONLY the dataset's own target answers
and its own provided `ocr_tokens` -- never a model's prediction, so filter membership can
never depend on how well any model happens to answer.
"""
from __future__ import annotations

from typing import Iterable, List

from .normalization import normalize_answer


def ocr_tokens_support_answer(answer: str, ocr_tokens: Iterable[str]) -> bool:
    """True iff `answer`'s normalized word sequence appears as a contiguous run within the
    normalized `ocr_tokens` sequence (order-preserving, one OCR token per matched word).
    """
    answer_words = normalize_answer(answer).split()
    if not answer_words:
        return False
    token_words: List[str] = [t for t in (normalize_answer(tok) for tok in ocr_tokens) if t]
    n, m = len(token_words), len(answer_words)
    if n < m:
        return False
    return any(token_words[start:start + m] == answer_words for start in range(n - m + 1))


def is_ocr_grounded(answers: Iterable[str], ocr_tokens: Iterable[str]) -> bool:
    """True iff ANY of `answers` is recoverable from `ocr_tokens` (see
    ocr_tokens_support_answer). `answers` is typically TextVQA's own 10-answer human-annotated
    list -- an example is grounded if even one of the ten is OCR-recoverable.
    """
    ocr_tokens = list(ocr_tokens)
    return any(ocr_tokens_support_answer(a, ocr_tokens) for a in answers)

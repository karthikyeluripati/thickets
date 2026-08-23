"""Shared answer normalization for the non-GQA adapters (ImageNet, CUB-200, TallyQA). GQA
adapters delegate entirely to GQAHandler's own normalization (never reimplemented) --
this project's existing policy of not introducing a second, incompatible GQA scoring path.

Deliberately simple/generic (lowercase, strip punctuation, drop leading articles, light
singularization) -- NOT a reimplementation of GQA's own richer synonym-table normalization,
which stays entirely inside the external, unmodified GQAHandler.
"""
from __future__ import annotations

import re
import string

_ARTICLES = {"a", "an", "the"}
# Hyphens/underscores separate words in compound terms (e.g. "Black-footed", CUB-200's own
# "Black_footed") -- treated as whitespace, not simply deleted, or "Black-footed" would
# collapse into the single jammed token "blackfooted" and never match "black footed".
_WORD_SEPARATOR_TABLE = str.maketrans("-_", "  ")
_REMAINING_PUNCT = string.punctuation.replace("-", "").replace("_", "")
_PUNCT_TABLE = str.maketrans("", "", _REMAINING_PUNCT)


def normalize_answer(text: str) -> str:
    text = text.strip().lower()
    text = text.translate(_WORD_SEPARATOR_TABLE)
    text = text.translate(_PUNCT_TABLE)
    tokens = [t for t in text.split() if t not in _ARTICLES]
    tokens = [_singularize(t) for t in tokens]
    return " ".join(tokens)


def _singularize(word: str) -> str:
    if len(word) > 3 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 2 and word.endswith("es") and word[-3] not in "aeiou":
        return word[:-2]
    if len(word) > 1 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def extract_integer(text: str) -> "int | None":
    """Robustly extracts a single integer answer from a free-form counting-style generation
    (e.g. "There are 4.", "4 objects", "The answer is four.") -- returns None (a parser
    failure the caller must count and report, never a guessed 0) if no clear integer is
    found. The first digit-run wins over a number word if both are present (e.g. "4 (four)").
    """
    cleaned = text.strip().lower()
    match = re.search(r"-?\d+", cleaned)
    if match:
        return int(match.group())
    for word, value in _NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", cleaned):
            return value
    return None

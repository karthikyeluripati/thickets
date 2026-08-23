"""Robust `\\boxed{...}` final-answer extraction for the GQA capability-benchmark adapters
(spatial_reasoning_gqa.py / relational_reasoning_gqa.py, via _gqa_filtered_base.py) -- this
module does NOT touch, and is never imported by, GQAHandler (external/RandOpt) or the
historical Gate-1 reproduction evaluator (eval_base_image_aware.py / run_randopt_image_aware.py
/ run_scoped_randopt.py), which keep calling GQAHandler's own extract_answer_for_voting
unchanged. This is a capability-benchmark-ONLY parsing layer.

Real RunPod finding this repairs: GQAHandler's own extract_answer_for_voting (external,
frozen -- not ours to fix, and presumably a naive non-balanced-brace regex) mis-extracted
`\\boxed{\\text{the person in the blue shirt}}` as `\\text{the person` (truncated at an inner
`}`), and for a SEPARATE generation that was cut off by the token ceiling before ever
producing a `\\boxed{}` at all, fabricated the nonsense string "step step" as if it were a
real answer -- with parser_failure_rate staying 0 throughout, because GQAHandler's own
"some non-empty string came back" was the only failure signal being used.

extract_boxed_answer() is used ONLY to decide parse_ok / parser_failure_rate for the
capability benchmark (a genuine "did the model give an extractable final answer" check, via
balanced-brace matching that correctly handles the nested-brace case above, and correctly
returns None -- a real parser failure -- for a truncated generation instead of inventing
something). Actual reward/correctness scoring is UNCHANGED: it still calls
GQAHandler.compute_reward on the untouched RAW generation text (see _gqa_filtered_base.py),
since that method does its own internal extraction/normalization and is the Gate-1-validated
scoring path -- not reimplemented or second-guessed here.

CONCISE-ANSWER FALLBACK (this repair pass): a real N=5 spatial run showed a genuinely valid,
common GQA answer style -- a bare "Yes" with no `\boxed{}` at all -- being counted as a parser
failure (parse_ok=False), even though it's an obviously usable short answer. extract_gqa_answer()
below adds a CONSERVATIVE fallback: if no `\boxed{}` answer exists, the ENTIRE generation is
accepted as-is only if it structurally looks like a concise answer (single line, non-empty, at
most _MAX_CONCISE_TOKENS whitespace-separated tokens, at most _MAX_CONCISE_CHARS characters) --
never a substring/last-token guess extracted FROM longer text. This is deliberately narrower
than "any short-looking text": it accepts the generation only when the ENTIRE thing already
qualifies as a short answer, which is exactly what distinguishes a genuine "Yes"/"keyboard"/
"two people" response from long or truncated chain-of-thought reasoning (which fails the
token/length/single-line checks and is correctly still a parser failure) -- the general shape
of the ORIGINAL bug this package's extract_boxed_answer() already fixed (fabricating "step
step" by guessing FROM truncated prose, rather than accepting a generation that already IS a
short answer).
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

_BOXED_MARKER = "\\boxed{"

# Extraction-mode labels, recorded alongside the extracted answer so a predictions.jsonl / the
# inspect_capability_predictions.py audit tool can distinguish "the model complied with the
# \boxed{} contract" from "we conservatively accepted a bare short answer" from "no usable
# answer at all" -- without needing three separate boolean flags.
EXTRACTION_MODE_BOXED = "boxed"
EXTRACTION_MODE_CONCISE_FALLBACK = "concise_fallback"
EXTRACTION_MODE_FAILURE = "failure"

# Conservative, documented thresholds for the concise-answer fallback -- NOT tuned against any
# particular example's score. "two people" (2 tokens, 10 chars) and "the person in the blue
# shirt" (6 tokens, 29 chars -- though that example arrives via \boxed{} in practice) both fit
# comfortably; a multi-sentence reasoning generation does not.
_MAX_CONCISE_TOKENS = 8
_MAX_CONCISE_CHARS = 60
# Simple, single-argument LaTeX text-style wrapper commands worth unwrapping one (or more,
# nested) layer(s) of -- e.g. "\text{the answer}" -> "the answer". Deliberately NOT a general
# LaTeX parser: anything else is left as-is rather than guessed at.
_SIMPLE_LATEX_WRAPPER = re.compile(r"^\\(text|mathrm|mathbf|textbf|textrm)\s*\{")


def _extract_balanced_group(text: str, open_brace_index: int) -> Optional[str]:
    """`text[open_brace_index]` must be '{'. Returns the substring strictly between that
    brace and its matching closing '}', correctly handling any nested '{'/'}' pairs in
    between (depth-counted, not a naive non-greedy regex). Returns None if the braces never
    balance before the end of the string -- the correct signal for a generation truncated
    mid-answer, never a guessed partial extraction.
    """
    if open_brace_index >= len(text) or text[open_brace_index] != "{":
        return None
    depth = 0
    for i in range(open_brace_index, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace_index + 1:i]
    return None  # unbalanced -- truncated before the boxed answer closed


def _unwrap_simple_latex(content: str) -> str:
    content = content.strip()
    match = _SIMPLE_LATEX_WRAPPER.match(content)
    if not match:
        return content
    inner = _extract_balanced_group(content, match.end() - 1)
    if inner is None:
        return content  # malformed wrapper -- leave as-is rather than guess
    return _unwrap_simple_latex(inner)  # handles nesting, e.g. \text{\textbf{x}}


def extract_boxed_answer(text: str) -> Optional[str]:
    """Finds the LAST "\\boxed{...}" in `text` (the final-answer convention the prompt asks
    for) via balanced-brace matching, unwraps one or more nested simple LaTeX text-style
    wrappers, and returns the cleaned answer string. Returns None -- a genuine parser
    failure, never a fabricated fallback like GQAHandler's observed "step step" -- if no
    "\\boxed{" marker is present, its braces never balance (a truncated generation), or the
    extracted content is empty after unwrapping/stripping.
    """
    idx = text.rfind(_BOXED_MARKER)
    if idx == -1:
        return None
    open_brace_index = idx + len(_BOXED_MARKER) - 1
    content = _extract_balanced_group(text, open_brace_index)
    if content is None:
        return None
    cleaned = _unwrap_simple_latex(content).strip()
    return cleaned or None


def _looks_like_concise_answer(stripped_text: str) -> bool:
    """True iff `stripped_text` (already whitespace-stripped) structurally looks like a
    complete short answer rather than reasoning/prose: non-empty, a single line, at most
    _MAX_CONCISE_TOKENS whitespace-separated tokens, and at most _MAX_CONCISE_CHARS
    characters. Deliberately simple/structural -- no semantic judgment of WHAT the answer
    says, only whether its shape is consistent with "a short answer" rather than "a paragraph
    of reasoning."
    """
    if not stripped_text:
        return False
    if "\n" in stripped_text:
        return False
    if len(stripped_text) > _MAX_CONCISE_CHARS:
        return False
    tokens = stripped_text.split()
    if not tokens or len(tokens) > _MAX_CONCISE_TOKENS:
        return False
    return True


def extract_gqa_answer(text: str) -> Tuple[Optional[str], str]:
    """Returns (extracted_answer_or_None, extraction_mode). Preferred path: extract_boxed_answer()'s
    balanced-brace \\boxed{} extraction. Fallback (see CONCISE-ANSWER FALLBACK module docstring
    note): if no boxed answer exists, accept the ENTIRE generation as-is only if
    _looks_like_concise_answer() says it structurally looks like a short answer already --
    never a guess extracted from within longer text. Returns (None, EXTRACTION_MODE_FAILURE)
    if neither path succeeds -- a genuine parser failure.
    """
    boxed = extract_boxed_answer(text)
    if boxed is not None:
        return boxed, EXTRACTION_MODE_BOXED

    stripped = text.strip()
    if _looks_like_concise_answer(stripped):
        return stripped, EXTRACTION_MODE_CONCISE_FALLBACK

    return None, EXTRACTION_MODE_FAILURE

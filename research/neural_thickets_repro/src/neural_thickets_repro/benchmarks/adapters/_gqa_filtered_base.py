"""Shared base for the two GQA-derived capability adapters (spatial_reasoning_gqa.py,
relational_reasoning_gqa.py). Both reuse GQAHandler's own load_data/compute_reward pipeline
UNCHANGED for data loading and actual reward scoring -- no second, incompatible GQA scoring
path is introduced. The only things each subclass adds are (a) which question-ID filter
(built by gqa_raw_schema.py) selects its examples out of the records GQAHandler.load_data()
returns, and (b) capability-benchmark-only parsing/prompt behavior described below, which
never touches GQAHandler or the historical Gate-1 reproduction evaluator.

GQAHandler.compute_reward(raw_response, ground_truth) does its OWN internal answer
extraction/normalization -- it needs the RAW generation text, not whatever this module's own
parser already extracted. Since CapabilityBenchmark.score_example()'s signature only receives
the already-parsed ParsedPrediction (not the raw text directly), the raw generation is
carried through inside ParsedPrediction.parsed (an opaque, adapter-owned payload) as
{"raw": ..., "extracted": ...} specifically so score_example can still call compute_reward on
the untouched original text.

PARSER FIX (earlier repair pass): parse_prediction() no longer calls GQAHandler's own
extract_answer_for_voting() at all -- a real RunPod run showed it mis-extracting a nested
`\\boxed{\\text{...}}` answer via (presumably) a non-balanced-brace regex, and fabricating the
nonsense string "step step" for a generation truncated before any `\\boxed{}` appeared, with
parser_failure_rate staying 0 throughout. gqa_boxed_answer.extract_boxed_answer() (this
package's own code, balanced-brace-correct) decided parse_ok/parser_failure_rate instead;
compute_reward's own scoring behavior is untouched (still called on the raw generation).

PARSER FIX, round 2 (this repair pass): a real N=5 spatial run showed a bare "Yes" (no
`\\boxed{}` at all) -- a genuinely valid, common GQA answer style -- being counted as a parser
failure. parse_prediction() now uses gqa_boxed_answer.extract_gqa_answer(), which keeps
`\\boxed{}` extraction as the preferred path but adds a CONSERVATIVE concise-answer fallback:
the entire generation is accepted as-is only if it structurally looks like a short answer
(single line, <= 8 tokens, <= 60 chars) -- never a guess extracted from within longer text, so
the original "step step"-from-truncated-prose failure mode stays fixed. The extraction mode
("boxed" / "concise_fallback" / "failure") is recorded in ParsedPrediction.parsed["extraction_mode"].

PROMPT FIX (this repair pass): build_prompt() appends a short, capability-benchmark-ONLY
instruction on top of GQAHandler's own historical messages (never mutated in place) asking
for brief reasoning -- a real RunPod run showed the historical "reason step by step" wording
causing at least one generation to hit the token ceiling before ever producing a `\\boxed{}`
answer. Gate 1's own scripts call GQAHandler directly and never see this addition; the
historical prompt text itself is not modified, since we cannot and should not touch
GQAHandler (external/RandOpt).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Set

from ..base import CapabilityBenchmark, Example, ExampleScore, ParsedPrediction
from .gqa_boxed_answer import extract_gqa_answer
from .gqa_raw_schema import GQASchemaError, load_persisted_filter_ids

# Appended as an EXTRA text block on GQAHandler's own last message turn -- never replaces or
# edits GQAHandler's own instruction text (which we do not have access to modify or fully
# know the exact wording of). Keeps the \boxed{...} answer contract intact (both this
# module's own extract_boxed_answer() and, presumably, GQAHandler.compute_reward's own
# internal extraction depend on it) while directly addressing the observed truncation cause:
# long, unconstrained step-by-step reasoning eating the token budget before an answer appears.
CAPABILITY_BENCHMARK_ANSWER_STYLE_OVERRIDE = (
    "For this evaluation specifically, do not write out long step-by-step reasoning -- keep "
    "any reasoning brief (at most one short sentence) so your final answer is not cut off. "
    "Still give your final answer as a short phrase inside \\boxed{...}, exactly as "
    "instructed above."
)

# src/neural_thickets_repro/benchmarks/adapters/_gqa_filtered_base.py -> research/neural_thickets_repro
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_FILTER_IDS_DIR = REPO_ROOT / "artifacts" / "benchmark_subsets"

# The exact command that (re)generates both ID files -- referenced in the actionable error
# message below, so a fresh-pod user with an empty artifacts/ dir is told precisely what to
# run rather than just that something is missing. Kept as one named constant so the CLI
# script and this error message can never drift apart silently.
PREPARE_FILTERS_COMMAND = "python -m neural_thickets_repro.prepare_gqa_capability_filters --config configs/gqa_repro.yaml"


class GQAFilteredBenchmark(CapabilityBenchmark):
    """Not directly instantiated as a capability -- subclassed by
    GQASpatialReasoningBenchmark / GQARelationalReasoningBenchmark, each fixing
    `capability`/`name`/`dataset_source()`/`known_caveats()`/`DEFAULT_FILTER_IDS_FILENAME`.
    """

    # Overridden per subclass -- the stable artifact filename under
    # artifacts/benchmark_subsets/ that prepare_gqa_capability_filters.py writes and this
    # class reads by default. This is the actual fix for the previously-reported
    # "needs either question_ids or filter_ids_path" RuntimeError: the no-arg CLI
    # instantiation (AdapterClass()) never had any way to supply either, because no default
    # path existed anywhere -- now every subclass has one, resolved automatically.
    DEFAULT_FILTER_IDS_FILENAME: ClassVar[str] = ""

    def __init__(
        self,
        gqa_handler: Any = None,
        question_ids: Optional[Set[str]] = None,
        filter_ids_path: "str | Path | None" = None,
    ):
        """gqa_handler/question_ids: inject directly for tests, bypassing both the real
        GQAHandler and the on-disk artifact entirely. In production (the CLI's no-arg
        `adapter_cls()` instantiation), both default to None: gqa_handler resolves lazily to
        the real vlm_adapter.load_gqa_handler(), and filter_ids_path defaults to this
        subclass's own DEFAULT_FILTER_IDS_FILENAME under artifacts/benchmark_subsets/ --
        generated by `prepare_gqa_capability_filters.py` (see PREPARE_FILTERS_COMMAND).
        Passing filter_ids_path explicitly still overrides the default, e.g. for a
        differently-located artifact.
        """
        self._handler = gqa_handler
        self._question_ids = set(str(i) for i in question_ids) if question_ids is not None else None
        if filter_ids_path is not None:
            self._filter_ids_path: Optional[Path] = Path(filter_ids_path)
        elif self.DEFAULT_FILTER_IDS_FILENAME:
            self._filter_ids_path = DEFAULT_FILTER_IDS_DIR / self.DEFAULT_FILTER_IDS_FILENAME
        else:
            self._filter_ids_path = None

    def subset_selection_rule(self) -> str:
        return "prefix"  # matches the existing GQA-pilot convention (prepare_gqa_data.py's own dataset.select(range(n)))

    def _resolve_handler(self) -> Any:
        if self._handler is None:
            from ...vlm_adapter import load_gqa_handler
            self._handler = load_gqa_handler()
        return self._handler

    def _resolve_question_ids(self) -> Set[str]:
        if self._question_ids is not None:
            return self._question_ids

        if self._filter_ids_path is None:
            raise RuntimeError(
                f"{type(self).__name__} needs either question_ids or filter_ids_path, and "
                f"has no DEFAULT_FILTER_IDS_FILENAME to fall back to -- this indicates a "
                f"missing subclass attribute, not a missing artifact."
            )
        try:
            ids = load_persisted_filter_ids(self._filter_ids_path)
        except GQASchemaError as exc:
            raise RuntimeError(
                f"{type(self).__name__}: no capability filter IDs found at "
                f"{self._filter_ids_path} ({exc}). Generate it first with:\n"
                f"    {PREPARE_FILTERS_COMMAND}\n"
                f"(requires external/RandOpt/data/gqa/testdev.parquet to already exist -- run "
                f"`python -m neural_thickets_repro.prepare_gqa_data --config configs/gqa_repro.yaml` "
                f"first if it doesn't)."
            ) from exc
        self._question_ids = set(ids)
        return self._question_ids

    def load_examples(self, cfg: Any) -> List[Example]:
        from PIL import Image

        # Cheaper check first: the filter-IDs artifact is a local JSON file needing no
        # external/RandOpt setup at all, so a missing/not-yet-generated filter is reported
        # before ever touching the (heavier, external-clone-dependent) GQAHandler resolution.
        question_ids = self._resolve_question_ids()
        handler = self._resolve_handler()
        records = handler.load_data(cfg.dataset.source, split=cfg.dataset.split, max_samples=None)
        filtered = [r for r in records if str(r["question_id"]) in question_ids]

        examples: List[Example] = []
        for r in filtered:
            image = Image.open(r["image_path"]).convert("RGB") if "image_path" in r else None
            examples.append(Example(
                example_id=str(r["question_id"]), image=image, image_ref=r.get("image_path", ""),
                prompt_input={"messages": r["messages"]}, target=r["ground_truth"],
            ))
        return examples

    def build_prompt(self, example: Example) -> List[dict]:
        """Returns GQAHandler's own historical messages with CAPABILITY_BENCHMARK_ANSWER_STYLE_
        OVERRIDE appended to the last turn's content list (see module docstring's PROMPT FIX
        note) -- a new list/dicts are built rather than mutating example.prompt_input["messages"]
        in place, so the original GQAHandler-provided structure is never altered.
        """
        historical_messages = example.prompt_input["messages"]
        messages = [dict(m) for m in historical_messages]
        if messages:
            last = messages[-1]
            content = list(last.get("content", [])) + [{"type": "text", "text": CAPABILITY_BENCHMARK_ANSWER_STYLE_OVERRIDE}]
            messages[-1] = {**last, "content": content}
        return messages

    def parse_prediction(self, raw_generation: str, example: Example) -> ParsedPrediction:
        """See module docstring's PARSER FIX notes: uses this package's own
        gqa_boxed_answer.extract_gqa_answer() (balanced-brace \\boxed{} extraction, with a
        conservative concise-answer fallback), NOT GQAHandler.extract_answer_for_voting(), to
        decide parse_ok.
        """
        extracted, extraction_mode = extract_gqa_answer(raw_generation)
        ok = extracted is not None
        return ParsedPrediction(
            parsed={"raw": raw_generation, "extracted": extracted, "extraction_mode": extraction_mode},
            parse_ok=ok,
            parse_error=None if ok else "no \\boxed{...} final answer and generation is not a concise short-answer fallback",
        )

    def score_example(self, parsed: ParsedPrediction, example: Example) -> ExampleScore:
        if not parsed.parse_ok:
            return ExampleScore(score=0.0, correct=False, detail={
                "extracted": None, "reason": "parse_failure", "extraction_mode": parsed.parsed["extraction_mode"],
            })
        reward = self._resolve_handler().compute_reward(parsed.parsed["raw"], example.target)
        return ExampleScore(score=float(reward), correct=reward > 0, detail={
            "extracted": parsed.parsed["extracted"], "extraction_mode": parsed.parsed["extraction_mode"],
        })

    def aggregate_metrics(self, scores: List[ExampleScore]) -> Dict[str, float]:
        n = len(scores)
        if n == 0:
            return {"accuracy": 0.0, "primary_metric": 0.0, "parser_failure_rate": 0.0}
        parser_failures = sum(1 for s in scores if s.detail.get("reason") == "parse_failure")
        accuracy = sum(s.score for s in scores) / n
        return {"accuracy": accuracy, "primary_metric": accuracy, "parser_failure_rate": parser_failures / n}

"""Shared base for the two GQA-derived capability adapters (spatial_reasoning_gqa.py,
relational_reasoning_gqa.py). Both reuse GQAHandler's own load_data/compute_reward/
extract_answer_for_voting pipeline UNCHANGED -- no second, incompatible GQA scoring path is
introduced. The only thing each subclass adds is which question-ID filter (built by
gqa_raw_schema.py) selects its examples out of the records GQAHandler.load_data() returns.

GQAHandler.compute_reward(raw_response, ground_truth) does its OWN internal answer
extraction/normalization -- it needs the RAW generation text, not whatever
extract_answer_for_voting() already extracted. Since CapabilityBenchmark.score_example()'s
signature only receives the already-parsed ParsedPrediction (not the raw text directly), the
raw generation is carried through inside ParsedPrediction.parsed (an opaque, adapter-owned
payload) as {"raw": ..., "extracted": ...} specifically so score_example can still call
compute_reward on the untouched original text.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from ..base import CapabilityBenchmark, Example, ExampleScore, ParsedPrediction
from .gqa_raw_schema import load_persisted_filter_ids


class GQAFilteredBenchmark(CapabilityBenchmark):
    """Not directly instantiated as a capability -- subclassed by
    GQASpatialReasoningBenchmark / GQARelationalReasoningBenchmark, each fixing
    `capability`/`name`/`dataset_source()`/`known_caveats()`.
    """

    def __init__(
        self,
        gqa_handler: Any = None,
        question_ids: Optional[Set[str]] = None,
        filter_ids_path: "str | None" = None,
    ):
        """gqa_handler/question_ids: inject directly for tests. In production, both default
        to None and are resolved lazily on first use (real GQAHandler via
        vlm_adapter.load_gqa_handler(), real IDs via the persisted filter_ids_path) -- this
        keeps the no-arg-constructor convention every other adapter uses for the CLI's
        dotted-class-path instantiation, while still allowing direct dependency injection in
        tests.
        """
        self._handler = gqa_handler
        self._question_ids = set(str(i) for i in question_ids) if question_ids is not None else None
        self._filter_ids_path = filter_ids_path

    def subset_selection_rule(self) -> str:
        return "prefix"  # matches the existing GQA-pilot convention (prepare_gqa_data.py's own dataset.select(range(n)))

    def _resolve_handler(self) -> Any:
        if self._handler is None:
            from ...vlm_adapter import load_gqa_handler
            self._handler = load_gqa_handler()
        return self._handler

    def _resolve_question_ids(self) -> Set[str]:
        if self._question_ids is None:
            if self._filter_ids_path is None:
                raise RuntimeError(f"{type(self).__name__} needs either question_ids or filter_ids_path")
            self._question_ids = set(load_persisted_filter_ids(self._filter_ids_path))
        return self._question_ids

    def load_examples(self, cfg: Any) -> List[Example]:
        from PIL import Image

        handler = self._resolve_handler()
        question_ids = self._resolve_question_ids()
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
        return example.prompt_input["messages"]

    def parse_prediction(self, raw_generation: str, example: Example) -> ParsedPrediction:
        extracted = self._resolve_handler().extract_answer_for_voting(raw_generation)
        ok = bool(extracted)
        return ParsedPrediction(
            parsed={"raw": raw_generation, "extracted": extracted},
            parse_ok=ok, parse_error=None if ok else "no answer extracted (extract_answer_for_voting)",
        )

    def score_example(self, parsed: ParsedPrediction, example: Example) -> ExampleScore:
        reward = self._resolve_handler().compute_reward(parsed.parsed["raw"], example.target)
        return ExampleScore(score=float(reward), correct=reward > 0, detail={"extracted": parsed.parsed["extracted"]})

    def aggregate_metrics(self, scores: List[ExampleScore]) -> Dict[str, float]:
        n = len(scores)
        if n == 0:
            return {"accuracy": 0.0, "primary_metric": 0.0, "parser_failure_rate": 0.0}
        parser_failures = sum(1 for s in scores if not s.detail.get("extracted"))
        accuracy = sum(s.score for s in scores) / n
        return {"accuracy": accuracy, "primary_metric": accuracy, "parser_failure_rate": parser_failures / n}

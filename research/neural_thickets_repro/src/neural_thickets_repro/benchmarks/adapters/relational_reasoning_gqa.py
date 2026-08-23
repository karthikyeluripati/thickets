"""GQA relational_reasoning adapter. Reuses GQAHandler's known-good load_data/compute_reward/
extract_answer_for_voting pipeline unchanged (see _gqa_filtered_base.py) -- the only new
logic is the relational question-ID filter built by gqa_raw_schema.py.

relational_reasoning is defined as GQA's own "relation" semantic-type questions EXCLUDING the
spatial_reasoning subset -- an EXPLICIT experimental choice (made so the two capability
benchmarks measure distinct skills), not a claim that GQA itself provides two naturally
disjoint categories. See gqa_raw_schema.py's module docstring and
build_spatial_relational_filters()'s stats output for the actual measured overlap.
"""
from __future__ import annotations

from typing import List

from ._gqa_filtered_base import GQAFilteredBenchmark


class GQARelationalReasoningBenchmark(GQAFilteredBenchmark):
    capability = "relational_reasoning"
    name = "gqa_testdev_relational"
    DEFAULT_FILTER_IDS_FILENAME = "gqa_relational_ids.json"

    def dataset_source(self) -> str:
        return (
            "lmms-lab-encoder/GQA (raw annotations, filtered to relation-type questions "
            "EXCLUDING the spatial subset -- see adapters/gqa_raw_schema.py)"
        )

    def known_caveats(self) -> List[str]:
        return [
            "relational_reasoning = non-spatial relation-type questions -- an explicit "
            "experimental exclusion, not a claim GQA itself provides two naturally disjoint "
            "categories (spatial questions ARE relation-type questions in GQA's own schema; "
            "see gqa_raw_schema.py's build_spatial_relational_filters() stats output for the "
            "actual measured overlap before this exclusion is applied).",
            "The relational-question ID filter depends on gqa_raw_schema.py's ASSUMED raw-GQA "
            "field names, not yet confirmed against the real raw dataset -- see "
            "CAPABILITY_BENCHMARK_GATE.md's first pod-side investigation step.",
        ]

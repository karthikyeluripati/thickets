"""GQA spatial_reasoning adapter. Reuses GQAHandler's known-good load_data/compute_reward/
extract_answer_for_voting pipeline unchanged (see _gqa_filtered_base.py) -- the only new
logic is the spatial question-ID filter built by gqa_raw_schema.py from GQA's own raw
annotation type metadata (semantic type "relation" + a spatial-keyword-matched relation
name).
"""
from __future__ import annotations

from typing import List

from ._gqa_filtered_base import GQAFilteredBenchmark


class GQASpatialReasoningBenchmark(GQAFilteredBenchmark):
    capability = "spatial_reasoning"
    name = "gqa_testdev_spatial"
    DEFAULT_FILTER_IDS_FILENAME = "gqa_spatial_ids.json"

    def dataset_source(self) -> str:
        return "lmms-lab-encoder/GQA (raw annotations, filtered to the spatial subset -- see adapters/gqa_raw_schema.py)"

    def known_caveats(self) -> List[str]:
        return [
            "The spatial-question ID filter depends on gqa_raw_schema.py's ASSUMED raw-GQA "
            "field names, not yet confirmed against the real raw dataset -- see "
            "CAPABILITY_BENCHMARK_GATE.md's first pod-side investigation step "
            "(gqa_raw_schema.inspect_raw_schema()).",
        ]

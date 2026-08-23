"""Deterministic fixed-N subset construction + ID persistence for the Capability Benchmark
Gate. Two rules:

  - "prefix": examples[:n] -- matches the existing GQA-pilot convention
    (prepare_gqa_data.py's `dataset.select(range(n))`), appropriate when the underlying
    pool's row order doesn't encode class/category structure.
  - "shuffled_prefix" (the ABC's default): a seeded shuffle of a COPY of the example list,
    then prefix -- appropriate whenever row order might encode structure (e.g. ImageNet-1K's
    validation split and most CUB-200 mirrors are ordered by class; a raw prefix slice would
    sample only a handful of classes total, which is not a meaningful evaluation). Still
    fully deterministic and reproducible: the same (pool order, n, seed) always produces the
    same subset, and the chosen IDs are persisted to disk so a re-run never needs to
    regenerate/re-shuffle at all -- it just replays the persisted list.

No GPU/ray/vllm/datasets import -- pure stdlib + the Example dataclass.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Optional, Sequence

from .base import Example

SUBSET_SELECTION_RULES = ("prefix", "shuffled_prefix")


class SubsetSelectionError(RuntimeError):
    """Raised on an unrecognized rule, an oversized request, or when persisted IDs don't
    match the current pool (dataset-drift guard) -- never silently resamples/substitutes.
    """


def select_fixed_subset(examples: Sequence[Example], n: int, rule: str, seed: Optional[int] = None) -> List[Example]:
    if rule not in SUBSET_SELECTION_RULES:
        raise SubsetSelectionError(f"Unknown subset_selection_rule {rule!r}, expected one of {SUBSET_SELECTION_RULES}")
    if n > len(examples):
        raise SubsetSelectionError(f"Requested subset size {n} exceeds pool size {len(examples)}")
    if n <= 0:
        raise SubsetSelectionError(f"Requested subset size must be positive, got {n}")

    if rule == "prefix":
        return list(examples[:n])

    if seed is None:
        raise SubsetSelectionError("subset_selection_rule='shuffled_prefix' requires a seed")
    ordered = list(examples)
    random.Random(seed).shuffle(ordered)
    return ordered[:n]


def persist_subset_ids(examples: Sequence[Example], path: "str | Path") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([e.example_id for e in examples], indent=2))


def load_persisted_ids(path: "str | Path") -> List[str]:
    path = Path(path)
    if not path.exists():
        raise SubsetSelectionError(f"No persisted subset IDs found at {path}")
    return json.loads(path.read_text())


def build_or_load_subset(
    examples: Sequence[Example], n: int, rule: str, seed: Optional[int], ids_path: "str | Path",
) -> List[Example]:
    """If ids_path already exists, replay the exact persisted IDs -- hard-fails if any
    persisted ID is missing from the freshly-loaded pool (a dataset-drift guard: the upstream
    dataset revision changed, or a filter changed, in a way that silently invalidated the
    persisted subset) rather than silently resampling around the gap. Otherwise selects a
    fresh subset via select_fixed_subset() and persists it for every future call.
    """
    ids_path = Path(ids_path)
    by_id = {e.example_id: e for e in examples}

    if ids_path.exists():
        persisted_ids = load_persisted_ids(ids_path)
        missing = [i for i in persisted_ids if i not in by_id]
        if missing:
            raise SubsetSelectionError(
                f"{len(missing)} persisted subset ID(s) from {ids_path} are missing from the "
                f"current example pool (dataset drift?): {missing[:10]}. Refusing to silently "
                f"resample -- investigate the dataset source before proceeding."
            )
        return [by_id[i] for i in persisted_ids]

    subset = select_fixed_subset(examples, n, rule, seed)
    persist_subset_ids(subset, ids_path)
    return subset

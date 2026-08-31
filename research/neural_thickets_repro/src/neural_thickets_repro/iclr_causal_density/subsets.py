"""Frozen, deterministic, disjoint selection/audit subset construction for the causal-density
pilot. The disjointness/two-subset-from-one-pool logic below is new (Phase 2's own requirement
-- benchmarks.subset_selection.select_fixed_subset only ever builds ONE subset per call, with
no built-in notion of a second, disjoint one drawn from the same pool), but persists its own
richer manifest (both subsets' IDs + hashes together) rather than subset_selection.py's own
single-subset persist_subset_ids/load_persisted_ids format, which has no place for a second,
paired subset.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from ..benchmarks.base import Example
from .design import AUDIT_SET_SIZE, MIN_DISJOINT_EXAMPLES_REQUIRED, SELECTION_SET_SIZE, SUBSET_SELECTION_SEED


class InsufficientExamplesError(RuntimeError):
    """Fewer than MIN_DISJOINT_EXAMPLES_REQUIRED (400) valid examples are available for a
    capability -- the task spec requires stopping with INCONCLUSIVE in this case, never
    shrinking the subset sizes or reusing overlapping examples.
    """


def _subset_hash(example_ids: Sequence[str]) -> str:
    canonical = json.dumps(list(example_ids), sort_keys=False)  # order matters -- selection is index-0..199, audit is 200..399, never resorted
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CapabilitySubsetManifest:
    capability: str
    seed: int
    pool_size: int
    selection_example_ids: Tuple[str, ...]
    audit_example_ids: Tuple[str, ...]
    selection_hash: str
    audit_hash: str

    def to_dict(self) -> Dict:
        return {
            "capability": self.capability, "seed": self.seed, "pool_size": self.pool_size,
            "selection_set_size": len(self.selection_example_ids), "audit_set_size": len(self.audit_example_ids),
            "selection_example_ids": list(self.selection_example_ids), "audit_example_ids": list(self.audit_example_ids),
            "selection_hash": self.selection_hash, "audit_hash": self.audit_hash,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "CapabilitySubsetManifest":
        return cls(
            capability=d["capability"], seed=d["seed"], pool_size=d["pool_size"],
            selection_example_ids=tuple(d["selection_example_ids"]), audit_example_ids=tuple(d["audit_example_ids"]),
            selection_hash=d["selection_hash"], audit_hash=d["audit_hash"],
        )


def build_selection_and_audit_subsets(
    examples: Sequence[Example], capability: str, *, seed: int = SUBSET_SELECTION_SEED,
    selection_size: int = SELECTION_SET_SIZE, audit_size: int = AUDIT_SET_SIZE,
) -> Tuple[List[Example], List[Example], CapabilitySubsetManifest]:
    """ONE seeded shuffle of the full example pool (deterministic given (pool identity order,
    seed)) -- the first `selection_size` examples become the selection set, the NEXT
    `audit_size` become the audit set. Disjoint BY CONSTRUCTION (non-overlapping index ranges
    of a single permutation, never two independently-sampled subsets that could coincide).
    Raises InsufficientExamplesError if the pool is smaller than selection_size+audit_size.
    """
    required = selection_size + audit_size
    if selection_size == SELECTION_SET_SIZE and audit_size == AUDIT_SET_SIZE:
        assert required == MIN_DISJOINT_EXAMPLES_REQUIRED  # frozen-design consistency check, never re-derived independently
    if len(examples) < required:
        raise InsufficientExamplesError(
            f"Capability {capability!r} has only {len(examples)} examples; needs at least "
            f"{required} ({selection_size} selection + {audit_size} audit, disjoint) -- per "
            f"task spec, stop with INCONCLUSIVE rather than shrinking subset sizes."
        )

    ordered = list(examples)
    random.Random(seed).shuffle(ordered)
    selection = ordered[:selection_size]
    audit = ordered[selection_size:selection_size + audit_size]

    selection_ids = tuple(e.example_id for e in selection)
    audit_ids = tuple(e.example_id for e in audit)
    assert set(selection_ids).isdisjoint(audit_ids)  # structural guarantee, asserted rather than merely assumed

    manifest = CapabilitySubsetManifest(
        capability=capability, seed=seed, pool_size=len(examples),
        selection_example_ids=selection_ids, audit_example_ids=audit_ids,
        selection_hash=_subset_hash(selection_ids), audit_hash=_subset_hash(audit_ids),
    )
    return selection, audit, manifest


def write_subset_manifest(manifests: Dict[str, CapabilitySubsetManifest], path: "str | Path") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({cap: m.to_dict() for cap, m in sorted(manifests.items())}, indent=2))


def load_subset_manifest(path: "str | Path") -> Dict[str, CapabilitySubsetManifest]:
    data = json.loads(Path(path).read_text())
    return {cap: CapabilitySubsetManifest.from_dict(d) for cap, d in data.items()}


class SubsetManifestMismatchError(RuntimeError):
    """A freshly-rebuilt subset manifest does not exactly match a previously-persisted one --
    refuses to silently resample; the frozen subsets must never change after preregistration.
    """


def ensure_subset_manifest_unchanged(persisted: CapabilitySubsetManifest, current: CapabilitySubsetManifest) -> None:
    if persisted != current:
        raise SubsetManifestMismatchError(
            f"Subset manifest for capability {current.capability!r} has changed since it was "
            f"frozen -- persisted={persisted.to_dict()} current={current.to_dict()}. Refusing "
            f"to silently re-select examples after preregistration."
        )

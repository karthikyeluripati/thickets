"""Deterministic four-way disjoint data-role partition (spec section F): D_map / D_confirm /
D_select / D_test. Operates purely on example IDs (strings) -- no model outputs, no dataset
content, involved in constructing the partition itself, so this is fully CPU-testable and
independent of which capability/dataset the IDs came from.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

ROLE_NAMES: Tuple[str, ...] = ("map", "confirm", "select", "test")


class DataRoleError(RuntimeError):
    """Requested role sizes cannot be satisfied by the available example-ID pool, or the pool
    itself contains duplicate IDs -- never silently truncated or deduplicated.
    """


class DataRoleOverlapError(RuntimeError):
    """Two roles were found to share at least one example ID -- never silently tolerated."""


class DataRoleDriftError(RuntimeError):
    """A persisted role assignment references an example ID no longer present in a freshly
    -loaded ID pool (dataset drift) -- never silently dropped or resampled, mirroring
    ..benchmarks.subset_selection's existing dataset-drift guard.
    """


@dataclass(frozen=True)
class DataRolePartition:
    roles: Dict[str, Tuple[str, ...]]
    sizes: Dict[str, int]
    seed: int
    manifest_hash: str

    def to_dict(self) -> Dict:
        return {"roles": {k: list(v) for k, v in self.roles.items()}, "sizes": self.sizes, "seed": self.seed, "manifest_hash": self.manifest_hash}

    @classmethod
    def from_dict(cls, d: Dict) -> "DataRolePartition":
        return cls(roles={k: tuple(v) for k, v in d["roles"].items()}, sizes=dict(d["sizes"]), seed=d["seed"], manifest_hash=d["manifest_hash"])


def _compute_manifest_hash(roles: Dict[str, Tuple[str, ...]]) -> str:
    canonical = json.dumps({k: sorted(v) for k, v in roles.items()}, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_disjoint(partition: DataRolePartition) -> None:
    """Defensive re-check: partition_data_roles() already guarantees disjointness by
    construction (slicing a single shuffled permutation), but this is called on every
    partition before it is returned or reloaded, so a future refactor that breaks that
    invariant fails loudly rather than silently.
    """
    seen: Dict[str, str] = {}
    for role in ROLE_NAMES:
        for example_id in partition.roles.get(role, ()):
            if example_id in seen:
                raise DataRoleOverlapError(f"Example ID {example_id!r} assigned to both role {seen[example_id]!r} and {role!r}.")
            seen[example_id] = role


def partition_data_roles(example_ids: Sequence[str], sizes: Dict[str, int], seed: int) -> DataRolePartition:
    """Deterministic seeded-shuffle-then-slice partition: `example_ids` is shuffled with
    random.Random(seed), then sliced into contiguous, non-overlapping blocks in ROLE_NAMES
    order with the requested `sizes` (missing roles default to size 0). Zero overlap is
    guaranteed by construction (disjoint slices of one permutation) and re-verified by
    validate_disjoint() before returning. Hard-fails if `sizes` sum exceeds the pool, or if
    `example_ids` itself contains duplicates.
    """
    ids = list(example_ids)
    if len(ids) != len(set(ids)):
        raise DataRoleError("example_ids contains duplicate IDs -- refusing to partition an ambiguous pool.")

    unknown_roles = set(sizes) - set(ROLE_NAMES)
    if unknown_roles:
        raise DataRoleError(f"Unknown role name(s) in sizes: {sorted(unknown_roles)}, expected a subset of {ROLE_NAMES}.")

    total_requested = sum(sizes.get(role, 0) for role in ROLE_NAMES)
    if total_requested > len(ids):
        raise DataRoleError(f"Requested role sizes sum to {total_requested}, exceeding the available pool of {len(ids)} example IDs.")

    rng = random.Random(seed)
    shuffled = ids.copy()
    rng.shuffle(shuffled)

    roles: Dict[str, Tuple[str, ...]] = {}
    cursor = 0
    for role in ROLE_NAMES:
        n = sizes.get(role, 0)
        roles[role] = tuple(sorted(shuffled[cursor:cursor + n]))
        cursor += n

    partition = DataRolePartition(
        roles=roles, sizes={role: len(roles[role]) for role in ROLE_NAMES}, seed=seed, manifest_hash=_compute_manifest_hash(roles),
    )
    validate_disjoint(partition)
    return partition


def write_data_role_manifest(partition: DataRolePartition, path: "str | Path") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(partition.to_dict(), indent=2))


def load_data_role_manifest(path: "str | Path") -> DataRolePartition:
    partition = DataRolePartition.from_dict(json.loads(Path(path).read_text()))
    validate_disjoint(partition)
    return partition


def validate_against_pool(partition: DataRolePartition, current_example_ids: Sequence[str]) -> None:
    """Hard-fails if any persisted role ID is missing from a freshly-loaded ID pool (dataset
    drift) -- never silently resamples or drops the missing ID.
    """
    pool = set(current_example_ids)
    missing = {role: [i for i in ids if i not in pool] for role, ids in partition.roles.items()}
    missing = {role: ids for role, ids in missing.items() if ids}
    if missing:
        raise DataRoleDriftError(f"Persisted data-role manifest references example ID(s) no longer present in the current pool: {missing}")

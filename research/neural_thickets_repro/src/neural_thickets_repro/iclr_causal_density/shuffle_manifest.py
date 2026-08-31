"""Frozen, deterministic shuffled-image derangement manifest -- built ONCE per capability per
subset (selection, audit), reused identically across all 600 candidates x 5 capabilities. This
module NEVER reimplements the derangement algorithm -- it reuses
benchmarks.image_sanity.make_shuffled_variant (already validated: true derangement, no self-
maps, delegates the per-example visual-input swap to the owning adapter's own
make_shuffled_image_variant so capability-specific visual-input construction -- e.g. a
localized crop -- is handled correctly) and simply MATERIALIZES its result once, persists a
provenance manifest (original vs. shuffled image_ref per example_id), and hands the caller a
single frozen Example list to reuse for the rest of the run.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from ..benchmarks.base import CapabilityBenchmark, Example
from ..benchmarks.image_sanity import ImageSanityError, make_shuffled_variant
from .design import SHUFFLE_SEED


class ShuffleManifestIntegrityError(RuntimeError):
    """The materialized shuffled-image variant did not actually change the visual input for
    some example (image_ref identical to the original) -- most likely two source examples
    coincidentally sharing the same image. Fail closed rather than silently evaluating a
    'shuffled' condition that is not actually shuffled for that example.
    """


@dataclass(frozen=True)
class ShuffleMapping:
    example_id: str
    original_image_ref: str
    shuffled_image_ref: str


@dataclass(frozen=True)
class CapabilityShuffleManifest:
    capability: str
    subset_role: str  # "selection" or "audit"
    seed: int
    n: int
    mappings: Tuple[ShuffleMapping, ...]

    def to_dict(self) -> Dict:
        return {
            "capability": self.capability, "subset_role": self.subset_role, "seed": self.seed, "n": self.n,
            "mappings": [m.__dict__ for m in self.mappings],
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "CapabilityShuffleManifest":
        return cls(
            capability=d["capability"], subset_role=d["subset_role"], seed=d["seed"], n=d["n"],
            mappings=tuple(ShuffleMapping(**m) for m in d["mappings"]),
        )


def build_frozen_shuffled_variant(
    examples: Sequence[Example], capability: str, subset_role: str, benchmark: CapabilityBenchmark, *, seed: int = SHUFFLE_SEED,
) -> Tuple[List[Example], CapabilityShuffleManifest]:
    """Calls make_shuffled_variant EXACTLY ONCE and returns the resulting list alongside its
    provenance manifest -- the caller must reuse the returned list for every subsequent
    candidate's shuffled-image evaluation, never call this function again for the same
    (capability, subset_role) within one run ("never reshuffle between candidates").
    """
    try:
        shuffled = make_shuffled_variant(list(examples), seed, benchmark)
    except ImageSanityError as exc:
        raise ImageSanityError(f"Cannot build shuffled-image variant for {capability!r}/{subset_role!r}: {exc}") from exc

    mappings: List[ShuffleMapping] = []
    for orig, shuf in zip(examples, shuffled):
        if shuf.example_id != orig.example_id:
            raise ShuffleManifestIntegrityError(
                f"Shuffled variant reordered examples for {capability!r}/{subset_role!r}: "
                f"expected example_id {orig.example_id!r} at this position, got {shuf.example_id!r}."
            )
        if shuf.image_ref == orig.image_ref:
            raise ShuffleManifestIntegrityError(
                f"Shuffled variant did not actually change the visual input for example "
                f"{orig.example_id!r} in {capability!r}/{subset_role!r} (image_ref unchanged: "
                f"{orig.image_ref!r}) -- refusing to score this as a genuine shuffled condition."
            )
        mappings.append(ShuffleMapping(example_id=orig.example_id, original_image_ref=orig.image_ref, shuffled_image_ref=shuf.image_ref))

    manifest = CapabilityShuffleManifest(capability=capability, subset_role=subset_role, seed=seed, n=len(examples), mappings=tuple(mappings))
    return shuffled, manifest


def write_shuffle_manifest(manifests: Dict[str, CapabilityShuffleManifest], path: "str | Path") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({key: m.to_dict() for key, m in sorted(manifests.items())}, indent=2))


def load_shuffle_manifest(path: "str | Path") -> Dict[str, CapabilityShuffleManifest]:
    data = json.loads(Path(path).read_text())
    return {key: CapabilityShuffleManifest.from_dict(d) for key, d in data.items()}

"""The 600 unique (scope, radius, seed) perturbation candidates -- 6 scope-radius cells x 100
seeds each. Reuses candidate_sampling.sample_candidate_seeds BY IMPORT (the exact same seed-
draw convention as run_randopt_image_aware.py's own sample_candidates, proven byte-identical
by that module's own test suite) for the per-cell seed sequence, namespaced per (scope, radius)
via thicket.seeds.derive_seed so no two cells can ever coincidentally draw from the same
np.random.default_rng stream.

The SAME 100-seed sequence for a given cell is shared across all 5 capabilities (task spec) --
this module builds the candidate list ONCE, capability-agnostic; the evaluator (evaluator.py)
is what loops capabilities for each candidate, never resampling seeds per capability.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from ..candidate_sampling import sample_candidate_seeds
from ..thicket.seeds import derive_seed
from .design import CANDIDATE_SEED_BASE, N_SEEDS_PER_CELL, RADII, SCOPES


class DuplicateCandidateError(RuntimeError):
    """Two candidates in the built population share an identical candidate_id -- refuses to
    proceed with a population that would corrupt checkpoint/resume identity.
    """


def _format_radius(r: float) -> str:
    return f"{r:.6f}".replace(".", "")


def compute_candidate_id(scope: str, radius: float, seed: int) -> str:
    return f"{scope}_r{_format_radius(radius)}_seed{seed}"


@dataclass(frozen=True)
class PerturbationCandidate:
    candidate_id: str
    scope: str
    radius: float
    seed: int
    seed_index: int  # 0..N_SEEDS_PER_CELL-1 -- the SAME index corresponds to the SAME seed value across every capability for this cell


def build_cell_seeds(scope: str, radius: float, *, base_seed: int = CANDIDATE_SEED_BASE, n: int = N_SEEDS_PER_CELL) -> List[int]:
    cell_seed = derive_seed(base_seed, "iclr_causal_density_candidate_seeds", scope, str(radius))
    return sample_candidate_seeds(n, cell_seed)


def build_candidate_population(*, scopes: Sequence[str] = SCOPES, radii: Sequence[float] = RADII, base_seed: int = CANDIDATE_SEED_BASE, n_per_cell: int = N_SEEDS_PER_CELL) -> List[PerturbationCandidate]:
    candidates: List[PerturbationCandidate] = []
    seen_ids = set()
    for scope in scopes:
        for radius in radii:
            seeds = build_cell_seeds(scope, radius, base_seed=base_seed, n=n_per_cell)
            for idx, seed in enumerate(seeds):
                candidate_id = compute_candidate_id(scope, radius, seed)
                if candidate_id in seen_ids:
                    raise DuplicateCandidateError(f"Duplicate candidate_id {candidate_id!r} in cell (scope={scope!r}, radius={radius!r}).")
                seen_ids.add(candidate_id)
                candidates.append(PerturbationCandidate(candidate_id=candidate_id, scope=scope, radius=radius, seed=seed, seed_index=idx))
    return candidates


def validate_candidate_population(candidates: Sequence[PerturbationCandidate], *, expected_scopes: Sequence[str] = SCOPES, expected_radii: Sequence[float] = RADII, expected_n_per_cell: int = N_SEEDS_PER_CELL) -> None:
    """Structural invariants: expected total count, unique candidate_ids, exactly the expected
    seed count per cell, and (task spec) the SAME ordered seed sequence used for the SAME
    (scope, radius) cell no matter which capability is being evaluated -- trivially true here
    since this module builds ONE capability-agnostic population, never one per capability.
    """
    ids = [c.candidate_id for c in candidates]
    if len(ids) != len(set(ids)):
        raise DuplicateCandidateError(f"Population has {len(ids)} candidates, {len(set(ids))} unique candidate_ids.")
    expected_total = len(expected_scopes) * len(expected_radii) * expected_n_per_cell
    if len(candidates) != expected_total:
        raise ValueError(f"Population has {len(candidates)} candidates, expected {expected_total}.")

    by_cell: Dict[Tuple[str, float], List[PerturbationCandidate]] = {}
    for c in candidates:
        by_cell.setdefault((c.scope, c.radius), []).append(c)
    expected_cells = {(s, r) for s in expected_scopes for r in expected_radii}
    if set(by_cell) != expected_cells:
        raise ValueError(f"Population cells {sorted(by_cell)} do not match expected cells {sorted(expected_cells)}.")
    for cell, cell_candidates in by_cell.items():
        if len(cell_candidates) != expected_n_per_cell:
            raise ValueError(f"Cell {cell} has {len(cell_candidates)} candidates, expected {expected_n_per_cell}.")
        seeds = [c.seed for c in cell_candidates]
        if len(seeds) != len(set(seeds)):
            raise DuplicateCandidateError(f"Cell {cell} has duplicate seeds: {seeds}")


def compute_population_hash(candidates: Sequence[PerturbationCandidate]) -> str:
    canonical = json.dumps([[c.scope, c.radius, c.seed, c.seed_index] for c in candidates], sort_keys=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_candidate_manifest(candidates: Sequence[PerturbationCandidate], path: "str | Path") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "n_candidates": len(candidates), "population_hash": compute_population_hash(candidates),
        "candidates": [c.__dict__ for c in candidates],
    }, indent=2))

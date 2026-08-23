"""Low-rank expert geometry -- INTERFACE / DESIGN WORK ONLY in Stage 5 (spec section I). Do
NOT run large SVDs here, and do NOT concatenate billion-dimensional perturbation vectors into
RAM anywhere in this project's CPU-only code.

THE SCALABLE STORAGE STRATEGY (the actual design decision this module exists to record):
we never persist a perturbation's raw high-dimensional epsilon vector at all. Every
perturbation is exactly reconstructible on demand from its small PerturbationManifest (seed +
anatomy_region + radius/sigma + parameter_mask_hash) via the identical per-tensor-reseed noise
generation already used to apply it (..perturbation.apply_anatomical_relative_l2 /
..perturb_cpu._generate_noise -- "same manifest+seed reproduces the same perturbation" is
tested directly in tests/test_thicket_perturbation.py). Later low-rank analysis (effective
rank, singular spectrum, split-half subspace reproducibility, principal angles between
capability-specific expert subspaces, low-rank consolidation) therefore only needs to persist
PerturbationVectorHandle-shaped records (manifest + per-capability scalar deltas) -- a few
hundred bytes each, trivially fitting in RAM even for large populations -- and can regenerate
the actual per-region noise chunk-by-chunk (e.g. one transformer layer's flattened parameters
at a time) directly on a GPU worker at analysis time, accumulating a randomized/streaming SVD
(e.g. a blocked randomized range-finder over layer-chunks) rather than ever materializing a
[n_perturbations x n_parameters] matrix. That streaming/chunked SVD driver is Stage 6+ GPU work
and is explicitly out of scope here; this module implements only the small, generic
linear-algebra primitives that operate on ALREADY-COMPUTED, small subspace bases or singular
-value arrays (never on raw per-perturbation vectors).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict

import numpy as np

if TYPE_CHECKING:
    from .perturbation import PerturbationManifest


@dataclass(frozen=True)
class PerturbationVectorHandle:
    """What gets PERSISTED per perturbation for later geometry analysis -- deliberately NOT
    the raw vector itself. `manifest` fully determines the reconstructible epsilon;
    `capability_deltas` is the small set of scalars needed to select WHICH perturbations
    belong in a later subspace (e.g. only confirmed experts for capability t).
    """
    perturbation_id: str
    manifest: "PerturbationManifest"
    capability_deltas: Dict[str, float]


def effective_rank(singular_values: np.ndarray) -> float:
    """Roy & Vetterli's effective rank: exp(H(p)) where p is the singular-value spectrum
    normalized to a probability distribution and H is its Shannon entropy (nats). A spectrum
    with a single nonzero singular value has effective rank 1; a spectrum with k equal nonzero
    singular values has effective rank k. Non-positive singular values are dropped before
    normalizing (a numerically negative-but-should-be-zero SVD output is never treated as
    "negative probability mass").
    """
    s = np.asarray(singular_values, dtype=float)
    s = s[s > 0]
    if s.size == 0:
        return 0.0
    p = s / s.sum()
    entropy = -float(np.sum(p * np.log(p)))
    return float(np.exp(entropy))


def principal_angles(basis_a: np.ndarray, basis_b: np.ndarray) -> np.ndarray:
    """Principal angles (radians, ascending) between the subspaces spanned by the COLUMNS of
    `basis_a` and `basis_b` (already-computed, small, dense bases -- e.g. the top-k left
    singular vectors of two capabilities' own already-reduced expert-perturbation matrices,
    never the raw per-parameter vectors) -- via QR-orthonormalization followed by the SVD of
    Q_a^T Q_b (the singular values are the cosines of the principal angles).
    """
    q_a, _ = np.linalg.qr(np.asarray(basis_a, dtype=float))
    q_b, _ = np.linalg.qr(np.asarray(basis_b, dtype=float))
    cosines = np.linalg.svd(q_a.T @ q_b, compute_uv=False)
    cosines = np.clip(cosines, -1.0, 1.0)
    return np.arccos(cosines)

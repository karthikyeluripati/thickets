"""Diversity / specialization metrics (spec section H) -- operate on a perturbation x
capability Delta matrix (rows = perturbation IDs, aligned across columns per spec D1;
columns = capability performance deltas). Pure numpy, no scipy/pandas dependency (neither is
in requirements-cpu.txt) -- Spearman rank correlation is implemented directly as Pearson
correlation of ranks (mathematically identical to scipy.stats.spearmanr), via numpy's own
np.corrcoef.

SPECTRAL DISCORDANCE (spec H2): external/RandOpt is not checked out in this repository (it is
cloned dynamically at a pinned commit on the pod -- see external/setup_external_repo.py -- and
has no declared license, per REPRO_SPEC.md's existing "read, described, and invoked externally,
never transcribed" discipline). The exact scientific definition below is ported from the
PUBLISHED paper this project reproduces (Neural Thickets: Diverse Task Experts Are Dense Around
Pretrained Weights, arXiv:2603.12228, Definition 2.2) -- public mathematics, not transcribed
code -- rather than invented from scratch:

    D = 1 - (1 / (M * (M - 1))) * sum_{j != k} C_jk

where P in [0,1]^(N x M) is the percentile-rank matrix (N perturbations/seeds, M tasks) and
C = corr(P) in R^(M x M) is the Pearson correlation matrix of P's columns. D -> 1 implies
orthogonal task rankings (specialists: a perturbation that helps one task tends to hurt
another); D -> 0 implies parallel rankings (generalists). The paper reports D bounded in
[0, M/(M-1)]. This is UNRESOLVED against the actual upstream RandOpt code (never inspected
directly, since it is not checked out here) -- see VISUAL_THICKET_EXPERIMENT_SPEC.md's decision
table; if the pod-side upstream implementation differs in some detail (e.g. a different rank
-normalization convention), that difference must be reconciled there before this metric is
treated as frozen for the paper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np


class DiversityInputError(ValueError):
    pass


def percentile_rank_matrix(delta_matrix: np.ndarray) -> np.ndarray:
    """Column-wise percentile ranks in (0, 1): rank 0 (smallest) maps to 0.5/N, rank N-1
    (largest) maps to (N-0.5)/N -- ties broken by np.argsort's stable order (documented, not
    a randomized tie-break). Matches the paper's P in [0,1]^(N x M) percentile-rank matrix.
    """
    arr = np.asarray(delta_matrix, dtype=float)
    if arr.ndim != 2:
        raise DiversityInputError(f"delta_matrix must be 2-D (N perturbations x M tasks), got shape {arr.shape}")
    n = arr.shape[0]
    ranks = np.argsort(np.argsort(arr, axis=0), axis=0)
    return (ranks + 0.5) / n


def task_rank_correlation_matrix(delta_matrix: np.ndarray) -> np.ndarray:
    """M x M Pearson correlation matrix of the percentile-rank matrix's columns -- equal to
    the Spearman rank correlation matrix of the raw `delta_matrix` columns (spec H1's "pairwise
    Spearman correlation of perturbation rankings between tasks"), and exactly the C matrix in
    spectral_discordance()'s definition (spec H2) -- one computation serves both.
    """
    p = percentile_rank_matrix(delta_matrix)
    if p.shape[1] < 2:
        raise DiversityInputError("task_rank_correlation_matrix requires at least 2 tasks (columns)")
    return np.corrcoef(p, rowvar=False)


def spectral_discordance(delta_matrix: np.ndarray) -> float:
    """D = 1 - mean of the off-diagonal entries of C (spec H2, Definition 2.2 of the paper).
    Requires at least 2 tasks (M=1 has no off-diagonal entries to average).
    """
    c = task_rank_correlation_matrix(delta_matrix)
    m = c.shape[0]
    off_diag_sum = float(c.sum() - np.trace(c))
    return 1.0 - off_diag_sum / (m * (m - 1))


def top_q_indices(deltas: Sequence[float], q: float, q_is_fraction: bool = True) -> np.ndarray:
    """Indices of the top-q performers by delta (descending), sorted ascending for a stable,
    order-independent set representation. `q` is a fraction of N (q_is_fraction=True, rounded
    to the nearest integer, minimum 1) or an absolute top-K count (q_is_fraction=False).
    """
    arr = np.asarray(deltas, dtype=float)
    n = arr.size
    k = max(1, int(round(q * n))) if q_is_fraction else int(q)
    k = min(k, n)
    order = np.argsort(-arr, kind="stable")
    return np.sort(order[:k])


def jaccard(indices_a: Sequence[int], indices_b: Sequence[int]) -> float:
    """|A intersect B| / |A union B|. An empty union (both sets empty) is defined as 0.0 --
    documented convention, not "trivially identical".
    """
    a, b = set(indices_a), set(indices_b)
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def expert_overlap_matrix(delta_matrix: np.ndarray, q: float, q_is_fraction: bool = True) -> np.ndarray:
    """M x M Jaccard-overlap matrix of each pair of tasks' top-q expert-index sets (spec H3).
    Diagonal is always 1.0 (a set's overlap with itself).
    """
    arr = np.asarray(delta_matrix, dtype=float)
    m = arr.shape[1]
    top_sets = [top_q_indices(arr[:, t], q, q_is_fraction) for t in range(m)]
    out = np.eye(m, dtype=float)
    for j in range(m):
        for k in range(j + 1, m):
            value = jaccard(top_sets[j], top_sets[k])
            out[j, k] = out[k, j] = value
    return out


def cross_capability_transfer_matrix(delta_matrix: np.ndarray, q: float, q_is_fraction: bool = True) -> np.ndarray:
    """T[t, u] = mean(Delta_u | perturbation selected as a top-q expert for capability t)
    (spec H4) -- a directional M x M matrix (T[t, u] need not equal T[u, t]).
    """
    arr = np.asarray(delta_matrix, dtype=float)
    m = arr.shape[1]
    out = np.empty((m, m), dtype=float)
    for t in range(m):
        selected = top_q_indices(arr[:, t], q, q_is_fraction)
        for u in range(m):
            out[t, u] = float(arr[selected, u].mean())
    return out


@dataclass(frozen=True)
class CapabilitySignatureMatrix:
    """Every perturbation's complete capability-delta vector v_i = [Delta_1, ..., Delta_T]
    (spec H5) -- the same delta_matrix used throughout this module, labeled with perturbation
    IDs and task names for later clustering/PCA/expert-family analysis. No visualization is
    implemented here (spec H5: "do not implement expensive visualizations now").
    """
    perturbation_ids: Tuple[str, ...]
    task_names: Tuple[str, ...]
    matrix: np.ndarray

    def __post_init__(self) -> None:
        if self.matrix.shape != (len(self.perturbation_ids), len(self.task_names)):
            raise DiversityInputError(
                f"matrix shape {self.matrix.shape} does not match "
                f"(len(perturbation_ids)={len(self.perturbation_ids)}, len(task_names)={len(self.task_names)})"
            )

    def row(self, perturbation_id: str) -> np.ndarray:
        return self.matrix[self.perturbation_ids.index(perturbation_id)]

    def column(self, task_name: str) -> np.ndarray:
        return self.matrix[:, self.task_names.index(task_name)]

"""Stage 7A Section 3: analytical EXPECTED-NORM anchors mapping Stage 6's frozen upstream
sigma grid onto a relative-L2 radius scale -- NOT measured realized radii (that is Section 4's
job, an empirical single-perturbation GPU sanity check) and NOT final anatomical calibration
hyperparameters (Section 5 only PROPOSES an initial common grid mechanically from these; nothing
here or downstream may select among them by capability score).

For an isotropic per-tensor-reseed Gaussian perturbation theta' = theta + sigma * epsilon,
epsilon ~ N(0, I) over d perturbable elements: E[||sigma * epsilon||_2^2] = sigma^2 * d, so
E[||epsilon_applied||_2] ~= sigma * sqrt(d) (an approximation of E[||X||] by sqrt(E[||X||^2])
for a chi-distributed norm -- exact for large d, which every real anatomical region here has by
several orders of magnitude). The relative-L2 anchor is then this expected norm divided by the
region's own ||theta||_2:

    r_hat = sigma * sqrt(d) / ||theta||_2
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

from ..run_global_visual_thicket_pilot import UPSTREAM_SIGMA_GRID as FROZEN_STAGE6_SIGMAS

ANCHOR_LABEL = "analytical_expected_norm_anchor"


def expected_epsilon_l2_norm(sigma: float, d: int) -> float:
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")
    if d <= 0:
        raise ValueError(f"d must be positive, got {d}")
    return sigma * math.sqrt(d)


def relative_l2_from_sigma(sigma: float, d: int, theta_l2_norm: float) -> float:
    if theta_l2_norm <= 0:
        raise ValueError(f"theta_l2_norm must be positive, got {theta_l2_norm}")
    return expected_epsilon_l2_norm(sigma, d) / theta_l2_norm


def build_sigma_relative_l2_mapping(
    sigmas: Sequence[float], d: int, theta_l2_norm: float, *, scope_label: str,
) -> List[Dict[str, object]]:
    """One row per sigma -- always labeled ANCHOR_LABEL so a downstream reader can never
    mistake this for a measured realized radius.
    """
    rows = []
    for sigma in sigmas:
        expected_norm = expected_epsilon_l2_norm(sigma, d)
        r_hat = relative_l2_from_sigma(sigma, d, theta_l2_norm)
        rows.append(
            {
                "sigma": sigma,
                "scope": scope_label,
                "d": d,
                "theta_l2_norm": theta_l2_norm,
                "expected_epsilon_l2_norm": expected_norm,
                "r_hat": r_hat,
                "kind": ANCHOR_LABEL,
            }
        )
    return rows


def select_common_calibration_radii(r_hat_values: Sequence[float], *, round_sig_figs: int = 6) -> List[float]:
    """Mechanical, order-preserving dedup of the translated Stage-6 sigma scales into a single
    COMMON radius grid -- never a per-region or per-capability selection, and never filtered by
    any downstream accuracy/behavior signal. Drops a value only if it is numerically pathological
    (non-finite or <= 0) or a near-duplicate (same value to `round_sig_figs` significant figures
    as one already kept), keeping the FIRST occurrence in `r_hat_values`'s own order (i.e. the
    sigma-grid order Section 3 already computed them in) rather than re-sorting.
    """

    def _round_sig(x: float, sig: int) -> float:
        if x == 0:
            return 0.0
        return round(x, -int(math.floor(math.log10(abs(x)))) + (sig - 1))

    seen_rounded = set()
    kept: List[float] = []
    for value in r_hat_values:
        if not math.isfinite(value) or value <= 0:
            continue
        key = _round_sig(value, round_sig_figs)
        if key in seen_rounded:
            continue
        seen_rounded.add(key)
        kept.append(value)
    return kept

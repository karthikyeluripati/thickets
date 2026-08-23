"""Deterministic, namespace-separated seed derivation (spec section P).

All randomness in this package must be explicit and reproducible: perturbation sampling,
subset/data-role construction, and bootstrap analysis each need their OWN seed stream so that,
e.g., changing the bootstrap resample count never perturbs which perturbations were sampled.
`derive_seed` turns one root/global seed plus a free-form namespace path into a stable 63-bit
integer seed -- the same (base_seed, *namespace_parts) always derives the same seed, and
changing ANY part (including their order) changes it, without maintaining a manual registry of
"the next unused seed offset".
"""
from __future__ import annotations

import hashlib

_DERIVED_SEED_BITS = 63  # stays a valid non-negative Python int / numpy/torch seed on any platform


def derive_seed(base_seed: int, *namespace_parts: object) -> int:
    """Deterministic derived seed: sha256(f"{base_seed}|{part}|{part}|...") truncated to the
    low 63 bits. `namespace_parts` should uniquely name what this seed is FOR (e.g.
    ("perturbation_population", mode, region, str(radius), str(i)) or ("data_roles", split_name)
    or ("bootstrap", metric_name)) -- two call sites with different namespace parts get
    independent seed streams even when given the same base_seed.
    """
    payload = "|".join([str(base_seed), *(str(p) for p in namespace_parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) & ((1 << _DERIVED_SEED_BITS) - 1)

"""Neutral, standalone candidate-seed sampling for the scoped RandOpt (relative_l2) path.

run_randopt_image_aware.py (Gate 2, frozen) is NOT touched, imported from, or refactored by
this module in any way -- per explicit instruction, Gate 2 stays exactly as it is. This file
independently reproduces only the SEED-drawing portion of that file's own
sample_candidates(seed draw: np.random.default_rng(seed).choice(2**31, size=n,
replace=False)), because relative_l2-mode scoped experiments need the identical seed
sequence a raw_sigma run would use, without ever drawing (and then discarding) a meaningless
sigma value from a sigma_candidate set that doesn't apply in relative_l2 mode at all.

tests/test_candidate_sampling.py proves this reproduces run_randopt_image_aware.py's
sample_candidates() seeds exactly, for several (N, seed) combinations -- not asserted by
inspection alone.
"""
from __future__ import annotations

from typing import List

import numpy as np


def sample_candidate_seeds(n: int, seed: int) -> List[int]:
    """N unique seeds drawn without replacement from a single np.random.default_rng(seed)
    stream -- identical to the first draw run_randopt_image_aware.py:sample_candidates()
    makes from a freshly seeded generator with the same (n, seed). Since both this function
    and that one perform this exact draw as the FIRST operation on an independently
    constructed np.random.default_rng(seed) (deterministic given the seed), the two produce
    byte-identical seed sequences without this function calling into, or depending on,
    sample_candidates() at all.
    """
    rng = np.random.default_rng(seed=seed)
    return [int(s) for s in rng.choice(2**31, size=n, replace=False)]

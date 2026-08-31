"""FROZEN preregistered design for the isolated 7B causal-density pilot. Single source of
truth: preregistration.md is generated FROM these constants (never hand-typed a second time),
and every other module in this package imports its design parameters from here rather than
re-declaring them. Changing any value in this file after results exist would be exactly the
"change thresholds after observing results" violation the task prohibits -- this module is
frozen the moment reports/iclr_causal_density/preregistration.md is written and must not be
edited afterward without a new, separately preregistered pilot.

Nothing here touches vllm/ray/torch/GPU -- pure Python constants and pure arithmetic, so the
whole design is importable and testable with zero dependencies beyond the stdlib.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

# =================================================================================================
# Model -- the repository's verified identifier only, never substituted.
# =================================================================================================
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_SCALE = "7B"

# =================================================================================================
# Capabilities -- exactly these five, using the ALREADY-VALIDATED benchmarks/adapters/*.py
# implementations (Capability Benchmark Gate infrastructure) reused unmodified.
# =================================================================================================
CAPABILITIES: Tuple[str, ...] = (
    "visual_grounding",       # RefCOCO/RefCOCO+ -- adapters/visual_grounding_refcoco.py
    "counting",                # TallyQA -- adapters/counting_tallyqa.py
    "ocr_text_recognition",    # TextVQA -- adapters/ocr_text_recognition_textvqa.py
    "spatial_reasoning",       # GQA-spatial -- adapters/spatial_reasoning_gqa.py
    "relational_reasoning",    # GQA-relational -- adapters/relational_reasoning_gqa.py
)
assert len(CAPABILITIES) == 5

# Explicitly excluded (task spec) -- listed so the exclusion is visible in code, not just prose.
EXCLUDED_CAPABILITIES: Tuple[str, ...] = ("attribute_recognition", "fine_grained_recognition")

# =================================================================================================
# Scopes -- exactly these three, mapped 1:1 onto scopes.py's existing canonical scope names.
# Parameters are never redefined here -- scopes.py's own build_scope_manifest is the only source
# of which parameters belong to a scope.
# =================================================================================================
SCOPES: Tuple[str, ...] = ("vision_encoder", "full_lm", "full_vlm")
assert len(SCOPES) == 3

# =================================================================================================
# Radii -- exactly these two, using scopes.py's existing relative-L2 norm-matching formula
# (compute_relative_l2_sigma) -- never a new perturbation-magnitude convention.
# =================================================================================================
RADII: Tuple[float, ...] = (0.02, 0.04)
assert len(RADII) == 2

# =================================================================================================
# Perturbations
# =================================================================================================
N_SEEDS_PER_CELL = 100
N_SCOPE_RADIUS_CELLS = len(SCOPES) * len(RADII)
assert N_SCOPE_RADIUS_CELLS == 6
N_UNIQUE_PERTURBATIONS = N_SCOPE_RADIUS_CELLS * N_SEEDS_PER_CELL
assert N_UNIQUE_PERTURBATIONS == 600

# The base seed the shared 100-seed sequence is derived from -- distinct from every other
# frozen base seed in this repository (Stage 6/7B/8/9/11 all use their own, this pilot needs
# its own too, per this project's established "never coincidentally reuse a seed root" discipline).
CANDIDATE_SEED_BASE = 20261005  # distinct from every prior Stage base seed in this repo

# =================================================================================================
# Frozen evaluation sets
# =================================================================================================
SELECTION_SET_SIZE = 200
AUDIT_SET_SIZE = 200
MIN_DISJOINT_EXAMPLES_REQUIRED = SELECTION_SET_SIZE + AUDIT_SET_SIZE  # 400 -- else INCONCLUSIVE
SUBSET_SELECTION_SEED = 20261005  # frozen BEFORE any results exist; never re-derived per capability differently

# =================================================================================================
# Visual conditions
# =================================================================================================
VISUAL_CONDITIONS: Tuple[str, ...] = ("correct_image", "shuffled_image", "text_only")
assert len(VISUAL_CONDITIONS) == 3

SHUFFLE_SEED = 20261006  # frozen once, applied identically to every capability's audit+selection sets

# =================================================================================================
# Search-budget analysis
# =================================================================================================
SEARCH_BUDGETS: Tuple[int, ...] = (10, 25, 50, 100)
N_MONTE_CARLO_SUBSAMPLES = 1000
SEARCH_BUDGET_ANALYSIS_SEED = 20261007  # the one preregistered analysis seed for all resampling
TOP_K_POOL_SIZE = 10

# =================================================================================================
# Grounded selection -- the fixed coefficient is NEVER tuned.
# =================================================================================================
GROUNDED_COEFFICIENT = 0.5

# =================================================================================================
# Decision-gate thresholds -- frozen BEFORE any results exist. See decision_gate.py for the
# literal, code-enforced application of these numbers; they are never re-typed anywhere else.
# =================================================================================================
DECISION_D_THRESHOLD = 2.0
DECISION_D_MIN_CAPABILITIES_ABOVE_THRESHOLD = 4       # of 5
DECISION_D_CI_EXCLUDES_ONE_MIN_CAPABILITIES = 3        # of 5
DECISION_BUDGET_DIVERGENCE_MIN_CAPABILITIES = 4        # of 5
DECISION_GROUNDED_RETENTION_FRACTION = 0.8
DECISION_GROUNDED_G_IMPROVEMENT_MIN_CAPABILITIES = 4   # of 5

BOOTSTRAP_N_RESAMPLES = 10_000
BOOTSTRAP_CI_LEVEL = 0.95
# The ONE preregistered analysis seed for Phase 7's paired bootstrap (distinct from
# SEARCH_BUDGET_ANALYSIS_SEED, which governs Phase 8's separate Monte Carlo subsampling) --
# frozen before any results exist, never re-drawn after seeing data.
BOOTSTRAP_ANALYSIS_SEED = 20261008


PREREGISTERED_BOOTSTRAP_METHOD_NOTE = (
    "Phase 7 uses ONE shared paired-bootstrap resample matrix of the audit-set EXAMPLE indices "
    "(B=10,000 resamples, seed=BOOTSTRAP_ANALYSIS_SEED, with replacement, same index set applied "
    "to every candidate and the base model for that resample -- 'paired' in both senses: "
    "correct/shuffled/text-only scores for one candidate always come from the SAME resampled "
    "examples, and every candidate's resample b uses the SAME example-index set as every other "
    "candidate's resample b). Per candidate i: CI_low^95%(G_i) is the 2.5th percentile of "
    "{G_i^(b)}_{b=1..B} computed from that shared matrix -- this determines a FIXED (not random) "
    "causally-visual-expert classification for candidate i from the observed data. For the "
    "population-level D=rho_standard/rho_visual's own 95% CI, each resample b's OWN "
    "Delta_i^R(b)>0 and G_i^(b)>0 are used as that resample's plug-in per-candidate "
    "classification (never a nested 10,000x10,000 bootstrap-of-bootstrap), giving rho_standard^(b)/"
    "rho_visual^(b)/D^(b) per resample and hence a 95% CI for D from {D^(b)}. This convention is "
    "fixed HERE, before any results exist, and is never altered after seeing data."
)


@dataclass(frozen=True)
class FrozenDesign:
    """A single object carrying every frozen value above, for callers (preregistration
    renderer, manifest builders, decision gate) that want one thing to pass around/hash rather
    than importing two dozen module-level constants individually.
    """
    model_name: str = MODEL_NAME
    model_scale: str = MODEL_SCALE
    capabilities: Tuple[str, ...] = CAPABILITIES
    scopes: Tuple[str, ...] = SCOPES
    radii: Tuple[float, ...] = RADII
    n_seeds_per_cell: int = N_SEEDS_PER_CELL
    n_unique_perturbations: int = N_UNIQUE_PERTURBATIONS
    candidate_seed_base: int = CANDIDATE_SEED_BASE
    selection_set_size: int = SELECTION_SET_SIZE
    audit_set_size: int = AUDIT_SET_SIZE
    subset_selection_seed: int = SUBSET_SELECTION_SEED
    visual_conditions: Tuple[str, ...] = VISUAL_CONDITIONS
    shuffle_seed: int = SHUFFLE_SEED
    search_budgets: Tuple[int, ...] = SEARCH_BUDGETS
    n_monte_carlo_subsamples: int = N_MONTE_CARLO_SUBSAMPLES
    search_budget_analysis_seed: int = SEARCH_BUDGET_ANALYSIS_SEED
    top_k_pool_size: int = TOP_K_POOL_SIZE
    grounded_coefficient: float = GROUNDED_COEFFICIENT


FROZEN_DESIGN = FrozenDesign()


def expected_row_count() -> int:
    """Expected long-form result rows for the DECISIVE pilot (Phase 6) -- 600 unique
    perturbations x 5 capabilities x 3 visual conditions on the audit set, PLUS the same shape
    again for base-model rows would be a separate, smaller base-control count (see
    expected_base_control_row_count) -- this counts candidate rows only.
    """
    return N_UNIQUE_PERTURBATIONS * len(CAPABILITIES) * len(VISUAL_CONDITIONS)


def expected_base_control_row_count() -> int:
    """Phase 5 base-control gate: unperturbed model x 5 capabilities x both subsets x 3
    conditions.
    """
    return len(CAPABILITIES) * 2 * len(VISUAL_CONDITIONS)

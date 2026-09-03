# Preregistration Amendment — 2026-09-03

This is a **post-hoc amendment** to `reports/iclr_causal_density/preregistration.md`, written
after the decisive-pilot (Phase 6) audit-set pass had completed but **before** the full 6-cell
search-budget analysis (Phase 8) existed for any capability. It records two gaps the original
preregistration left unspecified, discovered while wiring the Phase 7-10 analysis driver against
real collected data, and how each was resolved. Neither resolution touches any of the four
frozen analysis modules (`metrics.py`, `search_budget.py`, `grounded_selection.py`,
`decision_gate.py`), any preregistered constant in `design.py`, or any threshold, radius, seed,
subset, or scope. `design.py` itself remains completely unedited, per its own stated
frozen-on-write discipline.

## 1. `visual_grounding` exclusion from Phase 7–9

**Gap**: The preregistration's Visual Conditions section states `correct_image`, `shuffled_image`,
`text_only` are "evaluated for every base-model row and every candidate row," without carving out
an exception. In practice, RefCOCO/RefCOCO+ grounding has no meaningful text-only condition —
localizing an object's position in an image is undefined without the image — and this was
already, correctly, treated as unsupported throughout Phases 0–6 (`benchmark.
supports_text_only_condition()` returning `False` for this one capability, consistently, from the
Capability Benchmark Gate infrastructure onward).

`metrics.ConditionScores` (Phase 7) requires all three condition arrays, of equal length, by
construction (`__post_init__` raises otherwise). `visual_grounding` structurally cannot supply a
`text` array — this is not a missing-data problem correctable by more GPU time; it is unclassifiable
under the frozen Phase 7 formula, given the frozen formula's own equal-length requirement.

**Resolution**: `visual_grounding` is excluded from Phase 7, 8, and 9's computations. No
substitute value is fabricated for its missing text-only score, and the frozen `metrics.py`
formula is not modified to accept a two-condition input. The consequence flows entirely through
`decision_gate.py`'s own **pre-existing, unmodified** precedence rule 2 — `evaluate_decision_gate`
already requires exactly 5 `CapabilityGateInputs`, and returns `INCONCLUSIVE` with reason
`"expected exactly 5 capabilities with valid results, got 4"` otherwise. This is the frozen
module's own designed behavior for exactly this situation, not a new rule invented to handle it.

## 2. Capability-level search-budget divergence aggregation

**Gap**: `search_budget.py`'s own docstring is explicit that Monte Carlo search-budget analysis
and `check_registered_divergence` run **per (capability, scope, radius) cell** — "1,000
deterministic Monte Carlo subsamples per (capability, scope, radius) cell, drawn from that cell's
100-candidate pool." The decision gate's CONFIRMED criterion, per the preregistration, is stated
**per capability** — "the registered search-budget divergence holds in ≥4/5 capabilities" — matching
`decision_gate.CapabilityGateInputs.search_budget_divergence_confirmed: bool`, one boolean per
capability. Neither document specifies how one capability's **six** per-cell divergence booleans
collapse into that single per-capability boolean.

**Resolution (user-authorized, frozen 2026-09-03T[session], explicitly before the full 6-cell
results existed for any capability — only `vision_encoder`'s 2 cells were complete at decision
time, for any capability)**:

> A capability's `search_budget_divergence_confirmed` is `True` if and only if:
> 1. At least 4 of its 6 `(scope, radius)` cells have `divergence_confirmed=True`
>    (`search_budget.check_registered_divergence`'s own, unmodified output), **and**
> 2. Those passing cells cover **both** radii (0.02 and 0.04), **and**
> 3. Those passing cells cover **at least two** of the three scopes.

Implemented as tested, standalone code — never a prose-only rule — in
`src/neural_thickets_repro/iclr_causal_density/capability_divergence_aggregation.py`
(commit `1420151`). That module never calls, wraps, or reimplements `search_budget.py`'s or
`decision_gate.py`'s own logic; both remain completely unmodified. It is pure post-hoc
aggregation over already-computed per-cell booleans, matching this project's own
"implemented in code, never assigned subjectively" discipline (`decision_gate.py`'s own docstring).

## 3. Confidence-interval semantics (audited, not amended — no gap found)

A third item was raised for audit: the observed divergence between each capability's `D`
**point estimate** (10.8–90.7 across the four eligible capabilities) and its **95% CI**
(consistently ≈[1.01, 1.6]). This was investigated and found to be **already fully specified** in
the original preregistration — no amendment needed. `design.PREREGISTERED_BOOTSTRAP_METHOD_NOTE`
states explicitly that the population-level `D`'s CI uses each bootstrap resample's own
`Δ_i^R(b) > 0` and `G_i^(b) > 0` as that resample's per-candidate classification (a single,
un-CI'd threshold), while the **point-estimate** `D` (and each candidate's own conventional/
causally-visual classification) uses the stricter, individually-bootstrapped
`g_ci_low > 0` rule (`metrics.classify_candidate`) — explicitly to avoid a computationally
infeasible nested (10,000×10,000) bootstrap-of-bootstrap. With very few candidates meeting the
strict per-candidate CI bar (3–30 of 600, depending on capability), `D`'s point estimate is an
unstable small-denominator ratio; the laxer per-resample rule used for the CI classifies far more
candidates as causally-visual per resample, pulling the CI down toward 1. Both computations run
the frozen, unmodified `metrics.py` exactly as authored — this is the preregistered method
working as designed, not a defect.

## Non-negotiables preserved

- No change to any of the four frozen Phase 7–10 modules.
- No change to `design.py` or any preregistered constant, threshold, radius, seed, subset, or scope.
- No change made after seeing the full analysis outcome — the aggregation rule (§2) was frozen
  before the full 6-cell results existed for any capability.
- `iclr-causal-density-pilot` branch only; `main` untouched.

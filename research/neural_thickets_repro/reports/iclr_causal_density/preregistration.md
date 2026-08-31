# Preregistration — Isolated 7B Causal-Density Pilot

Frozen before any GPU execution, any result row, or any figure exists. Source of truth for
every constant below: `src/neural_thickets_repro/iclr_causal_density/design.py` — this document
must never be edited to match a result; any design change is a new, separately preregistered
pilot on its own branch.

Branch: `iclr-causal-density-pilot`, based on commit `9305cc8` (the completed 32B S1/S2
milestone, read-only, never modified by this pilot).

## Frozen hypothesis

> Standard RandOpt selects nearby VLMs that improve benchmark accuracy by exploiting language
> shortcuts. Consequently, conventional expert density overestimates causally visual expert
> density. Counterfactual selection recovers genuinely visual experts.

Not treated as true before the registered gates (Phase 10) pass.

## Model

`Qwen/Qwen2.5-VL-7B-Instruct` — the repository's verified `SCALING_MODEL_REGISTRY["7B"]`
identifier, resolved to its immutable HF revision via `resolve_immutable_model_revision` at run
time (never a hand-typed SHA, never substituted for another checkpoint).

## Capabilities (exactly five)

| Capability | Dataset | Adapter |
|---|---|---|
| `visual_grounding` | RefCOCO/RefCOCO+ | `benchmarks/adapters/visual_grounding_refcoco.py` |
| `counting` | TallyQA | `benchmarks/adapters/counting_tallyqa.py` |
| `ocr_text_recognition` | TextVQA | `benchmarks/adapters/ocr_text_recognition_textvqa.py` |
| `spatial_reasoning` | GQA-spatial | `benchmarks/adapters/spatial_reasoning_gqa.py` |
| `relational_reasoning` | GQA-relational | `benchmarks/adapters/relational_reasoning_gqa.py` |

Excluded: attribute_recognition (Visual Genome), fine_grained_recognition (CUB), every other
capability in the Capability Benchmark Gate. All five reuse their already-validated adapters,
prompts, subsets, parsers, normalizers, metrics, and deterministic (temperature=0) decoding
unmodified.

## Scopes (exactly three)

`vision_encoder`, `full_lm`, `full_vlm` — `scopes.py`'s existing canonical scope names.
Parameter membership is never redefined here; `scopes.build_scope_manifest` is the only source
of which parameters belong to a scope.

## Radii (exactly two)

`0.02`, `0.04` — relative-L2 norm matching via `scopes.compute_relative_l2_sigma` (unchanged
formula: `sigma_m = r * ||theta_m||_2 / sqrt(d_m)`).

## Perturbations

- 100 perturbation seeds per scope-radius cell, drawn via `candidate_sampling.sample_candidate_seeds`
  (the same seed-draw convention as `run_randopt_image_aware.py`'s own `sample_candidates`),
  namespaced per (scope, radius) via `thicket.seeds.derive_seed` from base seed
  `CANDIDATE_SEED_BASE = 20261005`.
- The SAME ordered 100-seed sequence for a given (scope, radius) cell is shared across all five
  capabilities — one perturbed weight state per candidate is evaluated against every capability
  before restoring, never re-perturbed per capability.
- 6 scope-radius cells × 100 seeds = 600 unique perturbations total.
- Failed candidates (restoration/norm/isolation/provenance/integrity failure) are recorded, not
  replaced — the population is fixed at build time, before any result exists.

## Frozen evaluation sets

- Selection set: 200 examples. Causal audit set: 200 disjoint examples. Built via
  `iclr_causal_density.subsets.build_selection_and_audit_subsets`: one seeded shuffle
  (`SUBSET_SELECTION_SEED = 20261005`) of each capability's full candidate pool; indices
  `[0:200)` → selection, `[200:400)` → audit — disjoint by construction, never two
  independently-sampled subsets that could coincide.
- If fewer than 400 valid examples exist for any of the five capabilities: stop with
  `INCONCLUSIVE` for that capability (task spec) — evaluated in Phase 0's artifact audit before
  any GPU execution is attempted.
- Frozen once; never regenerated or reshuffled after this document is written.

## Visual conditions (exactly three)

`correct_image`, `shuffled_image`, `text_only` — evaluated for every base-model row and every
candidate row.

- **Shuffled image**: one deterministic within-capability derangement (`SHUFFLE_SEED =
  20261006`), reusing `benchmarks.image_sanity.make_shuffled_variant` unmodified (true
  derangement, no self-maps, delegates the per-example visual-input swap to the owning
  adapter's `make_shuffled_image_variant`). Built ONCE per (capability, subset) and reused for
  every one of the 600 candidates — never reshuffled between candidates. Original and shuffled
  `image_ref` are persisted for every example; a mismatch check (`image_ref` must differ from
  the original) is enforced at manifest-build time, hard-failing rather than silently scoring a
  non-shuffled "shuffled" condition. Scored against the ORIGINAL example's target.
- **Text-only**: the already-validated image-stripping path (`benchmarks.image_sanity.make_text_only_variant`
  + `run_benchmark(..., allow_missing_image=True)`) — no image tensor, no image tokens, no
  cached visual embedding, no stale image metadata; textual prompt preserved exactly.

## Search-budget analysis (Phase 8)

N ∈ {10, 25, 50, 100}, 1,000 deterministic Monte Carlo subsamples per (capability, scope,
radius) cell, drawn from that cell's 100-candidate pool, using the ONE preregistered
`SEARCH_BUDGET_ANALYSIS_SEED = 20261007`. Top-10 is a selected population, never an
ensemble/voting method. Registered divergence (comparing the smallest vs. largest budget,
never a mid-range comparison chosen after seeing data): audit real-image gain increases, G
decreases or fails to increase proportionally, and the top-10 shortcut-expert fraction
increases.

## Grounded selection (Phase 9)

Standard ranking: `R_i^standard = S_i^real` (selection-set correct-image aggregate score).
Grounded ranking: `R_i^grounded = Δ_i^R − (1/2)(Δ_i^T + Δ_i^S)`, all computed on the SELECTION
set only; eligible only when the selection-set `Δ_i^R > 0`. The coefficient `1/2` is frozen and
never tuned. Evaluation uses the audit set only.

## Bootstrap method (Phase 7)

One shared paired-bootstrap resample matrix of audit-set example indices (10,000 resamples,
seed `BOOTSTRAP_ANALYSIS_SEED = 20261008`, with replacement). Per candidate, `CI_low^95%(G_i)`
is the 2.5th percentile of that candidate's resampled `G_i` distribution from the shared
matrix — this fixes each candidate's causally-visual-expert classification from the observed
data. The population-level `D`'s own 95% CI reuses the same matrix: for each resample, every
candidate's *that-resample's-own* `Δ_i^R(b) > 0` and `G_i^(b) > 0` serve as a per-resample
plug-in classification (never a nested bootstrap-of-bootstrap), giving a `D^(b)` per resample
and hence a 95% CI for `D` from that distribution. This convention is fixed here, before any
results exist.

## Decision-gate thresholds (Phase 10 — see `decision_gate.py` for the code that applies them)

- **CONFIRMED** requires all: `D ≥ 2` in ≥4/5 capabilities; `D`'s 95% CI excludes 1 in ≥3/5
  capabilities; the registered search-budget divergence holds in ≥4/5 capabilities; grounded
  selection retains ≥80% of standard selection's positive audit real-image gain in *every*
  capability where standard's gain is positive; grounded selection materially (strictly)
  improves audit `G` (top-10 pool mean) in ≥4/5 capabilities; every integrity/restoration/
  isolation/provenance/completeness gate passes.
- **REJECTED**: a complete, valid experiment with adequate precision (≥4/5 capabilities carry a
  defined `D`, i.e. `ρ_visual > 0`) that fails one or more CONFIRMED criteria.
- **INCONCLUSIVE**: integrity/controls fail, execution is blocked, required cells are
  incomplete, fewer than 4/5 capabilities carry a defined `D`, or a required capability cannot
  be validly evaluated. Never converted into confirmation through pooling.

No threshold in this document is altered after any result is observed.

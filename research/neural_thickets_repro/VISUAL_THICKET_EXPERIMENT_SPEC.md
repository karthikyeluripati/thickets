# Visual Thicket Experiment Specification (Stage 5)

## Where Do Visual Experts Live? Mapping Neural Thickets in Vision-Language Models

This document is the frozen experimental protocol every future GPU perturbation experiment on
this branch follows. It is CPU-only infrastructure: nothing in this document or the code it
describes runs GPU inference, perturbs real model weights, or executes a RandOpt/perturbation
sweep. Treat it as **frozen unless a serious methodological issue is later discovered** — the
same discipline `REPRO_SPEC.md` and `SCOPED_PERTURBATION_DESIGN.md` already use for this
branch's other specs (a decision table with Confirmed / Resolved-by-assumption / UNRESOLVED
status, updated in place rather than silently overwritten).

Deliberately separate from:
- `CAPABILITY_BENCHMARK_GATE.md` / `src/neural_thickets_repro/benchmarks/` — the measurement
  -instrument validation gate (8 capability adapters, zero-perturbation base-model evaluation).
  This spec assumes that gate's capability probes exist and are trustworthy; it does not modify
  or re-validate them.
- `REPRO_SPEC.md` / `SCOPED_PERTURBATION_DESIGN.md` and the code they govern
  (`scopes.py`, `scoped_perturbation.py`, `thicket_metrics.py`, `ledger.py`, `topk_voting.py`,
  `candidate_sampling.py`) — the frozen, GQA-only Gate-1/2 reproduction path and historical
  RandOpt compatibility. This spec's code (`src/neural_thickets_repro/thicket/`) **generalizes**
  ideas first proven there (scoped relative-L2 perturbation, expert-density statistics via
  Wilson confidence intervals) to an architecture-scale-independent, multi-capability setting,
  reusing several of their already-validated pieces directly (see section-by-section notes
  below) rather than duplicating or silently diverging from them.

The central object of study is not merely `P(nearby model improves)` but

```
P( nearby model improves | visual capability, model anatomy, perturbation radius, model scale )
```

characterizing: (1) solution density, (2) performance-density distributions, (3) anatomical
localization, (4) expert diversity/specialization, (5) perturbation-radius dependence, (6)
scaling with model size, (7) low-dimensional expert geometry, (8) whether anatomical knowledge
can later improve post-training search, (9) whether discovered experts can later be
distilled/consolidated. This document and its code build the common experimental foundation for
those claims — no claim itself is made or tested here.

---

## A. Base model, performance change, and the scientific quantities of interest

**Code:** `src/neural_thickets_repro/thicket/metrics.py`

Let `theta_0` be the pretrained VLM's parameters. For visual capability `t`, `S_t(theta)` is its
evaluation score (from the Capability Benchmark Gate's own scoring, unmodified). For a
perturbation `epsilon`:

```
Delta_t(epsilon) = S_t(theta_0 + epsilon) - S_t(theta_0)
```

implemented as `metrics.performance_delta(perturbed_score, base_score)`. Every sampled
perturbation retains its complete capability score vector (spec D1's shared-population design
guarantees this — see section D below) rather than being reduced to one capability's number.

### A2. Performance density

The empirical distribution `p_{t,a,r,s}(Delta)` (t=capability, a=anatomical region, r=radius,
s=model scale) is represented by the **raw stored per-perturbation Delta values themselves** —
never immediately collapsed to a mean. `metrics.empirical_density()` provides an optional
histogram view purely for later visualization; it is not the primary representation.

### A3. Solution density

```
delta_{t,a,r,s}(m) = P[Delta_t >= m]
```

`metrics.solution_density(deltas, margins)` accepts an arbitrary list of margins — `m=0` is
never hardcoded as the only supported value. `metrics.solution_density_confidence_interval()`
reuses `thicket_metrics.wilson_confidence_interval` (already implemented and tested for Gate-1
expert-density statistics) directly, since `delta(m)` is exactly a binomial proportion of the
sample.

### A4. Positive thicket mass

```
M_{t,a,r,s} = E[max(Delta_t, 0)]
```

`metrics.positive_thicket_mass(deltas)`. The relationship `M = integral_0^inf delta(m) dm` is
not just documented — `tests/test_thicket_metrics.py::test_positive_thicket_mass_equals_
integral_of_solution_density` verifies it numerically (fine-grid trapezoidal integration of
`solution_density()` against the direct `positive_thicket_mass()` computation, on a synthetic
2000-perturbation population). `M` distinguishes a neighborhood with many tiny improvements
(high `delta(m)` near `m=0`, low `M`) from one with fewer, stronger experts (lower `delta(0)`,
potentially higher `M` if the few improvements are large).

### A5. Candidate vs. confirmed expert

- **Candidate expert**: `metrics.is_candidate_expert(delta)` — strictly `delta > 0` on
  mapping/selection data (D_map/D_select — see section F). A tie (`delta == 0`) never counts.
- **Confirmed expert**: `metrics.confirm_expert(held_out_deltas, ci_lower_bound_threshold=0.0,
  ...)` — the SAME candidate's Delta values on held-out D_confirm data, tested via a paired
  bootstrap confidence interval (`metrics.paired_bootstrap_confidence_interval`, percentile
  bootstrap, deterministic given a seed) whose lower bound must exceed
  `ci_lower_bound_threshold`. **No arbitrary statistical threshold is frozen at this stage**:
  the default `0.0` is the natural "still an improvement after accounting for sampling
  uncertainty" boundary, not a tuned constant, and every argument (threshold, confidence,
  bootstrap count, seed) is caller-supplied. `tests/test_thicket_metrics.py` explicitly verifies
  the threshold is not hardcoded (`test_confirm_expert_threshold_is_not_hardcoded_by_the_
  library`).

---

## B. Model anatomy registry

**Code:** `src/neural_thickets_repro/thicket/anatomy.py`, `src/neural_thickets_repro/
inspect_model_anatomy.py`

### B1. Hierarchy

| Level | Regions |
|---|---|
| 0 | `full_model` |
| 1 | `vision`, `multimodal_connector_or_merger`, `language` |
| 2 (vision) | `vision_early`, `vision_middle`, `vision_late` |
| 2 (language) | `language_early`, `language_middle`, `language_late` |
| 3 (structural only, not executed) | `attention`, `mlp` per name (`anatomy.classify_attention_or_mlp`) |

Level-1 regions partition `full_model` exactly (verified by `validate_atlas`, tested in
`test_level1_regions_partition_full_model_exactly`). Level-2 vision/language bands partition
their own Level-1 parent's block/layer-indexed parameters, but generally **do not** cover every
parameter in that parent (e.g. `language`'s own embeddings/final-norm/lm_head sit outside any
numbered layer) — this is expected and reported (`validate_atlas`'s `uncovered_by_parent`), not
treated as a defect. Level 3 is a callable classifier only; it is not built into the default
atlas and no Level-3 sweep is executed in Stage 5, per the explicit non-negotiable-scope
instruction not to run individual layer/head sweeps yet.

### B2. Depth-band definition (exact, documented rule — no hardcoded layer counts)

For `n` contiguous discovered indices `0..n-1`: let `base = n // 3`, `remainder = n % 3`. The
first `remainder` bands (in early, middle, late order) get `base + 1` indices each; the rest get
`base`. Implemented in `anatomy.partition_into_thirds`. This generalizes, rather than
hardcodes, the two special cases `scopes.py` already committed to for the fixed 3B model:
- `n=32` (vision blocks) → 11/11/10, exactly matching `scopes.py`'s existing hardcoded
  `vision_early/middle/late` boundaries (`test_partition_into_thirds_matches_established_32_
  block_11_11_10_convention` checks this is not a coincidence).
- `n` divisible by 3 (e.g. the 3B model's 36 LM layers) → exact equal thirds, matching
  `scopes.partition_layers_into_thirds`'s existing behavior for that case.

Unlike `scopes.py`'s `partition_layers_into_thirds` (hard-requires `n % 3 == 0`) and
`partition_vision_blocks` (hard-requires exactly 32), `anatomy.partition_into_thirds` works for
**any** `n >= 3` — required because a 7B/72B ladder member is not guaranteed to have a layer/
block count divisible by 3 or equal to 32. `n < 3` still hard-fails (three non-empty contiguous
bands are impossible), never silently returning fewer/empty bands.

Block/layer **counts** are always *discovered* from the real parameter names handed to
`anatomy.build_anatomy_atlas`, never hardcoded — it reuses `scopes.py`'s already-validated,
multi-convention LM-layer discovery (`scopes.discover_lm_layer_indices`, which itself calls
`scopes.detect_lm_namespace_convention` against the registered `LM_NAMESPACE_CONVENTIONS`) for
the language side, and a fixed (but count-agnostic) `visual\.blocks\.(\d+)\.` pattern plus the
new generic `anatomy.discover_contiguous_block_indices` for the vision side.

### B3. Parameter inventory: `inspect_model_anatomy.py`

`inspect_model_anatomy(named_parameters, model_name, model_revision, model_scale)` reports,
per region: level, parent, tensor count, element (numel) count, percentage of total elements,
dtype(s) present, `||theta_a||_2`, mask hash, and a handful of representative parameter names.
Output is plain JSON (`json.dumps`-safe, tested). Two usage modes:
1. `inspect_model_anatomy(...)` itself — pure, CPU-testable, works against any `(name, tensor)`
   iterable; unit tests (`tests/test_inspect_model_anatomy.py`) exercise it against the existing
   synthetic 32-vision-block / 12-LM-layer dummy VLM fixture (`runtime_wrapped_vlm_32vision_
   factory`), never a real download.
2. `main()` — a thin CLI that lazily imports `torch`/`transformers` **only inside the function
   body** (matching this project's established convention of keeping heavy/GPU imports out of
   module scope), for real pod-side use against an actual checkpoint. Not exercised by any
   Stage-5 test, and not run in Stage 5.

### B4. Mask validation

`anatomy.validate_atlas(atlas)`:
- **deterministic**: pure function of the atlas's own (already-deterministic) region data, no
  RNG anywhere in the discovery/partition/validation path;
- **fails on unexpectedly empty regions**: raises `AnatomyValidationError` for any region with
  zero parameters (an explicit `allow_empty` override exists for genuinely optional regions,
  unused by the default atlas);
- **reports sibling overlap**: pairwise intersection of every pair of same-parent regions —
  expected to be empty for a correctly-built atlas, and treated as a real bug (raises) if not,
  unlike uncovered-parameter reporting below;
- **reports uncovered parameters**: parent-region parameters not covered by the union of its
  own children, reported (not raised) per parent;
- **exposes a stable mask hash**: `AnatomyRegion.mask_hash`, a sha256 of the region's sorted
  parameter-name tuple — order-independent, membership-sensitive (tested).

---

## C. Two perturbation modes

**Code:** `src/neural_thickets_repro/thicket/perturbation.py`

### C1. `global_gaussian_upstream`

A **thin, non-modifying wrapper** around the existing `perturb_cpu.perturb`/`perturb_cpu.restore`
(theta' = theta + sigma*epsilon, epsilon ~ N(0, I), per-tensor reseed by `seed`, skipping
`visual.*`/`model.visual.*`). `perturbation.apply_global_gaussian_upstream`/`undo_global_
gaussian_upstream` call these functions directly and unchanged; nothing about the historical
reproduction path (Gate 1/2, or `scoped_perturbation.py`'s own `raw_sigma` scale mode) is
rewritten. This mode exists purely so results stay comparable to Neural Thickets / RandOpt.

### C2. `anatomical_relative_l2`

For anatomical region `a`:

```
r = ||epsilon_a||_2 / ||theta_a||_2
```

**This is scientifically distinct from `scoped_perturbation.py`'s existing `relative_l2` scale
mode**, and the distinction is deliberate, not an oversight:
- `scoped_perturbation.py` (via `scopes.compute_relative_l2_sigma`) derives a single scalar
  `sigma_m = r * ||theta_m||_2 / sqrt(d_m)` such that `E[||epsilon_m||_2] ≈ r * ||theta_m||_2`
  **in expectation only** — the realized norm of any one finite-dimensional sample is not
  exactly `r * ||theta_m||_2`.
- `thicket.perturbation.apply_anatomical_relative_l2` instead: (1) samples independent Gaussian
  noise over only the region's own parameters (identical per-tensor-reseed convention, via the
  same `perturb_cpu._generate_noise`); (2) measures the realized combined L2 norm of that raw
  sampled noise; (3) rescales by a single exact scalar factor so the **applied** perturbation's
  L2 norm equals `r * ||theta_a||_2` up to floating-point rounding, not merely in expectation;
  (4) never even reads parameters outside the region.

Spec section C3 requires numerically verifying the realized ratio to tolerance — an
expectation-only scalar cannot guarantee this for any single sample, which is exactly why C2 is
implemented via exact post-hoc rescaling rather than reusing `compute_relative_l2_sigma`
directly. `scoped_perturbation.py` itself is untouched; both modes now coexist as documented,
separate scientific choices (`PERTURBATION_MODES = ("global_gaussian_upstream",
"anatomical_relative_l2")`).

### C3. Numerical validation (tests/test_thicket_perturbation.py)

- `test_anatomical_relative_l2_hits_requested_ratio_exactly`: realized
  `||epsilon_a||_2 / ||theta_a||_2` matches the requested `r` to `1e-6`.
- `test_anatomical_relative_l2_outside_region_is_exactly_unchanged`: every parameter outside
  the region is bit-identical before/after (`torch.equal`, not merely close).
- `test_anatomical_relative_l2_different_regions_get_different_scale`: confirms two
  differently-sized regions asked for the same `r` generally receive different scale factors —
  the central "must not use identical per-weight sigma across regions of different
  dimensionality" requirement.
- Determinism / distinctness: same-seed reproducibility and different-seed divergence, both
  tested directly.

---

## D. Perturbation identity

**Code:** `thicket/perturbation.py` (`PerturbationManifest`, `compute_perturbation_id`)

Every perturbation carries an immutable manifest: `seed`, `perturbation_mode`, `anatomy_region`,
`radius`, `sigma`, `model_family`, `model_scale`, `model_revision`, `parameter_mask_hash`. The
`perturbation_id` is `sha256(canonical_json(all_fields_including_seed))[:24]` — the SAME fields
(including seed) always produce the SAME id; changing the seed alone always changes the id
(tested). `PerturbationManifest` validates `perturbation_mode` against the closed
`PERTURBATION_MODES` registry at construction time.

### D1. Shared population across capabilities

`perturbation.generate_perturbation_population(mode, n, base_seed, anatomy_region, radius, ...,
model_family, model_scale, model_revision, parameter_mask_hash)` derives each member's seed via
`thicket.seeds.derive_seed(base_seed, "perturbation_population", mode, region, radius, sigma,
i)` — a **pure function of the cell's own identifying fields**, so calling it twice for the
identical `(mode, region, radius-or-sigma)` cell always returns the identical population (same
seeds, same perturbation_ids; tested). A future GPU evaluation driver that calls this once per
cell and then evaluates every capability against each returned manifest, in order, gets
perturbation `i` aligned across `Delta_grounding(i), Delta_counting(i), ...` automatically —
never an independently-resampled population per task, satisfying spec D1's diversity-analysis
requirement without any additional bookkeeping.

---

## E. Radius protocol

No giant final radius sweep is frozen here. `configs/visual_thicket_experiment.yaml` explicitly
separates `radii.calibration` (used to discover the three qualitative regimes: near-zero/
base-like, useful-thicket, destructive) from `radii.final_paper` (`null` — **UNRESOLVED**, and
must never be chosen by maximizing downstream benchmark results). Historical upstream sigma
values remain available and listed separately under the `global_gaussian_upstream` perturbation
-mode entry, for direct comparison against the historical reproduction path.

---

## F. Data role separation

**Code:** `src/neural_thickets_repro/thicket/data_roles.py`

Four disjoint roles — `map`, `confirm`, `select`, `test` — over plain example-ID strings (never
model outputs): `partition_data_roles(example_ids, sizes, seed)` performs a deterministic
seeded shuffle (`random.Random(seed)`) followed by contiguous, non-overlapping slicing in
`map, confirm, select, test` order. Disjointness is guaranteed by construction (slices of one
permutation) and independently re-verified by `validate_disjoint()` before every partition is
returned or reloaded (defense in depth, not merely an assumption). Overlap or a duplicate-ID
pool hard-fails (`DataRoleOverlapError`/`DataRoleError`) rather than silently proceeding.
Manifests persist to JSON with a `manifest_hash` (sha256 of the sorted role→ID mapping) and can
be reloaded; `validate_against_pool()` hard-fails (`DataRoleDriftError`) if a persisted ID is
missing from a freshly-loaded pool — the identical dataset-drift-guard discipline
`benchmarks/subset_selection.py` already established for the Capability Benchmark Gate.

---

## G. Visual thicket metrics

**Code:** `src/neural_thickets_repro/thicket/metrics.py` (see also section A above)

| Statistic | Function | Notes |
|---|---|---|
| Performance delta | `performance_delta` | `S_t(theta_0+epsilon) - S_t(theta_0)` |
| Performance density | `empirical_density` | Histogram view; raw deltas are the real object |
| Solution density | `solution_density(deltas, margins)` | Arbitrary margin grid, Wilson CI via `solution_density_confidence_interval` |
| Positive thicket mass | `positive_thicket_mass` | `E[max(Delta,0)]`, numerically tied to solution density's integral |
| P(improvement) / P(degradation) | `probability_of_improvement` / `probability_of_degradation` | Strict `>`/`<` |
| Catastrophic degradation rate | `catastrophic_degradation_rate(deltas, c)` | `c` is REQUIRED, never defaulted |
| Best-of-N | `best_of_n_single_order` / `best_of_n_expected` | Single realized order vs. permutation-averaged expectation |
| Quantiles | `quantiles(deltas, qs=(.5,.75,.9,.95,.99))` | Arbitrary `qs` |
| Mean/std | `mean_std` | — |
| Paired/bootstrap CI | `paired_bootstrap_confidence_interval` | Percentile bootstrap, deterministic given seed; covers non-proportion statistics Wilson's interval does not |

None of these functions load a model or require GPU — all operate on plain sequences of floats.

---

## H. Diversity / specialization

**Code:** `src/neural_thickets_repro/thicket/diversity.py`

Input throughout: a perturbation x capability Delta matrix (rows = perturbation IDs, aligned
across columns per section D1; columns = capability deltas).

### H1. Task rank correlation

`diversity.task_rank_correlation_matrix(delta_matrix)` — the full task x task matrix. Computed
as the Pearson correlation of the **percentile-rank matrix** (`percentile_rank_matrix`), which
is mathematically identical to Spearman rank correlation of the raw deltas — implemented via
plain `numpy.corrcoef` (no scipy/pandas dependency; neither is in `requirements-cpu.txt`).

### H2. Spectral Discordance

`external/RandOpt` is **not checked out in this repository** (cloned dynamically at a pinned
commit on the pod, per `external/setup_external_repo.py`; has no declared license, per
`REPRO_SPEC.md`'s existing discipline). It could not be inspected directly for this milestone.
Instead, the exact definition below was retrieved from the **published paper** this project
reproduces (arXiv:2603.12228, Definition 2.2 — public mathematics, not upstream source code,
and consistent with this project's existing "reuse public formulas, never transcribe
unlicensed code" convention already used by `vqa_soft_accuracy.py`):

```
D = 1 - (1 / (M*(M-1))) * sum_{j != k} C_jk
```

where `P in [0,1]^(N x M)` is the percentile-rank matrix (N perturbations, M tasks) and
`C = corr(P)` is the M x M Pearson correlation matrix of its columns (exactly
`task_rank_correlation_matrix`'s own output — one computation serves both H1 and H2).
`D -> 1` implies orthogonal task rankings (specialists); `D -> 0` implies parallel rankings
(generalists); the paper reports `D` bounded in `[0, M/(M-1)]`.

`diversity.spectral_discordance(delta_matrix)` implements this directly.
`tests/test_thicket_diversity.py` checks both bounds against known fixtures: perfectly
correlated tasks give `D ≈ 0`, and perfectly anti-correlated `M=2` tasks give `D ≈ 2 =
M/(M-1)`, the paper's own reported upper-bound case.

> **UNRESOLVED**: this definition is ported from the paper, not verified against the actual
> upstream RandOpt *implementation* (never inspected directly here). If the pod-side clone of
> `external/RandOpt` implements a materially different rank-normalization convention (e.g. a
> different percentile-rank tie-break, or Kendall's tau instead of Pearson-of-ranks), that
> difference must be reconciled and this row updated **before** Spectral Discordance is treated
> as frozen for the paper's Figure 4.

### H3. Expert overlap

`diversity.expert_overlap_matrix(delta_matrix, q, q_is_fraction)` — pairwise Jaccard overlap
(`diversity.jaccard`) of each pair of tasks' top-`q` expert-index sets (`diversity.
top_q_indices`), `q` as either a fraction of N or an absolute top-K. Diagonal is always 1.0.

### H4. Cross-capability transfer

`diversity.cross_capability_transfer_matrix(delta_matrix, q, q_is_fraction)`:

```
T[t, u] = mean(Delta_u | perturbation selected as a top-q expert for capability t)
```

A genuinely **directional** M x M matrix (`T[t,u]` need not equal `T[u,t]`; tested against
random continuous data, where a symmetric coincidence is vanishingly unlikely).

### H5. Capability signature

`diversity.CapabilitySignatureMatrix(perturbation_ids, task_names, matrix)` — every
perturbation's complete `v_i = [Delta_1, ..., Delta_T]` vector, labeled for later clustering/
PCA/expert-family analysis. No visualization is implemented in Stage 5.

---

## I. Low-rank expert geometry — interface/design only

**Code:** `src/neural_thickets_repro/thicket/geometry.py`

No large SVD is run, and no billion-dimensional perturbation vector is ever concatenated into
RAM, anywhere in Stage 5. **The scalable design decision**: a perturbation's full weight-space
delta is never persisted — it is exactly reconstructible on demand from its (small)
`PerturbationManifest` via the identical per-tensor-reseed noise generation already used to
apply it (`perturbation.apply_anatomical_relative_l2`; "same manifest+seed reproduces the same
perturbation" is tested directly, not merely assumed). Later low-rank analysis (effective rank,
singular spectrum, split-half subspace reproducibility, principal angles between
capability-specific expert subspaces, low-rank consolidation) therefore only needs to persist
`PerturbationVectorHandle`-shaped records (manifest + per-capability scalar deltas — a few
hundred bytes each) and can regenerate/stream the actual high-dimensional noise chunk-by-chunk
(e.g. one transformer layer at a time) directly on a GPU worker at analysis time, accumulating a
randomized/streaming SVD over layer-chunks rather than ever materializing an
`[n_perturbations x n_parameters]` matrix. That streaming SVD driver is explicitly **Stage 6+
GPU work**, out of scope here.

`geometry.py` implements only the small, generic linear-algebra primitives that operate on
already-computed, small subspace bases or singular-value arrays: `effective_rank` (Roy &
Vetterli's entropy-based definition, `exp(H(p))` over the normalized spectrum) and
`principal_angles` (via QR-orthonormalization + SVD of `Q_a^T Q_b`), both tested against known
closed-form cases (a single singular value → rank 1; `k` equal singular values → rank `k`;
identical subspaces → zero angles; orthogonal subspaces → `pi/2` angles).

---

## J. Scale dimension

Every manifest field-set (`PerturbationManifest`, `ExperimentResultRecord`,
`configs/visual_thicket_experiment.yaml`'s `models:` list) explicitly carries `model_family`,
`model_scale`, `model_revision`, alongside architecture metadata discovered (never hardcoded)
by `thicket.anatomy`. The primary intended same-family ladder is
`Qwen2.5-VL-{3B,7B,72B}-Instruct`; only the 3B entry is currently accessible (`status:
accessible` in the pilot manifest; 7B/72B are listed with `status: not_yet_run`). No code path
in `thicket/` assumes 32 vision blocks or 36 LM layers, or any other 3B-specific constant — see
section B2's generalized `partition_into_thirds`. **No scaling-law claim is made anywhere in
this specification or its code.**

---

## K. Experiment size estimator

**Code:** `src/neural_thickets_repro/thicket/experiment_size.py`

Purely arithmetic, **not** a dollar-cost or throughput estimator (tested:
`test_report_has_no_dollar_cost_field`). `ExperimentSizeInputs` takes `n_models,
n_capabilities, n_anatomy_regions, n_radii, n_perturbations_per_condition,
n_examples_per_capability, n_repeats=1, n_sanity_runs=0, ensemble_k=1`.
`estimate_experiment_size` reports:

- `unique_candidate_models = n_models * n_anatomy_regions * n_radii * n_perturbations_per_condition`
- `conditions = n_capabilities * n_anatomy_regions * n_radii`
- `total_model_example_evaluations = unique_candidate_models * n_capabilities * n_examples_per_capability * n_repeats * ensemble_k`
- `evaluations_per_capability` / `evaluations_per_anatomy` / `evaluations_per_radius` (the total, divided evenly)
- `baseline_evaluations = n_models * n_capabilities * n_examples_per_capability` (one full baseline sweep per model)
- `sanity_evaluations = n_sanity_runs * n_models * n_capabilities * n_examples_per_capability`
- `multiplier_vs_one_baseline = total_model_example_evaluations / baseline_evaluations`

**Worked example (item 10 of the Stage-5 task spec)**: `models=1, capabilities=3, anatomy=3,
radii=3, perturbations_per_condition=64, examples_per_capability=50`:

| Quantity | Value |
|---|---|
| unique_candidate_models | 576 |
| total_model_example_evaluations | 86,400 |
| evaluations_per_capability / anatomy / radius | 28,800 each |
| baseline_evaluations | 150 |
| multiplier_vs_one_baseline | 576.0 |

(`tests/test_thicket_experiment_size.py::test_spec_worked_example_matches_item_10` pins this
exact arithmetic.) Note `multiplier_vs_one_baseline` reduces to `unique_candidate_models /
n_models` in general, and only equals `unique_candidate_models` directly when `n_models == 1`
(as in this worked example) — both `unique_candidate_models` and `baseline_evaluations` scale
with `n_models`, since each model gets its own candidate sweep AND its own baseline sweep.

---

## L. Versioned experiment manifest

**File:** `configs/visual_thicket_experiment.yaml`, explicitly labeled `"PILOT / NOT
PAPER-FINAL"` throughout. Sections: `experiment`, `models`, `capabilities`, `anatomy`,
`perturbation_modes`, `radii`, `population`, `seeds`, `data_roles`, `metrics`, `outputs` — no
giant final sweep is populated; `n_perturbations_per_condition: 8` and
`n_examples_per_capability: 50` are deliberately small pilot-scale values, and `radii.
final_paper` is explicitly `null`/UNRESOLVED.

---

## M. Current core capability probes

The currently frozen, usable capability probes (do not modify their adapters as part of this
specification or its code):

```
visual_grounding, counting, spatial_reasoning, ocr_text_recognition_grounded,
relational_reasoning, fine_grained_recognition
```

ImageNet object recognition is pending gated HF access and may later join. **Visual Genome
attribute recognition is explicitly NOT part of the current core suite** (see
`CAPABILITY_BENCHMARK_GATE.md`'s own N=50 visual-dependence repair-pass notes for why it needed
a separate localized-crop protocol repair before it could be trusted at all).

---

## N. Paper figure mapping

| Figure | Requires | Question |
|---|---|---|
| 2 — Visual thickets exist | Performance-density distributions, solution-density curves, radius dependence, best-of-N behavior (`metrics.py`) | Do multiple visual capabilities exhibit dense nearby expert populations? |
| 3 — Where do visual experts live? (HERO result) | Capability x anatomy x radius atlas (`anatomy.py` + `metrics.py` + the experiment manifest's own grid) | How does expert density/strength vary across vision, connector, language, and depth? |
| 4 — Are visual experts different? | Spectral Discordance, task-rank correlations, expert overlap, cross-capability transfer, capability signatures (`diversity.py`) | Are perturbations that help one capability specialists or generalists across others? |
| 5 — Geometry and scale | Low-rank expert subspaces, principal angles, 3B/7B/72B comparison (`geometry.py` interface + section J's scale metadata) | What is the effective dimensionality of expert regions, and how does it scale? |
| 6 (later) — Controlled thicket emergence | Not implemented in Stage 5 | — |
| 7 (later) — Atlas-guided search/post-training | Not implemented in Stage 5 | — |
| 8 (later) — Distillation/consolidation | Not implemented in Stage 5 | — |

Stage 5 implements the machinery for Figures 2–5 (largely via CPU-testable statistics and
interfaces) and explicitly does not implement Figures 6–8's experiments; today's output schema
(section O) is designed so those later figures can be built on the same records.

---

## O. Experiment output schema

**Code:** `src/neural_thickets_repro/thicket/schema.py` (`ExperimentResultRecord`)

One record per evaluated (perturbation, capability) pair: `experiment_id`, `perturbation_id`,
`model_family`, `model_scale`, `model_revision`, `perturbation_mode`, `anatomy_region`,
`radius`, `sigma`, `seed`, `parameter_mask_hash`, `capability`, `dataset_role`, `subset_hash`,
`base_score`, `perturbed_score`, `delta`, `parser_failure_rate`, `per_example_result_path`,
`per_example_result_hash`, `runtime_metadata`. `delta` is validated at construction time to
equal `perturbed_score - base_score` (raises otherwise). Per-example predictions are referenced
by path/hash, never embedded, to avoid duplicating giant payloads across records.

---

## P. Reproducibility

**Code:** `src/neural_thickets_repro/thicket/seeds.py`

`derive_seed(base_seed, *namespace_parts)` — a sha256-based deterministic derivation, so
perturbation-sampling, subset/data-role construction, and bootstrap-analysis seed streams stay
independent of each other (different namespace parts) while all being reproducible from one
root/global seed. No implicit process randomness is relied on anywhere in `thicket/`. Every
manifest (`PerturbationManifest`, `DataRolePartition`, the experiment YAML itself) persists its
own hash (`parameter_mask_hash`, `manifest_hash`, mask hashes) so drift is always detectable.

---

## Q. Tests

CPU-only, `pytest tests/ -q`, zero GPU/ray/vllm/real-dataset access required. New/changed files
this stage: `tests/test_thicket_seeds.py`, `tests/test_thicket_anatomy.py`, `tests/
test_thicket_perturbation.py`, `tests/test_thicket_data_roles.py`, `tests/test_thicket_
metrics.py`, `tests/test_thicket_diversity.py`, `tests/test_thicket_experiment_size.py`, `tests/
test_thicket_geometry.py`, `tests/test_thicket_schema.py`, `tests/test_inspect_model_
anatomy.py`. Every category from the task spec's section Q is covered — see each section above
for the specific test names.

---

## R. Relationship to existing code (explicit reuse / non-modification ledger)

| Existing module | Touched? | Relationship |
|---|---|---|
| `perturb_cpu.py` | No | `thicket.perturbation`'s `global_gaussian_upstream` mode calls its `perturb`/`restore`/`_generate_noise` directly, unmodified |
| `scopes.py` | No | `thicket.anatomy` reuses `LM_NAMESPACE_CONVENTIONS`, `detect_lm_namespace_convention` (via `discover_lm_layer_indices`), `VISUAL_MERGER_PREFIXES` directly; does not import or depend on `PERTURBATION_SCOPES`, `compute_relative_l2_sigma`, or any hardcoded count |
| `scoped_perturbation.py` | No | Untouched; `anatomical_relative_l2` is a deliberately distinct (exact-rescale, not expectation-based) scientific choice, not a replacement |
| `thicket_metrics.py` | No | `thicket.metrics` reuses `wilson_confidence_interval` directly for proportion CIs |
| `ledger.py`, `topk_voting.py`, `candidate_sampling.py` | No | Not needed by any Stage-5 CPU code path |
| `benchmarks/` (Capability Benchmark Gate) | No | Assumed as a trustworthy, external capability-scoring oracle; not modified, not re-validated |
| `external/RandOpt` | Not checked out / not modified | Read indirectly via the published paper for Spectral Discordance's definition only (see section H2's UNRESOLVED note); no code inspected or transcribed |

`external/RandOpt` is not present in this checkout (cloned dynamically at a pinned commit on the
pod, per `external/setup_external_repo.py`) — this is stated plainly rather than assumed
available, since it directly affects section H2's confidence level.

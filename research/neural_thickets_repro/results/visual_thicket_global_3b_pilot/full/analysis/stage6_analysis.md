# Stage 6 Analysis: 3B Global Visual-Thicket Pilot (full run)

Source: `results/visual_thicket_global_3b_pilot/full/results.jsonl` (1152 rows, 384 unique perturbations, restoration_mode=fixed_base, perturbation_semantics=global_gaussian_upstream). Analysis only -- no model run, no perturbation applied, no existing result altered.

## A) Scope of this experiment

Stage 6 uses upstream-compatible **non-visual/language-side** Gaussian perturbations (`global_gaussian_upstream`: every parameter NOT prefixed `visual.`/`model.visual.` is perturbed; the vision encoder is frozen). **It is NOT the anatomical whole-VLM experiment** -- no anatomical region localization (vision encoder, connector, language depth bands) has been tested yet; that is Stage 7+.

## Baseline scores and headroom

| capability | baseline_score | headroom (1 - baseline) |
|---|---|---|
| ocr_text_recognition_grounded | 0.9100 | 0.0900 |
| spatial_reasoning | 0.7000 | 0.3000 |
| visual_grounding | 0.8400 | 0.1600 |

Headroom is reported for interpretation only -- raw Delta remains the metric used throughout every other table in this document; Delta is never renormalized by headroom.

## Radius regime table (descriptive classification only)

Regime rule (fixed, applied identically to every cell -- see `classify_regime` in the analysis script -- never a "best sigma" selection): `destructive` if P(Delta<0)>=0.5 and mean<=-0.05; `near_base` if P(Delta>0)<0.1 and P(Delta<0)<0.1; `useful` if mean>0 and density(>=0.02)>=0.3 and P(Delta<0)<0.5; else `transition`.

### ocr_text_recognition_grounded

| sigma | mean | std | median | P(>0) | P(<0) | d>=.02 | d>=.05 | mass | max | min | regime |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0001 | -0.0044 | 0.0040 | -0.0080 | 0.000 | 0.547 | 0.000 | 0.000 | 0.0000 | 0.0000 | -0.0080 | transition |
| 0.0005 | -0.0030 | 0.0039 | 0.0000 | 0.000 | 0.375 | 0.000 | 0.000 | 0.0000 | 0.0000 | -0.0080 | transition |
| 0.001 | -0.0028 | 0.0039 | 0.0000 | 0.016 | 0.359 | 0.000 | 0.000 | 0.0001 | 0.0040 | -0.0080 | transition |
| 0.002 | -0.0081 | 0.0106 | -0.0080 | 0.062 | 0.578 | 0.000 | 0.000 | 0.0005 | 0.0120 | -0.0380 | transition |
| 0.005 | -0.2776 | 0.2536 | -0.1450 | 0.000 | 1.000 | 0.000 | 0.000 | 0.0000 | -0.0200 | -0.8700 | destructive |
| 0.01 | -0.9065 | 0.0082 | -0.9100 | 0.000 | 1.000 | 0.000 | 0.000 | 0.0000 | -0.8700 | -0.9100 | destructive |

### spatial_reasoning

| sigma | mean | std | median | P(>0) | P(<0) | d>=.02 | d>=.05 | mass | max | min | regime |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0001 | 0.0044 | 0.0130 | 0.0000 | 0.344 | 0.125 | 0.344 | 0.000 | 0.0069 | 0.0200 | -0.0200 | useful |
| 0.0005 | 0.0097 | 0.0206 | 0.0000 | 0.484 | 0.188 | 0.484 | 0.031 | 0.0134 | 0.0600 | -0.0200 | useful |
| 0.001 | 0.0175 | 0.0382 | 0.0200 | 0.625 | 0.203 | 0.625 | 0.188 | 0.0256 | 0.1000 | -0.1000 | useful |
| 0.002 | 0.0013 | 0.0606 | 0.0200 | 0.516 | 0.297 | 0.516 | 0.156 | 0.0216 | 0.0800 | -0.2000 | useful |
| 0.005 | -0.2003 | 0.2273 | -0.1100 | 0.125 | 0.781 | 0.125 | 0.016 | 0.0044 | 0.0600 | -0.7000 | destructive |
| 0.01 | -0.6953 | 0.0121 | -0.7000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.0000 | -0.6400 | -0.7000 | destructive |

### visual_grounding

| sigma | mean | std | median | P(>0) | P(<0) | d>=.02 | d>=.05 | mass | max | min | regime |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0001 | -0.0069 | 0.0095 | 0.0000 | 0.000 | 0.344 | 0.000 | 0.000 | 0.0000 | 0.0000 | -0.0200 | transition |
| 0.0005 | -0.0100 | 0.0112 | 0.0000 | 0.000 | 0.469 | 0.000 | 0.000 | 0.0000 | 0.0000 | -0.0400 | transition |
| 0.001 | -0.0141 | 0.0172 | -0.0200 | 0.047 | 0.562 | 0.047 | 0.000 | 0.0009 | 0.0200 | -0.0600 | transition |
| 0.002 | -0.0156 | 0.0233 | -0.0200 | 0.125 | 0.562 | 0.125 | 0.000 | 0.0028 | 0.0400 | -0.0600 | transition |
| 0.005 | -0.3606 | 0.2936 | -0.3000 | 0.031 | 0.969 | 0.031 | 0.000 | 0.0006 | 0.0200 | -0.8400 | destructive |
| 0.01 | -0.8400 | 0.0000 | -0.8400 | 0.000 | 1.000 | 0.000 | 0.000 | 0.0000 | -0.8400 | -0.8400 | destructive |

## Statistical uncertainty (D_map population-level, exploratory)

Wilson 95% CIs (P(Delta>0), density>=.02, density>=.05) and bootstrap 95% CIs (mean Delta; 10000 resamples, deterministic seed=20260824) are in `radius_table.json` alongside every point estimate above. These are D_map population-level exploratory CIs over the 64 shared perturbations -- **not** held-out expert confirmation (that requires D_confirm, not evaluated in this pilot).

## Within-sigma diversity (the valid specialization diagnostic)

Pooled diversity (the pilot's own `diversity_summary.json`) mixes all six sigmas together, which is confounded: sigma=.005/.01 cause common, near-universal degradation across capabilities, which manufactures spurious agreement that has nothing to do with specialization at any one radius. The per-sigma numbers below (full detail in `diversity_by_sigma.json`) are the statistics that should actually be read for a specialization claim.

| sigma | Spectral Discordance | improving 0 caps | 1 cap | 2 caps | all 3 |
|---|---|---|---|---|---|
| 0.0001 | 0.8337 | 42 | 22 | 0 | 0 |
| 0.0005 | 0.9115 | 33 | 31 | 0 | 0 |
| 0.001 | 0.9414 | 23 | 38 | 3 | 0 |
| 0.002 | 1.1233 | 23 | 37 | 4 | 0 |
| 0.005 | 0.5409 | 55 | 8 | 1 | 0 |
| 0.01 | 0.4027 | 64 | 0 | 0 | 0 |

**Concretely**: the pilot's own pooled (all-384-perturbations-combined) Spectral Discordance is **0.3885**, while the per-sigma values above range from **0.4027** (sigma=0.01, the fully-collapsed destructive regime, where every capability degrades together and 'discordance' is nearly meaningless) up to **1.1233** (at the useful/transition radii). The pooled figure is pulled down toward the destructive-regime value, understating how discordant (specialist-like) the useful-radius perturbations actually are -- a direct, numerical demonstration of the pooling confound this section exists to fix. No perturbation ever improved all 3 capabilities simultaneously at any sigma (the "all 3" column is 0 throughout).

## Directional transfer (sigma in 0.0005, 0.001, 0.002)

For each source capability t, mean Delta on every target capability u, restricted to perturbations where Delta_t > 0 ("positive source"), and repeated with the stronger criterion Delta_t >= 0.02 ("strong source", reported only where the selection is non-empty). Full matrices + exact sample counts in `directional_transfer.json`.

## Top expert overlap

Top-5 / top-10 / top-20% Jaccard overlap between each pair of capabilities' own top-ranked perturbations, per sigma, with the actual perturbation IDs persisted in `expert_overlap.json` for audit.

## OCR / grounding threshold numeric diagnosis

- OCR sigma=0.001: n_delta_gt_0=1/64, min_positive_delta=0.004000, max_positive_delta=0.004000, n_delta_ge_0.02=0, max_abs_distance_to_nearest_0.02_multiple=0.004000.
- OCR sigma=0.002: n_delta_gt_0=4/64, min_positive_delta=0.006000, max_positive_delta=0.012000, n_delta_ge_0.02=0, max_abs_distance_to_nearest_0.02_multiple=0.008000.

**Diagnosis** (see `delta_numeric_audit.json` for every capability x sigma cell): `ocr_text_recognition_grounded`'s `primary_metric` is the mean of the continuous VQA soft-accuracy score per example (`vqa_soft_accuracy.py`, a 10-choose-9 leave-one-out fractional score), NOT a binary per-example correctness flag -- unlike `visual_grounding` (`accuracy_at_iou_0.5`, binary) and `spatial_reasoning` (GQA exact-match, binary), whose aggregate deltas are therefore near-exact multiples of 1/50=0.02 (with only floating-point-representation-scale noise, ~1e-14 to 1e-16 in magnitude -- see those capabilities' own `max_abs_distance_to_nearest_0.02_multiple` values in `delta_numeric_audit.json`, which stay at that tiny scale). OCR's positive deltas at sigma=.001/.002 are **genuinely fine-grained** (their distance to the nearest 0.02 multiple is on the order of the deltas themselves, not floating-point noise) -- a partial-credit shift in soft-accuracy on one or a few examples, without any example's score crossing a full correctness threshold. This is a real property of the OCR metric's own granularity, **not** a floating-point artifact, and metrics were not changed to accommodate it.

## Scientific interpretation

**B)** Spatial reasoning exhibits a dense useful nearby thicket: at sigma in {0.0001, 0.0005, 0.001, 0.002} its regime classifies as `useful` (mean Delta > 0, density(>=0.02) >= 0.3, degradation probability < 0.5 -- see the radius table), with density(>=0.02) peaking at sigma=0.001.
**C)** Grounding and OCR do not exhibit comparable density under this perturbation scope: both classify as `near_base` or `transition` at every sigma tested here (see their own radius-table rows), never reaching `useful` -- OCR in particular never crosses the 0.02 reporting margin at any sigma below the destructive regime.
**D)** Therefore this result supports capability-conditioned local structure (the same global, non-visual perturbation neighborhood behaves very differently across capabilities) and motivates anatomical localization as the next step; it does **not** establish *where* grounding/OCR expertise resides in the model -- that question is explicitly out of scope for a global, undifferentiated perturbation and requires the anatomical (Stage 7+) experiment.
**E)** Pooled cross-task diversity (the pilot's own aggregate `diversity_summary.json`) is confounded by perturbation radius, for the reason given above; the within-sigma statistics in this document (`diversity_by_sigma.json`) are the valid specialization diagnostic and should be used in place of the pooled numbers for any claim about whether experts are specialists or generalists.

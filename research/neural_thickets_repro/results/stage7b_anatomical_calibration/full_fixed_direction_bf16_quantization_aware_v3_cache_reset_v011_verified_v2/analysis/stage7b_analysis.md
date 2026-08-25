# Stage 7B Analysis: Anatomical Calibration (full run, v3 quantization-aware)

Source: `results.jsonl` (432 rows, 144 unique perturbations, radius_realization_method=fixed_direction_bf16_quantization_aware_v3, multimodal_cache_policy=full_encoder_reset_vllm011_verified_v2, enable_prefix_caching=False, restoration_mode=fixed_base). Analysis only -- no model run, no perturbation applied, no existing result altered.

## Stage disambiguation

- **Stage 6**: historical `global_gaussian_upstream` run -- now proven (see `thicket.anatomy`'s own exclusion of `visual.`/`model.visual.`-prefixed parameters) to perturb exactly the language region, not a separate 'language-only' protocol by original design.
- **Stage 7B** (this document): norm-controlled anatomical calibration -- 3 regions (vision, multimodal_connector_or_merger, language) x 6 common relative-L2 radii x 8 perturbations x 3 capabilities, D_map N=20 per capability, exact-norm-controlled `anatomical_relative_l2` (distinct from Stage 6's own raw-sigma Gaussian protocol -- radii and sigmas are NEVER compared as numerically identical anywhere in this document). Calibration-scale evidence, not the paper atlas.
- **Stage 8** (future, NOT implemented here): paper-scale anatomical atlas, built on the radius set this document recommends.

## Cache-artifact regression check: PASSED

**scientific_status = `valid`** -- old_cache_artifact_reproduced = **False** -- valid_regions = ['language', 'multimodal_connector_or_merger', 'vision'] (all 3), invalid_regions = [] (none).

CACHE-ARTIFACT REGRESSION CHECK: PASSED. The same detector applied to the prior no-cache-reset run (0 of 432 rows flagged here, vs. 288 of 432 there) found every (capability, region) group's delta NOT identically zero and per_example_result_hash NOT collapsed to a single value for every region, including vision and multimodal_connector_or_merger -- minimum distinct per_example_result_hash count observed across any (capability, region) group, by region: {'language': 33, 'multimodal_connector_or_merger': 6, 'vision': 22}. Generation output is no longer invariant to vision/connector perturbation magnitude.

**Conclusion**: No stale-encoder-cache artifact detected in this run.

For comparison, the SAME detector applied to the prior no-cache-reset run at `results/stage7b_anatomical_calibration/full_fixed_direction_bf16_quantization_aware_v3/` found `old_cache_artifact_reproduced=True`, `invalid_row_count=288` of 432 -- that run remains on disk, marked `scientific_status=partially_invalid`, as no-cache-reset PROVENANCE only; its vision/connector rows are never mixed into this analysis, and its language rows are reference-only, never merged with this run's own language rows.

| capability | region | n_rows | all delta==0 | unique hashes | suspected artifact |
|---|---|---|---|---|---|
| ocr_text_recognition_grounded | language | 48 | False | 33 | False |
| ocr_text_recognition_grounded | multimodal_connector_or_merger | 48 | False | 6 | False |
| ocr_text_recognition_grounded | vision | 48 | False | 22 | False |
| spatial_reasoning | language | 48 | False | 38 | False |
| spatial_reasoning | multimodal_connector_or_merger | 48 | False | 9 | False |
| spatial_reasoning | vision | 48 | False | 27 | False |
| visual_grounding | language | 48 | False | 48 | False |
| visual_grounding | multimodal_connector_or_merger | 48 | False | 48 | False |
| visual_grounding | vision | 48 | False | 48 | False |

## 1) Run integrity

`overall_pass=True`: 144/144 unique perturbations, 432/432 rows, 3x6x8 grid complete=True, model_revision consistent=True, mask hashes consistent=True, method==fixed_direction_bf16_quantization_aware_v3: True, run_complete=True.

Max actual relative-radius error observed across all accepted candidates: **0.000392** (admissibility bound: 0.001).

Quantization-limited acceptance counts by region x radius:

| region | radius | strict | quantization_limited |
|---|---|---|---|
| language | 0.003569882854 | 7 | 1 |
| language | 0.01784941427 | 6 | 2 |
| language | 0.03569882854 | 6 | 2 |
| language | 0.07139765709 | 6 | 2 |
| language | 0.1784941427 | 2 | 6 |
| language | 0.3569882854 | 1 | 7 |
| multimodal_connector_or_merger | 0.003569882854 | 7 | 1 |
| multimodal_connector_or_merger | 0.01784941427 | 8 | 0 |
| multimodal_connector_or_merger | 0.03569882854 | 7 | 1 |
| multimodal_connector_or_merger | 0.07139765709 | 3 | 5 |
| multimodal_connector_or_merger | 0.1784941427 | 2 | 6 |
| multimodal_connector_or_merger | 0.3569882854 | 1 | 7 |
| vision | 0.003569882854 | 5 | 3 |
| vision | 0.01784941427 | 6 | 2 |
| vision | 0.03569882854 | 7 | 1 |
| vision | 0.07139765709 | 6 | 2 |
| vision | 0.1784941427 | 2 | 6 |
| vision | 0.3569882854 | 0 | 8 |

## 2) Baseline scores and headroom

| capability | baseline_score | headroom (1 - baseline) |
|---|---|---|
| ocr_text_recognition_grounded | 0.8100 | 0.1900 |
| spatial_reasoning | 0.8500 | 0.1500 |
| visual_grounding | 0.8000 | 0.2000 |

Raw Delta (never headroom-normalized) is the metric used throughout every other table in this document.

**Baseline consistency across regions**: True -- every candidate row's own `base_score` (regardless of which anatomy region it perturbs) was checked against the single canonical baseline in `baseline_scores.json` ({'visual_grounding': 0.8, 'ocr_text_recognition_grounded': 0.8099999999999999, 'spatial_reasoning': 0.85}); a baseline is computed exactly once against theta_0, before any candidate loop, so it must not depend on anatomy region.

## 3) Capability x region x radius calibration table (compact: mean Delta / P(>0) / density>=.02)

### ocr_text_recognition_grounded

| region | radius | mean | P(>0) | P(<0) | d>=.02 | mass | regime (common) |
|---|---|---|---|---|---|---|---|
| language | 0.003570 | +0.0113 | 0.750 | 0.000 | 0.000 | 0.0113 | transition |
| language | 0.017849 | +0.0113 | 0.750 | 0.000 | 0.000 | 0.0113 | transition |
| language | 0.035699 | +0.0113 | 0.750 | 0.000 | 0.000 | 0.0113 | transition |
| language | 0.071398 | +0.0256 | 0.875 | 0.125 | 0.500 | 0.0300 | transition |
| language | 0.178494 | -0.5162 | 0.000 | 1.000 | 0.000 | 0.0000 | destructive |
| language | 0.356988 | -0.8100 | 0.000 | 1.000 | 0.000 | 0.0000 | destructive |
| multimodal_connector_or_merger | 0.003570 | +0.0150 | 1.000 | 0.000 | 0.000 | 0.0150 | transition |
| multimodal_connector_or_merger | 0.017849 | +0.0150 | 1.000 | 0.000 | 0.000 | 0.0150 | transition |
| multimodal_connector_or_merger | 0.035699 | +0.0113 | 0.750 | 0.000 | 0.000 | 0.0113 | transition |
| multimodal_connector_or_merger | 0.071398 | +0.0113 | 0.750 | 0.000 | 0.000 | 0.0113 | transition |
| multimodal_connector_or_merger | 0.178494 | +0.0094 | 0.625 | 0.000 | 0.000 | 0.0094 | destructive |
| multimodal_connector_or_merger | 0.356988 | +0.0150 | 1.000 | 0.000 | 0.000 | 0.0150 | destructive |
| vision | 0.003570 | +0.0131 | 0.875 | 0.000 | 0.000 | 0.0131 | transition |
| vision | 0.017849 | +0.0113 | 0.750 | 0.000 | 0.000 | 0.0113 | transition |
| vision | 0.035699 | +0.0131 | 0.875 | 0.000 | 0.000 | 0.0131 | transition |
| vision | 0.071398 | +0.0138 | 0.875 | 0.000 | 0.125 | 0.0138 | transition |
| vision | 0.178494 | -0.0131 | 0.125 | 0.500 | 0.125 | 0.0063 | destructive |
| vision | 0.356988 | -0.3537 | 0.000 | 1.000 | 0.000 | 0.0000 | destructive |

### spatial_reasoning

| region | radius | mean | P(>0) | P(<0) | d>=.02 | mass | regime (common) |
|---|---|---|---|---|---|---|---|
| language | 0.003570 | +0.0250 | 0.500 | 0.000 | 0.500 | 0.0250 | transition |
| language | 0.017849 | +0.0375 | 0.750 | 0.000 | 0.750 | 0.0375 | transition |
| language | 0.035699 | -0.0125 | 0.000 | 0.250 | 0.000 | 0.0000 | transition |
| language | 0.071398 | -0.0562 | 0.125 | 0.500 | 0.125 | 0.0063 | transition |
| language | 0.178494 | -0.6187 | 0.000 | 1.000 | 0.000 | 0.0000 | destructive |
| language | 0.356988 | -0.8500 | 0.000 | 1.000 | 0.000 | 0.0000 | destructive |
| multimodal_connector_or_merger | 0.003570 | +0.0313 | 0.625 | 0.000 | 0.625 | 0.0313 | transition |
| multimodal_connector_or_merger | 0.017849 | +0.0188 | 0.375 | 0.000 | 0.375 | 0.0188 | transition |
| multimodal_connector_or_merger | 0.035699 | +0.0250 | 0.500 | 0.000 | 0.500 | 0.0250 | transition |
| multimodal_connector_or_merger | 0.071398 | +0.0375 | 0.750 | 0.000 | 0.750 | 0.0375 | transition |
| multimodal_connector_or_merger | 0.178494 | +0.0063 | 0.125 | 0.000 | 0.125 | 0.0063 | destructive |
| multimodal_connector_or_merger | 0.356988 | +0.0313 | 0.625 | 0.000 | 0.625 | 0.0313 | destructive |
| vision | 0.003570 | +0.0500 | 1.000 | 0.000 | 1.000 | 0.0500 | transition |
| vision | 0.017849 | +0.0250 | 0.500 | 0.000 | 0.500 | 0.0250 | transition |
| vision | 0.035699 | +0.0125 | 0.375 | 0.125 | 0.375 | 0.0188 | transition |
| vision | 0.071398 | +0.0250 | 0.500 | 0.000 | 0.500 | 0.0250 | transition |
| vision | 0.178494 | -0.0312 | 0.000 | 0.500 | 0.000 | 0.0000 | destructive |
| vision | 0.356988 | -0.3500 | 0.000 | 1.000 | 0.000 | 0.0000 | destructive |

### visual_grounding

| region | radius | mean | P(>0) | P(<0) | d>=.02 | mass | regime (common) |
|---|---|---|---|---|---|---|---|
| language | 0.003570 | -0.0625 | 0.000 | 0.875 | 0.000 | 0.0000 | transition |
| language | 0.017849 | -0.0438 | 0.000 | 0.750 | 0.000 | 0.0000 | transition |
| language | 0.035699 | -0.0313 | 0.250 | 0.500 | 0.250 | 0.0125 | transition |
| language | 0.071398 | -0.0313 | 0.375 | 0.625 | 0.375 | 0.0250 | transition |
| language | 0.178494 | -0.7875 | 0.000 | 1.000 | 0.000 | 0.0000 | destructive |
| language | 0.356988 | -0.8000 | 0.000 | 1.000 | 0.000 | 0.0000 | destructive |
| multimodal_connector_or_merger | 0.003570 | -0.0375 | 0.000 | 0.625 | 0.000 | 0.0000 | transition |
| multimodal_connector_or_merger | 0.017849 | -0.0438 | 0.000 | 0.625 | 0.000 | 0.0000 | transition |
| multimodal_connector_or_merger | 0.035699 | -0.0375 | 0.000 | 0.625 | 0.000 | 0.0000 | transition |
| multimodal_connector_or_merger | 0.071398 | -0.0375 | 0.000 | 0.500 | 0.000 | 0.0000 | transition |
| multimodal_connector_or_merger | 0.178494 | -0.0375 | 0.000 | 0.625 | 0.000 | 0.0000 | destructive |
| multimodal_connector_or_merger | 0.356988 | -0.0250 | 0.125 | 0.500 | 0.125 | 0.0062 | destructive |
| vision | 0.003570 | -0.0250 | 0.000 | 0.500 | 0.000 | 0.0000 | transition |
| vision | 0.017849 | -0.0188 | 0.250 | 0.375 | 0.250 | 0.0125 | transition |
| vision | 0.035699 | -0.0250 | 0.000 | 0.500 | 0.000 | 0.0000 | transition |
| vision | 0.071398 | -0.0125 | 0.125 | 0.375 | 0.125 | 0.0062 | transition |
| vision | 0.178494 | -0.0125 | 0.250 | 0.500 | 0.250 | 0.0187 | destructive |
| vision | 0.356988 | -0.4313 | 0.000 | 1.000 | 0.000 | 0.0000 | destructive |

## 4) Matched-radius region comparison

At the SAME relative-L2 radius, mean Delta by region (relative-L2 normalization is already the cross-region control -- no separate parameter-count correction applied).

### ocr_text_recognition_grounded

| radius | language | multimodal_connector_or_merger | vision |
|---|---|---|---|
| 0.003570 | +0.0113 | +0.0150 | +0.0131 |
| 0.017849 | +0.0113 | +0.0150 | +0.0113 |
| 0.035699 | +0.0113 | +0.0113 | +0.0131 |
| 0.071398 | +0.0256 | +0.0113 | +0.0138 |
| 0.178494 | -0.5162 | +0.0094 | -0.0131 |
| 0.356988 | -0.8100 | +0.0150 | -0.3537 |

### spatial_reasoning

| radius | language | multimodal_connector_or_merger | vision |
|---|---|---|---|
| 0.003570 | +0.0250 | +0.0313 | +0.0500 |
| 0.017849 | +0.0375 | +0.0188 | +0.0250 |
| 0.035699 | -0.0125 | +0.0250 | +0.0125 |
| 0.071398 | -0.0562 | +0.0375 | +0.0250 |
| 0.178494 | -0.6187 | +0.0063 | -0.0312 |
| 0.356988 | -0.8500 | +0.0313 | -0.3500 |

### visual_grounding

| radius | language | multimodal_connector_or_merger | vision |
|---|---|---|---|
| 0.003570 | -0.0625 | -0.0375 | -0.0250 |
| 0.017849 | -0.0438 | -0.0438 | -0.0188 |
| 0.035699 | -0.0313 | -0.0375 | -0.0250 |
| 0.071398 | -0.0313 | -0.0375 | -0.0125 |
| 0.178494 | -0.7875 | -0.0375 | -0.0125 |
| 0.356988 | -0.8000 | -0.0250 | -0.4313 |

## 5) Collapse / destructive regime by region x radius

| region | radius | mean capability Delta | P(Delta<0) | P(Delta<=-0.10) |
|---|---|---|---|---|
| language | 0.003570 | -0.0088 | 0.292 | 0.125 |
| language | 0.017849 | +0.0017 | 0.250 | 0.042 |
| language | 0.035699 | -0.0108 | 0.250 | 0.125 |
| language | 0.071398 | -0.0206 | 0.417 | 0.250 |
| language | 0.178494 | -0.6408 | 1.000 | 1.000 |
| language | 0.356988 | -0.8200 | 1.000 | 1.000 |
| multimodal_connector_or_merger | 0.003570 | +0.0029 | 0.208 | 0.042 |
| multimodal_connector_or_merger | 0.017849 | -0.0033 | 0.208 | 0.083 |
| multimodal_connector_or_merger | 0.035699 | -0.0004 | 0.208 | 0.042 |
| multimodal_connector_or_merger | 0.071398 | +0.0038 | 0.167 | 0.083 |
| multimodal_connector_or_merger | 0.178494 | -0.0073 | 0.208 | 0.042 |
| multimodal_connector_or_merger | 0.356988 | +0.0071 | 0.167 | 0.042 |
| vision | 0.003570 | +0.0127 | 0.167 | 0.000 |
| vision | 0.017849 | +0.0058 | 0.125 | 0.042 |
| vision | 0.035699 | +0.0002 | 0.208 | 0.000 |
| vision | 0.071398 | +0.0088 | 0.125 | 0.000 |
| vision | 0.178494 | -0.0190 | 0.500 | 0.042 |
| vision | 0.356988 | -0.3783 | 1.000 | 1.000 |

## 6) Common radius regime classification (pooled across all 3 regions)

All 3 regions are scientifically valid in this run -- this pooled classification is the authoritative COMMON-radius signal (never diluted by a contaminated region). Uses `classify_regime`, byte-identical/UNCHANGED from the prior (contaminated-run) analysis -- no threshold was retuned after seeing these corrected results.

| radius | mean | P(>0) | P(<0) | d>=.02 | regime (pooled) |
|---|---|---|---|---|---|
| 0.003570 | +0.0023 | 0.528 | 0.222 | 0.236 | transition |
| 0.017849 | +0.0014 | 0.486 | 0.194 | 0.208 | transition |
| 0.035699 | -0.0037 | 0.389 | 0.222 | 0.125 | transition |
| 0.071398 | -0.0027 | 0.486 | 0.236 | 0.278 | transition |
| 0.178494 | -0.2224 | 0.125 | 0.569 | 0.056 | destructive |
| 0.356988 | -0.3971 | 0.194 | 0.722 | 0.083 | destructive |

### Language-only radius classification (supplementary)

| radius | mean | P(>0) | P(<0) | d>=.02 | regime |
|---|---|---|---|---|---|
| 0.003570 | -0.0088 | 0.417 | 0.292 | 0.167 | transition |
| 0.017849 | +0.0017 | 0.500 | 0.250 | 0.250 | transition |
| 0.035699 | -0.0108 | 0.333 | 0.250 | 0.083 | transition |
| 0.071398 | -0.0206 | 0.458 | 0.417 | 0.333 | transition |
| 0.178494 | -0.6408 | 0.000 | 1.000 | 0.000 | destructive |
| 0.356988 | -0.8200 | 0.000 | 1.000 | 0.000 | destructive |

## 7) Exploratory anatomical signal (CALIBRATION-SCALE / EXPLORATORY)

Radii used (non-destructive per the pooled common classification): [0.0035698828543799426, 0.017849414271899712, 0.035698828543799424, 0.07139765708759885].

All three anatomical regions are scientifically valid in this run (no cache artifact detected) -- every column below is genuinely interpretable, calibration-scale anatomical signal (still N=8 per cell, still exploratory, not a paper-final 'experts live in X' claim).

| capability | language | multimodal_connector_or_merger | vision (mean Delta) |
|---|---|---|---|
| ocr_text_recognition_grounded | +0.0148 | +0.0131 | +0.0128 |
| spatial_reasoning | -0.0016 | +0.0281 | +0.0281 |
| visual_grounding | -0.0422 | -0.0391 | -0.0203 |

## 8) Same-direction cross-capability diversity (region x radius, N=8 diagnostic)

| region | radius | spectral discordance | improving: none | 1 cap | 2 caps | all 3 |
|---|---|---|---|---|---|---|
| language | 0.003570 | 0.6905 | 1 | 4 | 3 | 0 |
| language | 0.017849 | 0.6508 | 1 | 2 | 5 | 0 |
| language | 0.035699 | 0.5873 | 2 | 4 | 2 | 0 |
| language | 0.071398 | 1.0714 | 1 | 3 | 4 | 0 |
| language | 0.178494 | 0.5000 | 8 | 0 | 0 | 0 |
| language | 0.356988 | 0.0000 | 8 | 0 | 0 | 0 |
| multimodal_connector_or_merger | 0.003570 | 0.1270 | 0 | 3 | 5 | 0 |
| multimodal_connector_or_merger | 0.017849 | 0.8413 | 0 | 5 | 3 | 0 |
| multimodal_connector_or_merger | 0.035699 | 0.8016 | 0 | 6 | 2 | 0 |
| multimodal_connector_or_merger | 0.071398 | 0.7698 | 0 | 4 | 4 | 0 |
| multimodal_connector_or_merger | 0.178494 | 1.0317 | 2 | 6 | 0 | 0 |
| multimodal_connector_or_merger | 0.356988 | 0.1984 | 0 | 3 | 4 | 1 |
| vision | 0.003570 | 0.3016 | 0 | 1 | 7 | 0 |
| vision | 0.017849 | 0.8175 | 1 | 2 | 5 | 0 |
| vision | 0.035699 | 0.7460 | 0 | 6 | 2 | 0 |
| vision | 0.071398 | 1.3095 | 0 | 4 | 4 | 0 |
| vision | 0.178494 | 0.8810 | 5 | 3 | 0 | 0 |
| vision | 0.356988 | 0.9524 | 8 | 0 | 0 | 0 |

Full Spearman rank correlation matrices, top-perturbation-overlap Jaccard (top-2-of-8), and sign-agreement matrices are in `diversity_by_region_radius.json`, alongside this same improving-count breakdown per cell -- 'general improvement' cells (mass concentrated in the 'all 3' / '2 caps' columns) vs. 'specialized' cells (mass concentrated in the '1 cap' column, with high Spectral Discordance) is the direct evidence for section 9's specialization question.

## 9) Quantization audit

All accepted candidates within v3 admissibility rule: **True** (0 violations).

| region | radius | n | strict | quantization_limited | mean realized/requested | max rel. error |
|---|---|---|---|---|---|---|
| language | 0.003570 | 8 | 7 | 1 | 1.000139 | 0.000392 |
| language | 0.017849 | 8 | 6 | 2 | 1.000022 | 0.000088 |
| language | 0.035699 | 8 | 6 | 2 | 1.000007 | 0.000038 |
| language | 0.071398 | 8 | 6 | 2 | 1.000004 | 0.000034 |
| language | 0.178494 | 8 | 2 | 6 | 1.000003 | 0.000021 |
| language | 0.356988 | 8 | 1 | 7 | 1.000000 | 0.000024 |
| multimodal_connector_or_merger | 0.003570 | 8 | 7 | 1 | 1.000052 | 0.000352 |
| multimodal_connector_or_merger | 0.017849 | 8 | 8 | 0 | 1.000026 | 0.000056 |
| multimodal_connector_or_merger | 0.035699 | 8 | 7 | 1 | 0.999997 | 0.000053 |
| multimodal_connector_or_merger | 0.071398 | 8 | 3 | 5 | 1.000002 | 0.000039 |
| multimodal_connector_or_merger | 0.178494 | 8 | 2 | 6 | 0.999990 | 0.000029 |
| multimodal_connector_or_merger | 0.356988 | 8 | 1 | 7 | 0.999997 | 0.000021 |
| vision | 0.003570 | 8 | 5 | 3 | 1.000068 | 0.000306 |
| vision | 0.017849 | 8 | 6 | 2 | 1.000012 | 0.000064 |
| vision | 0.035699 | 8 | 7 | 1 | 0.999999 | 0.000032 |
| vision | 0.071398 | 8 | 6 | 2 | 1.000000 | 0.000026 |
| vision | 0.178494 | 8 | 2 | 6 | 0.999979 | 0.000064 |
| vision | 0.356988 | 8 | 0 | 8 | 1.000001 | 0.000081 |

## 10) Comparison to Stage 6 (qualitative only -- sigma and radius are NEVER numerically compared)

Stage 6's `global_gaussian_upstream` protocol is now proven (via `thicket.anatomy`'s own parameter exclusion) to perturb exactly the language region, not a separate protocol -- so the only valid comparison is qualitative: does Stage 6's spatial-reasoning finding (`results/visual_thicket_global_3b_pilot/full/analysis/stage6_analysis.md`: 'a dense useful nearby thicket' at sigma in {0.0001, 0.0005, 0.001, 0.002}, `useful`/active regime, density(>=0.02) peaking at sigma=0.001) reproduce here, in the language-region row of THIS run's own regime table (section 6)?

**Reproduces: False.** Language-region spatial_reasoning regime by radius (this run): {'0.00356988': 'transition', '0.0178494': 'transition', '0.0356988': 'transition', '0.0713977': 'transition', '0.178494': 'destructive', '0.356988': 'destructive'}. Stage 6's own sigma-indexed radii are NOT the same numeric scale as Stage 7B's relative-L2 radii (raw Gaussian sigma vs. exact-norm-controlled relative-L2 -- see the Stage disambiguation section above), so only the QUALITATIVE pattern (a dense, active/useful, non-destructive small-radius language-side neighborhood for spatial reasoning) is being checked, never a sigma==radius numeric identity.

## 11) Stage-8 recommendation

**proceed_to_stage8 = True.** Selected COMMON radii: ['0.00356988', '0.0713977'].

- `0.0035698828543799426`: R_small: smallest frozen radius (0.00356988), classified 'transition' under the pooled common-radius regime -- the near-base/weakly-active anchor, representing the gentlest perturbation regime tested.
- `0.07139765708759885`: R_transition (optional): largest frozen radius classified 'transition' (0.0713977) -- demarcates the boundary between the active and destructive regimes, included only because a genuine transition-labeled radius exists in this run's own classification.

Excluded as destructive: ['0.1784941427189971', '0.3569882854379942'].

Full machine-readable recommendation (selected_common_radii, classifications, rationale, excluded_radii, proceed_to_stage8, blocking_issue) is in `stage8_radius_recommendation.json`.

## 12) Decision gate

**A. Did the stale-cache artifact disappear?** True (`old_cache_artifact_reproduced=False`).
**B. Do vision/connector/language now show distinguishable behavioral landscapes?** True (see section 5's matched-radius matrices for the exact per-cell values).
**C. Evidence of capability x anatomy interaction?** See section 7 (exploratory anatomical signal) and section 8's improving-count histograms -- calibration-scale signal only (N=8), not yet a paper-final claim.
**D. Does the spatial language-side thicket reproduce?** False (see section 10).
**E. Which COMMON radii should Stage 8 use?** ['0.00356988', '0.0713977'] (see section 11).
**F. Which radii are destructive and should be dropped?** ['0.1784941427189971', '0.3569882854379942'].
**G. Is Stage 7B strong enough to proceed to Stage 8?** True (see section 11 rationale).
**H. Remaining instrumentation concern?** None identified in this analysis pass. Quantization admissibility: True (0 violations); baseline region-independence: True.

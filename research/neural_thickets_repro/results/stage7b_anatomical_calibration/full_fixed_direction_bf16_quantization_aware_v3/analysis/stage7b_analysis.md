# Stage 7B Analysis: Anatomical Calibration (full run, v3 quantization-aware)

Source: `results.jsonl` (432 rows, 144 unique perturbations, radius_realization_method=fixed_direction_bf16_quantization_aware_v3, restoration_mode=fixed_base). Analysis only -- no model run, no perturbation applied, no existing result altered.

## Stage disambiguation

- **Stage 6**: language-only global Gaussian landscape (vision encoder frozen, never perturbed).
- **Stage 7B** (this document): norm-controlled anatomical calibration -- 3 regions (vision, multimodal_connector_or_merger, language) x 6 common relative-L2 radii x 8 perturbations x 3 capabilities, D_map N=20 per capability. Calibration-scale evidence, not the paper atlas.
- **Stage 8** (future, NOT implemented here): paper-scale anatomical atlas, built on the radius set this document recommends.

## CRITICAL FINDING: stale multimodal-encoder cache invalidates vision/connector results

**scientific_status = `partially_invalid`** -- valid_regions = ['language'], invalid_regions = ['multimodal_connector_or_merger', 'vision'], invalid_reason = 'stale multimodal encoder cache after anatomical weight changes', **invalid_row_count = 288 of 432 total rows** (NOT all 432 rows -- only the vision + connector rows).

Every (capability, region) row-group for region in {vision, multimodal_connector_or_merger} has delta EXACTLY 0.0 across all 6 radii x 8 seeds AND collapses to a single per_example_result_hash, despite a real, nonzero perturbation being applied (epsilon_region_l2_norm > 0 confirmed per-candidate). Generation output is completely invariant to vision/connector perturbation magnitude. Root cause (confirmed by source inspection, not assumed): run_stage7b_anatomical_calibration.py launches its engine via launch_stage6_engine()/build_stage6_engine_config() -- the exact path GATE2_CACHE_SAFETY_REVIEW.md analyzed and declared safe ONLY because 'the visual encoder is never perturbed' under Stage 6. Stage 7B perturbs both vision and connector regions, violating that precondition, and never calls vlm_adapter.ensure_full_encoder_cache_reset_exposed() / vlm_adapter.reset_vllm_encoder_cache_full() anywhere (confirmed by direct grep: zero references to either name in run_stage7b_anatomical_calibration.py) -- vLLM's cached multimodal-encoder output for the fixed image inputs is therefore never invalidated, so every generation call under a vision/connector-perturbed candidate silently reuses the BASE model's cached image embeddings. language-region rows show real, radius-dependent deltas and many distinct hashes and are NOT affected (no analogous caching layer sits between language weights and the token-generation forward pass, per GATE2_CACHE_SAFETY_REVIEW.md section 1/3).

**Affected regions**: multimodal_connector_or_merger, vision.

**Conclusion**: vision and multimodal_connector_or_merger results in THIS run are SCIENTIFICALLY INVALID -- an instrumentation artifact, not a near-base finding. The cache-lifecycle fix (reset_vllm_encoder_cache_full wired into evaluate_one_calibration_candidate_rpc, multimodal_cache_policy=full_reset_on_weight_change_v1) has since been implemented in run_stage7b_anatomical_calibration.py, but THIS specific run predates that fix and must be preserved as no-cache-reset PROVENANCE only, never consumed by Stage 8 or any later anatomical analysis -- a corrected run must be executed under the NEW full_fixed_direction_bf16_quantization_aware_v3_cache_reset_v1 run_signature/output_dir before vision/connector conclusions can be drawn. language-region results in this run are NOT affected by this bug and may be used as-is.

| capability | region | n_rows | all delta==0 | unique hashes | suspected artifact |
|---|---|---|---|---|---|
| ocr_text_recognition_grounded | language | 48 | False | 28 | False |
| ocr_text_recognition_grounded | multimodal_connector_or_merger | 48 | True | 1 | True |
| ocr_text_recognition_grounded | vision | 48 | True | 1 | True |
| spatial_reasoning | language | 48 | False | 36 | False |
| spatial_reasoning | multimodal_connector_or_merger | 48 | True | 1 | True |
| spatial_reasoning | vision | 48 | True | 1 | True |
| visual_grounding | language | 48 | False | 48 | False |
| visual_grounding | multimodal_connector_or_merger | 48 | True | 1 | True |
| visual_grounding | vision | 48 | True | 1 | True |

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
| visual_grounding | 0.7500 | 0.2500 |

Raw Delta (never headroom-normalized) is the metric used throughout every other table in this document.

## 3) Capability x region x radius calibration table (compact: mean Delta / P(>0) / density>=.02)

### ocr_text_recognition_grounded

| region | radius | mean | P(>0) | P(<0) | d>=.02 | mass | regime (common) |
|---|---|---|---|---|---|---|---|
| language | 0.003570 | +0.0131 | 0.875 | 0.000 | 0.000 | 0.0131 | transition |
| language | 0.017849 | +0.0131 | 0.875 | 0.000 | 0.000 | 0.0131 | transition |
| language | 0.035699 | +0.0094 | 0.625 | 0.000 | 0.000 | 0.0094 | transition |
| language | 0.071398 | +0.0150 | 0.750 | 0.000 | 0.125 | 0.0150 | transition |
| language | 0.178494 | -0.3631 | 0.000 | 1.000 | 0.000 | 0.0000 | transition |
| language | 0.356988 | -0.8037 | 0.000 | 1.000 | 0.000 | 0.0000 | transition |
| multimodal_connector_or_merger | 0.003570 | +0.0000 | 0.000 | 0.000 | 0.000 | 0.0000 | transition |
| multimodal_connector_or_merger | 0.017849 | +0.0000 | 0.000 | 0.000 | 0.000 | 0.0000 | transition |
| multimodal_connector_or_merger | 0.035699 | +0.0000 | 0.000 | 0.000 | 0.000 | 0.0000 | transition |
| multimodal_connector_or_merger | 0.071398 | +0.0000 | 0.000 | 0.000 | 0.000 | 0.0000 | transition |
| multimodal_connector_or_merger | 0.178494 | +0.0000 | 0.000 | 0.000 | 0.000 | 0.0000 | transition |
| multimodal_connector_or_merger | 0.356988 | +0.0000 | 0.000 | 0.000 | 0.000 | 0.0000 | transition |
| vision | 0.003570 | +0.0000 | 0.000 | 0.000 | 0.000 | 0.0000 | transition |
| vision | 0.017849 | +0.0000 | 0.000 | 0.000 | 0.000 | 0.0000 | transition |
| vision | 0.035699 | +0.0000 | 0.000 | 0.000 | 0.000 | 0.0000 | transition |
| vision | 0.071398 | +0.0000 | 0.000 | 0.000 | 0.000 | 0.0000 | transition |
| vision | 0.178494 | +0.0000 | 0.000 | 0.000 | 0.000 | 0.0000 | transition |
| vision | 0.356988 | +0.0000 | 0.000 | 0.000 | 0.000 | 0.0000 | transition |

### spatial_reasoning

| region | radius | mean | P(>0) | P(<0) | d>=.02 | mass | regime (common) |
|---|---|---|---|---|---|---|---|
| language | 0.003570 | +0.0188 | 0.375 | 0.000 | 0.375 | 0.0188 | transition |
| language | 0.017849 | +0.0438 | 0.875 | 0.000 | 0.875 | 0.0438 | transition |
| language | 0.035699 | +0.0125 | 0.250 | 0.000 | 0.250 | 0.0125 | transition |
| language | 0.071398 | +0.0000 | 0.125 | 0.125 | 0.125 | 0.0063 | transition |
| language | 0.178494 | -0.3688 | 0.000 | 0.875 | 0.000 | 0.0000 | transition |
| language | 0.356988 | -0.8187 | 0.000 | 1.000 | 0.000 | 0.0000 | transition |
| multimodal_connector_or_merger | 0.003570 | +0.0000 | 0.000 | 0.000 | 0.000 | 0.0000 | transition |
| multimodal_connector_or_merger | 0.017849 | +0.0000 | 0.000 | 0.000 | 0.000 | 0.0000 | transition |
| multimodal_connector_or_merger | 0.035699 | +0.0000 | 0.000 | 0.000 | 0.000 | 0.0000 | transition |
| multimodal_connector_or_merger | 0.071398 | +0.0000 | 0.000 | 0.000 | 0.000 | 0.0000 | transition |
| multimodal_connector_or_merger | 0.178494 | +0.0000 | 0.000 | 0.000 | 0.000 | 0.0000 | transition |
| multimodal_connector_or_merger | 0.356988 | +0.0000 | 0.000 | 0.000 | 0.000 | 0.0000 | transition |
| vision | 0.003570 | +0.0000 | 0.000 | 0.000 | 0.000 | 0.0000 | transition |
| vision | 0.017849 | +0.0000 | 0.000 | 0.000 | 0.000 | 0.0000 | transition |
| vision | 0.035699 | +0.0000 | 0.000 | 0.000 | 0.000 | 0.0000 | transition |
| vision | 0.071398 | +0.0000 | 0.000 | 0.000 | 0.000 | 0.0000 | transition |
| vision | 0.178494 | +0.0000 | 0.000 | 0.000 | 0.000 | 0.0000 | transition |
| vision | 0.356988 | +0.0000 | 0.000 | 0.000 | 0.000 | 0.0000 | transition |

### visual_grounding

| region | radius | mean | P(>0) | P(<0) | d>=.02 | mass | regime (common) |
|---|---|---|---|---|---|---|---|
| language | 0.003570 | +0.0125 | 0.625 | 0.250 | 0.625 | 0.0313 | transition |
| language | 0.017849 | +0.0000 | 0.375 | 0.375 | 0.375 | 0.0188 | transition |
| language | 0.035699 | +0.0125 | 0.375 | 0.375 | 0.375 | 0.0312 | transition |
| language | 0.071398 | +0.0313 | 0.625 | 0.125 | 0.625 | 0.0375 | transition |
| language | 0.178494 | -0.3813 | 0.000 | 1.000 | 0.000 | 0.0000 | transition |
| language | 0.356988 | -0.7500 | 0.000 | 1.000 | 0.000 | 0.0000 | transition |
| multimodal_connector_or_merger | 0.003570 | +0.0000 | 0.000 | 0.000 | 0.000 | 0.0000 | transition |
| multimodal_connector_or_merger | 0.017849 | +0.0000 | 0.000 | 0.000 | 0.000 | 0.0000 | transition |
| multimodal_connector_or_merger | 0.035699 | +0.0000 | 0.000 | 0.000 | 0.000 | 0.0000 | transition |
| multimodal_connector_or_merger | 0.071398 | +0.0000 | 0.000 | 0.000 | 0.000 | 0.0000 | transition |
| multimodal_connector_or_merger | 0.178494 | +0.0000 | 0.000 | 0.000 | 0.000 | 0.0000 | transition |
| multimodal_connector_or_merger | 0.356988 | +0.0000 | 0.000 | 0.000 | 0.000 | 0.0000 | transition |
| vision | 0.003570 | +0.0000 | 0.000 | 0.000 | 0.000 | 0.0000 | transition |
| vision | 0.017849 | +0.0000 | 0.000 | 0.000 | 0.000 | 0.0000 | transition |
| vision | 0.035699 | +0.0000 | 0.000 | 0.000 | 0.000 | 0.0000 | transition |
| vision | 0.071398 | +0.0000 | 0.000 | 0.000 | 0.000 | 0.0000 | transition |
| vision | 0.178494 | +0.0000 | 0.000 | 0.000 | 0.000 | 0.0000 | transition |
| vision | 0.356988 | +0.0000 | 0.000 | 0.000 | 0.000 | 0.0000 | transition |

## 4) Matched-radius region comparison

At the SAME relative-L2 radius, mean Delta by region (relative-L2 normalization is already the cross-region control -- no separate parameter-count correction applied).

### ocr_text_recognition_grounded

| radius | language | multimodal_connector_or_merger | vision |
|---|---|---|---|
| 0.003570 | +0.0131 | +0.0000 | +0.0000 |
| 0.017849 | +0.0131 | +0.0000 | +0.0000 |
| 0.035699 | +0.0094 | +0.0000 | +0.0000 |
| 0.071398 | +0.0150 | +0.0000 | +0.0000 |
| 0.178494 | -0.3631 | +0.0000 | +0.0000 |
| 0.356988 | -0.8037 | +0.0000 | +0.0000 |

### spatial_reasoning

| radius | language | multimodal_connector_or_merger | vision |
|---|---|---|---|
| 0.003570 | +0.0188 | +0.0000 | +0.0000 |
| 0.017849 | +0.0438 | +0.0000 | +0.0000 |
| 0.035699 | +0.0125 | +0.0000 | +0.0000 |
| 0.071398 | +0.0000 | +0.0000 | +0.0000 |
| 0.178494 | -0.3688 | +0.0000 | +0.0000 |
| 0.356988 | -0.8187 | +0.0000 | +0.0000 |

### visual_grounding

| radius | language | multimodal_connector_or_merger | vision |
|---|---|---|---|
| 0.003570 | +0.0125 | +0.0000 | +0.0000 |
| 0.017849 | +0.0000 | +0.0000 | +0.0000 |
| 0.035699 | +0.0125 | +0.0000 | +0.0000 |
| 0.071398 | +0.0313 | +0.0000 | +0.0000 |
| 0.178494 | -0.3813 | +0.0000 | +0.0000 |
| 0.356988 | -0.7500 | +0.0000 | +0.0000 |

## 5) Collapse / destructive regime by region x radius

| region | radius | mean capability Delta | P(Delta<0) | P(Delta<=-0.10) |
|---|---|---|---|---|
| language | 0.003570 | +0.0148 | 0.083 | 0.000 |
| language | 0.017849 | +0.0190 | 0.125 | 0.000 |
| language | 0.035699 | +0.0115 | 0.125 | 0.000 |
| language | 0.071398 | +0.0154 | 0.083 | 0.000 |
| language | 0.178494 | -0.3710 | 0.958 | 0.833 |
| language | 0.356988 | -0.7908 | 1.000 | 1.000 |
| multimodal_connector_or_merger | 0.003570 | +0.0000 | 0.000 | 0.000 |
| multimodal_connector_or_merger | 0.017849 | +0.0000 | 0.000 | 0.000 |
| multimodal_connector_or_merger | 0.035699 | +0.0000 | 0.000 | 0.000 |
| multimodal_connector_or_merger | 0.071398 | +0.0000 | 0.000 | 0.000 |
| multimodal_connector_or_merger | 0.178494 | +0.0000 | 0.000 | 0.000 |
| multimodal_connector_or_merger | 0.356988 | +0.0000 | 0.000 | 0.000 |
| vision | 0.003570 | +0.0000 | 0.000 | 0.000 |
| vision | 0.017849 | +0.0000 | 0.000 | 0.000 |
| vision | 0.035699 | +0.0000 | 0.000 | 0.000 |
| vision | 0.071398 | +0.0000 | 0.000 | 0.000 |
| vision | 0.178494 | +0.0000 | 0.000 | 0.000 |
| vision | 0.356988 | +0.0000 | 0.000 | 0.000 |

## 6) Common radius regime classification (pooled across all 3 regions)

**WARNING**: this pooled classification currently averages 2 contaminated (constant-zero, see the critical finding above) regions together with the 1 real (language) region -- it is diluted, not a valid common-radius decision, until the encoder-cache bug is fixed and vision/connector are re-run. Shown for completeness; the language-only table immediately below is the currently trustworthy signal.

| radius | mean (pooled, contaminated) | P(>0) | P(<0) | d>=.02 | regime (pooled) |
|---|---|---|---|---|---|
| 0.003570 | +0.0049 | 0.208 | 0.028 | 0.111 | transition |
| 0.017849 | +0.0063 | 0.236 | 0.042 | 0.139 | transition |
| 0.035699 | +0.0038 | 0.139 | 0.042 | 0.069 | transition |
| 0.071398 | +0.0051 | 0.167 | 0.028 | 0.097 | transition |
| 0.178494 | -0.1237 | 0.000 | 0.319 | 0.000 | transition |
| 0.356988 | -0.2636 | 0.000 | 0.333 | 0.000 | transition |

### Language-only radius classification (supplementary, currently the trustworthy signal)

| radius | mean | P(>0) | P(<0) | d>=.02 | regime |
|---|---|---|---|---|---|
| 0.003570 | +0.0148 | 0.625 | 0.083 | 0.333 | active |
| 0.017849 | +0.0190 | 0.708 | 0.125 | 0.417 | active |
| 0.035699 | +0.0115 | 0.417 | 0.125 | 0.208 | transition |
| 0.071398 | +0.0154 | 0.500 | 0.083 | 0.292 | transition |
| 0.178494 | -0.3710 | 0.000 | 0.958 | 0.000 | destructive |
| 0.356988 | -0.7908 | 0.000 | 1.000 | 0.000 | destructive |

## 7) Exploratory anatomical signal (CALIBRATION-SCALE / EXPLORATORY)

Radii used (non-destructive per the language-only classification -- the only trustworthy per-radius signal): [0.0035698828543799426, 0.017849414271899712, 0.035698828543799424, 0.07139765708759885].

vision and multimodal_connector_or_merger columns are contaminated by the stale encoder-cache artifact documented in radius_regime_summary.json's data_integrity_warning -- their mean_delta=0.0 / p_delta_gt_0=0.0 values reflect the caching bug, not an anatomical finding about where grounding/OCR/spatial reasoning expertise resides. Only the language column is currently interpretable.

| capability | language | multimodal_connector_or_merger | vision (mean Delta) |
|---|---|---|---|
| ocr_text_recognition_grounded | +0.0127 | +0.0000 | +0.0000 |
| spatial_reasoning | +0.0188 | +0.0000 | +0.0000 |
| visual_grounding | +0.0141 | +0.0000 | +0.0000 |

## 8) Same-direction cross-capability diversity (region x radius, N=8 diagnostic)

| region | radius | spectral discordance |
|---|---|---|
| language | 0.003570 | 0.9444 |
| language | 0.017849 | 0.6032 |
| language | 0.035699 | 0.7937 |
| language | 0.071398 | 1.2857 |
| language | 0.178494 | 0.3333 |
| language | 0.356988 | 0.0476 |
| multimodal_connector_or_merger | 0.003570 | 0.0000 |
| multimodal_connector_or_merger | 0.017849 | 0.0000 |
| multimodal_connector_or_merger | 0.035699 | 0.0000 |
| multimodal_connector_or_merger | 0.071398 | 0.0000 |
| multimodal_connector_or_merger | 0.178494 | 0.0000 |
| multimodal_connector_or_merger | 0.356988 | 0.0000 |
| vision | 0.003570 | 0.0000 |
| vision | 0.017849 | 0.0000 |
| vision | 0.035699 | 0.0000 |
| vision | 0.071398 | 0.0000 |
| vision | 0.178494 | 0.0000 |
| vision | 0.356988 | 0.0000 |

Full Spearman rank correlation matrices, top-perturbation-overlap Jaccard, and sign-agreement matrices are in `diversity_by_region_radius.json`. The exact 0.0 values for vision/multimodal_connector_or_merger above are a SPURIOUS perfect-agreement artifact of ranking constant-zero columns (see that file's own per-cell `note` field), not evidence of low specialization -- there is no real variation in those columns to agree or disagree about.

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

## 11) Stage-8 recommendation

**D) Does calibration give enough evidence to proceed?** **NO, not yet.** 2 of 3 anatomical regions (vision, multimodal_connector_or_merger) in this run are contaminated by the stale encoder-cache artifact documented above; a genuinely COMMON radius set cannot be chosen across all three regions from this data. **E) Issue that would invalidate Stage 8**: exactly this bug, if Stage 8 were launched on the current codebase, would silently repeat -- Stage 8 must not launch until `reset_vllm_encoder_cache_full()` is wired into `evaluate_one_calibration_candidate_rpc`'s RPC path (or equivalent) and vision/connector are re-run and re-validated with this same analysis.

**A/B/C (language region only, the one trustworthy signal in this run)**: near_base radii = [], active radii = ['0.003570', '0.017849']. A principled COMMON radius set, once vision/connector are re-run and confirmed to behave consistently with language's regime boundaries, should retain one near-base radius and one active radius from this set (dropping ['0.178494', '0.356988'] as destructive) -- see the RETURN summary for the specific proposal. This is an interim, language-only-informed proposal, not a final Stage-8 decision.

# Stage 6 cache-safe reproduction audit

Compares the historical Stage-6 full run (`results/visual_thicket_global_3b_pilot/full/`, `enable_prefix_caching` left at its vLLM default True, `stage6_cache_safety_status=cache_suspect`) against the cache-safe reproduction (`results/visual_thicket_global_3b_pilot/stage6_global_gaussian_upstream_cache_safe_v2/`, `enable_prefix_caching=False`, `multimodal_cache_policy=full_encoder_reset_vllm011_verified_v2`), same frozen scientific config throughout (six sigmas, 64/sigma, 3 capabilities, D_map N=50, `global_gaussian_upstream` semantics, `fixed_base` restoration).

## Validation

All hard verification checks pass: **True**. Candidate-for-candidate alignment exact: **True** (1152/1152 candidate IDs in common, 0 seed mismatches, 0 mask-hash mismatches).

## Spatial thicket survival

`spatial_thicket_reproduces` = **partially** (2/4 small sigmas classify `useful` in the clean run, vs 4/4 historically; 2 in both).

| sigma | hist mean Delta | hist regime | clean mean Delta | clean regime | unchanged |
|---|---|---|---|---|---|
| 0.0001 | 0.0044 | useful | 0.0025 | transition | False |
| 0.0005 | 0.0097 | useful | 0.0163 | useful | True |
| 0.001 | 0.0175 | useful | 0.0150 | useful | True |
| 0.002 | 0.0013 | useful | -0.0103 | transition | False |

## Candidate-level agreement

Exact `perturbed_score` agreement across all 1152 rows: **43.0%**. `per_example_result_hash` exact match: **1.0%** (of 1152 rows with a hash on both sides).

## Cache-impact classification

**B_QUALITATIVELY_ROBUST_BUT_NUMERICALLY_CONTAMINATED**

Candidate-level agreement between the historical and cache-safe runs is weak (exact perturbed_score match on only 43.0% of the 1152 rows, mean improvement-sign flip rate 29.7%, mean top-10 Jaccard 0.30) -- individual candidate rankings are NOT stable across the two runs. However, the central scientific conclusions this experiment is used to support -- existence of a useful (mean Delta>0, density>=.02>=0.3, degradation<0.5) spatial-reasoning thicket at small sigma, and non-positive transfer of useful spatial perturbations to grounding/OCR (specialization) -- are preserved in the clean run, though the useful regime's boundary narrows (see stage8_radius_final_recommendation.json and clean_stage6_summary.json for the per-sigma detail). This is the profile of 'qualitatively robust, numerically contaminated' historical prefix-cache reuse, not a negligible effect and not a reversal of the central claim.

## Stage 8 radius recommendation

Selected common radii: [0.0035698828543799426, 0.017849414271899712, 0.07139765708759885]. proceed_to_stage8 = **True**.

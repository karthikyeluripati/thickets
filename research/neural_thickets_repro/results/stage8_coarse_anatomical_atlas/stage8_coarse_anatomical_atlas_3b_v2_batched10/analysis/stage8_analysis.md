# Stage 8: paper-scale coarse anatomical atlas -- analysis

Integrity gate: **PASS**. Model revision: `66285546d2b821cf421d4f5eb2576359d3770cd3`.

## Baselines

| capability | baseline | headroom |
|---|---|---|
| visual_grounding | 0.8800 | 0.1200 |
| counting | 0.6800 | 0.3200 |
| spatial_reasoning | 0.7000 | 0.3000 |
| ocr_text_recognition_grounded | 0.9380 | 0.0620 |
| relational_reasoning | 0.5400 | 0.4600 |
| fine_grained_recognition | 0.4200 | 0.5800 |

## Stage-9 drilldown recommendation

priority_1_region = **language**, priority_2_region = **vision**, connector_action = **keep_whole**.

multimodal_connector_or_merger is architecturally far smaller than vision/language (36.7M vs 632.0M / 3086.0M parameters, Stage 7A's own live inventory) -- shows no capability-selective stable dominance and no density advantage over the other two regions in this Stage-8 atlas, so it should remain a single undivided L1 region for Stage 9.

## CUB / fine_grained_recognition stability

fraction_exact_zero_delta = 0.783, n_distinct_nonzero_abs_delta_values = 9.

## Stage 6 / Stage 7B / Stage 8 bridge

Sigma (Stage 6, global upstream Gaussian perturbation) and relative-L2 radius (Stage 7B/8, anatomically-scoped perturbation) are DIFFERENT parameterizations over DIFFERENT perturbation scopes and are NOT numerically equated anywhere in this file.

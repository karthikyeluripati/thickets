# Stage 11 S2: interim 3B-vs-7B anatomy-resolved scale analysis

Cross-scale integrity gate: **PASS**.

This is NOT a scaling-law claim (only 2 scale points). Terminology guard: ['scale trend', 'cross-scale comparison'].

## Claim gate (A1-A6)

- A1_coarse_anatomy_structures_density_both_scales: **strongly_supported_3B_to_7B**
- A2_anatomical_distribution_changes: **supported_3B_to_7B**
- A3_scale_effects_capability_dependent: **strongly_supported_3B_to_7B**
- A4_scale_effects_anatomically_non_uniform: **strongly_supported_3B_to_7B**
- A5_radius_and_scale_jointly_reorganize: **supported_3B_to_7B**
- A6_specialization_changes_differently_by_region: **strongly_supported_3B_to_7B**

## Anatomical preference transitions (dominant region, per capability x radius)

| capability | radius | dominant 3B | dominant 7B | classification |
|---|---|---|---|---|
| visual_grounding | small | None | None | diffuse_no_clear_preference |
| visual_grounding | mid | None | language | diffuse_no_clear_preference |
| visual_grounding | transition | None | None | diffuse_no_clear_preference |
| counting | small | None | None | diffuse_no_clear_preference |
| counting | mid | None | None | diffuse_no_clear_preference |
| counting | transition | None | None | diffuse_no_clear_preference |
| spatial_reasoning | small | None | vision | diffuse_no_clear_preference |
| spatial_reasoning | mid | language | None | diffuse_no_clear_preference |
| spatial_reasoning | transition | language | vision | anatomical_preference_reorganizes |
| ocr_text_recognition_grounded | small | None | None | diffuse_no_clear_preference |
| ocr_text_recognition_grounded | mid | None | None | diffuse_no_clear_preference |
| ocr_text_recognition_grounded | transition | None | None | diffuse_no_clear_preference |
| relational_reasoning | small | None | None | diffuse_no_clear_preference |
| relational_reasoning | mid | None | language | diffuse_no_clear_preference |
| relational_reasoning | transition | vision | None | diffuse_no_clear_preference |
| fine_grained_recognition | small | None | None | diffuse_no_clear_preference |
| fine_grained_recognition | mid | None | None | diffuse_no_clear_preference |
| fine_grained_recognition | transition | None | None | diffuse_no_clear_preference |

DO NOT START 32B. DO NOT START 72B. DO NOT REDESIGN THE FROZEN SCALE EXPERIMENT. DO NOT START ATLAS-GUIDED SEARCH YET.
]0;root@24abe5fd1a92: /workspace/thickets/research/neural_thickets_reproroot@24abe5fd1a92:/workspace/thickets/research/neural_thickets_repro# echo MARKER
R_MD_END

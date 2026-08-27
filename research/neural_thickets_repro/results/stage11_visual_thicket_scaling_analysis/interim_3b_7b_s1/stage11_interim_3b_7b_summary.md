# Stage 11 S1: interim 3B-vs-7B whole-model scale analysis

Cross-scale integrity gate: **PASS**.

This is NOT a scaling-law claim (only 2 scale points). Terminology guard: ['scale trend', 'cross-scale comparison'].

## Baselines

| capability | baseline_3B | baseline_7B | headroom_3B | headroom_7B |
|---|---|---|---|---|
| visual_grounding | 0.8800 | 0.8200 | 0.1200 | 0.1800 |
| counting | 0.6800 | 0.8200 | 0.3200 | 0.1800 |
| spatial_reasoning | 0.7000 | 0.7200 | 0.3000 | 0.2800 |
| ocr_text_recognition_grounded | 0.9380 | 0.9140 | 0.0620 | 0.0860 |
| relational_reasoning | 0.5400 | 0.5400 | 0.4600 | 0.4600 |
| fine_grained_recognition | 0.4200 | 0.4600 | 0.5800 | 0.5400 |

## Interim claim gate (S1-S5)

- S1_nearby_specialists_exist_both_scales: **strongly_supported_3B_to_7B**
- S2_solution_density_changes_systematically: **mixed**
- S3_specialist_strength_changes: **supported_3B_to_7B**
- S4_useful_radius_behavior_changes: **supported_3B_to_7B**
- S5_specialization_diversity_changes: **unsupported**

## Capability-by-capability scale response

| capability | classification |
|---|---|
| visual_grounding | thicket_expands_3B_to_7B |
| counting | thicket_contracts_3B_to_7B |
| spatial_reasoning | thicket_contracts_3B_to_7B |
| ocr_text_recognition_grounded | little_change |
| relational_reasoning | thicket_contracts_3B_to_7B |
| fine_grained_recognition | mixed_scale_response |

DO NOT START 7B ANATOMY. DO NOT ENABLE 32B. DO NOT ENABLE 72B. DO NOT CHANGE THE FROZEN SCALE EXPERIMENT.

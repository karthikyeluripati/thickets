# Stage 9: hierarchical anatomical localization -- analysis

Integrity gate: **PASS**. Model revision: `66285546d2b821cf421d4f5eb2576359d3770cd3`.
Baselines match Stage-8 authoritative values: **True**.

## Hero question: spatial_reasoning language depth

Answer: **E** (changes substantially with radius).

## Stage 8 -> Stage 9 story

- **A_did_stage9_sharpen_stage8_localization**: False
- **B_did_spatial_reasoning_language_signal_resolve_to_a_depth**: False
- **C_did_vision_capabilities_separate_by_depth**: False
- **D_are_experts_more_localized_at_depth_than_at_l1**: False
- **E_does_radius_still_reorganize_expert_identity_after_depth_conditioning**: True

## Paper claim gate

- **C1_nearby_visual_specialists_exist**: strongly_supported
- **C2a_expert_density_strength_depends_on_coarse_anatomy**: strongly_supported
- **C2b_expert_density_strength_exhibits_hierarchical_depth_structure**: supported
- **C3_nearby_experts_are_capability_specialized**: supported
- **C_radius_expert_identity_density_changes_with_scale**: supported

## Next-stage recommendation

**geometry_low_dimensional_structure** -- n_capabilities_with_stable_depth_dominance=4, n_significant_depth_density_contrasts=0, language_depth_answer=E (changes substantially with radius). This does not cross the frozen exceptional-evidence bar for jumping ahead of the roadmap's default next step -- geometry / low-dimensional useful perturbation structure remains the recommended next stage.

## Numerical patch audit

runtime_metadata does not persist a per-row bracket_expansion_used flag -- exact per-candidate bracket-expansion identification is not recoverable from results.jsonl alone. quantization_limited_count below is a STRICT SUPERSET of bracket-expansion- resolved candidates (most quantization_limited candidates resolve within the original <=20-attempt v2/v3 search, exactly as in Stage 8; only candidates that ALSO had no bracket within the original 20 attempts needed expansion).
strict=850, quantization_limited=302, admissibility_violations=0.

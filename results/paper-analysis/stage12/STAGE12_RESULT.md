# Stage 1+2 result: PASS. A perturbation's answer-content tilt is predictable from its noise

Lock: `plan_lock.md` (committed in 09c1f1f before any measurement). 80 frozen perturbations × 96 frozen probe items.
Spend $2.86 of the $8.00 cap; pod stopped after a verified pull.

| Locked test | Result | Rule | Verdict |
|---|---|---|---|
| S1a tilt reliability (split-half, Spearman–Brown) | 0.878 | ≥ 0.7 | pass |
| S1b fidelity | 80/80 fingerprints match; vision reset and base restore exact; 0 missing letter scores | all | pass |
| **S2 primary** r(pred_NOV, T_NOV) | **0.938** (slope 0.83) | ≥ 0.6 PASS | **PASS** |
| S2 share r(T_NOV, T_FULL)² | 0.843 | ≥ 0.5 | pass |
| **Stage 3 GO** | | PASS and share ≥ 0.5 | **GO** |

Reported, not gating:
- r(pred_NOV, T_FULL) = 0.839 (slope 0.81).
- Adding the vision first-order term lowers the fit: r(pred_NOV + pred_V, T_FULL) = 0.525. This matches the earlier
  finding that vision is nonlinear.
- r(pred_V, T_FULL − T_NOV) = 0.18.
- Winner 9504111: T_FULL = 1.16 nats (97.5th percentile of the 80); pred_NOV = 1.42.

**Post-hoc check (after the verdict), content vs letter position** (`position_check.json`):
- The letter-position (A–D) part of the tilt has sd 0.04, against a total of 0.59. **99% of the tilt variance is
  option content.**
- The prediction tracks the content part (r = 0.94 with vision reset) and not the position part (r = 0.07).

## What this establishes
For σ = 0.002 RandOpt perturbations, the shift in the model's preference for "front" options is, outside the vision
tower, first-order predictable from the noise stream alone:

T ≈ σ⟨fold_nonvision(∇T̄), ε⟩

This holds across 80 perturbations not used for any fitting (the prediction has no free parameters). This is the
reproducible component that earlier CPU work showed carries part of the winner's selection-set gain.

## What it does not establish
- That per-example accuracy changes are first-order (H★ remains falsified).
- Which layers or representation carry ∇T̄ (Stage 3).
- That selection on SEARCH/RERANK operates through this projection for the whole population. That needs
  predicted tilts for all candidates.
- Anything outside this model, benchmark and σ.

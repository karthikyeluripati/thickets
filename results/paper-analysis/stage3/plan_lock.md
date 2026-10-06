# Stage 3 plan lock: does search select along the tilt axis, and where in the network does the tilt come from?

Locked before any Stage 3 computation. Follows the Stage 2 locked PASS (`../stage12/STAGE12_RESULT.md`:
r = 0.938, 80 perturbations). GPU budget: **pending explicit user approval**; proposed cap $6.00 for one pod session.
H★ remains falsified. Nothing here re-tests per-example accuracy linearity.

## Frozen inputs (`frozen_inputs.json`)
- **Probe sets.** Items have ≥ 1 option containing front/forward and ≥ 1 without.
  - S: 180 SEARCH items.
  - R: 183 RERANK items.
  - P96: the Stage-2 probe.
- **Groups:** embed, six 6-layer language blocks (L00–05 … L30–35), final_norm_head, vision. This partition is
  exhaustive.
- **Candidates:** all 5000 in `ALL_candidates` (all σ). Accuracy outcomes come from the master predictions.
- **Localization perturbations:** the first 20 of the Stage-2 frozen order (9504111, 12 controls, 7 hash-sampled).

## Predictor
- For probe set X, F_X^g = fold of the gradient of mean tilt T̄_X restricted to group g. It is computed in HF fp32 at
  BASE, exactly as in Stage 2.
- τ_X(c) = σ_c Σ_{g ≠ vision} ⟨F_X^g, ε_c⟩ is the predicted non-vision tilt.
- There are no fitted parameters.

## Stage 3A: population-level selection (HF only)

| Test | Statistic | Rule |
|---|---|---|
| 3A-0 pipeline consistency | r(τ_P96, measured Stage-2 T_NOV) on the 80 | ≥ 0.9, else stop (pipeline error) |
| **3A-1** | r(τ_R, SEARCH accuracy gain) over all 5000 candidates (R items disjoint from SEARCH) | **≥ 0.25 PASS; < 0.10 FALSIFIED** |
| **3A-2** | r(τ_S, RERANK accuracy gain) over the 543 RERANK candidates (S items disjoint from RERANK) | **≥ 0.25 PASS; < 0.10 FALSIFIED** |
| 3A-3 selection enrichment | within each σ, z-score τ_R over that σ's 1250; mean z of RERANK-advanced candidates minus mean z of the rest | ≥ 0.3 SD supports; reported with a bootstrap CI |
| reported | r(τ_R, SEARCH gain) within σ = 0.002; percentiles of 9504111 for τ_S and τ_R; r(τ_S+R, TEST gain) over the 61 TEST-measured candidates (expected ≈ 0; no decision) | – |

**3A conclusion:**
- "Search selects along the tilt axis" is supported only if 3A-1 **and** 3A-2 PASS.
- It is falsified if both are falsified.
- Anything else is mixed or inconclusive.

## Stage 3B: block localization of the tilt (vLLM)
- For each of the 20 perturbations and each non-vision group g, build INSERTION_g: BASE plus the perturbation
  restricted to g, every other tensor exactly BASE (verified). Measure its tilt on P96 with the Stage-2 metric and
  engine.
- Predicted partial: σ⟨F_P96^g, ε⟩.

| Test | Statistic | Rule |
|---|---|---|
| **3B-1** | pooled r(predicted partial, measured insertion tilt) over 20 × 8 (perturbation, group) pairs | **≥ 0.6 PASS; < 0.3 FALSIFIED** |
| 3B-2 additivity | r(Σ_g measured insertion tilts, Stage-2 T_NOV) over the 20 | reported; ≥ 0.9 means block decomposition is valid |
| 3B-3 localization | each group's share of the variance of τ_P96 across the 5000 candidates, plus its share of the measured insertion variance | a group is **dominant** if it carries ≥ 50% of both; otherwise "distributed" |

- **Stage 4** (representation-level causal test) is proposed only if 3B-1 PASSES. It targets the dominant group, or
  the top two if the tilt is distributed. It would need its own lock and approval.
- If 3B-1 is falsified, the tilt is first-order predictable as a whole but not decomposable by blocks. Report that,
  and stop mechanistic GPU work.

## Budget and order
- **Order:** 3A gradients (P96 first, then R, then S) and projections → 3B insertions in frozen order.
- **Estimate:** 3A ≈ $1.5, 3B ≈ $3, setup ≈ $0.4.
- **Guards:** a hard pod guard at the approved cap; an in-script guard skips remaining 3B perturbations; partial
  results are kept. 3B needs ≥ 10 complete perturbations for a verdict, else INCONCLUSIVE.

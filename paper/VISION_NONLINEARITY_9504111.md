# Does the nonlinear vision response explain 9504111's RERANK→TEST mirage?

- **Status:** post-hoc, CPU-only, existing artifacts only. Hypothesis-generating. H★ stays falsified (GPU-A, A1 lock:
  F1 r = 0.137, S2d r = 0.239) and is not reinterpreted here.
- **Code:** `scripts/vision_nonlinearity_analysis.py`. **Outputs:** `results/paper-analysis/vision-nonlinearity/`
  (`vision_nonlinearity_results.json`, `loc_decomposition_per_example.csv`,
  `whole_model_linearization_winner_vs_controls.csv`, `gain_decomposition_test_measured_candidates.csv`).
- **Answer: NO.** Vision is strongly nonlinear in its own parameters, but it does not carry the RERANK/TEST asymmetry,
  is no more context-dependent than the language layers, and cannot be compared with controls because no control has
  group-level measurements. **Classification: INTERESTING NONLINEARITY, BUT NOT THE WHY. Recommendation: STOP
  MECHANISTIC GPU WORK.**

## 1. What GPU-A falsified

H★: the candidate's per-example score change equals its whole-model first-order term σ⟨fold(∇s_j), ε⟩, so search
selects ε aligned with the RERANK folded gradient and transfer is set by cross-set gradient cosines. At σ = 0.002 the
whole-model first-order term does not predict the measured change (F1 pooled r = 0.137, slope 0.027, 13 candidates ×
761 examples), and the group partials do not predict the measured group insertions (S2d pooled r = 0.239). The
"implicit gradient step" reading is ruled out for this candidate at this radius. The tests that did not falsify
(F2 0.53, F3-RERANK r = 0.38) are weak and do not rescue it.

## 2. What GPU-A discovered

**Established (measured, reproducible from the artifacts):**
- The engineering is sound: candidate bytes are exact, and the checkpointed forward reproduces the stored contrast
  to ≤ 7.6e-6.
- Language-layer groups respond close to linearly at σ = 0.002: r = 0.71–0.96, slope 0.82–0.99 (lm_q1–q4); final
  norm + head r = 0.86, slope 1.03.
- The vision group does not. Its first-order term has sd 7.17 nats against a measured sd of 1.53, and r = 0.05.
  The nonlinear residual (measured − linear) has sd 7.26, which is essentially the linear term with its sign flipped.
- The seven measured group insertions add up in score space: sum vs candidate r = 0.93, slope 0.99, 86% of energy
  explained (as previously reported in margin-additivity).
- Set folded gradients are nearly orthogonal (RERANK·TEST cosine 0.031).

**Not established:** any causal role for vision in the mirage, or why vision linearization fails (for example
saturation, normalization, the patch-merger, or DeepStack).

## 3. Does nonlinear vision explain the mirage?

**NO.** Four independent checks:

1. **Vision does not carry the phase asymmetry.** Within each transition cell, measured vision pushes in the same
   direction, with similar size, on both sets: repairs RERANK +1.10 vs TEST +0.50; regressions −0.63 vs −0.76 nats.
   The language-layer sum is the larger driver in both phases: repairs +3.30 vs +1.83, regressions −1.25 vs −2.00.
   The population-weighted phase gap of the vision component is +0.27 (95% CI −0.76 to 1.60), against +0.36 for LM.
   Both CIs span zero; see the power caveat below.
2. **The nonlinear residual adds nothing beyond the measured vision effect.** In LOO regression the residual model
   (R² 0.77) beats LM-linear alone (0.48), but its fitted coefficients on vision-linear (1.25) and vision-residual
   (1.27) are equal. The model is reconstructing the measured vision insertion, which is mechanical given additivity.
   Phase alone gives LOO R² 0.04, and adding phase to the residual model changes nothing (0.771).
   The residual's correlation with the total is r = 0.26. Its effect sizes are changed vs unchanged d = 0.33 and
   RERANK vs TEST d = 0.13.
3. **Analytical counterfactual** (not causal). Removing the measured vision component leaves the contrast-sign phase
   gap essentially unchanged: 16.7 pp vs 16.8 pp. Removing the LM component collapses it to 2.1 pp. Vision alone
   gives a gap in the opposite direction (−5.0 pp).
4. **Vision is not unusually context-dependent.** Its insertion effect (base context) matches its removal effect
   (candidate context) with r = 0.78, the same as the LM quarters (0.80–0.86). Its context difference shows no phase
   gap (−0.39, inside the LM groups' range of −0.37 to +0.25).

**Power caveat (important).** LOCALIZATION is an 83-example sample stratified by transition cell. It was designed to
localize changed answers, not to estimate phase means: 6 examples stand in for 75–259 unchanged examples per cell.
Reweighted, it gives a RERANK mean shift of +0.02, against a true population mean of +0.33 over all 200 RERANK
examples. Because the cells were fixed by sampling, LOC cannot explain why RERANK has more repairs than regressions
(20:4 vs TEST 23:38). It can only show how each component contributes within a cell, and there vision is
phase-symmetric.

## 4. Quantitative decomposition (83 LOC examples, contrast s = z[gold] − z[r_B], vLLM)

| Predictor of Δ_total | r | slope | resid. SD | explained energy | MAE | changed contrast flips predicted (of 44) | false flips (of 39 unchanged) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Σ all 7 groups | 0.93 | 0.99 | 1.39 | **0.86** | 1.04 | 30 | 2 |
| vision + LM | 0.94 | 1.01 | 1.30 | 0.88 | 0.95 | 35 | 3 |
| LM only | 0.87 | 1.17 | 1.92 | 0.74 | 1.45 | 26 | 3 |
| vision only | 0.55 | 1.35 | 3.22 | 0.28 | 2.36 | 11 | 2 |
| LM-linear + other-linear + measured vision | 0.89 | 1.08 | 1.74 | 0.78 | 1.23 | 30 | 2 |
| all first-order (H★) | 0.12 | 0.06 | 7.68 | −3.12 | 4.66 | 26 | 7 |

Full A–D vectors: the sum gives the candidate's answer on 45/59 changed examples.

The interaction/residual (Δ_total − Σ groups) has sd 1.39 and carries 14% of the energy. It is not systematic:
- phase means −0.19 (RERANK) vs +0.10 (TEST), d = 0.12;
- it is not concentrated at close margins (mean |·| 0.97 for |m_B| < 1 vs 1.08 otherwise);
- Spearman(|res|, |m_B|) = 0.08.

**Reject the interaction hypothesis.**

Share of the population-weighted phase gap (+0.43; CI −1.4 to 2.7): LM 0.82, vision 0.62, other 0.24, interaction
−0.68. These shares are ill-determined because the total gap's CI spans zero.

## 5. Candidate vs controls

- **Vision-specific comparison: NOT AVAILABLE.** No control has group insertions, removals, or first-order group
  partials. Session 1 and the GPU-A group partials cover 9504111 only. Whether the controls' vision is equally
  nonlinear cannot be determined without new GPU work. It is not estimated here.
- **Whole-model linearization** (HF-internal, all 761 examples; 9504111 vs 12 controls):

| Quantity | 9504111 | Controls min / median / max | Winner rank (desc, of 13) |
|---|---:|---|---:|
| r(first-order, measured), RERANK | 0.14 | 0.06 / 0.20 / 0.32 | 9 |
| r(first-order, measured), TEST | 0.07 | 0.06 / 0.13 / 0.32 | 11 |
| SD of nonlinear residual, RERANK | 10.9 | 5.3 / 8.3 / 14.3 | 2 |
| SD of nonlinear residual, TEST | 16.3 | 7.7 / 10.5 / 18.1 | 3 |
| Phase gap of measured shift | 0.57 | −0.40 / 0.25 / 0.62 | 2 |
| Phase gap of first-order term | 1.85 | −1.85 / −0.02 / 0.91 | 1 |
| Phase gap of nonlinear residual | −1.28 | −0.34 / 0.12 / 2.26 | 13 |

Linearization failure is generic: every control also has r ≤ 0.32. The winner is unusual only in that its
**first-order** term favors RERANK (consistent with selection on RERANK: GPU-A winner at the 96.6th percentile of the
RERANK set projection). The nonlinear part cancels most of that tilt, so it opposes the winner's RERANK advantage
rather than producing it. Its measured phase gap (0.57) lies inside the control range.

## 6. Simpler explanations

- **Selection on an evaluation-specific component fits all the data.** Among 543 RERANK candidates the winner is at
  the 99.8th percentile of observed gain. Of its +8.0 pp, about +2.5 pp is linear-predicted (87th percentile) and
  +5.5 pp is the non-first-order remainder (98th percentile). Neither component transfers:
  - across the 61 TEST-measured candidates, RERANK→TEST correlations are −0.15 (observed), −0.14 (linear) and
    −0.21 (nonlinear);
  - within the top 50 they are −0.27 (observed) and −0.06 (nonlinear).

  Whatever made the winner look good on RERANK is not a reproducible property of the perturbation.
- **Base margin / generic flattening do not explain it**, and do not need to. The winner's shift *rises* with base
  margin (slope +0.12, R² 0.06; controls −0.17 to +0.01), so it sharpens rather than flattens. Its margin- and
  correctness-adjusted phase gap (0.67) is rank 1 of 13, but only just above the controls' maximum (0.55). With 12
  controls that is p ≈ 1/13 and does not establish specificity.
- **LM linear shift:** it is the main score-space component of the changed answers in both phases. It too is
  phase-symmetric within cells, so it describes how the change happens, not why RERANK got more repairs.
- **Vision score shift:** weakly related to base margin (r = −0.18); not a better account.
- **Answer position:** prior letter-offset models explain at most 15/59 changed answers (margin-additivity),
  far fewer than the additive decomposition.

The simplest account that fits everything is selection of a high-variance, non-transferring RERANK fluctuation,
consistent with the earlier selection-vs-specificity and random-control reports.

## 7. Scientific conclusion

**INTERESTING NONLINEARITY, BUT NOT THE WHY.**

- *We discovered a nonlinear vision response*: at σ = 0.002 the vision tower's response to a random perturbation is
  far from its first-order term (r = 0.05, first-order sd ≈ 5× measured), while the language layers are near-linear.
  This is shown for one perturbation (9504111) on 83 examples. Its generality across perturbations is untested.
- *We did not explain the expert mirage.* Vision is phase-symmetric within cells, removing it analytically leaves the
  asymmetry intact, its nonlinear residual adds no information beyond its measured effect, there is no systematic
  interaction, and no control comparison exists at the group level.

## 8. Recommendation

**STOP MECHANISTIC GPU WORK.** No strong lead qualifies. The single remaining open question, whether vision
nonlinearity is generic across random perturbations, would characterize the nonlinearity, not explain the mirage. It
should be proposed, if at all, as a separate pre-registered characterization study.

## 9. Conference assessment

Bluntly, the mechanistic path to a strong ML-conference paper is closed for this candidate. The pre-registered
mechanism (H★) failed. The post-hoc clue (vision nonlinearity) does not explain the phenomenon, and everything
remaining is consistent with selection on a non-transferring fluctuation, which the earlier reports had already
suggested.

What remains is an honest empirical/analysis paper:
- the RNG/subspace structural finding;
- the RERANK→TEST failure, with a random-control population and selection analyses;
- a pre-registered, falsified first-order mechanism (a useful negative result);
- the observation that, at σ = 0.002, the language layers respond near-linearly while the vision tower does not.

That can be a solid workshop or analysis-track paper. Calling it a mechanism paper would overclaim.

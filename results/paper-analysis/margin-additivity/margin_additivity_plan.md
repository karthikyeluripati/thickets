# Margin and additivity follow-up: analysis plan lock

**Post-hoc plan lock** (not a preregistration). It was written and committed
before any of the analyses below were computed. The original causal-diagnostic
report and its stopping decision (`DISTRIBUTED_OR_INTERACTING_EFFECTS`; no
targeted route; reserved examples unused) stay unchanged. This is a separate
post-hoc analysis.

## Inputs

- **Branch:** `research/perspective-margin-and-additivity`, from local commit
  `a5fe0ab5860c7abad20f1abd25d093213c5f906d`, which is **not on the remote**.
- **Hashes:** SHA-256 of all 27 input files are in `input_hashes.json`. They
  cover:
  - the example manifest;
  - session-1 BASE and CANDIDATE scores for all 761 examples;
  - the 7 insertion and 7 removal LOCALIZATION outputs, plus 2 controls;
  - the session-1 analysis and per-example CSV;
  - the group inventories and norms;
  - the cost log;
  - the frozen RERANK and TEST JSONL files.
- **Model and candidate:**
  - `Qwen/Qwen3-VL-8B-Instruct` @ `0c351dd01ed8…`;
  - candidate `25a60f0b…b843`, seed 9504111, σ 0.002, all parameters;
  - engine state hashes base `2582817f…` and candidate `7d7ef38b…`, as
    verified in session 1.
- **Scores:** vLLM raw first-token log-probabilities of A, B, C and D (token
  IDs 32–35) at the same prompt prefix. All 4 are present and finite for every
  BASE and CANDIDATE output.
- **Verification already run (counts only):**
  - 200 RERANK and 561 TEST examples, all with 4 options;
  - gold letters match the manifest;
  - RERANK 79 → 95 (20 repairs, 4 regressions), TEST 259 → 244 (23
    repairs, 38 regressions);
  - the generated answer is always among the argmax letters;
  - exact top-score ties occur in 22 BASE and 17 CANDIDATE outputs (BF16
    precision), and the engine's emitted letter is one of the tied letters.

## Definitions

**Notation.**

- L_B[j,a] and L_C[j,a] are the BASE and CANDIDATE log-probabilities of
  option a.
- y[j] is the gold option.
- *Correctness* uses the generated (parsed) answer, as in the study. Score-based
  quantities use L.

**Analysis A, base opportunities** (all 761, BASE scores only):

- BASE correctness.
- Gold rank among A–D: 1 is best. Ties count as the best rank shared, and a
  tie flag is recorded.
- Top-two gap.
- Signed correct-answer margin m_B = L_B[y] − max_{a≠y} L_B[a]. An exact
  m_B = 0 is a tie; it is flagged and reported separately.
- Fixed bins on |m_B| in nats: [0, 0.5), [0.5, 1), [1, 2), [2, 4),
  [4, ∞).
- Reported per phase: counts, ranks, bins, repair rate among BASE-wrong,
  regression rate among BASE-correct, and wrong→wrong changes.

**Analysis B, like-for-like comparison:**

- **Identity:** gain_pp = 100[(1 − b) r − b d] is verified exactly.
- **Strata:** h = BASE correctness × |m_B| bin, so 10 strata. The gold rank
  is a sensitivity check only.
- **Common support:** strata with ≥ 5 examples in *each* phase.
- **Effect:** e_D[h] = mean(candidate correct − base correct).
- **Standardization:** reference weights are the average of the two phases'
  within-common-support stratum proportions.
- **Reported:**
  - the full gap;
  - the common-support gap;
  - the standardized gap;
  - the excluded examples and their outcomes;
  - the symmetric decomposition
    g_T − g_R = 100 Σ (w_T − w_R)(e_T + e_R)/2 + 100 Σ ((w_T + w_R)/2)(e_T − e_R),
    computed on common support with within-support proportions. The two terms
    are the opportunity-composition component and the within-stratum residual.
    These are accounting terms, not causal effects.
- **Uncertainty:** bootstrap, resampling examples within phase, 2,000
  replicates, seed 20261010. It is post hoc and descriptive.

**Analysis C, score movement** (all 761):

- **Fixed comparator:** r_B = argmax_{a≠y} L_B[a]. For a tie among wrong
  options, take the lowest letter.
- **Movement terms:**
  - t = (L_C[y] − L_C[r_B]) − (L_B[y] − L_B[r_B]);
  - s = max_{a≠y} L_C[a] − L_C[r_B] (always ≥ 0);
  - m_C = L_C[y] − max_{a≠y} L_C[a].
- **Identity check:** m_C = m_B + t − s, to floating precision.
- **Reported:**
  - distributions of t and s;
  - the share with t > 0;
  - margin crossings, m_B > 0 → m_C < 0 and m_B < 0 → m_C > 0, with ties
    reported separately;
  - all of these by phase × BASE correctness, and within the Analysis-B
    strata, using the same reference weights for a standardized mean t.
- **Bootstrap:** same unit and seed. Outcome-selected repair and regression
  groups are descriptive follow-ups only.

**Analysis D, parameter-free additive account** (LOCALIZATION, 83 examples,
exploratory):

- **Centering:** z = L − mean_{A–D} L for every condition. This preserves
  contrasts and argmax.
- **Contributions:**
  - insertion u_g = z_{I_g} − z_B;
  - removal-context v_g = z_C − z_{R_g}.
- **Predictor:** z_add = z_B + Σ_{g ∈ 7 groups} u_g, with every coefficient
  exactly 1. The seven groups are vision, embed, lm_q1, lm_q2, lm_q3, lm_q4
  and final_norm_head; they are exhaustive and disjoint.
- **Explicitly excluded:**
  - no fitted weights;
  - no group selection;
  - no per-example normalization by the candidate shift;
  - no separate coefficients by transition;
  - no use of z_C as a predictor.
- **Metrics,** each separately for original repairs, regressions,
  wrong→wrong changes and unchanged controls, plus pooled:
  - per-example L2 error ‖z_add − z_C‖;
  - mean absolute error;
  - pooled relative residual energy Σ‖z_add − z_C‖² / Σ‖z_C − z_B‖²;
  - argmax agreement with the actual candidate answer;
  - correct-answer-margin prediction (MAE and correlation of m_add with m_C);
  - all 6 fixed pairwise contrast changes, predicted vs actual (correlation
    and MAE).
- **Comparators for decoded answers:**
  - predict BASE's answer;
  - the frozen session-1 letter-offset model, whose offsets A +0.3023,
    B −0.6404, C +0.1201, D +0.2180 were fitted on these same LOCALIZATION
    examples (in-sample).
- **Consistency check:** u_g vs v_g, as the per-group correlation and the
  pooled correlation of the stacked vectors.
- **Ties:** argmax ties in z_add are reported, with no tie-breaking in favour
  of agreement.

**Optional reserved-example check** (not run without new explicit approval):

- **Examples:** the same 82 MECHANISM-CHECK examples listed in
  `results/paper-analysis/causal-diagnostic/example_manifest.json`; they are
  not replaced.
- **Conditions:** 7 single-group insertions, plus BASE and CANDIDATE fidelity
  passes.
- **Predictor and metrics:** the same frozen z_add and the same metrics as
  above.

## What this analysis cannot establish

- Which semantic or visual computation changed.
- Why the perturbation population transfers poorly.
- Whether the candidate gained or lost a coherent skill.
- That any group, vision included, is the single cause.

Score identities are exact descriptions of the scores, not models of the
computation. The standardization is limited to *measured* base opportunities.
The within-stratum residual mixes selection, finite-sample variation,
unmeasured item differences and genuine response differences.

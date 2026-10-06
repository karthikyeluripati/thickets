# Why 9504111 helped RERANK but hurt TEST: margin and additivity follow-up

> **POST HOC, CPU only so far. No GPU used.** This is a separate analysis. The
> causal-diagnostic report and its stopping decision
> (`DISTRIBUTED_OR_INTERACTING_EFFECTS`; no targeted route; reserved examples
> unused) are unchanged.

**Provenance:**

- **Branch:** `research/perspective-margin-and-additivity`, from local commit
  `a5fe0ab` (not on the remote).
- **Plan:** locked and committed before any analysis (`5585680`) in
  `results/paper-analysis/margin-additivity/margin_additivity_plan.md`, with
  SHA-256 of all 27 inputs.
- **Code:** `scripts/margin_additivity.py`.
- **Tests:** `tests/test_margin_additivity.py` (6 passing): the margin
  identity, the gain identity, the symmetric decomposition, bins, centering,
  and the parameter-free additive predictor.

## Inputs verified

- **Examples:** 200 RERANK and 561 TEST, all with 4 options. Gold letters match
  the frozen JSONL and the manifest.
- **Scores:** first-token A–D log-probabilities from session 1 for BASE and
  CANDIDATE are all present, finite and taken at the same prompt prefix.
- **Identity:** candidate seed 9504111, σ 0.002, all parameters. The base and
  candidate engine hashes match the forensic ones.
- **Counts:** RERANK 79 → 95 (20 repairs, 4 regressions); TEST 259 → 244
  (23 repairs, 38 regressions).
- **Generated answers versus score argmax:** the generated answer is always
  among the argmax letters. BF16 ties at the top occur in 22 BASE and 17
  CANDIDATE outputs (the emitted letter is one of the tied ones). Correctness
  uses the generated answers.

## Question A: why did the balance differ?

### Base opportunities (all 761; BASE scores only)

| | RERANK | TEST |
|---|---:|---:|
| BASE correct / wrong | 79 / 121 | 259 / 302 |
| Gold rank 1 / 2 / 3 / 4 | 81 / 47 / 38 / 34 | 264 / 214 / 45 / 38 |
| BASE errors within 1 nat of correct | **8.3%** (10) | **13.2%** (40) |
| BASE successes within 1 nat of losing | 19.0% (15) | 15.1% (39) |
| Repair rate among BASE-wrong | 16.5% (20/121) | 7.6% (23/302) |
| Regression rate among BASE-correct | 5.1% (4/79) | 14.7% (38/259) |
| Wrong → different-wrong changes | 12 | 20 |
| Gain, observed = 100[(1 − b)r − b·d] (exact identity) | +8.00 pp | −2.67 pp |

**RERANK does not contain more close, repairable base errors.** TEST has
proportionally *more* base errors near the decision boundary, and far more
with the gold answer in second place (214 vs 47). It also has a similar share
of close base successes. The opportunities, if anything, favour TEST repairs.

### Like-for-like comparison (`fig_standardized_gap`, `fig_opportunity_by_margin`)

**Strata** are BASE correctness × |correct-answer margin| bin (10 strata).
**Common support** (≥ 5 per phase) keeps 9 strata, covering 98% of RERANK and
96.8% of TEST. The excluded stratum is BASE-wrong with |m| in [0.5, 1)
(RERANK 4, TEST 18 examples; net +1 and +5).

| TEST − RERANK gain gap | Value | Bootstrap 95% (examples within phase, 2,000 replicates; post hoc) |
|---|---:|---|
| Observed, all examples | −10.67 pp | [−15.9, −5.4] |
| Within common support | −11.34 pp | [−16.1, −5.6] |
| Standardized to the same stratum weights | **−10.61 pp** | [−14.8, −5.6] |
| Opportunity-composition component | **−0.73 pp** | [−3.3, +1.8] |
| Within-stratum residual | −10.61 pp | [−14.8, −5.6] |

In stratum after stratum, the candidate does better on RERANK than on TEST:

| Stratum | RERANK | TEST |
|---|---:|---:|
| BASE-wrong, margin 1–2: repair rate | 54% (7/13) | 21% (6/28) |
| BASE-correct, margin < 0.5: regression rate | 17% (1/6) | 50% (6/12) |
| BASE-correct, margin 2–4: regression rate | 0/12 | 16% (9/55) |

### Which way did the scores move? (exact identity m_C = m_B + t − s, max error 0)

*t* is the change in the gold-vs-BASE-strongest-wrong contrast. *s* is the
extra gap when a different wrong option becomes strongest. *s* is small (mean
0.24 / 0.10 nats; positive in 14% / 7% of examples), so competitor switches
are rare.

| | RERANK | TEST |
|---|---:|---:|
| Mean t, all examples | +0.33 | −0.30 |
| Mean t, standardized to the same strata | **+0.40** | **−0.32** |
| Standardized TEST − RERANK | **−0.72 nats** [−1.14, −0.28] | |
| BASE-wrong: mean t / P(t > 0) / upward crossings | +0.16 / 45% / 15 | −0.72 / 38% / 17 |
| BASE-correct: mean t / P(t > 0) / downward crossings | +0.58 / 58% / 4 | +0.19 / 55% / 33 |

The gold-rank strata give the same picture as a sensitivity check (BASE
correct; BASE wrong with gold at rank 2; BASE wrong with gold at rank 3+): a
standardized gap of −12.2 pp, with a composition component of +1.6 pp.

**Classification: `SCORE_SHIFT_DIFFERENCES_REMAIN_AFTER_OPPORTUNITY_MATCHING`.**

- **What composition explains:** essentially none of the 10.7-pp difference
  (−0.7 pp, interval spanning 0).
- **What remains:** among base cases matched on correctness and
  correct-answer margin, the candidate's scores moved toward the correct
  answer on RERANK and away from it on TEST.
- **What the residual could be:** selection (RERANK is the set this candidate
  was chosen on), finite-sample variation, unmeasured image or question
  differences (subtask mix and question forms differ), and genuinely different
  perturbation responses. **This analysis cannot separate these.**

## Question B: do the seven group contributions add up?

**Prediction.** The predictor is parameter-free: z_add = z_B + Σ_g (z_{I_g} −
z_B), with all 7 groups and coefficient 1. It is applied to centered A–D
scores of the *existing* LOCALIZATION insertion hybrids (83 examples).
**Exploratory; not reserved-example validation.**

| LOCALIZATION | n | Relative residual energy | Margin r | Pairwise contrast-change r | Candidate answer predicted (sum) | (BASE answer) | (letter-offset model, in-sample) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Repairs | 22 | 0.14 | 0.95 | 0.94 | **16/22** | 0 | 6 |
| Regressions | 21 | 0.16 | 0.99 | 0.92 | **15/21** | 0 | 6 |
| Wrong → wrong | 16 | 0.12 | 0.97 | 0.94 | **14/16** | 0 | 3 |
| All changed | 59 | 0.14 | 0.98 | 0.93 | **45/59** | 0 | 15 |
| Unchanged controls | 24 | 0.18 | 0.99 | 0.91 | 23/24 | 24 | 24 |

By phase, the sum predicts:

| Cell | Predicted |
|---|---:|
| RERANK repairs | 8/10 |
| RERANK regressions | **0/2** |
| TEST repairs | 8/12 |
| TEST regressions | 15/19 |

The sum's argmax produced no ties. The median per-example error is 1.36 nats,
against a median candidate shift of 3.13. The correct-answer-margin MAE is
0.86 nats on changed examples.

**Context consistency.** Each group's insertion effect (in the base context)
matches its removal effect (in the candidate context) with r = 0.75–0.88 per
group; the pooled r is 0.83. The exception is embeddings, whose effects are
tiny (r = 0.20). Interaction is limited at this scale.

**Classification: `ADDITIVITY_IS_ONLY_A_LOCALIZATION_OBSERVATION`.**

- On the localization examples, the candidate's score changes are well
  described by the plain sum of the seven groups' separate effects. That is a
  distributed but approximately additive account, consistent with the earlier
  finding that no single group dominates.
- It has **not** been checked on the 82 reserved examples. Near-additivity is
  also roughly what one would expect for a small perturbation (σ = 0.002,
  relative ‖δ‖/‖W‖ ≈ 0.04–0.09), so it describes how the change combines, not
  what it computes.

## What this does not establish

- Which visual or semantic computation changed.
- Why the population transfers poorly.
- That the candidate gained or lost a coherent skill.
- That vision, or any single group, is the cause.

The residual in Question A has no assigned source.

## Proposed GPU check (NOT run; needs explicit approval)

| Item | Value |
|---|---|
| Examples | The SAME 82 MECHANISM-CHECK examples (`example_manifest.json`), not replaced |
| Conditions | BASE and CANDIDATE fidelity passes, which must equal the stored session-1 text and scores; then 7 single-group insertions. No removals. |
| Engine and checks | Original engine and prompt; exact BF16 copy hybrids with tensor verification; stop on any missing letter score |
| Script | `scripts/reserved_additivity_9504111.py` (prepared) |
| Frozen analysis | `additivity()` in `scripts/margin_additivity.py`, with the same metrics and breakdowns and no retuning |
| Estimate, startup to shutdown | ≈ 10.5 min at the last measured $3.49/h: setup and load 5.4 min, 2 identity hashes ≈ 1 min, 2 fidelity passes ≈ 0.5 min, 7 hybrids ≈ 2 min, retrieval and stop ≈ 1.5 min. **≈ $0.61** |
| Proposed hard cap | **$1.20**, with the pod's guard stopping it at $1.10 and the in-script guard stopping before any step that would exceed it |

## Artifacts (`results/paper-analysis/margin-additivity/`)

| Artifact | Contents |
|---|---|
| `margin_additivity_plan.md`, `input_hashes.json` | Plan lock and input hashes |
| `per_example_margins_all761.csv` | Margins, ranks, t, s and strata for all 761 examples |
| `margin_additivity_results.json` | Opportunity tables, standardization, bootstrap, movement, localization additivity |
| `additivity_localization_per_example.csv` | Per-example additive predictions on LOCALIZATION |
| `paper/figures/margin-additivity/` | `fig_opportunity_by_margin`, `fig_standardized_gap`, `fig_additive_vs_actual` |

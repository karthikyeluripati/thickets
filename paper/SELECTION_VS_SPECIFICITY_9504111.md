# Selection vs candidate-specificity for 9504111: analysis and verdict

> **POST HOC, CPU only, existing predictions.** No GPU used.
>
> - **Branch:** `research/perspective-selection-vs-specificity`, from local
>   `5abe9a4`. The parent branches are not on the remote.
> - **Plan lock:** committed before computation (`faf824a`,
>   `results/paper-analysis/selection-vs-specificity/analysis_plan.md`).
> - **Code:** `scripts/selection_vs_specificity.py`.
> - **Outputs:** `selection_vs_specificity_results.json`,
>   `crossfit_top50_trials.csv`, `test_measured_candidates.csv`,
>   `rerank_reliability_eb.json`.
> - **One addition not in the plan lock:** the empirical-Bayes reliability
>   check (D2). It was added after seeing the cross-fit result, because the
>   per-candidate cross-fit is mild by construction for a dominant candidate.
>   It is reported as a sensitivity check, not a replacement.

## Evidence status

| Evidence | Status |
|---|---|
| SEARCH → top 50 → RERANK winner → TEST chain; 500-candidate audit (RERANK) | **Frozen / precommitted** |
| 12 random-control TEST runs | Frozen sample rule (plan-locked before outputs); post-hoc study |
| Everything below (accounting, cross-fit, shrinkage, conditional comparisons) | **Post hoc, descriptive** |

**Data available:**

- RERANK for 543 candidates;
- TEST for 61 (the SEARCH top 50, plus 11 random controls not in it);
- A–D scores only for 9504111, BASE and the 12 controls.

## Refined accounting (exact; relative to the 12 random controls)

The earlier "winner-specific −9.03 pp" mixes two different things:

| Component | pp |
|---|---:|
| 9504111 gap (TEST − RERANK) | −10.67 |
| Shared phase tilt (control mean gap) | −1.65 |
| RERANK excess over controls (enters the gap with a minus sign) | **+6.92** → −6.92 |
| TEST deficit vs controls | **−2.11** |

Most of the "winner-specific" part is the RERANK *excess*. Only about 2 pp is
a TEST *deficit*. The two parts need different explanations.

## A/B. Selection: does the RERANK excess survive held-out RERANK questions?

**Cross-fit.** This uses the frozen 2,000 RERANK 100/100 partitions with the
original rule (selection-half count, then SEARCH count, then frozen hash),
among the fixed SEARCH top 50. TEST is never used to select. The effective
information is the 200 RERANK questions; the partitions are not independent.

| | Value |
|---|---:|
| P(9504111 selected on a half) | 0.65 (20 distinct winners) |
| 9504111, selection-half gain when selected | +8.82 [5.0, 13.0] |
| **9504111, held-out RERANK gain when selected** | **+7.18** [3.0, 11.0] |
| Implied within-RERANK optimism for 9504111 | ≈ +0.8 pp |
| Any selected winner, held-out RERANK gain | +5.19; uniform member +1.37 |
| TEST gain of the selected candidate (all trials) | −2.06 (when not 9504111: −0.89) |
| Uniform top-50 TEST / random-control TEST | −0.35 / −0.56 |

**D2 sensitivity: population reliability and empirical-Bayes shrinkage.**
Split-half reliability of RERANK gains at 200 questions (Spearman–Brown from
100/100 halves) is low:

| Pool | Reliability |
|---|---:|
| Top 50 | 0.35 |
| All 500 audit | 0.30 |
| σ = 0.002 audit | 0.53 |

Shrinking 9504111's +8.0 toward the top-50 mean gives **+3.7** (optimism
≈ 4.3 pp). With σ = 0.002 audit reliability it gives **+4.5** (optimism
≈ 3.4 pp). Among the 500 audit candidates, the cross-fitted optimism of
whichever candidate is selected averages **+4.7 pp**.

**Reading:**

- 9504111's RERANK advantage is not pure noise. Under every estimate its
  RERANK-pool gain stays above the selected-group and control means (+1.4 /
  +1.1).
- Its size is not identified. The candidate's own split halves suggest about
  +7; population reliability suggests about +4.
- Selection optimism is therefore somewhere between about 0.8 and 4.3 pp of
  the 6.9-pp excess. Both "selection" and "a real RERANK-pool advantage"
  contribute, in unidentified proportions.

## C. Candidate-specificity: among candidates with similar RERANK gains, is 9504111's TEST unusually bad?

**High-RERANK group** (g_R ≥ +4 pp, threshold fixed in the plan), all 6
TEST-measured candidates:

| Seed | Sample | RERANK | TEST |
|---|---|---:|---:|
| 9504179 | random control | +6.0 | +0.18 |
| 9500931 | top 50 | +5.5 | −0.89 |
| 9504875 | top 50 | +4.5 | −0.71 |
| 9502763 | top 50 | +4.0 | −1.78 |
| 9502699 | top 50 | +4.0 | −0.71 |
| 9500875 | top 50 | +4.0 | −1.07 |
| **9504111** | winner | **+8.0** | **−2.67** |

**Descriptive results:**

- **Rank.** 9504111's TEST is the worst of all 61 TEST-measured candidates and
  below every high-RERANK peer.
- **No regression relationship.** Among the 60 others, TEST gain is unrelated
  to RERANK gain (OLS slope −0.003, r = −0.007, residual SD 0.87). The
  prediction at g_R = 8 is −0.36, an extrapolation (the largest other g_R is
  +6.0). 9504111's residual is −2.31, about 2.6 residual SDs.
- **Question bootstrap** (TEST questions resampled, candidates fixed;
  10,000 replicates):

  | Comparison | Observed | 95% interval |
  |---|---:|---:|
  | 9504111 − 12 controls | −2.11 pp | [−4.28, 0.00] |
  | 9504111 − high-RERANK group | −1.84 pp | [−4.10, +0.42] |

- **Score-change vectors** (only the 12 controls have A–D scores). 9504111's
  per-question shifts resemble the control mean much less than a typical
  control does (cosine about 0.2 vs about 0.5). Its profile differs in kind:
  it protects BASE-correct answers rather than flattening them. These were
  already reported and are not extended; 12 controls give no conditional
  reference for the score vector.

**Reading.** Conditional on high RERANK, 9504111's TEST is consistently the
worst observed. The deficit (about 2 pp) is borderline against
TEST-question sampling noise (the interval touches or crosses 0). The
high-RERANK reference has only 6 candidates. This is suggestive of
candidate-specific behaviour, not conclusive.

## D. Causal diagram

```
random perturbation (seed 9504111, σ = 0.002)
        │  (1)
        ▼
shared phase response: ordinary σ = 0.002 neighbours do ~1.65 pp better on RERANK than TEST
        │  (2)
        ▼
RERANK selection (SEARCH top 50 → RERANK max) ──(3a) finite-sample optimism on the 200 RERANK questions
        │                                        └─(3b) a real advantage on RERANK-pool items, found by selection
        ▼
candidate-specific residual: RERANK-pool advantage + atypical score profile
        │  (4)
        ▼
TEST: −2.67 pp (worst of 61); deficit vs peers ≈ 2 pp
```

| Arrow | Supported by existing evidence? |
|---|---|
| (1) The perturbation family has a shared RERANK > TEST tilt | **Supported.** 10 of 12 random controls; mean −1.65 pp (bootstrap over candidates −2.7 to −0.7); the selected top 50 agree (−1.72). |
| (2) That tilt accounts for the bulk of the winner's gap | **Not supported.** It accounts for about −1.65 of −10.67. |
| (3a) Finite-sample selection optimism on RERANK | **Supported to exist; size not identified.** About 0.8 pp (own cross-fit) to about 4.3 pp (population shrinkage). |
| (3b) A real RERANK-pool advantage specific to this candidate | **Supported to exist; size not identified.** A positive held-out RERANK gain under every estimate (about +3.7 to +7.2, against a group mean of +1.4). |
| (4) A candidate-specific TEST deficit beyond the shared tilt and similar-RERANK peers | **Suggestive only.** Worst TEST of 61, −2.6 residual SD, but the question-bootstrap intervals touch or cross 0 and there are only 6 peers. |
| Why (3b) fails to transfer: which computation | **Unidentified** (out of scope). |

## E. Verdict

**`BOTH`**

- **Selection** contributes: within-RERANK optimism is non-zero under every
  estimate, and the winner was chosen as the maximum of noisy, low-reliability
  RERANK scores.
- **Candidate-specific behaviour** contributes:
  - a RERANK-pool advantage that survives held-out RERANK questions under
    every estimate;
  - an atypical score-change profile;
  - the worst TEST of all 61 measured candidates.
- **The proportions are not identified.** The selection-optimism share of the
  6.9-pp RERANK excess lies between about 12% and 62%, depending on whether
  the candidate's own split-half consistency or population reliability is
  trusted. The 2-pp TEST deficit is borderline against question noise.

## Would another experiment materially improve the workshop paper?

**Recommendation: stop.**

The paper's supported claims do not depend on resolving this split. Those
claims are:

- selection optimism;
- non-transfer to TEST;
- a shared phase tilt plus a winner-specific reversal.

Of the two unresolved quantities:

- **The size of the RERANK-pool advantage, (3a) vs (3b), cannot be resolved by
  any small run on existing questions.** It needs *fresh* RERANK-like
  questions, which means building a new evaluation set from the official
  train split. The Allocentric follow-up showed such fresh items are scarce,
  and this is outside the "no new splits" constraint.
- **Whether the TEST deficit is specific to 9504111 (arrow 4)** is the only
  question a small run could sharpen.

**The single experiment that directly tests arrow (4), if wanted:**

| Item | Value |
|---|---|
| **Hypothesis** | Candidates chosen *only* for extreme RERANK gains (not by SEARCH) also show TEST failures of 9504111's size, which would make its TEST deficit a pool-selection effect rather than candidate-specific. |
| **Candidates** | Fixed RERANK-only rule (pre-specified in the plan): all audit candidates with RERANK gain ≥ +5 pp not already TEST-measured. Six, all σ = 0.002: seeds **9500991 (+7.5, the closest analogue to the winner)**, 9502643 (+6.0), 9500487 (+5.5), 9500579 (+5.5), 9501527 (+5.0), 9502155 (+5.0). Full IDs and expected `candidate_state_id`s are in `selection_vs_specificity_results.json` → `gpu_proposal_candidates`. |
| **Questions** | TEST561 only. RERANK outputs are already stored and used as the fidelity check. |
| **Conditions** | The same engine, fingerprints, prompt, parser and A–D scores as the random-control run. |
| **Cost** | Measured basis ≈ 3.2 min per candidate for TEST plus apply and fingerprint, setup ≈ 6 min, base TEST fidelity ≈ 2.6 min, stop ≈ 1.5 min: about 29 min ≈ **$1.70** at $3.49/h. **Hard cap $2.50.** |
| **Decision rule** | If these 6 have TEST gains near the control mean (≈ −0.5), arrow (4) strengthens. If several fall near −2.5, the TEST deficit is a pool-selection phenomenon and the verdict moves toward selection. |
| **Value** | Footnote-level refinement for one candidate. **Not required** for the workshop paper. |

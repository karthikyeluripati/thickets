# RESULTS_MASTER: the only numbers the paper may use

Every number below is reproduced from a committed result file. Pre-registered tests cite their lock commit; post-hoc
analyses are labelled **POST-HOC**. If a number is not here, it must not appear in the paper.

Abbreviations:
- **SC@K:** self-consistency, majority vote over K samples of the unperturbed base model.
- **NOV:** perturbation with vision tensors reset to base.
- **LM-only:** RandOpt PERTURB_VISUAL=0.

---

## R1. The motivating case: a "mirage" that was partly an evaluation artifact
**Source:** `results/paper-analysis/paper_master_predictions.parquet`, `paper/CORRECTION_SPLIT_MISMATCH.md`.

**Setup.** A precommitted N = 5000 RandOpt-style search on Qwen3-VL-8B-Instruct (all parameters incl. vision; σ
mixture {0.00025, 0.0005, 0.001, 0.002}), OmniSpatial Perspective_Taking, direct-letter prompt. Stages:
SEARCH200 → RERANK200 (543 candidates) → official TEST561.

**The winner, seed 9504111 (σ = 0.002):**
| Stage | Base → winner | Gain |
|---|---|---|
| SEARCH | 72 → 77 | +2.5 pp |
| RERANK | 79 → 95 | +8.0 pp (20 repairs / 4 regressions) |
| official TEST | 259 → 244 | −2.67 pp |

**The split mismatch (the format artifact):**
| | SEARCH / RERANK (OmniSpatial-train) | TEST (official test) |
|---|---|---|
| 8-way compass-format items | 185 / 189 of 200 | **2 of 561** |
| gold answer contains "front" | 94 / 92 | 16 |

The winner's RERANK gain lies entirely on compass items (+8.5 pp; 19 repairs / 3 regressions of 189).

## R2. Matched-split transfer on fresh same-format data (pre-registered, lock 5eb864b)
**Source:** `results/paper-analysis/m1/M1_C1_RESULT.md`, `m1/m1_results.json`.

**Setup.** H = 600 fresh compass-format OmniSpatial-train items (484 images; no image shared with
SEARCH/RERANK/TEST; 0 leaks). Base accuracy 34.7%. 59 of 61 frozen candidates measured: the winner, 12 random
controls, 46 other SEARCH-top-50 members.

| Pre-registered test | Result |
|---|---|
| Winner's gain on H | **+2.67 pp**, 95% image-cluster CI [+0.17, +4.93] (TRANSFERS, not FULL); rank 1 of 59 |
| r(RERANK gain, H gain) | 0.56 [0.36, 0.72] |
| SEARCH-top-50 mean minus control mean on H | +1.16 pp [+0.31, +2.03] |

**Selection optimism:** +8.5 (RERANK compass) → +2.7 (fresh). An empirical-Bayes shrinkage computed beforehand
predicted +3.7 (`selection-vs-specificity/rerank_reliability_eb.json`).

**POST-HOC (CPU), RandOpt-style votes on H** (`m1/M1_VOTE_POSTHOC.md`):
| Ensemble | Gain on H | 95% CI |
|---|---|---|
| **Top-50 majority vote** | **+0.2 pp** | [−1.0, +1.3] |
| Top-10 by RERANK | +2.8 | [+1.0, +4.7] |
| Top-10 by SEARCH | +2.0 | [+0.2, +4.0] |
| 12 random controls | −0.3 | [−2.2, +1.5] |

## R3. What selection finds in the direct-answer regime: answer-prior tilts
**Source:** `results/paper-analysis/answer-prior/*.json`, `paper/WHY_INVESTIGATION_9504111.md`. **POST-HOC.**

**Across candidates:**
- Answer-content shifts (option text containing front/back/left/right) explain gain variance with 5-fold CV
  R² = 0.32 over all 5000 SEARCH candidates, and 0.42 over the 543 RERANK candidates.
- The shift is a property of the perturbation: SEARCH vs RERANK "front" shift r = 0.62 over 543 candidates
  (independent items). The shift-explained gain replicates (r = 0.57); the residual gain does not (r = 0.10).

**The winner:**
- "Front" answer shift: +6.5 pp on SEARCH (99.2nd percentile), +10.0 pp on RERANK (96.6th).
- Score-level tilt toward front options +0.89 nats (t = 5.7), including on items whose gold is not front.
- Net +13 answer switches *into* front options where front is wrong (99.5th percentile).
- Gains concentrated on gold-front items: SEARCH +7.4 pp (99.3rd percentile), RERANK +15.2 pp (15 repairs / 1
  regression). Non-front items: −1.9 / +1.9 pp.

**Calibration (pre-registered C1, lock 5aa4c06):** removing a 4-parameter answer-content shift leaves the winner's
fresh gain at +2.33 of +2.67 pp (NOT EXPLAINED). The shift explains 9% of its score-change variance.

## R4. The first-order tilt law (pre-registered)
**Definitions.**
- Tilt T = mean over items of Δ[mean centred log-prob of options containing a content word − mean of the others].
- Prediction τ = σ·⟨fold(∇T̄), ε⟩ with **no fitted parameters**. ε is the perturbation's Gaussian stream; fold() maps
  every parameter tensor onto RandOpt's shared stream.
- All perturbation constructions were verified byte-exact against the vLLM/RandOpt realization.

| Test (lock) | Model | Task / content word | n | **r(pred, measured)** | Other |
|---|---|---|---|---|---|
| Stage 2 (09c1f1f) | Qwen3-VL-8B | Perspective / "front" | 80 perturbations × 96 items | **0.938** | slope 0.83; reliability 0.88; 99% of variance is content, not letter position |
| R1 (00155cc) | Qwen3-VL-8B | Complex Logic / "left" | 80 new × 98 | **0.915** | position share 0.18 |
| R2 (47d2102) | Qwen2.5-VL-7B | Perspective / "front" | 80 × 96 | 0.920 | **INCONCLUSIVE**: reliability gate 0.52 < 0.7 |
| R2b (1cb3252) | Qwen2.5-VL-7B | Perspective / "front" | 80 new × 363 | **0.925** | reliability 0.949; position share 0.07 |
| I5 per item (a1c2a7a) | Qwen3-VL-8B | Perspective / "front" | 7680 item × perturbation pairs | **0.771** pooled | per-item median r 0.88; 95/96 items ≥ 0.6 |
| P0 (d5152f7) | Qwen2.5-VL-3B, LM-only (RandOpt GQA setting) | GQA per-question answer contrast | 96 × 16 at σ ≤ 0.002 | **0.930** | σ = 0.001: 0.977; 0.002: 0.915; **σ = 0.005: 0.31 (breaks)** |

**Boundary (vision is nonlinear).**
- First-order prediction of whole-model, per-example gold-vs-wrong contrasts with vision perturbed was **falsified**
  (GPU-A A1, lock 79252d4: pooled r = 0.137).
- Per-group: vision r = 0.05 (first-order sd 7.1 vs measured 1.5 nats); language groups r = 0.71–0.96, slopes
  0.82–1.03.
- Adding vision's first-order term lowers the full-perturbation fit from 0.84 to 0.53 (Stage 2).

## R5. Localization (pre-registered Stage 3B, lock ce60c5d)
- **Block-level law (Qwen3-VL-8B):** 20 perturbations × 8 language blocks, exact single-block insertions. Pooled
  r = **0.978**, slope 0.95; additivity r = 0.96.
- **Share of tilt variance by block (predicted / measured):**

  | Block | embed | L0–5 | **L6–11** | **L12–17** | **L18–23** | L24–29 | L30–35 | head |
  |---|---|---|---|---|---|---|---|---|
  | Predicted | 0.00 | 0.04 | **0.44** | **0.22** | **0.23** | 0.03 | 0.04 | 0.00 |
  | Measured | 0.00 | 0.05 | **0.41** | **0.22** | **0.28** | 0.01 | 0.04 | 0.00 |
- **Replications:** layers 6–23 carry 0.73 of predicted variance in R1. In R2b (28-layer model), layers 4–19 carry
  0.77, peaking at layers 12–15.

## R6. The noise-only predictor and search (pre-registered Stage 3A, lock ce60c5d; plus POST-HOC τ-check)
- Cross-set predicted tilt (gradients from items disjoint from the outcome set) vs accuracy gain:
  - RERANK, 543 candidates: **r = 0.40 (PASS)**;
  - SEARCH, 5000 candidates: r = 0.24 (INCONCLUSIVE; bar 0.25); within σ = 0.002, 0.31.
- The winner's predicted tilt is at the 99.8th percentile of 5000.
- **On fresh matched data (M1-3):** r(τ, fresh gain) = 0.61 [0.42, 0.75].
- **POST-HOC τ-check** (criteria committed before computing):
  - holds within the top-50 (r = 0.59) and after partialling selection gains (0.31);
  - but works entirely through the front tilt: r(τ, gold-front gain) = +0.71, r(τ, non-front gain) = −0.42;
  - so it predicts **label-prior exploitation**, not general improvement.

## R7. The chain-of-thought regime (RandOpt GQA setting: Qwen2.5-VL-3B, LM-only, CoT + \boxed, RandOpt scorer)
**P0 (lock d5152f7):**
- Base CoT accuracy 56.25%; RandOpt reports 56.6%.
- At σ ≤ 0.002, **22% of item × candidate pairs flip CoT correctness**, unrelated to the first-order contrast
  (r ≈ 0.0; sign agreement 0.52).
- σ = 0.005 is destructive: per-candidate CoT change −6 to −56 pp.

**P1 (lock 464aed0), 64 perturbations; base sampled at T ∈ {0.3, 0.5, 0.7, 1.0}, n = 32:**
- **Same items flip:** Spearman of per-item flip propensity, perturbations vs sampling (T\* = 0.3) = **0.946**.
- **Perturbed models are persistent:** r(selection gain, held-out gain) = 0.74 [0.61, 0.84]; pseudo-models from
  samples give 0.10.
  - POST-HOC: much of this is format damage. The boxed-answer rate falls from 90.5% to as low as 50.7%.
  - The top-8 have held-out gains of 0 to +7, mean +2.9 pp.
- **Vote vs SC** (held-out 200, base 56.0): RandOpt-style top-8 vote 62.0; SC@8 (T = 0.7) 63.0; SC@8 (T\*) 59.5;
  random-8 perturbation vote 60.1. **NO ADVANTAGE** (CI of the difference vs SC@T\*: [−3.5, +8.5]; underpowered).

## R8. Self-consistency vs the paper's reported RandOpt (P2 lock 01efd9d; P3 lock c60b600)
**Protocol.** RandOpt's own prompts and scorers. SC@50 at T = 0.7 (top_p 1). The gate is: our greedy base within
±2.0 pp of the paper's Base (GQA ±2.5), else NOT COMPARABLE. **MATCHES** if SC@50 ≥ RandOpt − 1.0.

| Row | our greedy / paper Base | gate | **SC@50** [95% CI] | paper TT-MV | **paper RandOpt** | verdict | SC gain vs RandOpt gain |
|---|---|---|---|---|---|---|---|
| GSM8K, Qwen2.5-1.5B | 59.3 / 58.8 | pass | **79.8** [77.6, 82.0] | 69.1 | 76.4 | **MATCHES** | +20.5 vs +17.6 |
| GSM8K, Qwen2.5-3B | 80.3 / 79.8 | pass | **88.2** [86.5, 90.1] | 82.5 | 87.1 | **MATCHES** | +8.0 vs +7.3 |
| GSM8K, Qwen2.5-0.5B | 43.2 / 39.9 | fail | 56.0 [53.1, 58.6] | 41.0 | 54.1 | not comparable | +12.7 vs +14.2 |
| MATH-500, Qwen2.5-1.5B | 49.6 / 43.2 | fail | 62.4 [58.2, 66.6] | 50.0 | 59.7 | not comparable | +12.8 vs +16.5 |
| MATH-500, Qwen2.5-3B | 64.4 / 58.6 | fail | 73.8 [70.0, 77.6] | 60.8 | 68.7 | not comparable | +9.4 vs +10.1 |
| **GQA, Qwen2.5-VL-3B** (2000 of testdev) | 54.0 / 56.6 | fail | 59.2 [57.1, 61.4] | – | **69.0** | not comparable | **+5.2 vs +12.4** |

**SC@K curves (K = 1, 5, 10, 20, 50):**
- GSM8K-1.5B: 57.5, 69.9, 75.2, 78.7, 79.8
- GSM8K-3B: 78.7, 85.3, 87.0, 88.0, 88.2
- GQA: 47.7, 54.6, 57.7, 57.8, 59.2

At T = 0.3, SC@50 on GSM8K-1.5B is 73.3, close to the paper's TT-MV of 69.1. The paper's TT-MV is defined as a vote
"over test samples… with different seeds", with the temperature unstated.

## R9. Pre-registration ledger (report every test, including failures)
**Passed:**
- Stage 2;
- Stage 3B;
- R1;
- R2b;
- I5;
- P0 primary;
- M1-1, M1-3, M1-4;
- P1-A;
- P2/P3 GSM8K 1.5B and 3B.

**Failed or falsified:**
- GPU-A H★ F1 and S2d (whole-model first-order with vision);
- run-1 S2a (BF16 resolution; fixed by amendment A1);
- Stage 3A-1 (SEARCH r = 0.24 vs bar 0.25);
- R2 (reliability gate; remedied by R2b with a pre-specified fix);
- M1-2 (front concentration, not significant);
- C1-1 (not explained);
- P1-B (persistent, so not pure re-sampling);
- gates for 4 of 6 P2/P3 rows.

**Engineering failures that produced no data:** P0 attempt 1 (offline-mode bug); P2 attempt 1 (package conflict).

## R10. Prior-work overlap (must be cited and credited)
- Format effects: the original paper (§8, "format thickets": GSM8K 19.0% format vs 12.3% reasoning).
- Selection bias / overfitting to the selection set: arXiv 2608.10867 (Bayesian optimization in Neural Thickets).
- Informal geometric intuition: the blog "A Thicket by Any Other Name".
- Sequence-length dependence: Ziming Liu's blog "When does RandOpt work?".

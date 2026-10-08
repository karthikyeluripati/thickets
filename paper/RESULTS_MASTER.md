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

**EXPLORATORY (CPU), RandOpt-style votes on H** (`m1/M1_VOTE_POSTHOC.md`):
| Ensemble | Gain on H | 95% CI |
|---|---|---|
| **Vote of the 47 measured SEARCH-top-50 members** (the winner + 46 others; 2 frozen top-50 members not measured under the budget cap) | **+0.2 pp** | [−1.0, +1.3] |
| Top-10 by RERANK | +2.8 | [+1.0, +4.7] |
| Top-10 by SEARCH | +2.0 | [+0.2, +4.0] |
| 12 random controls | −0.3 | [−2.2, +1.5] |

- Mean single-member gain on H: top-50 +0.28 pp; controls −0.88 pp.
- **What this does and does not show.** The 47-model vote shows little measured ensemble gain. It does **not** establish
  a cancellation mechanism.
- **A descriptive reading consistent with the numbers:** the transferable gain sits in a few top candidates (the
  winner +2.67; top-10 votes +2.0 to +2.8). The other members average close to zero, so a plurality vote over many of
  them is dominated by near-base answers.

## R3. What selection favours in the direct-answer regime: answer-preference tilts (an incomplete explanation)
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

**Calibration (pre-registered C1, lock 5aa4c06): the key boundary result.** Removing a cross-fitted 4-parameter
answer-content shift leaves **+2.33 of the winner's +2.67 pp** fresh gain (NOT EXPLAINED). The shift explains 9% of
its score-change variance. The answer-preference tilt is therefore favoured by selection but does **not** account for
most of the winner's transferable gain.

## R4. The first-order tilt law (pre-registered)
**Definitions.**
- Tilt T = mean over items of Δ[mean centred log-prob of options containing a content word − mean of the others].
- Prediction τ = σ·⟨fold(∇T̄), ε⟩, a **first-order approximation** that needs the model-and-task gradient at the base
  model plus the perturbation's noise. It has **no fitted coefficients**. ε is the perturbation's Gaussian stream;
  fold() maps every parameter tensor onto RandOpt's shared stream.
- The prediction is highly correlated with the measurements but **not exact**: slopes are 0.83–0.95, and pooled
  per-item r is 0.77.
- All perturbation constructions were verified byte-exact against the vLLM/RandOpt realization.

| Test (lock) | Model | Task / content word | n | **r(pred, measured)** | Other |
|---|---|---|---|---|---|
| Stage 2 (09c1f1f) | Qwen3-VL-8B | Perspective / "front" | 80 perturbations × 96 items | **0.938** | slope 0.83; reliability 0.88; 99% of variance is content, not letter position |
| R1 (00155cc) | Qwen3-VL-8B | Complex Logic / "left" | 80 new × 98 | **0.915** | position share 0.18 |
| R2 (47d2102) | Qwen2.5-VL-7B | Perspective / "front" | 80 × 96 | 0.920 | **INCONCLUSIVE**: reliability gate 0.52 < 0.7 |
| R2b (1cb3252) | Qwen2.5-VL-7B | Perspective / "front" | 80 new × 363 | **0.925** | reliability 0.949; position share 0.07 |
| I5 per item (a1c2a7a) | Qwen3-VL-8B | Perspective / "front" | 7680 item × perturbation pairs | **0.771** pooled | per-item median r 0.88; 95/96 items ≥ 0.6 |
| P0 (d5152f7) | Qwen2.5-VL-3B, LM-only (RandOpt GQA setting) | GQA per-question answer contrast | 96 × 16 at σ ≤ 0.002 | **0.930** | σ = 0.001: 0.977; 0.002: 0.915; **σ = 0.005: 0.31 (breaks)** |
| S1-A (797fa96) | **OLMo-2-0425-1B-Instruct** (non-Qwen, text-only) | **ARC-Challenge** per-question answer contrast | 100 × 16 at σ ≤ 0.002 | **0.790** [0.753, 0.824] | σ = 0.001: 0.925; 0.002: 0.752; **σ = 0.005: 0.21 (breaks)**; slope 0.76; base acc 0.47 |
| S1-7B (d64a1ad) | **OLMo-2-1124-7B-Instruct** | ARC-Challenge per-question answer contrast | 100 × 16 at σ ≤ 0.002 | **0.787** [0.758, 0.820] | σ = 0.001: 0.914; 0.002: 0.755; **σ = 0.005: 0.25 (breaks)**; slope 0.77; base acc 0.68 |

**Boundary (vision is nonlinear).**
- First-order prediction of whole-model, per-example gold-vs-wrong contrasts with vision perturbed was **falsified**
  (GPU-A A1, lock 79252d4: pooled r = 0.137).
- Per-group: vision r = 0.05 (first-order sd 7.1 vs measured 1.5 nats); language groups r = 0.71–0.96, slopes
  0.82–1.03.
- Adding vision's first-order term lowers the full-perturbation fit from 0.84 to 0.53 (Stage 2).
- **The selected winner, per item (S1-B, lock 797fa96; 600 fresh matched items, all parameters):** r(pred, Δc) =
  **0.162** [0.067, 0.252] → **NO-GO**; sign agreement on the 52 items whose correctness changed 0.654
  (inconclusive). EXPLORATORY decomposition (pre-declared as descriptive, no rule): the vision part carries 93% of
  the prediction's variance with r = −0.008; the language part alone gives r = 0.621. Source:
  `results/paper-analysis/s1/S1_RESULT.md`.

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
- **Similarly susceptible questions:** Spearman of per-item flip propensity, perturbations vs sampling (T\* = 0.3) =
  **0.946**. Correlated flip propensities do **not** establish equivalent mechanisms. P1-B (next) rejects pure
  re-sampling.
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

**How to state R8:**
- **"MATCHES" is the pre-registered decision label** (SC@50 ≥ RandOpt − 1.0). On the two gated GSM8K rows SC's point
  estimates are **higher** than the published RandOpt numbers (+3.4 and +1.1 pp). This is **not** a same-run
  comparison and **not** a statistical equivalence or superiority test.
- **SC vs the paper's TT-MV on the gated rows:** +10.7 and +5.7 pp (**5.7–10.7 pp**).
- **GQA is an unresolved comparison, not an established counterexample.** The base gate fails there, so the row
  cannot establish a residual RandOpt advantage, but it also prevents any universal conclusion about SC.

## R8b. Same-run RandOpt vs self-consistency at the paper's own settings (lock 1437d44 + amendment 6118222)
**Source:** `results/paper-analysis/c-sameRun/C_RESULT.md`, `c_results.json`.

**Setup.** GSM8K test (1319), Qwen2.5-1.5B-Instruct @ 989aa79. RandOpt arm: RandOpt's own `randopt.py` @ 4000d34
(one logging line added), **N = 5000**, σ ∈ {0.0005, 0.001, 0.002}, 200 selection items, greedy, K = 50 vote (the
paper's settings), 6× H100. SC arm: 50 samples at T = 0.7, same prompt, same vote rule and scorer (RandOpt's).
Validity: recomputed K = 50 vote = randopt.py's printed 1018/1319 exactly.

| | Accuracy |
|---|---|
| Base (randopt.py / greedy) | 60.27 / 59.29 (paper 58.8; gate pass) |
| **RandOpt, N = 5000, K = 50** | **77.18** (paper 76.4; reproduced) |
| **SC@50, T = 0.7** | **79.83** |
| **D = RandOpt − SC (paired, 10,000 bootstrap)** | **−2.65 pp [−4.32, −0.99] → SC AHEAD** (81 SC-only vs 46 RandOpt-only items) |
| Secondary, K = 10 | RandOpt 77.18 vs SC@10 75.21: +1.97 [0.00, +3.87] → no difference detected |

**EXPLORATORY decomposition:** the 50 selected models individually average 64.3% (range 59.4–68.9; base 60.3); their
vote gives 77.2. SC single samples average 57.3; their vote gives 79.8. Vote curves K = 1/5/10/20/50: RandOpt
68.2/74.6/77.2/77.5/77.2; SC 57.5/69.9/75.2/78.7/79.8. Selection: top-50 train reward 0.780 vs population 0.676 and
base 0.730 (200 items).

**Further same-run rows (same protocol):**

| Row (lock) | Base | RandOpt K = 50 (paper) | SC@50 | **D = RandOpt − SC** [paired 95% CI] | Outcome | Members (mean, single model) |
|---|---|---|---|---|---|---|
| GSM8K / Qwen2.5-1.5B (C, 1437d44+6118222) | 60.3 | 77.18 (76.4) | 79.83 | **−2.65 [−4.32, −0.99]** | SC AHEAD | 64.3 |
| GSM8K / Qwen2.5-3B (C3B, e78cfd0) | 80.7 | 86.66 (87.1) | 88.25 | **−1.59 [−2.65, −0.53]** | SC AHEAD | 80.9 |
| GQA / Qwen2.5-VL-3B, 1238 q (G2, d2b3d68+5be1720) | 53.4 | 63.49 (69.0†) | 60.02 | **+3.47 [+1.62, +5.41]** | **RANDOPT AHEAD** | 58.4 |
| GQA, second population seed (G2R, 9e84c4f+0754376) | 53.4 | 63.65 | 60.02 | **+3.63 [+1.78, +5.49]** | **RANDOPT AHEAD (replicated)** | 57.8 |

K = 10: C +1.97 [0.00, 3.87] n.d.; C3B −1.06 [−2.35, 0.15] n.d.; G2 +4.36 [2.26, 6.54] RandOpt ahead; G2R +4.68 [2.50, 6.95].
GQA two-seed mean D (K = 50) = +3.55 [+1.78, +5.37]; seed-to-seed difference +0.16 [−0.97, +1.29].
**G4 (2ad4e46): with a direct-answer prompt, BASE greedy (1 generation) = 64.70 and SC@50 = 64.70 on the same questions;
RandOpt (CoT) − SC@50 (direct) = −1.21 [−2.83, +0.40], no difference detected. The GQA advantage is a prompt effect.**
† The paper's GQA number is on all of testdev with train-split selection; G2 uses image-disjoint testdev splits.
G2 uses a faithful re-implementation of randopt.py (released code cannot pass images); its environment gate
(base 53.39 vs P2 53.55 on the same questions) passed. One test worker OOM'd and was re-run with identical settings.

**EXPLORATORY (G2):** the GQA advantage is on open questions (vote 56.2 vs 51.3; yes/no 76.5 vs 75.6). The base model
often ends without a usable answer (heuristic non-answer rate: base 10.7%, SC samples 10.8%, selected members 5.5%);
on those 133 questions members are correct 43.4% (base 0%), elsewhere members gain +1.3 pp. Selected perturbations
repair answer termination/format, which sampling the base model cannot (cf. the original paper's format analysis).

**G3 (lock dc19fd4, confirmatory follow-up of the G2 exploratory reading):** with a 4× token budget (1024), RandOpt's
GQA advantage shrinks by 0.97 pp [0.32, 1.70] but persists: D_1024 = +2.67 [+0.73, +4.52] → **PARTIAL**. Termination
explains about a quarter of the advantage. Selected models answer in ~66 tokens vs 147 for base. The base model rarely
hits the limit; its failures are mostly missing final boxes. The G2 "termination repair" reading is superseded by G3.

**How to state R8b:** three same-run comparisons at the paper's settings, with RandOpt's published numbers reproduced
on GSM8K. On GSM8K (1.5B, 3B) SC@50 is ahead (CIs exclude 0) and selected members are only 0–4 pp above base, so the
gain is the vote. On GQA RandOpt is ahead (+3.47), and its members are +5 pp above base, mostly by fixing a format /
termination failure (exploratory). RandOpt's 5000 × 200 search generations are not charged to it; one run per row.

## R8c. Theory checks (criteria committed before computing, f06f3c6; status: criteria-first EXPLORATORY)
**Source:** `results/paper-analysis/theory/theory_results.json`, derivations in `paper/THEORY.md`. Data: P0 (96 GQA
questions × 24 perturbations, Qwen2.5-VL-3B, LM-only, direct one-word contrast).

| Check | Statistic | Rule | Result | Outcome |
|---|---|---|---|---|
| T1-a flip probability Φ(−\|c\|/(σ‖g‖)), σ ≤ 0.002 | AUC, 1536 pairs, 46 flips | ≥ 0.80 supports; < 0.65 not | **0.935** | supports |
| T1-b calibration | predicted / observed flips | within [0.67, 1.5] | 65.0 / 46 = 1.41 | calibrated (over-predicts) |
| T2 unselected vote returns base, σ ≤ 0.002 | share of questions, 16 perturbations | ≥ 0.95 supports; < 0.85 not | **96/96 = 1.00** | supports |
| σ = 0.005 (reported) | AUC; ratio; vote = base; mean Δc | – | 0.61; 0.63; 0.979; −0.62 [−1.00, −0.27] | breaks (second-order drift) |
| T4 (descriptive) | CV of ‖g‖; model-implied Spearman, perturbation vs sampling flip prob | – | 0.29; 0.94 | – |

## R9. Confirmatory-test ledger (every locked test, with its outcome)
How to describe the process:
- Confirmatory tests were specified in plan locks committed **before** their corresponding runs.
- Exploratory analyses (marked EXPLORATORY / POST-HOC throughout) were not locked and are identified separately.
- "Lock" = the commit that fixed the rule before the output existed.

| Study | Lock | Test | Statistic | Rule (fixed in the lock) | Result | Outcome |
|---|---|---|---|---|---|---|
| GPU-A run 1 | 198c352 | S2a HF vs vLLM fidelity | median per-item max \|Δlog p\|; argmax agreement | ≤ 0.10; ≥ 99% | 0.25; 98% | **NO-GO** (BF16 output resolution; led to amendment A1) |
| GPU-A A1 | 79252d4 | V0 score-definition gate; S2b candidate bytes | fp32 contrast checks; packed-tensor SHA | as locked | pass; 642/642 | pass |
| GPU-A A1 | 79252d4 | F1 whole-model first-order (vision perturbed) | pooled r, 13 candidates × 761 items | ≥ 0.6 support; < 0.3 falsify | 0.137 | **falsified** |
| GPU-A A1 | 79252d4 | F2 changed answers predicted to flip | share | < 0.5 falsify | 0.53 | not falsified (weak) |
| GPU-A A1 | 79252d4 | F3 RERANK gain prediction | r over 543 | ≥ 0.5 support; < 0.3 falsify | 0.376 | inconclusive |
| GPU-A A1 | 79252d4 | S2d group partials vs insertions | pooled r | < 0.5 falsify | 0.239 | **falsified** |
| Stage 1+2 | 09c1f1f | S1a reliability; S1b fidelity | Spearman–Brown; exactness | ≥ 0.7; all exact | 0.878; exact | pass |
| Stage 1+2 | 09c1f1f | **S2 tilt law** (Qwen3-VL-8B) | r(pred_NOV, T_NOV), 80 perturbations | ≥ 0.6 pass; < 0.3 falsify | 0.938 (slope 0.83) | **pass** |
| Stage 1+2 | 09c1f1f | S2 share carried by non-vision | r(T_NOV, T_FULL)² | ≥ 0.5 | 0.843 | pass |
| Stage 3 | ce60c5d | 3A-0 pipeline consistency | r | ≥ 0.9 | 0.938 | pass |
| Stage 3 | ce60c5d | 3A-1 τ vs SEARCH gain | r over 5000 | ≥ 0.25 pass; < 0.10 falsify | 0.240 | inconclusive |
| Stage 3 | ce60c5d | 3A-2 τ vs RERANK gain | r over 543 | same | 0.402 | pass |
| Stage 3 | ce60c5d | 3A-3 enrichment of advanced candidates | mean z difference | ≥ 0.3 SD | +0.09 [−0.01, 0.18] | not supported |
| Stage 3 | ce60c5d | 3B-1 block-level law | pooled r, 160 pairs | ≥ 0.6 pass; < 0.3 falsify | 0.978 (slope 0.95) | **pass** |
| Stage 3 | ce60c5d | 3B-2 additivity; 3B-3 dominance | r; variance share | ≥ 0.9; ≥ 50% for one block | 0.96; none | valid; distributed |
| R1 | 00155cc | reliability; position share; **law** | SB; share; r | ≥ 0.7; < 0.20; ≥ 0.6 | 0.866; 0.184; **0.915** | **replicated** |
| R1 | 00155cc | middle-layer share | share of predicted variance, layers 6–23 | ≥ 0.5 | 0.73 | supported |
| R2 | 47d2102 | reliability; **law** (Qwen2.5-VL-7B, 96 items) | SB; r | ≥ 0.7; ≥ 0.6 | **0.518**; 0.920 | **inconclusive** (reliability gate failed) |
| R2b | 1cb3252 | reliability; position share; **law** (363 items; remedy fixed in advance) | SB; share; r | ≥ 0.7; < 0.20; ≥ 0.6 | 0.949; 0.073; **0.925** | **replicated** |
| I5 | a1c2a7a | consistency gate; **per-item law** | r; pooled r, 7680 pairs | ≥ 0.99; ≥ 0.6 item-level, < 0.3 average-only | 0.99999; **0.771** | **item-level** |
| M1 | 5eb864b | M1-1 winner on fresh matched items | gain, image-cluster CI | transfers if CI > 0; full if also ≥ +6 | +2.67 [0.17, 4.93] | transfers (not full) |
| M1 | 5eb864b | M1-2 gold-front concentration | gain difference, CI | supports if CI > 0 | +1.95 [−2.57, 6.47] | not supported |
| M1 | 5eb864b | M1-3 τ vs fresh gain | r over 59 (Fisher CI) | ≥ 0.25 supports | 0.61 [0.42, 0.75] | supports |
| M1 | 5eb864b | M1-4 top-50 vs controls | mean difference, bootstrap CI | reported | +1.16 [0.31, 2.03] | reported |
| C1 | 5aa4c06 | **C1-1 prior calibration of the winner** | calibrated vs raw fresh gain | explains if ≤ 0.5×raw with CI; not explained if ≥ 0.8×raw | 2.33 vs 2.67 | **not explained** |
| C1 | 5aa4c06 | C1-2 surviving share (candidates with raw g > 0) | Σg_cal / Σg | ≤ 0.5 supports | ≈ 0.00 [−0.42, 0.32] | locked label "supports"; **flagged**: a cross-fitted exploratory check shows it is regression-to-mean, so it is not evidence |
| τ-check | 8c52ba1 | criteria fixed before computing (exploratory data) | within-top-50 r; partial r; control sign | ≥ 0.30; ≥ 0.20; ≥ 0 | 0.59; 0.31; 0.57 | holds (works through the front tilt) |
| P0 | d5152f7 | **per-question first-order** (LM-only GQA) | pooled r, σ ≤ 0.002 | ≥ 0.5 GO; < 0.3 NO-GO | **0.930** | **GO** (σ = 0.005: 0.31) |
| P1 | 464aed0 | P1-A flip propensities | Spearman over 400 items | ≥ 0.6 supports | 0.946 | supports |
| P1 | 464aed0 | P1-B persistence | r(selection, held-out) over 64 | persistent if ≥ 0.3 with CI > 0 | 0.74 [0.61, 0.84] | **persistent** (rejects pure re-sampling) |
| P1 | 464aed0 | P1-C top-8 vote vs SC at T\* | difference, item bootstrap CI | advantage if CI > 0 | +2.5 [−3.5, 8.5] | no advantage (underpowered) |
| P2 | 01efd9d | base gate; SC@50 vs published RandOpt, GSM8K-1.5B | greedy diff; SC − RandOpt | ±2.0; match if ≥ −1.0 | 0.5; +3.4 | gate pass; MATCHES (label) |
| P2 | 01efd9d | same, GSM8K-0.5B / GQA-VL-3B | greedy diff | ±2.0 / ±2.5 | +3.3 / −2.6 | gate **fail**: not comparable |
| P3 | c60b600 | base gate; SC@50 vs published RandOpt, GSM8K-3B | greedy diff; SC − RandOpt | ±2.0; ≥ −1.0 | 0.5; +1.1 | gate pass; MATCHES (label) |
| P3 | c60b600 | same, MATH-500-1.5B / 3B | greedy diff | ±2.0 | +6.4 / +5.8 | gate **fail**: not comparable |
| S1 | 797fa96 | **A-1 law on a non-Qwen model** (OLMo-2-1B, ARC-Challenge) | pooled r, σ ≤ 0.002, 1600 pairs | ≥ 0.5 GO; < 0.3 NO-GO | **0.790** [0.753, 0.824] | **GO** |
| S1 | 797fa96 | B-1 winner per-item first-order (all parameters) | r over 600 fresh items | ≥ 0.5 GO; < 0.3 NO-GO | 0.162 [0.067, 0.252] | **NO-GO** |
| S1 | 797fa96 | B-2 sign agreement on changed items | share, n = 52 | ≥ 0.70 supported; < 0.60 not | 0.654 | inconclusive |
| C | 1437d44 + 6118222 | validity; base gate | recomputed = printed; ±2.0 | exact; required | 1018 = 1018; +1.5 / +0.5 | valid; pass |
| C | 1437d44 + 6118222 | **C-1 same-run RandOpt (N = 5000, K = 50) − SC@50** | paired item bootstrap | CI > 0 RandOpt ahead; CI < 0 SC ahead | **−2.65 [−4.32, −0.99]** | **SC AHEAD** |
| C | 1437d44 + 6118222 | K = 10 (secondary) | same | same | +1.97 [0.00, 3.87] | no difference detected |
| C3B | e78cfd0 | validity; base gate | recomputed = printed; ±2.0 | required | 1143 = 1143; +0.9 / +0.5 | valid; pass |
| C3B | e78cfd0 | **C3B-1 same-run RandOpt − SC@50, GSM8K-3B** | paired item bootstrap | as C-1 | **−1.59 [−2.65, −0.53]** | **SC AHEAD** |
| C3B | e78cfd0 | K = 10 | same | same | −1.06 [−2.35, 0.15] | no difference detected |
| G2 | d2b3d68 + 5be1720 | environment gate | base vs P2 greedy, same items | ≤ 1.0 pp | 0.16 | valid |
| G2 | d2b3d68 + 5be1720 | **G2-1 same-run RandOpt − SC@50, GQA** | paired item bootstrap | as C-1 | **+3.47 [+1.62, +5.41]** | **RANDOPT AHEAD** |
| G2 | d2b3d68 + 5be1720 | K = 10 | same | same | +4.36 [+2.26, +6.54] | RandOpt ahead |
| G2R | 9e84c4f + 0754376 | **G2R-1 GQA advantage with a disjoint population (seed 43)** | paired item bootstrap | REPLICATED if CI > 0 | **+3.63 [+1.78, +5.49]**; D₂ − D₁ +0.16 [−0.97, +1.29] | **REPLICATED** |
| G4 | 2ad4e46 | **G4-1 RandOpt (CoT) − SC@50 with a direct-answer prompt, GQA** | paired item bootstrap | as C-1, + equivalence ±2 | **−1.21 [−2.83, +0.40]** | **NO DIFFERENCE DETECTED** (not equivalent) |
| G4 | 2ad4e46 | direct-prompt BASE − CoT BASE; search on top of prompt | paired bootstrap | reported | +11.31 [8.48, 14.14]; −0.08 [−0.97, +0.81] | prompt effect; search adds nothing |
| G3 | dc19fd4 | **G3-1 does a 1024-token budget remove the GQA advantage?** | Δ = D256 − D1024; D1024 | supported if Δ CI > 0 and D1024 CI ∋ 0 | Δ +0.97 [0.32, 1.70]; D1024 +2.67 [0.73, 4.52] | **PARTIAL** |
| G3 | dc19fd4 | G3-2 base vs members non-termination at 256 | difference, CI | supported if CI > 0 | +1.90 [0.31, 3.48] | supported (small) |
| S1-7B | d64a1ad | **7B-1 law at 7B** (OLMo-2-7B, ARC) | pooled r, σ ≤ 0.002 | ≥ 0.5 GO; < 0.3 NO-GO | **0.787** [0.758, 0.820] | **GO** |

**Engineering failures that produced no data:** P0 attempt 1 (offline-mode bug); P2 attempt 1 (package conflict);
S1 smoke attempt 1 (tar ownership); C on 1× H100 (RandOpt arm skipped by the locked budget rule, `N_DOES_NOT_FIT`).
The earlier OmniSpatial studies (random-control transfer, selection-vs-specificity, margin-additivity, causal
diagnostic) have their own locks. Their TEST-based conclusions are superseded by `paper/CORRECTION_SPLIT_MISMATCH.md`.

## R10. Prior-work overlap (must be cited and credited)
- Format effects: the original paper (§8, "format thickets": GSM8K 19.0% format vs 12.3% reasoning).
- Selection bias / overfitting to the selection set: arXiv 2608.10867 (Bayesian optimization in Neural Thickets).
- Informal geometric intuition: the blog "A Thicket by Any Other Name".
- Sequence-length dependence: Ziming Liu's blog "When does RandOpt work?".

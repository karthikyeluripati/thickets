# Why do some random directions win RERANK and fail TEST? Design pass and CPU results

> **Note (2026-10-07 cleanup):** earlier study write-ups referenced below were removed from the working tree because
> they are superseded or outside the paper's story. They remain available at git tag `pre-cleanup`
> (`git show pre-cleanup:paper/<FILE>.md`). Their raw data, where the paper still uses it, is kept under `results/`.

- **Status:** post-hoc, CPU-only, existing artifacts. Nothing here is pre-registered. H★ stays falsified, and vision
  nonlinearity is "not the why" (`VISION_NONLINEARITY_9504111.md`).
- **Code:** `scripts/answer_prior_analysis.py`, `scripts/answer_prior_checks.py`, `scripts/answer_prior_bias_model.py`,
  `scripts/answer_prior_goldsplit.py`. **Outputs:** `results/paper-analysis/answer-prior/`.
- **Feature choice disclosure:** the option-content features (front/back/left/right/"can not determine") were defined
  from the option vocabulary. "front" came to the fore after looking at gold composition (one look, no other features
  tried). The tests that do not depend on that choice are the cross-candidate reproducibility, the gold-independent
  switching, and the score-level bias.

## 0. A fact that changes the framing

SEARCH and RERANK come from OmniSpatial-train and have identical structure: the same gold-letter counts (B 56,
A 51, D 47, C 46) and the same subtask counts (119/65/16). **95% of their items use the 8-way compass option format**
("front-left", "back", …). TEST comes from the test split and **contains 2 compass items out of 561**. Its formats are
left/right/"can not determine" (266), counting (159) and other (134). Its subtask mix is also different.

The winner's RERANK gain lies entirely in compass items: 19 repairs, 3 regressions in 189 (+8.5 pp). RERANK→TEST
is therefore not a same-distribution transfer test. **TEST cannot evaluate the effect the winner was selected for.**
Earlier reports attributed "under half" of the collapse to subtask mix; the option format is the larger difference
and was not analysed before.

## 1. Competing hypotheses (ranked by plausibility after the CPU tests)

| # | Mechanism | Predicted signature | Why earlier work did not rule it out | Cheapest distinguishing test | Falsifier | Status after CPU tests |
|---|---|---|---|---|---|---|
| H1 | **Answer-content prior shift × label distribution.** The perturbation tilts the readout toward options containing a direction word ("front"). It pays off where that word is over-represented in gold relative to the base model's predictions. | Gold-independent score tilt; answers move *into* front options even where front is wrong; gains concentrated on gold-front items; the shift reproduces across independent item sets. | Letter-position offsets were tested (A–D), never option *content*. | CPU: score-level bias by gold; switch targets; SEARCH↔RERANK reproducibility across 543 candidates. | No gold-independent tilt; shift not reproducible (r ≈ 0); gains not concentrated on gold-front items. | **Supported, partial**: 2–4 of the 8 pp. |
| H2 | **Selection on item-level noise** (extreme value over 146 σ = .002 RERANK candidates). | The part of the gain not explained by reproducible components does not replicate across item sets. | Already the "simplest explanation"; never decomposed into reproducible and non-reproducible parts. | CPU: replicate the residual SEARCH↔RERANK. | Residual replicates (r ≫ 0). | **Supported** for the remaining ~4–6 pp. |
| H3 | **Format-specific skill.** The perturbation improves genuine compass-direction reasoning, which TEST barely contains. | Gains on non-front compass items as well as front ones; replicates on fresh compass items. | TEST never tested the format. | CPU: gold-front vs non-front split on SEARCH/RERANK. GPU: fresh compass items. | No gain on non-front items. | **Not supported**: SEARCH non-front −1.9 pp (33rd pct); RERANK non-front +1.9 pp (90th pct, n = 108). |
| H4 | **Out-of-format damage.** A large readout shift costs other formats, giving the TEST loss. | TEST gain decreases with shift magnitude across candidates. | No one related shift size to TEST. | CPU: across the 61 TEST-measured candidates. | r ≈ 0. | **Weak**: r(\|RERANK front shift\|, TEST gain) = −0.28, n = 61, one of several correlations. A pure bias model predicts **0** TEST effect. Unresolved. |
| H5 | Letter-position (A–D) bias | Repairs follow letter offsets | – | Done earlier | – | Minority (15/59 changed answers) |
| H6 | Subtask specialist | Gains confined to one subtask | – | Done earlier | – | Superseded by format (H1/H3) |
| H7 | Boundary-item volatility (small base margins flip under any perturbation) | Winner's flips are the pool's high-propensity items | – | Margin analyses done earlier | – | Does not single out the winner |
| H8 | **Construction artifact.** Replicated noise across equal-shape tensors amplifies coherent readout biases. | Same-norm i.i.d. noise produces smaller content biases. | Never compared. | GPU only | i.i.d. noise produces equal biases | Untested; explains why biases exist, not why this direction wins |
| H9 | Vision nonlinearity | – | – | Done | – | Not the why |
| H10 | First-order gradient geometry (H★) | – | – | Done | – | Falsified |

## 2. What existing artifacts can and cannot answer

**CAN TEST NOW** (done here):
- Correct/incorrect and predicted letter for the full SEARCH 5000 × 200 matrix, RERANK 543 × 200, and TEST 50 × 561,
  plus 12 controls on RERANK/TEST.
- Option texts.
- vLLM A–D letter log-probs for the winner and the 12 controls on all 761 RERANK+TEST items.
- Base margins, transitions, subtask, question form.
- Group insertions and removals for the winner (83 items).
- First-order predictions (GPU-A).

**REQUIRES GPU:**
- Scores (not just answers) for SEARCH candidates.
- Any fresh compass-format items (held-out confirmation of H1/H3).
- TEST behaviour for more high-shift candidates (H4).
- i.i.d.-noise comparison (H8).
- Group-level control measurements.

**NOT IDENTIFIABLE with current experiments:**
- Which internal computation turns a random direction into an option-content preference.
- Whether the TEST loss has any structured cause. TEST shares no format with the selection sets, and the pure bias
  model predicts no TEST effect.

## 3. The missing observable: an answer-content prior shift

| Test | Result |
|---|---|
| Across all 5000 SEARCH perturbations, gain explained by answer-content shifts (5-fold CV R²) | **0.32** (r_front = 0.53); σ = .002: 0.33; RERANK (543): 0.42 |
| Is the shift a property of the direction? SEARCH vs RERANK front shift across 543 candidates (independent items) | **r = 0.62** (back 0.38, left 0.44, right 0.39) |
| Gain explained by the shift model, SEARCH vs RERANK | **r = 0.57** |
| Remaining gain, SEARCH vs RERANK | **r = 0.10** (does not replicate) |
| Raw gain, SEARCH vs RERANK | r = 0.21 |
| Winner: front-answer shift | SEARCH +6.5 pp (99.2nd pct of 1249), RERANK +10.0 pp (96.6th pct of 145) |
| Winner: score-level tilt toward front options, RERANK | +0.89 nats, t = 5.7; +1.13 if gold contains front, **+0.66 if not**; away from back −0.72, t = −6.5 (13th of 13) |
| Winner: answers moved into front options, SEARCH+RERANK | net +20 where gold is front (99.7th pct), **net +13 where gold is not front** (99.5th pct) |
| Winner: accuracy change by gold content | front gold: SEARCH **+7.4 pp** (99.3rd pct), RERANK **+15.2 pp** (15 repairs / 1 regression; 98.6th pct). Non-front gold: SEARCH −1.9 pp (33rd pct), RERANK +1.9 pp (90th pct) |
| Winner's RERANK gain split | shift-explained **+4.1 pp** (99.1st pct) + remainder **+3.9 pp** (99.3rd pct) |
| Pure score-level bias model (5 option-content offsets) | winner RERANK: score R² 0.12; 2-fold out of sample **+2.0 of +8.0 pp**, 7/36 changed answers. TEST: predicts **0.0 pp** (actual −2.7), 2/81 changed answers |
| Controls | also carry sizeable content biases (front tilt −1.3 to +1.2 nats, \|t\| up to 11). Biasing is generic; the winner's direction (front ↑, back ↓) is the one this label distribution rewards. |

Base predictions under-use "front" relative to gold (SEARCH 75 vs 94; RERANK 88 vs 92) and over-use "right"
(85 vs 60; 80 vs 59), so a front tilt pays.

## 4. The shared-subset hypothesis

**Supported, with a named property.** The winner's reproducible gain concentrates on items whose gold answer
contains "front": about 46% of SEARCH/RERANK items, but 16 of 561 TEST items. It is not a broad improvement:
non-front items are at noise level on SEARCH. The shared latent property is a *label-content* property, not a visual
or semantic category.

## 5. Decomposing "selection on a fluctuation"

| Layer | What it is | Share of the winner's +8.0 pp RERANK gain |
|---|---|---|
| Evaluation/dataset | Selection sets reward a front-tilt because front is the modal gold answer and the base under-predicts it. TEST has neither the format nor the label skew. | the channel through which the reproducible part pays |
| Statistical selection | Search picks the extreme of (reproducible shift + non-reproducible item noise). The winner is ~99th pct on both. | ~4–6 pp non-reproducible |
| Model computation | Random directions (σ = .002) shift option-content preferences by up to ~1 nat. **How is unknown.** | ~2–4 pp via the shift |
| Construction artifact | Replicated noise might make such coherent biases more common (H8). | untested |

## 6. Criterion for "no identifiable why"

The winner would have no identifiable why if, after conditioning on available variables, nothing reproducible
distinguished it from the pool. That is **not** the case for the RERANK gain: a reproducible, gold-independent front
tilt (99th pct) accounts for part of it. It **is** the case for the TEST loss: no tested variable explains it, and a
pure bias model predicts none of it.

## 7. Recommended sequence

1. **Done (CPU):** format composition; answer-content shift; reproducibility; score-level bias; switch targets;
   gold-split; pure bias model.
2. **Optional confirmation (GPU, ≈ $1–2, only if the paper will claim H1).** This needs unused OmniSpatial-train
   compass items (existence to be confirmed). Pre-register the predictions:
   - the winner's gain on fresh gold-front items is > 0 and larger than on non-front items;
   - its overall fresh-compass gain is ≤ the shift-explained +4 pp, not +8 pp;
   - across the 12 controls, the SEARCH front shift predicts the fresh gold-front gain (r > 0.3).

   **Falsifier:** no front/non-front difference, or an overall gain near +8 pp (which would mean a skill, H3).
   This confirms a mechanism already supported in two independent sets; it is not a discovery experiment.
3. **Not recommended:** H8 (construction), H4 (needs many TEST runs), and any model-internal work, which has no
   decisive test.

## 8. Classification

**B. STRONG STATISTICAL EXPLANATION, MECHANISM UNKNOWN.**

The winner's RERANK advantage decomposes into a reproducible answer-content prior shift (front ↑, back ↓), which the
train-split compass format and its label skew reward, plus selection on non-reproducible item noise. TEST is a
different question format, so it cannot reward the shift. TEST also cannot test the gain at all. The TEST loss
itself has no identified cause. Why random perturbations induce option-content preferences, which is the
model-computation "why", is not identifiable with current data.

**Paper consequence:** the "expert mirage" should be described as selection on (a) a label-prior shift specific to
the selection sets' format and (b) item noise, evaluated on a test set of a different format. It is not a failure
of a learned skill to generalize within a distribution.

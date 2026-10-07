# τ-predictor check (post-hoc on M1 data; criteria committed before computing)

## Pre-set criteria: τ HOLDS (not a selection or group artifact)
| Criterion | Result | Rule |
|---|---|---|
| Within SEARCH-top-50 (n = 47): r(τ, H gain) | 0.59 [0.42, 0.74] | ≥ 0.30, pass |
| Partial r(τ, H gain \| RERANK gain, SEARCH gain), n = 59 | 0.31 [0.13, 0.48] | ≥ 0.20, pass |
| Within the 12 controls | r = 0.57 | sign preserved, pass |

## What τ predicts: largely a label-prior exploit
- **τ works through the front tilt.**
  - r(τ, measured H front tilt) = 0.89.
  - r(H tilt, H gain) = 0.69.
  - τ adds nothing beyond the measured tilt: partial r = −0.02.
- **The label skew is present in H.** Gold contains "front" for 50.5% of items; the base predicts a front option for
  41.2%.
- **Across candidates:**
  - r(τ, gain on gold-front items) = +0.71;
  - r(τ, gain on non-front items) = −0.42.

  A higher τ helps where the answer is "front" and hurts elsewhere. That is the signature of a prior shift, which
  transfers only to item sets with the same label skew.
- **9504111 is different:** +3.63 on front items and +1.68 on non-front items. It gains on both, consistent with C1
  (its gain is not removed by prior calibration).
- **A crude label-balanced reweighting** (front share set to the base's front-prediction rate) leaves r(τ, g) = +0.45.
  That reweighting is not an exact neutral point for a pure tilt, so the residual is **not interpretable**.

## Consequence
- The "noise-only predictor of transferable gains" framing is **misleading**. τ predicts which perturbations will
  exploit a label skew, not which generally improve the model.
- Step 2 (predict-then-verify on H) would only re-demonstrate the prior exploit, so it is **not run**.

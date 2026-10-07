# M1 (matched-split transfer) and C1 (prior calibration): results

Locks: `m1/plan_lock.md` (5eb864b) and `c1/plan_lock.md` (5aa4c06), both committed before any M1 measurement or
result.
- Measured 59 of 61 frozen candidates (in-script budget guard at the $6.00 cap; the 2 missing are top-50 members last
  in the frozen order). The winner and all 12 controls are complete.
- All fingerprints matched and every restore was exact.
- Hold-out H: 600 fresh compass-format OmniSpatial-train items, 484 images, 0 dropped for image leakage.
- Pod spend $5.80.
- After the run, `base.json.gz` was found stored as a list rather than {'HOLDOUT': …}. Both analysis scripts were made
  to accept either form; no analysis logic changed.

## M1: does the selected advantage transfer to fresh items of the same format?
| Test | Result | Locked reading |
|---|---|---|
| Base accuracy on H | 34.7% (208/600) | – |
| **M1-1** 9504111's gain on H | **+2.67 pp**, 95% image-cluster CI [+0.17, +4.93]; 34 repairs / 18 regressions; **rank 1 of 59** | **TRANSFERS** (not FULL; RERANK compass gain was +8.5) |
| M1-2 gold-front minus non-front gain | +1.95 pp (front +3.63, non-front +1.68), CI [−2.57, +6.47] | does not support concentration on front items |
| **M1-3** r(RERANK gain, H gain), 59 candidates | **0.56** [0.36, 0.72] | the selection signal transfers within format |
| **M1-3** r(τ, H gain); τ = noise-only predicted non-vision tilt | **0.61** [0.42, 0.75] | ≥ 0.25: supports |
| **M1-4** SEARCH-top-50 mean minus control mean on H | **+1.16 pp** [+0.31, +2.03] (top-50 +0.28; controls −0.88) | selection buys a modest real gain |

## C1: does removing each candidate's answer-content prior shift remove its gain?
| Test | Result | Locked reading |
|---|---|---|
| **C1-1** 9504111: raw vs calibrated gain | +2.67 → **+2.33** pp; (g − g_cal) CI [−0.51, +1.32]; content shift explains 9% of its score-change variance | **NOT EXPLAINED** |
| C1-2 share of positive gain surviving (34 candidates with raw g > 0) | ≈ 0.00 [−0.42, 0.32] | locked rule: "supports"; **see caveat** |
| C1-3 r(raw g, calibrated g) | 0.66 | top-50 mean 0.28 → −0.33; controls −0.88 → −0.96 |
| C1-4 median variance share explained by the 4-parameter shift | 0.08 | – |

**Caveat on C1-2 (post-hoc check, does not change the locked number).**
- C1-2 conditions on raw g > 0 measured on the *same* items, so regression to the mean inflates the apparent
  removal.
- Cross-fitted (select candidates with g > 0 on one image half, evaluate on the other), the selected candidates' raw
  gains are themselves near zero (+0.17 / +0.68 pp). The surviving share is unstable (−4.65 / 0.30).
- Across all 59 candidates, calibration lowers the mean gain by 0.5 pp (+0.05 → −0.45).
- C1-2 is therefore **not** evidence that "most expert gain is prior shift".

## What this changes
1. **There is real, modest within-format transfer.**
   - The winner keeps about a third of its selection-set advantage on fresh same-format items (+2.7 of +8.5 pp) and
     is the best of 59 there.
   - Selection signal correlates with fresh gains (r = 0.56).
   - The large selection-set number was mostly selection optimism. The remainder is real, *not* a mirage.
2. **The winner's transferring gain is not a global answer-content prior shift** (C1-1). The "experts = label-prior
   exploits" hypothesis is **not supported** for the winner.
3. **The noise-only first-order predictor τ predicts fresh-item gains across candidates (r = 0.61).** τ was computed
   from gradients at BASE on SEARCH/RERANK items with no candidate evaluation. Calibration does not remove the winner's
   gain, so τ appears to track more than a pure label prior. The direction it measures correlates with genuinely
   transferable improvement. This is a post-hoc interpretation and needs its own test.
4. **The original "fails on the official test" result remains a format artifact** (`paper/CORRECTION_SPLIT_MISMATCH.md`).

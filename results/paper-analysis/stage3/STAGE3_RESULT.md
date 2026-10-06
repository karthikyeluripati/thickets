# Stage 3 result: 3A MIXED, 3B PASS (tilt localized, distributed over middle language layers)

Lock: `plan_lock.md` (committed in ce60c5d before any computation). Spend: $3.01 of the $6.00 cap; pod stopped
after a verified pull. All numbers are from `stage3_results.json` (`scripts/stage3_analysis.py`). The only change to
that script after the lock was the JSON serializer (numpy bool); no analysis change.

## 3A: does search select along the tilt axis? (cross-set, no fitted parameters)
| Test | Result | Rule | Verdict |
|---|---|---|---|
| 3A-0 pipeline | r(τ_P96, Stage-2 measured T_NOV) = 0.938 (matches Stage-2 predictions at r = 0.99999) | ≥ 0.9 | pass |
| **3A-1** | r(τ_R, SEARCH gain), all 5000 = **0.240** | ≥ 0.25 PASS, < 0.10 FALSIFIED | **INCONCLUSIVE** |
| **3A-2** | r(τ_S, RERANK gain), 543 = **0.402** | same | **PASS** |
| 3A-3 enrichment | SEARCH-advanced vs rest: +0.09 SD (95% CI −0.01 to 0.18) | ≥ 0.3 | not supported |
| **3A conclusion** | | both PASS needed | **MIXED/INCONCLUSIVE** |

Reported:
- 3A-1 by σ: 0.137 (σ = 0.00025), 0.246 (0.0005), 0.287 (0.001), 0.309 (0.002).
- 9504111's predicted tilt, from noise alone: 99.76th percentile of 5000 (τ_R); 99.63rd of 543 (τ_S).
- r(τ, TEST gain) over 61 = −0.14.

## 3B: where is the tilt produced? (20 perturbations × 8 non-vision groups, exact single-group insertions)
| Test | Result | Rule | Verdict |
|---|---|---|---|
| **3B-1** | pooled r(predicted group partial, measured insertion tilt) = **0.978**, slope 0.95 (160 pairs) | ≥ 0.6 PASS | **PASS** |
| 3B-2 additivity | r(Σ insertions, Stage-2 T_NOV) = 0.960 | ≥ 0.9 → decomposition valid | valid |
| 3B-3 localization | no group ≥ 50% | | **distributed** |

Group shares (predicted across 5000 / measured across 20); per-group r (predicted vs measured):

| group | predicted share | measured share | per-group r |
|---|---:|---:|---:|
| embed | 0.00 | 0.00 | 0.71 |
| L00–05 | 0.04 | 0.05 | 0.98 |
| **L06–11** | **0.44** | **0.41** | 0.98 |
| **L12–17** | **0.22** | **0.22** | 0.98 |
| **L18–23** | **0.23** | **0.28** | 0.98 |
| L24–29 | 0.03 | 0.01 | 0.98 |
| L30–35 | 0.04 | 0.04 | 0.98 |
| final norm + head | 0.00 | 0.00 | 0.93 |

## What is established (pre-registered, out of sample)
1. A σ = 0.002 RandOpt perturbation's answer-content ("front") tilt, outside vision, is a first-order function of its
   noise stream (Stage 2, r = 0.94).
2. It decomposes block by block with the same first-order law (r = 0.98 per block, additive).
3. About 90% of it is produced in language layers 6–23. The embeddings, the final norm/head and the last 12 layers
   contribute almost nothing. So it is **not** a readout or letter-bias artifact.
4. The tilt predicts RERANK gains among RERANK candidates (r = 0.40). Across all 5000 SEARCH gains the prediction
   is weaker (r = 0.24, just below the PASS bar). SEARCH advancement was not enriched for it.

## What is not established
- What computation in layers 6–23 represents the "front" preference (Stage 4, representation-level).
- That search on SEARCH selects mainly through this axis: it is a modest component, at most about 6–16% of gain
  variance.
- Any generality beyond this model, benchmark and noise construction.

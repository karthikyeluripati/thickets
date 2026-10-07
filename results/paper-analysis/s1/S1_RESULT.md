# Session 1 result (lock 797fa96; analysis `scripts/s1_analysis.py`, output `s1_results.json`)

Run: one H100, smoke tests passed, pod stopped automatically after retrieval (~15 min, ≈ $1).

## A: tilt law on a non-Qwen model and new benchmark (OLMo-2-0425-1B-Instruct, ARC-Challenge, 100 × 24)

Validity gates passed: model restored exactly, base accuracy 0.47 (≥ 0.30), full (non-smoke) run.

- **A-1 (primary): r = 0.790 [0.753, 0.824] at σ ≤ 0.002 (1,600 pairs) → GO.** The first-order law transfers
  beyond Qwen and OmniSpatial.
- Per σ (secondary, no rule): 0.925 (σ 0.001), 0.752 (σ 0.002), 0.205 (σ 0.005). OLS slope 0.76 at σ ≤ 0.002;
  21/24 perturbations have per-perturbation r > 0. This is the same picture as P0: tight at small σ, breaking down
  by σ = 0.005.

## B: does the law explain which fresh items the winner 9504111 moves? (Qwen3-VL-8B, 600 M1 hold-out items)

No items were excluded for top-20 floors. HF and vLLM base contrasts agree (r = 0.9995).

- **B-1 (primary): r(pred, Δc) = 0.162 [0.067, 0.252] → NO-GO.** The full first-order prediction (all parameters)
  does not explain the winner's per-item changes.
- **B-2: sign agreement on the 52 items whose correctness changed (34 repairs, 18 regressions) = 0.654 →
  INCONCLUSIVE** (between 0.60 and 0.70).
- Descriptive, pre-declared without a rule (so exploratory in interpretation): the vision part of the prediction
  carries 93% of its variance and is uncorrelated with the measurement (r = −0.008). The non-vision part alone gives
  r = 0.621. This matches the earlier falsification of first-order prediction for vision parameters (0.137): the
  law describes the language-model contribution and fails on the vision tower, which dominates the total prediction.
  A "language-only prediction explains the winner" claim would need its own pre-registered test; it is not
  established here.

## Status for the paper

- R-generalisation (A): confirmatory GO; add to the R9 ledger and the tilt-law table/figure (new model family,
  new benchmark).
- Winner case study (B): confirmatory NO-GO for the full prediction; report the vision/non-vision decomposition as
  exploratory, consistent with the known vision limit.

# Session 1 plan lock (A: non-Qwen generalisation of the tilt law; B: does the law explain the winner per item?)

Committed before any Session 1 output exists. Runners: `scripts/s1_a_olmo.py`, `scripts/s1_b_winner.py`.
Analysis: `scripts/s1_analysis.py` (CPU, deterministic). Frozen inputs: `s1/frozen_A.json`, `s1/B_comparators.json`.
Budget: one H100, $5 hard cap (pod guard). Order: smoke (A `--smoke`, B `--smoke`) → full A → full B.

## A: OLMo-2-0425-1B-Instruct @ 48d788e on ARC-Challenge (100 frozen test items, 24 frozen perturbations)

- Perturbation: RandOpt-style shared stream per tensor; σ ∈ {0.001, 0.002, 0.005} (8 seeds each, `frozen_A.json`).
- Contrast c_j = h_last·(W[gold] − W[strongest wrong at base]) in fp32. Measurement Δc_jk = c_j(θ+σε_k) − c_j(θ).
- Prediction pred_jk = σ_k ⟨fold(∇c_j), ε_k⟩, no fitted coefficients.
- **Validity gates** (all must hold or A is reported as *uninformative*, not as a result): `restored_exact` true;
  base accuracy ≥ 0.30 (chance 0.25); smoke run completes.
- **Primary A-1:** Pearson r(pred, meas) pooled over items × perturbations with σ ≤ 0.002 (16 × 100 = 1600 pairs).
  GO (law generalises beyond Qwen/OmniSpatial) if r ≥ 0.5; NO-GO if r < 0.3; inconclusive otherwise.
  95% CI by item-cluster bootstrap (2000 resamples, seed 0) reported, not part of the rule.
- Secondary (reported, no rule): r per σ (incl. σ = 0.005, where earlier P0 showed breakdown), OLS slope meas~pred
  at σ ≤ 0.002, fraction of perturbations whose per-perturbation r > 0.

## B: Winner 9504111 (Qwen3-VL-8B @ 0c351dd, σ = 0.002, all parameters incl. vision) on M1 hold-out H (600)

- Comparators fixed at base from M1 stored vLLM letter logprobs (`B_comparators.json`: gold, strongest wrong; floor
  substituted for letters missing from top-20).
- Prediction (HF, fp32 contrast, bf16 model): pred_j = pred_NOV_j + pred_V_j, the non-vision and vision folds of ∇c_j
  against the winner's ε via the verified vLLM↔HF mapping.
- Measurement (no GPU): Δc_j = [lp_cand(g) − lp_cand(w)] − [lp_base(g) − lp_base(w)] from M1 `base.json.gz`,
  `cand_9504111.json.gz` (log-prob differences = logit differences). Items where g or w is missing from the top-20
  in base or candidate are excluded (count reported).
- **Primary B-1:** Pearson r(pred, Δc) over included items. GO (first-order law explains which items the winner
  moves) if r ≥ 0.5; NO-GO if r < 0.3; inconclusive otherwise. Item bootstrap CI reported.
- **Secondary B-2 (rule):** among items whose base-argmax correctness changes (repairs: wrong→right; regressions:
  right→wrong), sign agreement of pred with the change direction. Supported if agreement ≥ 0.70 and n ≥ 20
  changed items; not supported if < 0.60; otherwise inconclusive; n < 20 → underpowered.
- Descriptive: r of pred_NOV and pred_V separately; var share of pred_V; HF base c vs vLLM base c agreement (r).

## Interpretation guardrails

A NO-GO on A means the law is not shown to transfer to OLMo/ARC and is reported as such. B tests a *single*
selected perturbation, so it is a case study; it does not re-test selection. No thresholds are changed after data.

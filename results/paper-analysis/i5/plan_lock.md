# I5 plan lock: is the first-order law item-level, or only a property of the 96-item average?

Locked before computing any per-item prediction. It addresses the main expected objection to Stage 2: "a small
perturbation's effect on an *averaged* score is trivially linear". GPU: proposed cap **$2.50**, one pod session (needs a
pod).

## Data
- **Measured:** existing Stage-2 measurements (`../stage12/pod/measurements.json`), NOV condition. For item j and
  perturbation c: ΔB_j(c) = B_j(c) − B_j(BASE). There are 96 items × 80 perturbations; nothing new is measured.
- **Predicted (new, GPU):** pred_j(c) = σ_c⟨fold_nonvision(∇B_j), ε_c⟩. This is HF fp32 at BASE, the same predictor as
  Stage 2 but per item instead of averaged. There are no fitted parameters.
- **Consistency gate:** the mean over items of pred_j(c) must reproduce Stage-2 pred_NOV(c) with r ≥ 0.99, else stop
  (pipeline error).

## Tests
| Test | Statistic | Rule |
|---|---|---|
| **I5-1 (primary)** | pooled r over all 96 × 80 (item, perturbation) pairs of pred_j(c) vs ΔB_j(c) | **≥ 0.6: ITEM-LEVEL LAW** (the averaging objection is refuted); **< 0.3: AVERAGE-ONLY** (Stage-2 linearity arises from averaging); else INTERMEDIATE |
| I5-2 | median over items of the per-item r (each over 80 perturbations), with the IQR | reported |
| I5-3 | per-item r vs |base margin| of the item (Spearman) | reported: are near-tie items less linear? |
| reported | pooled slope; the share of items with per-item r ≥ 0.6 | – |

No re-tuning. These measurements are BF16 vLLM (quantized at 0.125 nats). The per-item noise floor is reported (base
re-scores are identical, so the floor comes from quantization only).

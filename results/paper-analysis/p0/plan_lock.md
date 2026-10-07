# P0 pilot lock: is the RandOpt GQA setting first-order at the level of each question's answer?

Locked before any P0 output. User-approved cap **$10** (one H100). This pilot decides whether "a first-order theory of
neural thickets" is viable before larger spending.

## Setting
- Qwen2.5-VL-3B-Instruct @ 66285546…, language-model-only perturbation (RandOpt PERTURB_VISUAL=0).
- 96 GQA held-out items: the first 96 of the frozen G1 held-out set (34 yes/no).
- 24 candidates: seed 9700000+k, σ ∈ {0.001, 0.002, 0.005}, 8 each. These are not part of the G1 population.

## Measurement (HF, fp32 contrast, no BF16 output quantization)
- **Answer contrast**, fixed per item at BASE from full fp32 logits under the direct one-word prompt:
  - g_j = the gold-answer variant first token with the highest base logit;
  - w_j = the highest non-gold token;
  - c_j = h_last·(W[g_j] − W[w_j]).
- **Measured Δc_j(k):** the candidate is built exactly as RandOpt: bf16 base + bf16(σ·ε) on language-model tensors.
- **Predicted:** σ_k⟨fold_LM(∇c_j), ε_k⟩ at BASE. No fitted parameters.

## Gates
- **Mapping:** seed 9600003 rebuilt in HF is byte-identical to the vLLM/RandOpt realization (job 01).
- **Recompute:** base contrasts reproduce to within 1e-3.
- **Restore:** exact.

## Decision (primary, pre-registered)
| Statistic | Rule |
|---|---|
| **pooled r(pred, measured Δc)** over items × candidates with σ ≤ 0.002 (96 × 16 = 1536 pairs) | **≥ 0.5: GO** (the theory paper is viable; proceed to the full G1 + theory study); **< 0.3: NO-GO** (write the narrower paper); else **WEAK** (proceed only with the critique framing) |
| reported | r at σ = 0.005; per-item median r; slope; the share of items with per-item r ≥ 0.5 |

## Secondary (reported only; uses job 03, our G1 vLLM evaluation code on the same items and candidates)
- **The evaluation code runs end to end.** Base CoT accuracy is reported against RandOpt's published 3B GQA baseline
  (56.6% on all of testdev, per the earlier reproduction config). It is not a gate: the item subset differs.
- **Spread of CoT accuracy change** across candidates (do perturbations change answers?).
- **Link to CoT outcomes:** r between predicted Δc_j(k) and the CoT correctness change (−1/0/+1), over item ×
  candidate pairs.

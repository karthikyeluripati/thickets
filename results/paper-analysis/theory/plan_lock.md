# Theory check lock (criteria fixed before computing; data already exist → reported as criteria-first EXPLORATORY,
# the same status as the τ-check, not as a confirmatory test)

Data: P0 (lock d5152f7) `p0/pod/pred.npy`, `p0/pod/meas.npy` (96 GQA items × 24 perturbations; σ = 0.001/0.002/0.005,
8 each, column k ↔ `p0/candidates.json` order), base contrasts `c_j` from `p0/pod/p0_info.json` (`contrast_tokens`).
None of the statistics below has been computed before this commit. Script: `scripts/theory_check.py`.

## Model (THEORY.md)
First-order law: Δc_jk = σ_k ⟨g_j, ε_k⟩ with ε_k ~ N(0, I) ⇒ Δc_jk ~ N(0, σ_k² ‖g_j‖²) across perturbations.
‖g_j‖ is estimated from the predictions only (never from measurements): ŝ_j² = mean over all 24 k of (pred_jk / σ_k)².

## T1: flip probability (σ ≤ 0.002; 96 × 16 = 1536 item × perturbation pairs)
Event: flip_jk = 1[sign(c_j + meas_jk) ≠ sign(c_j)]. Prediction: p_jk = Φ(−|c_j| / (σ_k ŝ_j)).
- **T1-a discrimination:** pooled AUC of p_jk for flip_jk. **Supports if ≥ 0.80; not supported if < 0.65**; else partial.
  If fewer than 20 flip events, T1-a is UNDERPOWERED.
- **T1-b calibration:** Σ p_jk / Σ flip_jk. **Calibrated if within [0.67, 1.5]**; else miscalibrated (direction reported).
- Reported: the same at σ = 0.005 (where the law breaks, so failure there is expected).

## T2: an unselected perturbation vote returns the base answer (σ ≤ 0.002)
Per item, majority over its 16 perturbations of sign(c_j + meas_jk) (ties → base). Statistic: share of items whose
majority equals sign(c_j). **Supports if ≥ 0.95; not supported if < 0.85.** Reported: σ = 0.005 (8 perturbations),
and the mean measured Δc per σ with item-bootstrap CI (a second-order drift would make it non-zero at large σ).

## T4 (descriptive): per-item effective temperature
Coefficient of variation of ŝ_j across items; Spearman over items between the perturbation flip probability
Φ(−|c_j|/(σ ŝ_j)) at σ = 0.002 and the two-way sampling flip probability 1/(1+exp(|c_j|/T)) at the T that matches
the mean flip rate. High agreement would explain "similarly susceptible questions" (P1-A) in the direct-answer regime.
No rule.

T3 (selection as a mean shift along the selection gradient) is derived in THEORY.md and linked to the existing
Stage 3A / M1-3 / τ-check results; no new test here (projections of 24 perturbations are too few to estimate
gradient cosines).

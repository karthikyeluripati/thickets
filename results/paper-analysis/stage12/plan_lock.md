# Stage 1+2 plan lock: can a perturbation's answer-content tilt be predicted from its noise?

Locked before any Stage 1/2 measurement. Approved by the user: GPU use, cap **$8.00** for this pod (hard guard).
Context: `paper/WHY_INVESTIGATION_9504111.md` (post-hoc discovery of the "front" tilt). H★ remains falsified.
This lock tests a new, narrower hypothesis on a new target.

## Hypothesis H_tilt
For σ = 0.002 RandOpt perturbations, the **non-vision** part of a perturbation's answer-content tilt is first-order
predictable from its noise stream ε:

T_NOV(c) ≈ σ⟨fold_nonvision(∇T̄), ε_c⟩

## Frozen inputs (`frozen_inputs.json`, outcome-blind)
- **Probe set:** 96 SEARCH+RERANK items that have ≥ 1 option containing "front"/"forward" and ≥ 1 option without.
  They are the 96 smallest SHA256('stage12-probe-v1:'+uid).
- **Perturbations (80):** 9504111, the 12 frozen controls, and 67 σ = 0.002 candidates by smallest
  SHA256('stage12-pert-v1:'+candidate_id). They are measured in that order. No new perturbations or search.

## Metric
- For item j with centered A–D letter log-probs z: B_j = mean z(front options) − mean z(non-front options).
- Per perturbation: T(c) = mean over j of [B_j(c) − B_j(BASE)].
- Measured with vLLM 0.11 in-process, the pinned RandOpt WorkerExtension (`apply_perturbation`, exact base restore),
  official direct prompt, first-token top-20 raw log-probs.
- A letter absent from the top-20 takes the top-20 floor (upper bound). Counts are reported. If more than 1% of
  item × condition scores are missing, the metric is flagged.

## Conditions per perturbation
- **FULL:** `apply_perturbation(seed, σ)`, fingerprint checked against the stored candidate_state_id.
- **NOV:** FULL with every vision tensor (`cd.assign_group == 'vision'`) reset exactly to BASE (verified).
- Then exact restore to BASE (verified).

## Prediction (HF, transformers 4.57.1, fp32 contrast at BASE)
- T̄ = mean over probe items of h_last · Σ_k w_jk W[ID_k] (fp32), with w = +1/|F_j| on front options and −1/|N_j|
  on non-front options. This equals the log-prob contrast, because the normalizer cancels.
- The gradient is folded per GPU-A (`geometry_lib`, tested) into two stream vectors: F_NOV (all non-vision tensors)
  and F_V (vision tensors).
- pred_NOV(c) = σ⟨F_NOV, ε_c⟩ and pred_V(c) = σ⟨F_V, ε_c⟩.
- There are no fitted parameters, so every perturbation is out of sample.

## Gates and decisions
| Test | Statistic | Rule |
|---|---|---|
| S1a reliability | Split-half (odd/even probe items) r of T_FULL across perturbations, Spearman–Brown corrected | ≥ 0.7, else the metric is unreliable and S2 is not interpreted |
| S1b fidelity | Fingerprints match; vision reset and base restore exact; missing scores ≤ 1% | any failure stops the run |
| **S2 primary** | Pearson r(pred_NOV, T_NOV) across perturbations | **≥ 0.6 PASS; < 0.3 FALSIFIED; else INCONCLUSIVE** |
| S2 share | r(T_NOV, T_FULL)² | reported; needed for GO |
| S2 secondary (reported only) | r(pred_NOV, T_FULL); r(pred_NOV + pred_V, T_FULL); r(pred_V, T_FULL − T_NOV); slopes | no decision |
| Minimum n | perturbations with both conditions measured | < 40: INCONCLUSIVE regardless |

- **Stage 3 GO** requires S2 primary PASS **and** r(T_NOV, T_FULL)² ≥ 0.5.
- Anything else means **STOP mechanistic GPU work**, with no re-tuning of items, metric, thresholds or perturbations.

## Budget
- Pod hard cap $8.00, enforced by the pod guard (stops at $7.90).
- An in-script guard skips the remaining perturbations before the cap; partial results are kept.
- Order: HF gradient and predictions first, then vLLM measurements in the frozen order.
- Estimate ≈ $6–7.

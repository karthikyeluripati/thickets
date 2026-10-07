# THEORY: a first-order account linking the tilt law (Claim 2) to voting and selection (Claims 1 and 3)

Scope: the **direct-answer regime at small σ**, where the first-order law is established (R4). It is **not** a theory of
chain-of-thought accuracy, which the law does not track (R7, P0). Empirical checks: `results/paper-analysis/theory/`
(criteria committed before computing, commit f06f3c6; status criteria-first EXPLORATORY, like the τ-check).

## Setup
- Perturbation θ' = θ + σε with ε the RandOpt shared stream. Each tensor's noise is a prefix of one N(0, I) stream, so the
  model sees fold-structured noise; the fold operator maps per-tensor gradients onto the stream (R4).
- For question j, the **answer margin** c_j(θ) = logit(gold) − logit(strongest wrong answer at base).
- **First-order law (tested, R4):** Δc_j ≈ σ⟨g_j, ε⟩ with g_j = fold(∇_θ c_j) at the base model. No fitted
  coefficients; r = 0.79–0.94 across models and tasks; slopes 0.76–0.95.

## T1: each question has its own flip probability, set by margin over gradient size
Because ε is isotropic Gaussian, ⟨g_j, ε⟩ ~ N(0, ‖g_j‖²) over perturbations. So Δc_j ~ N(0, σ²‖g_j‖²), and a random
perturbation flips question j's two-way answer with probability

  p_j(σ) = Φ(−|c_j| / (σ‖g_j‖)).

Questions with small margins or large gradients are "susceptible"; the rest essentially never flip at small σ.
**Check (P0, Qwen2.5-VL-3B / GQA, 96 × 16 at σ ≤ 0.002; ‖g_j‖ estimated from predictions only):** AUC 0.935 for the
46 observed flips; predicted 65 vs observed 46 flips (ratio 1.41, within the locked [0.67, 1.5]).

## T2: an unselected vote returns the base answer
Δc_j is symmetric about 0 to first order, so P(c_j + Δc_j has the sign of c_j) = Φ(|c_j|/(σ‖g_j‖)) > 1/2, and the
majority of K unselected perturbations converges to the base answer as K grows. Random perturbations add answer
noise; voting removes it. **Check:** majority over 16 perturbations equals the base answer on 96/96 questions
(σ ≤ 0.002). Consistent with the M1 random-control vote (−0.3 pp [−2.2, +1.5], R2).
Beyond first order, curvature adds a mean drift E[Δc_j] ≈ ½σ² tr(fold-Hessian). At σ = 0.005 the mean margin drift is
−0.62 [−1.00, −0.27] (perturbations hurt on average), T1 loses discrimination (AUC 0.61) and the vote departs from
base on 2 of 96 questions: the same σ where the law breaks (R4).

## T3: selection is a noisy first-order step along the selection-set gradient
Select the top K of N perturbations by a selection score that is, to first order, S(ε) ≈ S₀ + σ⟨G_S, ε⟩ (G_S = the
folded gradient of the selection objective, a smoothed accuracy over the selection items). For Gaussian ε, conditioning
on a high projection onto u = G_S/‖G_S‖ shifts only that component:

  E[ε | selected] = μ_{K/N} u,  with μ_{K/N} the mean of the top K/N tail of N(0,1) (≈ 2.67 for the top 1%),

and leaves the orthogonal components as fresh N(0, I) noise. For any question j, a selected perturbation therefore gives

  Δc_j ≈ σ μ_{K/N} ‖g_j‖ cos(g_j, G_S)  +  N(0, σ²‖g_j‖²(1 − cos²(g_j, G_S))).

Consequences, each matching an observation:
1. **Gains transfer in proportion to gradient alignment with the selection set.** A shift shared by many items (an
   answer-preference tilt toward content the selection labels reward, e.g. "front") has high alignment and transfers;
   item-specific fits do not. This is why selection favours label-aligned tilts (R3, R6: noise+gradient predictor
   r = 0.40 on RERANK, 0.61 on fresh items, working through the front tilt) and why selection-set gains shrink on fresh
   items (+8.5 → +2.67, R2).
2. **The vote of selected models ≈ the base model plus a deterministic shift along G_S.** Averaging over K selected
   perturbations keeps the common term σμ‖g_j‖cos and, by T2, votes away the orthogonal noise. In this regime
   RandOpt's ensemble acts like one small evolution-strategies step on the selection objective (Salimans et al. 2017),
   with step length σμ_{K/N}. (Interpretation, not separately tested.)
3. **Why individual selected models are only modestly better while their vote is much better** (R8b: 64.3% each vs
   77.2% voted): each selected model carries the common shift *plus* fresh noise of size σ‖g_j‖; the vote strips the
   noise. Note: R8b is chain-of-thought, outside this theory's tested regime, so this is an analogy there, not a claim.

## T4: why perturbations and sampling hit the same questions
Two-way temperature sampling flips question j with probability 1/(1 + exp(|c_j|/T)); a perturbation flips it with
Φ(−|c_j|/(σ‖g_j‖)). Both are decreasing in the margin |c_j|; they differ only through ‖g_j‖, which varies moderately
across questions (CV 0.29 in P0). So σ‖g_j‖ acts as a **per-question effective temperature**, and the two mechanisms
rank questions alike: model-implied Spearman 0.94 in P0's direct-answer setting (descriptive), in line with the 0.946
measured for chain-of-thought flip propensities (P1-A, a different regime). The ‖g_j‖ heterogeneity and the persistent
common component in T3 are what distinguish perturbations from re-sampling (P1-B, r = 0.74).

## What the theory does not cover
- Chain-of-thought accuracy (P0: correctness changes unrelated to the first-order margin signal). Multi-step generation
  compounds token-level shifts nonlinearly.
- Vision weights (first-order fails, R4) and σ ≥ 0.005 (second-order drift dominates).
- The winner's residual transferable gain (C1-1, S1-B).

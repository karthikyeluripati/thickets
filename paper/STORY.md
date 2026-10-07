# STORY: the canonical paper story (supersedes the spine in WRITING_PROMPT.md where they differ)

Numbers: only from `RESULTS_MASTER.md` (section refs in brackets). Status as of 2026-10-07.

## The one question
RandOpt samples thousands of Gaussian weight perturbations, keeps the best K on a small selection set and
majority-votes them, and reportedly rivals PPO/GRPO. **What does the weight search actually buy, and what do the
selected perturbations change?**

## The answer in three claims

### Claim 1: The gain is mostly the vote, and self-consistency votes better (practical headline)
- **Same run, paper's own settings** (N = 5000, K = 50, RandOpt's code, GSM8K / Qwen2.5-1.5B): RandOpt reproduces the
  published number (77.18 vs 76.4), but SC@50 is ahead: **−2.65 pp [−4.32, −0.99]** [R8b].
- Decomposition (exploratory): selected models are individually only ~4 pp above base (64.3 vs 60.3); the vote adds
  ~13 pp. SC with 50 samples of the unperturbed model votes to 79.8 [R8b].
- Supporting, cross-paper: SC@50 has higher point estimates than published RandOpt on both base-gated GSM8K rows
  (79.8 vs 76.4; 88.2 vs 87.1), 5.7–10.7 pp above the paper's TT-MV baseline [R8].
- Why they behave alike (CoT regime): perturbations and sampling flip similarly susceptible questions (ρ = 0.946), but
  perturbed models carry persistent effects (r = 0.74), so it is **not** pure re-sampling [R7].
- **Open:** one same-run row only; GQA (the paper's flagship VLM result) is unresolved [R8].

### Claim 2: What a perturbation does to answers is first-order, with sharp limits (mechanism)
- A folded-gradient prediction, base-model gradient × the perturbation's noise, no fitted coefficients, tracks
  answer-preference changes: r = 0.915–0.938 for tilts (2 tasks, 2 Qwen-VL models), 0.930 per question in RandOpt's
  GQA setting, **0.790 on a non-Qwen text model and new benchmark (OLMo-2-1B, ARC-Challenge)**; per item 0.771;
  localized to middle language layers (block-level r = 0.978) [R4, R5].
- Limits, each demonstrated: vision weights (r = 0.137), σ = 0.005 (r ≈ 0.2–0.3 in two models), chain-of-thought
  correctness (unrelated to the first-order signal; 22% of answers flip at σ ≤ 0.002) [R4, R7].

### Claim 3: Selection favours label-aligned tilts, and that explains only part of a "winner" (case study)
- One OmniSpatial search (N = 5000, Qwen3-VL-8B): the winner's +8.0 pp selection gain sat on a format the test set
  lacked (split mismatch); on fresh matched items it keeps **+2.67 [0.17, 4.93]** [R1, R2].
- Selection favours tilts toward answer content the selection labels reward (the "front" tilt; noise+gradient
  predictor works through label priors) [R3, R6].
- Boundary, stated as a main result: removing the content shift leaves +2.33 of +2.67 (C1-1 not explained), and the
  full first-order prediction does not explain which items the winner changes (S1-B r = 0.16; its language part
  alone r = 0.62, exploratory) [R3, R4].

### The theory that ties the claims together (`paper/THEORY.md`, R8c)
Under the law, each perturbation shifts question j's margin by N(0, σ²‖g_j‖²): flip probability Φ(−|c_j|/(σ‖g_j‖))
(AUC 0.935, calibrated), unselected votes return the base answer (96/96), σ‖g_j‖ acts as a per-question temperature
(why perturbations and sampling hit the same questions), and top-K selection is a noisy first-order step along the
selection-set gradient (why label-aligned tilts transfer and item-specific fits do not; why votes beat members).
Direct-answer regime only; CoT is outside it.

**Closing message.** Weight search mostly buys an ensemble, and a cheaper ensemble (sampling) does as well or better in
the setting we could test same-run. What individual perturbations change is a predictable, first-order tilt of answer
preferences, which selection exploits where labels reward it; beyond that (vision, large σ, CoT, the winner's
residual gain) the account stops, and we say so.

## Where each experiment goes
| Main text | Appendix | Dropped / superseded (mention once) |
|---|---|---|
| C same-run (R8b); P2/P3 gated rows (R8); SC@K curves | Non-comparable P2/P3 rows (0.5B, MATH-500, GQA gate fail) | Earlier OmniSpatial TEST-based conclusions (random-control transfer, selection-vs-specificity, margin-additivity, causal diagnostic): superseded by CORRECTION_SPLIT_MISMATCH |
| Tilt law: Stage 2, R1, R2/R2b, I5, P0, S1-A (R4); localization 3B (R5) | GPU-A fidelity gates (S2a NO-GO → A1), V0/S2b checks, per-σ and per-block tables | Expert/mirage framing of the first draft |
| Limits: GPU-A F1/S2d, σ breakdown, P0 CoT, C1-1, S1-B | F2/F3, 3A-1/3A-3, τ-check details, C1-2 (flagged) | |
| Case study: R1 split mismatch, M1 transfer, front tilt, Stage 3A (r = 0.40) | 47-model vs top-10 votes (exploratory), EB shrinkage | |
| P1-A/B (similar susceptibility; persistence) | P1-C (underpowered) | |
| Full R9 ledger (appendix table, referenced in Sec. 2) | Compute ledger | |

## Gaps a top-tier reviewer will press (to close before writing)
1. **Claim 1 rests on one same-run row.** Need a second, independent one at larger scale (GSM8K / Qwen2.5-3B,
   paper 87.1) and the paper's flagship VLM result (GQA / Qwen2.5-VL-3B, paper 69.0), same-run.
2. **Scale of Claim 2:** text-model evidence is ≤ 1.5B (VLMs to 8B). A 7B text model check.
3. ~~Theory~~ **done** (THEORY.md, R8c): T1, T2 supported on P0; T3 derived and linked to existing results; T4
   descriptive. Remaining theory gap: T3 has no direct test (needs gradient cosines; could be added to the 7B run).

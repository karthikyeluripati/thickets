# STORY: the canonical paper story (supersedes the spine in WRITING_PROMPT.md where they differ)

Numbers: only from `RESULTS_MASTER.md` (section refs in brackets). Status as of 2026-10-08 (all planned GPU runs complete).

## The one question
RandOpt samples thousands of Gaussian weight perturbations, keeps the best K on a small selection set and
majority-votes them, and reportedly rivals PPO/GRPO. **What does the weight search actually buy, and what do the
selected perturbations change?**

## The answer in three claims

### Claim 1: RandOpt's gain is a vote plus, where one exists, a shared shift; on Qwen GSM8K sampling votes better, on GQA a one-line prompt gives the shift (practical headline)
Three same-run comparisons at the paper's settings (N = 5000, K = 50) [R8b]:
- **GSM8K / Qwen2.5-1.5B and 3B** (RandOpt's own code; published numbers reproduced, 77.18 vs 76.4 and 86.66 vs 87.1):
  SC@50 is ahead, **−2.65 [−4.32, −0.99]** and **−1.59 [−2.65, −0.53]**. Selected models are individually only +4.0
  and +0.2 pp above base; the gain is the vote, and sampling the base model votes better.
- **GQA / Qwen2.5-VL-3B** (faithful re-implementation; released code cannot pass images): RandOpt is ahead,
  **+3.47 [+1.62, +5.41]**, replicated with a disjoint population of 5000 perturbations (+3.63 [+1.78, +5.49];
  two-seed mean +3.55). But the advantage exists only under RandOpt's chain-of-thought prompt, which costs this
  model 11 points: the selected perturbations mostly switch reasoning off (median 10 tokens vs 146; shorter members
  are more accurate, r = −0.89; exploratory). **Asked to answer directly, the base model with one greedy generation
  scores 64.7, vs RandOpt's 63.5** (G4, pre-registered: RandOpt − SC@50 with the direct prompt −1.21 [−2.83, +0.40],
  no difference detected; same with a second prompt and the second seed). Search adds nothing on top of the prompt
  (selected members under the direct prompt vs SC: −0.08 [−0.97, +0.81]). Termination repair explained only a
  quarter of the CoT-prompt advantage (G3).
- **GSM8K / OLMo-2-1B (non-Qwen, O1, pre-registered): RandOpt ahead, +8.72 [+6.75, +10.77].** Base scores only 35% under
  RandOpt's prompt; the selected models are +5.1 pp better individually (a shared shift, as on GQA). So on GSM8K the
  outcome is model-dependent. Whether OLMo's shift is also a prompt effect (as G4 showed for GQA) is **untested**:
  only extracted answers were saved, not texts.
- Supporting, cross-paper: SC@50 point estimates above published RandOpt on both gated GSM8K rows, 5.7–10.7 pp above
  the paper's TT-MV baseline [R8].
- CoT regime: perturbations and sampling flip similarly susceptible questions (ρ = 0.946), but perturbed models carry
  persistent effects (r = 0.74), so it is **not** pure re-sampling [R7].

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

**Closing message.** Read through the first-order picture, RandOpt is two things: a vote, which removes perturbation noise
and which sampling the base model does as well or better (GSM8K), and a selection step that moves the model along
whatever *shared* direction the selection set rewards, which pays when such a direction exists (GQA: switching off
chain-of-thought reasoning that a direct prompt also switches off; OmniSpatial's label-aligned tilt). That is where "thickets" are real and useful: shared,
transferable shifts, often of answer form rather than reasoning. Beyond that (vision weights, large σ, CoT correctness,
the winner's residual gain) the account stops, and we say so.

## Where each experiment goes
| Main text | Appendix | Dropped / superseded (mention once) |
|---|---|---|
| C, C3B, G2 same-run (R8b) incl. member-vs-vote decomposition and the GQA format analysis (exploratory); P2/P3 gated rows (R8); SC@K curves | Non-comparable P2/P3 rows (0.5B, MATH-500, GQA gate fail) | Earlier OmniSpatial TEST-based conclusions (random-control transfer, selection-vs-specificity, margin-additivity, causal diagnostic): superseded by CORRECTION_SPLIT_MISMATCH |
| Tilt law: Stage 2, R1, R2/R2b, I5, P0, S1-A, S1-7B (R4); localization 3B (R5) | GPU-A fidelity gates (S2a NO-GO → A1), V0/S2b checks, per-σ and per-block tables | Expert/mirage framing of the first draft |
| Limits: GPU-A F1/S2d, σ breakdown, P0 CoT, C1-1, S1-B | F2/F3, 3A-1/3A-3, τ-check details, C1-2 (flagged) | |
| Case study: R1 split mismatch, M1 transfer, front tilt, Stage 3A (r = 0.40) | 47-model vs top-10 votes (exploratory), EB shrinkage | |
| P1-A/B (similar susceptibility; persistence) | P1-C (underpowered) | |
| Full R9 ledger (appendix table, referenced in Sec. 2) | Compute ledger | |

## Gaps a top-tier reviewer will press
1. ~~One same-run row~~ **closed**: three rows (GSM8K 1.5B, 3B; GQA). They split by task, which the closing message
   explains; one RandOpt run per row (the paper averages 3).
2. ~~Text-model scale~~ **closed**: S1-7B r = 0.787 (OLMo-2-7B), same as 1B.
3. ~~Theory~~ **done** (THEORY.md, R8c). Remaining: T3 (selection as a step along the selection gradient) has no direct
   test; the GQA format finding is exploratory (keyword heuristic; texts / token counts not saved).
4. ~~GQA mechanism test~~ **done** (G3): PARTIAL; termination explains ~1 of ~3.6 pp. Remaining open: what the rest of
   the GQA shift is (answer length/form vs content). Countdown remains optional.

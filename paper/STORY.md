# STORY: the canonical paper story

Numbers only from `RESULTS_MASTER.md` (sections in brackets). Status 2026-10-09: all confirmatory runs complete.

## Title
*What Does Random Weight Search Buy? Votes, Prompts and First-Order Tilts in RandOpt*
(alternative: *Thickets or Tilts? A Pre-Registered Re-Examination of Random Weight Perturbation as Post-Training*)

## The one question
RandOpt samples thousands of Gaussian weight perturbations of a pretrained model, keeps the best K on a small
selection set and majority-votes them, and reportedly rivals PPO and GRPO. **What does the weight search buy, and
what do the selected perturbations change?**

## The answer in one paragraph
RandOpt's gain has two parts: a vote, and a shift in the model's answers that is shared across questions. Sampling
the unperturbed model votes better in three of four rows (its vote adds more over single samples than RandOpt's
vote adds over single members; on OLMo it adds less). The shared shifts selection found in our settings are ones
a prompt also reaches: on the two rows where RandOpt beats self-consistency, its own prompt costs the base model
11.3 and 31.6 points, and the selected perturbations recover part of that accuracy. Choosing the prompt on RandOpt's own selection
data (600 generations, 0.06% of its 1,000,000), then sampling, matches or beats RandOpt in all four same-run rows. In
direct-answer settings, a first-order account explains what a perturbation does to answers and why selection
favours shared shifts, within limits we demonstrate; the Claim 1 rows use step-by-step prompts, outside that regime,
so Claim 1 rests on its own experiments.

## Claim 1 (headline, practical): prompt selection plus sampling matches or beats RandOpt's weight search [R8b]
Four same-run rows at RandOpt's own settings (N = 5000, K = 50): GSM8K with Qwen2.5-1.5B, Qwen2.5-3B and OLMo-2-1B,
and GQA with Qwen2.5-VL-3B. Two model families, two tasks; RandOpt's published numbers reproduced on the Qwen GSM8K
rows (77.18 vs 76.4; 86.66 vs 87.1).
- **The headline result (PS) [R8b.5].** Choose among three prompts by greedy accuracy on RandOpt's 200 selection
  questions (600 generations vs RandOpt's 1,000,000), then take SC@50. RandOpt − SC: Qwen-1.5B **−2.96 [−4.62, −1.29]**,
  Qwen-3B **−0.38 [−1.67, +0.91]** (equivalent), OLMo **−23.65 [−26.23, −21.08]**, GQA **−1.21 [−2.83, +0.40]**.
  Ahead in two rows, no difference in two. **RandOpt is ahead in none.** Selection picks a non-default prompt for
  every model. Robust (exploratory, R8b.6): holds under strict scoring (where it strengthens, as the lenient scorer
  under-credits boxed answers), under Holm correction, and at K = 10 (where Qwen-1.5B becomes no-difference).
- **Decomposition under RandOpt's own prompts [R8b.1].**
  - *Where RandOpt loses* (Qwen GSM8K: −2.65, −1.59): the selected models are only +4.0 and +0.2 pp above base
    individually, so the gain is the vote, and sampling votes better.
  - *Where RandOpt wins* (GQA +3.47, replicated with a disjoint population at +3.63; OLMo +8.72): the selected
    models are about +5 pp above base individually, a shift shared across questions.
- **What that shift is [R8b.2, R8b.3].**
  - GQA: RandOpt's chain-of-thought prompt costs the base model 11.3 points. The selected models mostly stop
    reasoning and answer directly (median 10 tokens vs 146; shorter members are more accurate, r = −0.89;
    exploratory). One direct-prompt generation scores 64.7 vs RandOpt's 63.5 (G4). **Running RandOpt's search itself
    under the direct prompt (GD, same 5000 perturbations) gives 64.38 vs SC@50 64.70: −0.32 [−1.29, +0.65],
    equivalent within ±2 pp**; its selected models are no better than base (−0.60 [−1.27, +0.05]), and none of
    them overlaps the CoT search's top 50. With the prompt held fixed, weight search adds nothing measurable.
    Termination repair explained only a quarter of the CoT-prompt advantage (G3).
  - OLMo: RandOpt's "####" instruction halves the model's accuracy (33.7 vs 65.3 with just the question). The search
    partly repairs it (52.5); a prompt change repairs more (SC@50 74.7; one plain generation 65.3).
- **Damage is part of the story, not all of it [R8b.4].** On the locked measure (plain/direct − RandOpt's prompt)
  damage is +11.3 and +31.6 where RandOpt wins and +4.7 and −7.2 where it loses (prediction supported 4/4). But
  against the selection-chosen prompt, Qwen-1.5B is damaged by 10.2 points and RandOpt still lost (post-hoc).
- **Context.** Cross-paper, SC@50 has higher point estimates than published RandOpt on the two gated GSM8K rows and
  is 5.7–10.7 pp above the paper's TT-MV baseline [R8]. In the CoT regime, perturbations and sampling flip similarly
  susceptible questions (ρ = 0.946), but perturbed models carry persistent effects (r = 0.74): not pure re-sampling [R7].

## Claim 2 (mechanism): what a perturbation does to answers is first-order, with demonstrated limits [R4, R5, R7]
- A folded-gradient prediction (base-model gradient × the perturbation's noise, no fitted coefficients) tracks
  answer-preference changes: r = 0.915–0.938 for tilts (two tasks, two Qwen-VL models), 0.930 per question in
  RandOpt's GQA setting, 0.790 / 0.787 on OLMo-2-1B / 7B on ARC-Challenge (non-Qwen, text-only); per item 0.771.
- It is localized to the middle language layers (block-level r = 0.978).
- Limits, each demonstrated: vision weights (r = 0.137), σ = 0.005 (r ≈ 0.2–0.3 in two models), chain-of-thought
  correctness (unrelated to the first-order signal; 22% of answers flip at σ ≤ 0.002).

## Claim 3 (why selection finds shared shifts): theory and a case study [R8c, R1–R3, R6]
- **Theory (`THEORY.md`; criteria fixed before computing, data pre-existing, so exploratory).** Under the law, a perturbation shifts question j's margin by N(0, σ²‖g_j‖²). Consequences:
  flip probabilities are predictable (AUC 0.935, calibrated); an unselected vote returns the base answer (96/96);
  σ‖g_j‖ acts as a per-question temperature (why perturbations and sampling hit the same questions); top-K selection
  is a noisy first-order step along the selection set's gradient, so it favours shifts shared across the selection
  questions (why votes beat members). That the shared shifts we found are ones a prompt also gives is an empirical
  finding (Claim 1), not a consequence of the theory. T3 is derived,
  not directly tested. Direct-answer regime only.
- **Case study (one OmniSpatial search, N = 5000, Qwen3-VL-8B).** The winner's +8.0 pp selection gain sat on a format
  the test set lacked; on fresh matched items it keeps +2.67 [0.17, 4.93]. Selection favours tilts toward answer
  content the selection labels reward. Boundary, stated as a main result: removing that content shift leaves +2.33
  of +2.67 (C1-1, not explained), and the full first-order prediction does not explain which items the winner
  changes (S1-B r = 0.16). No prompt control was run here.

## Closing message
Nearby models that beat the base on a selection set are common (10.7–44.7% of 5000 perturbations on 200 questions,
though the population mean is below base in every row), and in two rows the selected ones transfer to test. In our
settings, what selection finds there is a vote plus shared shifts in answer form, and those are cheaper to get from
sampling and from choosing a prompt on the same selection data. Beyond that (vision weights, large σ, CoT
correctness, the OmniSpatial winner's residual gain) our account stops, and we say so.

## Scope and limits (say these in the paper)
Two tasks for the same-run comparisons (GSM8K, GQA); models ≤ 8B; one RandOpt search per GSM8K row (two on GQA);
three prompt candidates per task, fixed before the PS run but chosen after earlier results ("direct" after the GQA
shift analysis; "boxed" is the standard math prompt), stated as such; RandOpt's search under the chosen prompt was run on GQA only (GD), not
on the GSM8K rows; RandOpt's search cost not charged in the accuracy comparisons;
OLMo "search on top of the prompt" not valid (fidelity gate failed); GQA uses a re-implementation (released code
cannot pass images); exploratory analyses are labelled as such.

## Where each experiment goes
| Main text | Appendix | Superseded (mention once) |
|---|---|---|
| R8b: C, C3B, G2/G2R, O1 (same-run); PS (headline); G4, O2, Q2 (prompt controls); GQA shift (exploratory); Figures 7–8, Table 4 | G3 details; K = 10; vote curves; member decompositions; O2 fidelity failure; cross-paper P2/P3 (R8) incl. non-comparable rows | G2's exploratory "termination repair" reading (superseded by G3, G4) |
| Tilt law: Stage 2, R1, R2/R2b, I5, P0, S1-A, S1-7B (R4); localization 3B (R5) | GPU-A fidelity gates (S2a NO-GO → A1), V0/S2b checks, per-σ and per-block tables | Expert/mirage framing of the first draft |
| Limits: GPU-A F1/S2d, σ breakdown, P0 CoT, C1-1, S1-B | F2/F3, 3A-1/3A-3, τ-check details, C1-2 (flagged) | Earlier OmniSpatial TEST-based conclusions (CORRECTION_SPLIT_MISMATCH) |
| Theory T1, T2, T4 (R8c); case study R1, M1, front tilt, Stage 3A | P1-A/B (R7); P1-C; 47-model vs top-10 votes; EB shrinkage | |
| R9 ledger referenced in Sec. 2 | Full R9 ledger; compute ledger | |

## What a reviewer will still press (answer in the paper, not with new runs)
1. Only two tasks and ≤ 8B: stated as scope; Countdown and larger models are future work.
2. One search per GSM8K row: RandOpt reproduces the published numbers; GQA replicates across two populations.
3. "You picked prompts that help": the three candidates were fixed before any test output, and PS chose among them
   on selection data only.
4. "Damage explains everything": we say it does not (Qwen-1.5B, Figure 8).

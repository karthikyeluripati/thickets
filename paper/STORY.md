# STORY: the canonical paper story

Numbers only from `RESULTS_MASTER.md` (sections in brackets). Status 2026-10-10: all confirmatory runs complete and
the LaTeX draft is complete (8 pages main text + appendices A–I; every number audited against RESULTS_MASTER);
including the reviewer round (RV, R8b.5/R8b.9/R8b.10; RV-3 replicated the OLMo row with a second seed).

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
data (600 generations, 0.06% of its 1,000,000), then sampling, matches or beats RandOpt in all four same-run rows when
the candidates include a format-fixing prompt (with public harness templates alone, in three of four), and with that
prompt held fixed RandOpt's search itself adds nothing over sampling in all four rows (GQA equivalent; OLMo, Qwen-3B
and Qwen-1.5B behind; the selected models are no better than base on test). In
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
  under-credits boxed answers), under Holm correction, and at K = 10 (where Qwen-1.5B becomes no-difference); SC
  temperature 0.5-1.0 changes no outcome (RV-5, pre-registered).
- **Candidate independence (RV-1, pre-registered) [R8b.5].** Repeated with public evaluation templates
  (lm-evaluation-harness, simple-evals, LLaVA, BLIP-2) fixed before any was run: Qwen-3B −0.45 (equivalent), OLMo
  −5.23, GQA −2.67 (now SC ahead), but **Qwen-1.5B +4.17 [+1.97, +6.44], RandOpt ahead**: no public template beats
  RandOpt's own prompt there on the selection set (67.5 vs 68.5). Pooling all six candidates restores "matches or
  beats" in all four rows. Say it plainly: the cheap route needs a candidate that fixes the answer format.
- **Decomposition under RandOpt's own prompts [R8b.1].**
  - *Where RandOpt loses* (Qwen GSM8K: −2.65, −1.59): the selected models are only +4.0 and +0.2 pp above base
    individually, so the gain is the vote, and sampling votes better.
  - *Where RandOpt wins* (GQA +3.47, replicated with a disjoint population at +3.63; OLMo +8.72, replicated at
    +10.39 [+8.26, +12.51] with a second population, RV-3): the selected
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
    partly repairs it (52.5); a prompt change repairs more (SC@50 74.7; one plain generation 65.3). **Running RandOpt's
    search under the boxed prompt (GB, same 5000 perturbations) gives 74.00 vs SC@50 76.12: −2.12 [−3.49, −0.83],
    SC ahead**; the selected models are no better than base (−0.81 [−2.19, +0.57]).
  - Qwen GSM8K (RV-2a/2b, pre-registered): RandOpt's search under the boxed prompt is behind sampling on Qwen-3B
    (**−1.36 [−2.35, −0.38]**) and Qwen-1.5B (**−4.32 [−5.91, −2.81]**); selected models no better than base (−0.47,
    −0.51). **With the prompt held fixed, weight search adds nothing in all four rows** [R8b.9].
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
- **Theory (RESULTS_MASTER R11; criteria fixed before computing, data pre-existing, so exploratory).** Under the law, a perturbation shifts question j's margin by N(0, σ²‖g_j‖²). Consequences:
  flip probabilities are predictable (AUC 0.935, calibrated); an unselected vote returns the base answer (96/96);
  σ‖g_j‖ acts as a per-question temperature (why perturbations and sampling hit the same questions); top-K selection
  is a noisy first-order step along the selection set's gradient, so it favours shifts shared across the selection
  questions (why votes beat members). That the shared shifts we found are ones a prompt also gives is an empirical
  finding (Claim 1), not a consequence of the theory. T3 is derived,
  not directly tested. Direct-answer regime only.
- **Bridge to Claim 1 (exploratory, R8b.7).** In all ten N = 5000 searches the top 50 beat base on the 200
  selection questions while the population mean is at or below base. The selected models are better on test (+4.0 to
  +7.9) only where the prompt leaves ≥ 10 pp of accuracy unclaimed; with the prompt held fixed (GD, GB, RV-2a, RV-2b)
  they are not (−0.5 to −0.8). GD is in the theory's direct-answer regime: with no shared direction left, top-K
  selection picks up selection-set noise, as T3 implies. Descriptive.
- **The "experts" are prompt-specific (exploratory, R8b.8).** Over the same 5000 perturbations, selection rewards under
  two prompts are nearly uncorrelated (Spearman 0.025 OLMo, 0.171 GQA), the two top 50s share no member, and the
  winners under RandOpt's prompt sit at the 31st–37th percentile under the good prompt. On GQA the selected
  perturbations reproduce the direct prompt's answers (they side with it 48.7% vs 15.3% where the prompts disagree);
  on OLMo they do not. This speaks to the original thesis: in our settings the dense "experts" are repairs for a
  particular prompt, not task experts.
- **Case study (one OmniSpatial search, N = 5000, Qwen3-VL-8B).** The winner's +8.0 pp selection gain sat on a format
  the test set lacked; on fresh matched items it keeps +2.67 [0.17, 4.93]. Selection favours tilts toward answer
  content the selection labels reward. Boundary, stated as a main result: removing that content shift leaves +2.33
  of +2.67 (C1-1, not explained), and the full first-order prediction does not explain which items the winner
  changes (S1-B r = 0.16). **Prompt control (OS, pre-registered, R2b):** the winner's +2.67 exists only under the
  direct prompt it was selected with; under the benchmark's step-by-step prompts its gain is −0.67 [−3.95, +2.51]
  (manual_cot) or −3.83 [−6.81, −0.84] (zeroshot_cot), and the prompt chosen on its own selection data gives the base
  model 38.83 vs the winner's 37.33 (−1.50 [−5.35, +2.23], no difference detected). So the one transferable,
  unexplained gain in the study is also prompt-specific.

## Closing message
Nearby models that beat the base on a selection set are common (11.8–58.3% of 5000 perturbations on 200 questions;
the population mean is level with base on the Qwen rows and below it on GQA and OLMo). Under RandOpt's prompts the
selected ones are better on test in three of four rows (+4.0 to +7.9); with the prompt held fixed they are not, in any
row. In our
settings, what selection finds there is a vote plus shared shifts in answer form, and those are cheaper to get from
sampling and from choosing a prompt on the same selection data. In every setting where we tested it (GQA, OLMo,
and the OmniSpatial winner), the selected perturbation's advantage was specific to the prompt it was selected
under. Beyond that (vision weights, large σ, CoT correctness) our account stops, and we say so.

## Scope and limits (say these in the paper)
Two tasks for the same-run comparisons (GSM8K, GQA); models ≤ 8B; one RandOpt search per GSM8K row under RandOpt's
prompt for the Qwen rows (two populations on GQA and on OLMo); 200 selection questions (RandOpt's setting);
three PS candidates per task, fixed before the PS run but chosen after earlier results, plus five public templates
fixed before RV (with public templates alone the headline holds in three of four rows); RandOpt's search cost not
charged in the accuracy comparisons; randopt.py's base print for the Qwen rows is engine-specific (RV-4; disclosed, no
confirmatory test uses it); OLMo "search on top of the prompt" not valid (fidelity gate failed); GQA uses a
re-implementation (released code cannot pass images); GB-2 is INVALID-ENV in the ledger and was re-run as RV-2b;
exploratory analyses are labelled as such.

## Where each experiment goes
| Main text | Appendix | Superseded (mention once) |
|---|---|---|
| R8b: C, C3B, G2/G2R, O1 (same-run); PS (headline) + RV-1 (public templates); G4, O2, Q2 (prompt controls); GD, GB, RV-2a, RV-2b (prompt held fixed); GQA shift (exploratory); Figures 1, 2 (fig7), Table 1 | G3 details; K = 10; vote curves; member decompositions; O2 fidelity failure; RV-4 (base print), RV-5 (temperature); Figure 8 (damage); cross-paper P2/P3 (R8) incl. non-comparable rows | G2's exploratory "termination repair" reading (superseded by G3, G4) |
| Tilt law: Stage 2, R1, R2/R2b, I5, P0, S1-A, S1-7B (R4); localization 3B (R5) | GPU-A fidelity gates (S2a NO-GO → A1), V0/S2b checks, per-σ and per-block tables | Expert/mirage framing of the first draft |
| Limits: GPU-A F1/S2d, σ breakdown, P0 CoT, C1-1, S1-B | F2/F3, 3A-1/3A-3, τ-check details, C1-2 (flagged) | Earlier OmniSpatial TEST-based conclusions (split-mismatch correction, R1) |
| Theory T1, T2, T4 (R8c); case study R1, M1, front tilt, Stage 3A | P1-A/B (R7); P1-C; 47-model vs top-10 votes; EB shrinkage | |
| R9 ledger referenced in Sec. 2 | Full R9 ledger; compute ledger | |

## What a reviewer will still press (answer in the paper, not with new runs)
1. Only two tasks and ≤ 8B: stated as scope; Countdown and larger models are future work.
2. One search per Qwen GSM8K row under RandOpt's prompt: RandOpt reproduces the published numbers there; both rows
   where RandOpt wins (GQA, OLMo) replicate with a second population; every row also has an independent search under
   the chosen prompt (GD, GB, RV-2a, RV-2b).
3. "You picked prompts that help": answered by RV-1 (public templates fixed before the run): three of four rows hold;
   Qwen-1.5B does not, and we say the cheap route needs a format-fixing candidate.
4. "Damage explains everything": we say it does not (Qwen-1.5B, Figure 8).
5. "With a larger selection set, search would find transferable gains": all searches use RandOpt's 200 questions;
   stated as "at RandOpt's own setting".
6. "Prompt held fixed on Qwen GSM8K?": shown (RV-2a −1.36, RV-2b −4.32; SC ahead in both).
7. "Your runner disagrees with randopt.py on the base": RV-4 (73.00 reproducible in randopt.py, 68.00 in ours; test
   bases agree; perturbed rewards agree); the affected exploratory numbers are recomputed and the discrepancy disclosed.
8. "Is SC tuned?": T = 0.7 was fixed in advance; RV-5 shows T = 0.5 and 1.0 change nothing.

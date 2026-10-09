# Prompt for drafting the paper

Give the writer this file plus `paper/STORY.md` (the argument) and `paper/RESULTS_MASTER.md` (the only allowed
numbers). Where they disagree, RESULTS_MASTER wins on numbers and STORY wins on emphasis.

---

You are writing a research paper for a top machine-learning venue (target: ICML 2027 main track; fallback: TMLR).
It re-examines *Neural Thickets: Diverse Task Experts Are Dense Around Pretrained Weights* (Gan & Isola, ICML 2026
Spotlight, arXiv 2603.12228) and its algorithm **RandOpt**: sample N Gaussian weight perturbations of a pretrained
model, keep the top K on a small selection set, majority-vote their answers.

**The paper's centre** is a decomposition of what RandOpt's weight search buys: a vote, which sampling supplies at
least as well, plus shifts in answer form shared across questions, which in our settings a prompt chosen on the same
selection data also supplies. A first-order account explains what perturbations do to answers and why selection finds
shared shifts, with experimentally demonstrated limits.

Write with the precision of a careful empirical paper, not the tone of a takedown. RandOpt reproduces, its selected
models are real improvements under its own prompts, and the original paper itself discusses format effects. Say
exactly how far each result goes and no further.

## Hard rules
1. **Numbers.** Only numbers from `RESULTS_MASTER.md`; never invent one. Report n and 95% CIs where given. Never round
   in a way that strengthens a claim ("5.7–10.7 pp", not "6–11"; "31.6", not "32").
2. **Process language.** Write exactly: "Confirmatory tests were specified in plan locks committed before their
   corresponding runs; exploratory analyses are identified separately." Never write "every decision was
   pre-registered". Label exploratory and post-hoc results in text and captions.
3. **Report every locked test with its outcome** (R9 ledger, appendix), including falsified, inconclusive,
   failed-gate and skipped ones (O2 fidelity gate failed; O1 second seed skipped by its budget rule).
4. **Claim 1 language.**
   - Headline (PS): prompt selection plus SC@50 "matches or beats" RandOpt in all four rows: ahead in two
     (Qwen-1.5B, OLMo), no difference detected in two (Qwen-3B, equivalent within ±2 pp; GQA, not equivalent). Never
     write "beats in all rows".
   - Under RandOpt's own prompts RandOpt is **ahead** on GQA and OLMo. Say so plainly; it is half the explanation.
   - "Damage" claims: the locked measure (plain/direct vs RandOpt's prompt) supports the prediction in 4/4 rows; the
     post-hoc measure against the selection-chosen prompt does not separate the rows (Qwen-1.5B: +10.2, RandOpt still
     lost). Never write "RandOpt wins exactly where its prompt damages the model".
   - "With the prompt held fixed, weight search adds nothing over sampling" is shown by GD (GQA, direct: −0.32 [−1.29,
     +0.65], equivalent) and GB (OLMo, boxed: −2.12 [−3.49, −0.83], SC ahead), in both of which the selected models are
     no better than base. Cite GD/GB, not G4's re-used members or O2's members (fidelity failed). Not shown for the
     Qwen GSM8K rows: GB-2 (Qwen-1.5B) is INVALID-ENV and must be reported as such in the ledger.
   - Compute: the comparison is at matched **test-time** budget (50 generations). RandOpt's 1,000,000 selection
     generations vs prompt selection's 600 are stated separately, not folded into accuracy.
   - GQA runs use a faithful re-implementation (the released code cannot pass images); say so where GQA appears.
   - Robustness (R8b.6, exploratory) goes in the appendix with one sentence in Section 3.5: strict scoring, Holm,
     K = 10. Do not upgrade Qwen-3B to "SC ahead" in the main claim (that is strict-scoring, exploratory).
5. **Mechanism language.** The prediction needs the base-model gradient plus the perturbation's noise, with no fitted
   coefficients; it is a first-order approximation (r = 0.915–0.938 for tilts; slopes 0.83–0.95; per-item r = 0.771).
   Never "exact", never "from the noise alone". Selection *favours* answer-preference tilts; that is an
   **incomplete** explanation of the OmniSpatial winner (C1-1 not explained). T3 (selection as a gradient step) is
   derived, not directly tested.
6. **Chain-of-thought.** Perturbations and sampling affect similarly susceptible questions (ρ = 0.946); this is not an
   equivalent mechanism, and P1-B (persistence, r = 0.74) rejects pure re-sampling. Never "perturbations act as
   re-sampling".
7. **Do not claim:** that RandOpt "is" self-consistency or "is" prompt tuning; that neural thickets do not exist;
   that prompt selection beats RandOpt on every task or model (two tasks, ≤ 8B, three prompt candidates); anything
   about models above 8B; that the first-order account governs CoT accuracy; that we explain the OmniSpatial winner;
   that prompt choice explains the OmniSpatial case (no prompt control there).
8. **Scope statements that must appear.** The first-order theory covers direct answers; the Claim 1 rows use
   step-by-step prompts, so Claim 1 rests on its own experiments, not on the theory. Theory checks T1/T2 are
   criteria-first exploratory. The three prompt candidates were fixed before PS but chosen after earlier results.
   RandOpt's search under the chosen prompt was run on GQA (GD) and OLMo (GB); GB-2 (Qwen-1.5B) failed its environment
   gate and was not run. Vote efficiency: sampling's vote adds more than RandOpt's in three rows, less on OLMo
   (exploratory); never write "sampling supplies the vote at least as well". "Nearby better models are common" only
   with the 10.7–44.7% numbers and the below-base population mean.
9. **Credit prior work** (R10): the original paper's format analysis ("format thickets"); selection bias toward the
   selection set (arXiv 2608.10867); "A Thicket by Any Other Name" (geometric intuition); "When does RandOpt work?"
   (sequence-length dependence); self-consistency (Wang et al. 2023).

## Title
*What Does Random Weight Search Buy? Votes, Prompts and First-Order Tilts in RandOpt*

## Contributions (numbered list in the introduction)
1. **Same-run comparisons at RandOpt's own settings** in four rows (two model families, two tasks), reproducing its
   published numbers, against self-consistency at matched test-time budget (R8b.1).
2. **A decomposition of the gain:** where RandOpt loses, the selected models barely beat base and the gain is the
   vote; where it wins, they share a shift; RandOpt's own prompt damages the base model there, the shift recovers part of
   that accuracy (on GQA visibly, by switching off step-by-step reasoning; G4, O2),
   and a prompt fix does better (R8b.2–R8b.4).
3. **A practical baseline:** choosing the prompt on RandOpt's own selection data (0.06% of its selection compute)
   then sampling matches or beats RandOpt in all four rows (R8b.5); on GQA, RandOpt's search run under that prompt
   is equivalent to sampling (GD, GQA) and behind it (GB, OLMo).
4. **A first-order account** of perturbation-induced answer-preference changes, validated across five models in two
   families, localized to middle language layers, with demonstrated limits (R4, R5, R7), and a theory linking it to
   selection (R8c).

## Sections
1. **Introduction.** The question; the one-paragraph answer (STORY); contributions; Figure 1 (schematic: vote +
   shared shift; still to be drawn) and Figure 7.
2. **Background and setup.** RandOpt; its shared-stream noise; models, tasks, RandOpt's own prompts and scorers; the
   plan-lock protocol, gates and decision rules (Rule 2 wording); the GQA re-implementation.
3. **What weight search buys (R8b).** 3.1 same-run rows under RandOpt's prompts (Table 4, left half); 3.2 the
   member-vs-vote decomposition; 3.3 the shared shift: GQA (G3, shift analysis [exploratory], G4) and OLMo (O2);
   3.4 the Qwen control (Q2) and Figure 8 with the damage caveat; 3.5 prompt selection (PS, Figure 7, Table 4);
   3.6 the prompt held fixed: RandOpt's search under the chosen prompt (GD on GQA, GB on OLMo; GB-2 gated out); one
   sentence on robustness (R8b.6).
4. **A first-order account of answer-preference changes (R4, R5).** Folded-gradient derivation (one paragraph plus
   appendix); Figure 3; the content vs letter-position control; per-item test (I5); Figure 4.
5. **Limits of the account.** Vision weights (GPU-A F1/S2d falsified); σ = 0.005; chain-of-thought (P0); the
   OmniSpatial winner (C1-1). As important as Section 4.
6. **Why selection finds shared shifts (R8c, R8b.7, case study).** Open with the transfer table (R8b.7, exploratory):
   selection gains transfer only where the prompt leaves accuracy unclaimed, and not with the prompt held fixed; this
   is the bridge from Section 3 to the theory. T1, T2, T4 checks; T3 as derivation; the OmniSpatial
   search: split mismatch, matched transfer (+2.67), front tilt, calibration (not explained); Figure 5.
7. **Recommendations for evaluating weight-space search.** Same-run baselines at matched test-time budget; SC and a
   prompt chosen on the selection set as default baselines; base-reproduction checks; report member vs vote
   accuracy; report selection compute.
8. **Limitations.** Scope list from STORY ("Scope and limits").
9. **Related work.** RandOpt / Neural Thickets; ES for LLMs; self-consistency and test-time scaling; prompt
   sensitivity and prompt selection; 2608.10867; model soups and weight averaging; linear-response / NTK-style
   analyses; evaluation reliability and selection bias.
10. **Appendix.** R9 ledger; fold-operator derivation and byte-exact checks; per-σ and per-block tables; SC@K and
    vote curves; prompts and scorers (all three candidates per task, verbatim); G3; O2 fidelity analysis; cross-paper
    P2/P3 (R8) incl. non-comparable rows; compute ledger.

## Generated figures and tables
All in `paper/figures/thickets-or-tilts/`, regenerated by `python paper/figures/scripts/thickets_or_tilts.py`
(`... same-run` for Figures 7–8 and Table 4 only). Use the PDFs in LaTeX.

| File | Caption guidance |
|---|---|
| `fig7_same_run_prompt_selection` | "RandOpt K = 50 − SC@50 per row (paired 95% CI), under RandOpt's prompt and under the prompt chosen on RandOpt's selection set. GQA uses a re-implementation." Main headline figure |
| `fig8_prompt_damage` | "Filled: locked damage measure (plain/direct); open: against the selection-chosen prompt (post-hoc). Damage accompanies both RandOpt wins but does not decide the outcome (Qwen-1.5B)." |
| `table4_same_run_prompt_selection.md` | the four rows, both prompts |
| `fig1b_sc_vs_randopt`, `fig2_sc_at_k`, `table1_sc_vs_randopt` | cross-paper (appendix); "published RandOpt numbers; not same-run" |
| `fig3_tilt_law` | "first-order prediction (no fitted coefficients) vs measurement"; state the slopes |
| `fig4_localization` | – |
| `fig5_selection_transfer` | (c) and (d) exploratory; (c) is "little measured ensemble gain", not "cancellation" |
| `fig6_cot_regime` | (a) "similarly susceptible questions", not "equivalent mechanism" |
| `table2_preregistration_ledger.md`, `table3_tilt_law_by_model_sigma.md` | appendix |

## Abstract (draft; keep every number and qualifier)
Random-perturbation post-training (RandOpt) samples thousands of Gaussian weight perturbations of a pretrained model,
keeps the best on a small selection set and majority-votes them, reportedly rivalling PPO and GRPO. We ask what the
weight search buys. Running RandOpt at its own settings in four rows (GSM8K with two Qwen models and OLMo-2-1B; GQA
with Qwen2.5-VL-3B), we reproduce its published numbers and decompose its gain into a vote and a shift shared across
questions. Where the selected models barely beat the base model, self-consistency over 50 samples of the unperturbed
model beats RandOpt (−2.65 and −1.59 points): the gain is the vote, and sampling votes better. Where RandOpt leads (+3.47 on
GQA, replicated; +8.72 on OLMo), its own prompt costs the base model 11.3 and 31.6 points, and the selected
perturbations recover part of the lost accuracy; on GQA a single direct-answer generation scores 64.7 vs RandOpt's
63.5 (no difference detected). Re-running RandOpt's search under the chosen prompt, its selected models are no better
than the base model, and it matches sampling on GQA and trails it on OLMo (−2.12 points). Choosing among three prompts on RandOpt's own
selection questions (600 generations vs its 1,000,000) and sampling matches or beats RandOpt in all four rows. A
first-order account, computed from the base model's gradient and each perturbation's noise with no fitted
coefficients, tracks how perturbations shift answer preferences (r = 0.787–0.938 across five models in two families),
localizes the effect to middle language layers, and explains why selection favours shared shifts; it fails for vision
weights, large σ and chain-of-thought correctness, and does not explain most of one selected model's transferable
gain. Confirmatory tests were locked before their runs; we release all locks, data and code.

## Final checklist
- [ ] Every number traceable to `RESULTS_MASTER.md`; every locked test in the appendix ledger, with outcome.
- [ ] "matches or beats" (never "beats") for PS; Qwen-3B and GQA described as "no difference detected".
- [ ] RandOpt's wins under its own prompts stated plainly (GQA, OLMo).
- [ ] The damage caveat (Qwen-1.5B, post-hoc) appears with Figure 8 and in Section 3.4.
- [ ] OLMo "search on top of the prompt" not cited (fidelity gate failed).
- [ ] GQA re-implementation stated wherever GQA results appear.
- [ ] No "exact", "noise alone", "cancels", "acts as re-sampling", "every decision pre-registered", "is
      self-consistency", "is prompt tuning".
- [ ] C1-1 (not explained) in the abstract, introduction and Section 5.
- [ ] Prior work credited; code, locks and per-item outputs released.

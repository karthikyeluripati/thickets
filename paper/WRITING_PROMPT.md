# Prompt for drafting the paper

Give the writer this file plus `paper/STORY.md` (the argument) and `paper/RESULTS_MASTER.md` (the only allowed
numbers). Where they disagree, RESULTS_MASTER wins on numbers and STORY wins on emphasis.

---

You are writing a research paper for a top machine-learning venue (target: ICML 2027 main track; fallback: TMLR).
It re-examines *Neural Thickets: Diverse Task Experts Are Dense Around Pretrained Weights* (Gan & Isola, ICML 2026
Spotlight, arXiv 2603.12228) and its algorithm **RandOpt**: sample N Gaussian weight perturbations of a pretrained
model, keep the top K on a small selection set, majority-vote their answers.

**The paper's centre** is a decomposition of what RandOpt's weight search buys: a vote, plus shifts in answer form
shared across questions, which in our settings a prompt chosen on the same selection data also supplies (when the
candidates include a format-fixing prompt); with that prompt held fixed, the search adds nothing in any row. A first-order account explains what perturbations do to answers and why selection finds
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
     write "beats in all rows". **Always pair it with RV-1**: with public harness templates alone it holds in three of
     four rows, and on Qwen-1.5B RandOpt is ahead (+4.17 [+1.97, +6.44]); the pooled six-candidate rule holds in all
     four. The honest form: "when the candidates include a format-fixing (boxed-answer) prompt".
   - Under RandOpt's own prompts RandOpt is **ahead** on GQA and OLMo. Say so plainly; it is half the explanation.
   - "Damage" claims: the locked measure (plain/direct vs RandOpt's prompt) supports the prediction in 4/4 rows; the
     post-hoc measure against the selection-chosen prompt does not separate the rows (Qwen-1.5B: +10.2, RandOpt still
     lost). Never write "RandOpt wins exactly where its prompt damages the model".
   - "With the prompt held fixed, weight search adds nothing over sampling" is shown in **all four rows**: GD (GQA,
     direct: −0.32 [−1.29, +0.65], equivalent), GB (OLMo, boxed: −2.12 [−3.49, −0.83]), RV-2a (Qwen-3B, boxed: −1.36
     [−2.35, −0.38]) and RV-2b (Qwen-1.5B, boxed: −4.32 [−5.91, −2.81]), SC ahead in the last three; in all four the
     selected models are no better than base. Cite these, not G4's re-used members or O2's members (fidelity failed).
     GB-2 stays INVALID-ENV in the ledger; RV-2b is its re-run under the RV lock's gates.
   - randopt.py's base print (RV-4): for Qwen-1.5B it prints 73.00 reproducibly while our runner measures 68.00 on
     byte-identical prompts (test bases agree). Disclose it; use the same-engine base for selection-set statistics
     (R8b.1, R8b.7); never claim the population mean is below base on the Qwen rows (it is level: −0.4, −0.3 pp).
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
   that prompt selection beats RandOpt on every task or model (two tasks, ≤ 8B; public templates alone lose on
   Qwen-1.5B); that any prompt choice works (it needs a format-fixing candidate); anything
   about models above 8B; that the first-order account governs CoT accuracy; that we explain the OmniSpatial winner;
   that the OmniSpatial winner's gain is "explained" by the prompt: OS shows it is prompt-specific and not detectably
   better than a selection-chosen prompt (−1.50 [−5.35, +2.23]), but that is no difference, not equivalence.
8. **Scope statements that must appear.** The first-order theory covers direct answers; the Claim 1 rows use
   step-by-step prompts, so Claim 1 rests on its own experiments, not on the theory. Theory checks T1/T2 are
   criteria-first exploratory. The three prompt candidates were fixed before PS but chosen after earlier results.
   RandOpt's search under the chosen prompt was run in all four rows (GD, GB, RV-2a, RV-2b). Both RandOpt wins
   replicate with a second population (G2R; RV-3, fast runner). Five public templates were fixed in the RV lock. Vote efficiency: sampling's vote adds more than RandOpt's in three rows, less on OLMo
   (exploratory); never write "sampling supplies the vote at least as well". "Nearby better models are common" only
   with the 11.8–58.3% numbers (same-engine base) and "the population mean is at or below base".
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
3. **A practical baseline, and its condition:** choosing the prompt on RandOpt's own selection data (0.06% of its
   selection compute) then sampling matches or beats RandOpt in all four rows when the candidates include a
   format-fixing prompt, and in three of four with public harness templates alone (R8b.5, RV-1); RandOpt's search run
   under the chosen prompt adds nothing over sampling in all four rows (R8b.9).
4. **A first-order account** of perturbation-induced answer-preference changes, validated across five models in two
   families, localized to middle language layers, with demonstrated limits (R4, R5, R7), and a theory linking it to
   selection (R8c).

## Sections (as in `paper/latex/main.tex`)
1. **Introduction.** The question; the one-paragraph answer (STORY); contributions; Figure 1 (overview,
   `fig1_overview`) and Figure 2 (headline, `fig7`).
2. **Background and setup.** RandOpt; its shared-stream noise; models, tasks, RandOpt's own prompts and scorers; the
   plan-lock protocol, gates and decision rules (Rule 2 wording); the GQA re-implementation.
3. **What weight search buys (R8b).** 3.1 same-run rows under RandOpt's prompts (Table 1); 3.2 the member-vs-vote
   decomposition; 3.3 the shared shift repairs prompt damage: GQA (G3, shift analysis [exploratory], G4), OLMo (O2),
   the Qwen control (Q2) and the damage caveat; 3.4 prompt selection (PS, Figure 2, Table 1), robustness (R8b.6,
   RV-5) and candidate independence (RV-1); 3.5 the prompt held fixed in all four rows (GD, GB, RV-2a, RV-2b).
4. **A first-order account of answer-preference changes (R4, R5).** Folded-gradient derivation (one paragraph plus
   appendix); Figure 3; the content vs letter-position control; per-item test (I5); Figure 4.
5. **Limits of the account.** Vision weights (GPU-A F1/S2d falsified); σ = 0.005; chain-of-thought (P0); the
   OmniSpatial winner (C1-1). As important as Section 4.
6. **Why selection finds shared shifts (R8c, R8b.7, case study).** Open with the transfer table (R8b.7, exploratory):
   selection gains transfer only where the prompt leaves accuracy unclaimed, and not with the prompt held fixed; this
   is the bridge from Section 3 to the theory. Then R8b.8 (exploratory): the selected "experts" are prompt-specific
   (same population, near-zero rank correlation across prompts, top-50 overlap 0) and on GQA reproduce the direct
   prompt's answers; not on OLMo. Say "in our settings", never "there are no task experts". T1, T2, T4 checks; T3 as derivation; the OmniSpatial
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
(`... same-run` for Figures 7, 8, 10 and Tables 1 and 4). Use the PDFs in LaTeX; Table 1 is `table1_same_run.tex`.

| File | Caption guidance |
|---|---|
| `fig7_same_run_prompt_selection` | "RandOpt K = 50 − SC@50 per row (paired 95% CI): under RandOpt's prompt (circles), under the prompt chosen on the selection set (squares), under the public template chosen the same way (diamonds), and RandOpt searched under the chosen prompt (triangles). GQA uses a re-implementation." Main headline figure (Figure 2) |
| `fig10_transfer` | "Selection gain vs test gain, ten searches (exploratory); same-engine bases for the Qwen rows." |
| `fig9_prompt_specific_experts` | exploratory; ρ 0.025 / 0.171; top-50 overlap 0 |
| `fig1_overview` | Figure 1; built by `fig_overview.py` |
| `fig8_prompt_damage` | "Filled: locked damage measure (plain/direct); open: against the selection-chosen prompt (post-hoc). Damage accompanies both RandOpt wins but does not decide the outcome (Qwen-1.5B)." |
| `table4_same_run_prompt_selection.md` | the four rows, both prompts |
| `fig1b_sc_vs_randopt`, `fig2_sc_at_k`, `table1_sc_vs_randopt` | cross-paper (appendix); "published RandOpt numbers; not same-run" |
| `fig3_tilt_law` | "first-order prediction (no fitted coefficients) vs measurement"; state the slopes |
| `fig4_localization` | – |
| `fig5_selection_transfer` | (c) and (d) exploratory; (c) is "little measured ensemble gain", not "cancellation" |
| `fig6_cot_regime` | (a) "similarly susceptible questions", not "equivalent mechanism" |
| `table2_preregistration_ledger.md`, `table3_tilt_law_by_model_sigma.md` | appendix |

## Abstract (draft; keep every number and qualifier)
See `paper/latex/main.tex` (the abstract there is the current draft and follows these rules).

## Final checklist
- [ ] Every number traceable to `RESULTS_MASTER.md`; every locked test in the appendix ledger, with outcome.
- [ ] "matches or beats" (never "beats") for PS; Qwen-3B and GQA described as "no difference detected"; RV-1's
      Qwen-1.5B "RandOpt ahead" stated wherever the headline is (abstract, introduction, Section 3.4, Figure 2).
- [ ] Prompt held fixed: four rows (GD, GB, RV-2a, RV-2b); both RandOpt wins replicated (G2R, RV-3).
- [ ] RV-4 disclosed; no "population mean below base in every row".
- [ ] RandOpt's wins under its own prompts stated plainly (GQA, OLMo).
- [ ] The damage caveat (Qwen-1.5B, post-hoc) appears with Figure 8 and in Section 3.4.
- [ ] OLMo "search on top of the prompt" (O2 members) not cited (fidelity gate failed).
- [ ] GQA re-implementation stated wherever GQA results appear.
- [ ] No "exact", "noise alone", "cancels", "acts as re-sampling", "every decision pre-registered", "is
      self-consistency", "is prompt tuning".
- [ ] C1-1 (not explained) in the abstract, introduction and Section 5.
- [ ] Prior work credited; code, locks and per-item outputs released.

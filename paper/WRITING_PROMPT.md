# Prompt for drafting the paper (give this whole file plus `paper/RESULTS_MASTER.md` and `paper/STORY.md` to the writer)

> **2026-10-07:** `paper/STORY.md` is now the canonical story (three claims; Claim 1 led by the same-run result R8b).
> Where this file's spine, contributions or abstract differ from STORY.md, STORY.md wins. Hard rules below still apply,
> as updated in rules 5–7. A full rewrite of this prompt follows once the remaining GPU runs are in.

---

You are writing a research paper for a top machine-learning venue (target: ICML 2027 main track; fallback: TMLR).
It re-examines *Neural Thickets: Diverse Task Experts Are Dense Around Pretrained Weights* (Gan & Isola, ICML 2026
Spotlight, arXiv 2603.12228) and its algorithm **RandOpt**:
1. Sample N Gaussian weight perturbations of a pretrained model.
2. Keep the top K on a small selection set.
3. Majority-vote their answers.

**The paper's centre** is a validated first-order (linear-response) account of how small weight perturbations change
a model's answer preferences, **together with the experimentally demonstrated limits of that account**.
- A self-consistency baseline is a secondary, practical finding.
- The investigation of one selected "expert" is a case study.

Write with the precision of a careful empirical paper, not the tone of a takedown. Be fair to the original authors.
Say exactly how far each result goes and no further.

## Hard rules
1. **Numbers.**
   - Use only numbers from `RESULTS_MASTER.md`; never invent one.
   - Report n and CIs where given.
   - Do not round in a way that strengthens a claim (e.g. write "5.7–10.7 pp", never "6–11").
2. **Process language.** Write exactly: "Confirmatory tests were specified in plan locks committed before their
   corresponding runs; exploratory analyses are identified separately."
   - Never write "every decision was pre-registered".
   - Label exploratory results in the text and in figure captions.
3. **Report every locked test with its outcome** (the R9 ledger), including falsified, inconclusive and failed-gate
   results. Do not describe a set of tests as "validated" if any of them was inconclusive.
4. **Mechanism language: say only what was tested.**
   - **Prediction.** It needs the model-and-task gradient at the base model plus the perturbation's noise, with no
     fitted coefficients. It is a first-order approximation that is highly correlated with measurements (r =
     0.915–0.938 for tilts; slopes 0.83–0.95; pooled per-item r = 0.77). Never call it "exact", and never say
     "from the noise alone".
   - **Selection** *favours* answer-preference tilts. These tilts are an **incomplete** explanation: removing the
     4-parameter content shift leaves +2.33 of the winner's +2.67 pp fresh gain (C1-1, not explained). This
     failed intervention is part of the main story.
   - **The 47-model vote** shows little measured ensemble gain (+0.2 pp [−1.0, +1.3]). Do **not** claim it "cancels"
     tilts. Report descriptively: the transferable gain sits in a few top candidates (top-10 votes +2.0 to +2.8 pp;
     the winner +2.67), while the other members average near zero.
   - **Chain-of-thought.** Perturbations and temperature sampling affect *similarly susceptible questions* (ρ =
     0.946). That does **not** establish equivalent mechanisms, and P1-B (persistent candidate effects, r = 0.74)
     **rejects** pure re-sampling. Never write "perturbations act as re-sampling".
5. **Self-consistency language.**
   - SC@50 achieves **higher point estimates** than the published RandOpt results on the two base-reproduction-gated
     GSM8K settings (79.8 vs 76.4; 88.2 vs 87.1).
   - This is **not** a same-run comparison and **not** an equivalence test. "MATCHES" is our pre-registered decision
     label and must be presented as such.
   - Against the paper's TT-MV on gated rows: 5.7–10.7 pp.
   - **Same-run (R8b):** at N = 5000, K = 50, RandOpt reproduces 77.18 (paper 76.4) and SC@50 is ahead by 2.65 pp
     [−4.32, −0.99]. Say "in the one setting we ran same-run"; at K = 10 no difference was detected.
6. **GQA** is an **unresolved comparison**, not a counterexample and not a confirmation. Our base-reproduction gate
   fails there, so it neither establishes a residual RandOpt advantage nor supports a universal SC conclusion.
7. **Do not claim:**
   - that RandOpt "is" self-consistency;
   - that neural thickets do not exist;
   - anything about non-Qwen models beyond the OLMo-2-1B / ARC tilt-law test (S1-A), or about models above 8B;
   - that the first-order account governs chain-of-thought accuracy (P0: r ≈ 0 with CoT correctness changes);
   - that we explain the winner's improvement (we do not; C1-1, S1-B);
   - that SC beats RandOpt in general (one same-run row; GQA unresolved).
8. **Credit prior work** (R10):
   - "format thickets" (the original paper);
   - selection bias toward the selection set (arXiv 2608.10867);
   - the informal geometric intuition ("A Thicket by Any Other Name");
   - sequence-length dependence ("When does RandOpt work?").

## Title
*Thickets or Tilts? A Pre-Registered Re-Examination of Random Weight Perturbation as Post-Training*

## The spine (what the paper argues)
1. **A linear-response account of answer preferences, with tested limits.**
   - In direct-answer settings, a small perturbation of the language weights changes answer preferences in a way a
     first-order prediction tracks closely, with no fitted coefficients.
   - It was tested in six locked settings across three Qwen models. Five were confirmed (r = 0.915–0.938 for tilts;
     0.930 for per-question contrasts in RandOpt's own GQA setting), and one was inconclusive on a failed
     reliability gate, then replicated with a remedy fixed in advance.
   - The response is localized block by block to the middle language layers (r = 0.978).
2. **Where it stops.**
   - **Vision weights:** a whole-model first-order account fails (r = 0.137).
   - **σ ≥ 0.005:** r drops to 0.31.
   - **Chain-of-thought accuracy:** it is unrelated to the first-order signal.
   - **The selected winner:** its transferable gain is mostly not a content-prior shift.
3. **What this explains about selection** (case study: one OmniSpatial search).
   - Selection favours tilts toward answer content that the selection labels reward. This explains part of the
     apparent specialization and why its value depends on the evaluation distribution.
   - A train/test format mismatch made the selected model look like a pure "mirage". On fresh matched items it
     keeps +2.67 pp of a +8.5 pp selection-set gain.
   - Calibration shows the tilt does not account for most of that.
4. **A practical baseline.** Self-consistency at matched K has higher point estimates than published RandOpt on the
   two gated GSM8K settings. Perturbations and sampling affect similarly susceptible questions, but perturbed models
   carry persistent effects. GQA remains unresolved.

**Closing message:** we understand a reproducible mechanism behind answer-preference changes and its limits. We do
**not** have a complete causal explanation of the winning candidate's improvement, and the failed calibration test
is the clearest evidence of that boundary.

## Contributions (as a numbered list in the introduction)
1. **A first-order account of perturbation-induced answer-preference changes.**
   - A "folded gradient" handles RandOpt's shared Gaussian stream. Predictions use the base-model gradient and the
     noise, with no fitted coefficients.
   - Locked tests: three models, two tasks, per item, per layer block (R4, R5).
2. **Experimentally demonstrated limits of that account:**
   - vision nonlinearity (R4);
   - σ breakdown (R4);
   - no link to chain-of-thought correctness (R7);
   - failure to explain the selected winner's gain (R3, C1-1).
3. **A case study of what selection favours.** Answer-preference tilts aligned with the selection labels, an
   incomplete explanation (R3, R6). Selection inflation and a split-format artifact, corrected with matched splits
   (R1, R2).
4. **A practical baseline result.** Self-consistency vs published RandOpt, with base-reproduction gates and an
   unresolved GQA comparison (R8).

## Sections
1. **Introduction:**
   - motivation;
   - the question (what do selected perturbations change, and how much of the gain needs weight search?);
   - contributions;
   - Figure 1.
2. **Background and setup.**
   - RandOpt; its shared-stream noise (each tensor's noise is a prefix of one Gaussian stream); byte-exact
     verification.
   - Models, tasks, RandOpt's own prompts and scorers.
   - The plan-lock protocol, gates and decision rules (Rule 2 wording).
3. **A first-order account of answer-preference changes** (R4, R5).
   - Folded-gradient derivation (one paragraph plus an appendix).
   - Figure 3, the content vs letter-position control, the per-item test (I5), Figure 4.
   - Present the six locked tests with their outcomes, including R2.
4. **Limits of the account:**
   - vision nonlinearity (GPU-A F1/S2d falsified);
   - the σ = 0.005 breakdown;
   - chain-of-thought (P0: 22% of answers flip at σ ≤ 0.002, unrelated to the first-order signal);
   - the winner's gain (C1-1).

   This section is as important as Section 3.
5. **Case study: what selection favoured on OmniSpatial** (R1–R3, R6, Figure 5).
   - The "front" tilt and its gold-front concentration (exploratory).
   - The noise-plus-gradient predictor works through label priors (τ-check).
   - Matched-split transfer (+2.67).
   - The ensemble result (47-model vote +0.2; top-10 +2.0 to +2.8), reported descriptively.
   - Calibration (not explained).
6. **Self-consistency as a baseline** (R7, R8; Table 1; Figures 2 and 6).
   - Gated rows, non-comparable rows, GQA unresolved.
   - Similarly susceptible questions; persistence rejecting pure re-sampling; the TT-MV temperature observation,
     stated neutrally.
7. **Evaluation recommendations:**
   - matched held-out splits;
   - SC baselines at a stated, tuned temperature;
   - base-reproduction checks;
   - K-matched compute reporting.
8. **Limitations:**
   - Qwen only, ≤ 8B;
   - cross-paper comparisons;
   - 4 of 6 baseline rows fail the gate;
   - one case-study search (σ ≤ 0.002);
   - GQA unresolved;
   - no causal account of the winner's gain;
   - exploratory analyses not confirmatory.
9. **Related work:**
   - RandOpt / Neural Thickets;
   - ES for LLMs;
   - self-consistency (Wang et al. 2023);
   - test-time scaling;
   - 2608.10867;
   - model soups and weight averaging;
   - linear-response / NTK-style analyses;
   - evaluation reliability and selection bias.
10. **Appendix:**
    - the full R9 ledger;
    - the fold-operator derivation and byte-exact checks;
    - per-σ and per-block tables;
    - SC@K curves;
    - prompts and scorers;
    - compute ledger (≈ $60 of H100 time for the confirmatory runs).

## Generated figures and tables
All in `paper/figures/thickets-or-tilts/`, regenerated by `python paper/figures/scripts/thickets_or_tilts.py`. Use
the PDFs in LaTeX.

| File | Caption guidance |
|---|---|
| `fig1b_sc_vs_randopt` | "point estimates; published RandOpt numbers; GQA not comparable (base gate)". Figure 1(a), the schematic, is still to be drawn |
| `fig2_sc_at_k` | same caveats |
| `fig3_tilt_law` | "first-order prediction (no fitted coefficients) vs measurement"; state the slopes |
| `fig4_localization` | – |
| `fig5_selection_transfer` | (c) and (d) are exploratory; (c) is "little measured ensemble gain", not "cancellation" |
| `fig6_cot_regime` | (a) "similarly susceptible questions", not "equivalent mechanism" |
| `table1_sc_vs_randopt` (.md/.tex) | – |
| `table2_preregistration_ledger.md` | – |
| `table3_tilt_law_by_model_sigma.md` | – |

## Abstract (draft; keep every number and qualifier)
Random-perturbation post-training (RandOpt) samples thousands of Gaussian weight perturbations of a pretrained model,
keeps the best on a small selection set and majority-votes them, reportedly rivalling PPO and GRPO. We ask what the
selected perturbations change. In direct-answer settings, we find that a first-order approximation, computed from
the base model's gradient and each perturbation's noise with no fitted coefficients, closely tracks how
language-weight perturbations shift answer preferences (r = 0.915–0.938 across two tasks and two Qwen models; r = 0.930
per question in RandOpt's own GQA setting). The shift is localized to the middle language layers. The account has
sharp, experimentally demonstrated limits: it fails for vision weights, breaks down at σ = 0.005, does not predict
chain-of-thought correctness, and does not explain most of a selected model's transferable gain (+2.33 of +2.67
points survive removing the content shift). In a case study, selection favoured perturbations whose answer-preference
tilts matched the selection labels, and a train/test format mismatch made the selected model look like a pure
"mirage"; on fresh matched items it keeps +2.67 of +8.5 points. As a practical baseline, self-consistency over 50
samples, with no weight search, has higher point estimates than published RandOpt results on the two GSM8K settings
where we reproduce the base model, while a GQA comparison remains unresolved. Confirmatory tests were locked before
their runs; we release all locks, data and code.

## Final checklist
- [ ] Every number traceable to `RESULTS_MASTER.md`; every locked test in the appendix ledger, with outcome.
- [ ] No "exact", "noise alone", "cancels", "acts as re-sampling", "every decision pre-registered", or "6–11".
- [ ] C1-1 (not explained) appears in the abstract, the introduction and Section 4.
- [ ] GQA described as unresolved in the abstract, Section 6 and Limitations.
- [ ] The 47-model vote membership stated (the winner + 46 measured top-50 members).
- [ ] Prior work credited; code, locks and per-item outputs released.

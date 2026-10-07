# Prompt for drafting the paper (give this whole file plus `paper/RESULTS_MASTER.md` to the writer)

---

You are writing a research paper for a top machine-learning venue (target: ICML 2027 main track; fallback: TMLR).
The paper is a **pre-registered, mechanistic re-examination** of *Neural Thickets: Diverse Task Experts Are Dense
Around Pretrained Weights* (Gan & Isola, ICML 2026 Spotlight, arXiv 2603.12228) and its algorithm **RandOpt**:
1. Sample N Gaussian weight perturbations of a pretrained model.
2. Keep the top K on a small selection set.
3. Majority-vote their answers.

Write with the precision of a careful empirical paper, not the tone of a takedown. Be fair to the original authors.
Credit what they got right, and state plainly what holds and what does not.

## Hard rules
1. **Use only numbers from `RESULTS_MASTER.md`.** Never round in a way that changes a conclusion, never invent a
   number, and never report a result without its n and CI where one is given.
2. **Separate pre-registered from post-hoc results** everywhere. Pre-registered tests cite their lock; post-hoc
   results are labelled "post hoc" in the text and in figure captions.
3. **Report every pre-registered test, including failures and inconclusive results** (R9 ledger). Failures are part
   of the contribution, because they mark the boundaries of the account.
4. **Do not claim:**
   - that RandOpt "is just self-consistency" in general (GQA contradicts it);
   - that neural thickets do not exist;
   - anything about non-Qwen models or about models above 8B;
   - that the first-order law governs chain-of-thought accuracy (P0 shows it does not);
   - that our self-consistency numbers are same-run head-to-heads with RandOpt (they are cross-paper comparisons,
     gated by base reproduction).
5. **Credit prior work explicitly** (R10):
   - "format thickets" is the original paper's own finding;
   - selection bias toward the selection set was reported by arXiv 2608.10867;
   - the geometric intuition appears informally in "A Thicket by Any Other Name".
6. Every claim in the abstract must map to a table or figure.

## Title
**Recommended:** *Thickets or Tilts? A Pre-Registered Re-Examination of Random Weight Perturbation as Post-Training*

Alternatives:
- *What Do Random Weight Perturbations Buy? Revisiting Neural Thickets with Self-Consistency and First-Order
  Analysis*
- *Neural Thickets, Revisited: Self-Consistency, Selection, and First-Order Answer Tilts*

## The one-paragraph story (the paper's spine)
Random-perturbation search reports large gains, but what it buys depends on how the model answers.
- **On chain-of-thought reasoning**, a small weight perturbation behaves much like re-sampling the reasoning path:
  it flips the same questions that temperature sampling flips. A properly sampled self-consistency baseline, with no
  weight search, **matches or exceeds RandOpt's reported accuracy wherever we reproduce the paper's base model**
  (GSM8K, Qwen2.5-1.5B and 3B). It also exceeds the paper's own majority-vote baseline by 6–11 points, which suggests
  that baseline was under-tuned.
- **On direct-answer tasks**, a perturbation's effect is a **first-order tilt of the model's answer preferences**.
  The tilt is predictable from the noise vector alone (r ≈ 0.92–0.94 across two tasks and two model families; per
  item; localized to the middle language layers). Selection preferentially picks tilts toward whatever answer
  content the selection set's labels reward. Such gains transfer only where the label distribution matches. Voting
  over ~50 selected perturbations averages the tilts back to the base model (+0.2 pp on fresh matched data).
- **The vision tower is the nonlinear exception**, which also explains why a whole-model first-order account fails.
- **Methodologically**, selection inflation (+8.5 → +2.7 pp) and a train/test format mismatch can manufacture
  apparent "expert mirages". We show how matched splits fix this.
- **The one result we cannot explain:** RandOpt's reported GQA gain (+12.4) is twice that of self-consistency
  (+5.2). We report it as an open problem.

## Contributions (as a numbered list in the introduction)
1. **A strong, cheap baseline the field should use.** Self-consistency at matched K matches or exceeds RandOpt's
   reported reasoning accuracy on comparable rows; on the other rows its gains are similar in size (R8).
2. **A first-order law for random weight perturbations of answer preferences.**
   - Exact, parameter-free prediction from the noise stream, using a "folded gradient" that handles RandOpt's shared
     Gaussian stream.
   - Validated in 6 pre-registered tests across 3 Qwen models (R4).
   - Localized block by block to the middle language layers (R5).
   - Bounded by vision nonlinearity and by σ ≥ 0.005.
3. **What selection finds.** On direct-answer tasks, selection finds label-prior tilts (R3, R6), and the ensemble
   vote cancels them (R2). On CoT tasks, perturbations act as re-sampling (R7).
4. **Evaluation methodology.** Selection inflation quantified on fresh matched data, and a split-format artifact
   exposed and corrected (R1, R2). Every decision was pre-registered, and all outcomes are reported (R9).

## Sections and what goes in each
1. **Introduction** (~1 page).
   - Motivation: random-perturbation post-training is cheap and parallel, and the claim that "experts are dense"
     matters for how we think about pretraining.
   - The question: *what* do the selected perturbations change, and how much of the gain needs weight search at all?
   - The contributions list; Figure 1.
2. **Background and setup.**
   - RandOpt; the shared-stream structure of its noise. Each tensor's noise is a prefix of one Gaussian stream, so
     equal-shape tensors share noise. Byte-exact verification.
   - Models, tasks, scorers. We use RandOpt's own prompts and scorers.
   - Pre-registration protocol: plan locks committed before outputs; gates; decision rules.
3. **Self-consistency is a strong baseline for RandOpt** (R8).
   - Table 1, Figure 2.
   - The base-reproduction gate and why 4 of 6 rows are "not comparable".
   - The TT-MV temperature issue, stated carefully as an observation, not an accusation.
   - The GQA exception.
4. **Direct-answer regime: perturbations are first-order answer tilts** (R4, R5).
   - The folded-gradient derivation (one paragraph plus an appendix).
   - Figures 3 and 4. Content vs letter-position control. Per-item law (I5).
   - Vision nonlinearity and the σ boundary.
5. **What selection finds and why votes collapse** (R1–R3, R6).
   - The OmniSpatial case: label-prior tilt toward "front"; gains concentrated on gold-front items.
   - The noise-only predictor forecasts fresh gains through label priors.
   - Matched-split transfer: winner +2.7; top-50 vote +0.2.
   - Figure 5.
6. **Chain-of-thought regime: perturbations as re-sampling** (R7).
   - Flip-pattern equivalence with sampling (Figure 6).
   - Persistent models, but much of that persistence is format damage.
   - Vote ≈ SC on GQA at K ≤ 32.
   - Why this explains Section 3.
7. **Evaluation pitfalls** (R1, R2): selection inflation; the split-format artifact; recommendations (matched held-out
   splits, SC baselines at tuned temperature, reporting K-matched compute).
8. **Limitations and open problems.**
   - The GQA gap.
   - Cross-paper comparisons.
   - Qwen only.
   - Our N = 5000 search used a smaller σ range than RandOpt.
   - MATH-500 and small-model bases not reproduced (the gate failed).
9. **Related work:**
   - RandOpt / Neural Thickets;
   - ES for LLMs;
   - self-consistency (Wang et al. 2023);
   - test-time scaling;
   - the BO follow-up (2608.10867);
   - weight-space ensembles / model soups;
   - linear-response / NTK-style analyses;
   - evaluation reliability and selection bias.
10. **Appendix:**
    - full pre-registration ledger with lock hashes;
    - derivation of the fold operator and the byte-exact verification;
    - all per-σ and per-block tables;
    - full SC@K curves;
    - prompt and scorer details;
    - compute ledger (≈ $60 of H100 time for all confirmatory experiments).

## Figures (exact specifications; data sources in brackets)
**Figure 1 (teaser, two panels).**
- (a) Schematic: base model → N perturbations → select top-K → vote. Annotate the two regimes (direct answer =
  first-order tilt; CoT = re-sampling).
- (b) Grouped bars per comparable row:

  | Row | Base | Paper RandOpt | SC@50 |
  |---|---|---|---|
  | GSM8K-1.5B | 58.8 | 76.4 | 79.8 |
  | GSM8K-3B | 79.8 | 87.1 | 88.2 |

  Add GQA as a hatched "not comparable" group (base 56.6 / RandOpt 69.0 / SC 59.2, our base 54.0) to show the
  exception honestly. [R8; `results/paper-analysis/p2/p2_results.json`, `p3/p3_results.json`]

**Figure 2.** SC@K curves (K = 1, 5, 10, 20, 50) for GSM8K-1.5B, GSM8K-3B and GQA, with horizontal lines for the
paper's Base, TT-MV and RandOpt. Annotate the K at which SC crosses RandOpt (GSM8K-1.5B: K ≈ 10–20; GSM8K-3B:
K ≈ 10). [R8]

**Figure 3 (the law).** Four scatter panels of predicted vs measured tilt, each with r, slope and the identity line:
| Panel | Setting | r | Source |
|---|---|---|---|
| a | Stage 2, Qwen3-VL "front" | 0.938 | `stage12/stage12_per_perturbation.json` |
| b | R1, Complex Logic "left" | 0.915 | `r1/r1_per_perturbation.json` |
| c | R2b, Qwen2.5-VL | 0.925 | `r2b/r2b_per_perturbation.json` |
| d | P0, GQA per-question contrast, coloured by σ | 0.98 / 0.91 / 0.31 | `p0/pod/pred.npy`, `meas.npy` |

Inset in (a): the per-item r distribution from I5 (median 0.88).

**Figure 4 (localization).**
- Paired bars of predicted vs measured variance share per 6-layer block (Stage 3B), with an inset scatter of block
  partial vs measured insertion (r = 0.978). [`stage3/stage3_results.json`]
- Small panel: share by relative depth for R1 and R2b.

**Figure 5 (selection and transfer, OmniSpatial).**
- (a) Format composition of SEARCH/RERANK vs official TEST. [R1]
- (b) Winner gain on RERANK (+8.0), official TEST (−2.67) and fresh matched H (+2.67, CI).
- (c) On H: single winner, top-10 vote, top-50 vote, control vote, with CIs. [R2]
- (d) Across 5000 candidates, "front" answer shift vs SEARCH gain (hexbin), with r. [R3]

**Figure 6 (CoT regime, GQA).**
- (a) Per-item flip propensity under 64 perturbations vs under T = 0.3 sampling (ρ = 0.946). [`p1/p1_results.json`
  and the per-item recomputation in `scripts/p1_analysis.py`]
- (b) Held-out accuracy: base, RandOpt-style top-8 vote, random-8 vote, SC@8 at T\* and T = 0.7. [R7]

**Table 1:** the full R8 table, including the gate column and the "not comparable" rows.
**Table 2:** the pre-registration ledger (R9): test, lock hash, statistic, threshold, outcome.
**Table 3 (appendix):** the tilt law by σ and by model.

**Style:**
- Colour-blind-safe palette; one consistent colour per method (Base grey, RandOpt orange, SC blue, controls light
  grey).
- No dual axes. CIs as error bars.
- "Post hoc" in the caption of any post-hoc panel (Figure 5d, Figure 6 annotations).

## Generated figures and tables (ready to use)
All in `paper/figures/thickets-or-tilts/`, regenerated by `python paper/figures/scripts/thickets_or_tilts.py`:
- `fig1b_sc_vs_randopt`: Figure 1(b). Figure 1(a), the schematic, still has to be drawn.
- `fig2_sc_at_k`
- `fig3_tilt_law`: inset shows the I5 per-item r distribution.
- `fig4_localization`: (a) block shares, (b) block-level scatter, (c) relative-depth replication.
- `fig5_selection_transfer`
- `fig6_cot_regime`
- `table1_sc_vs_randopt` (.md / .tex), `table2_preregistration_ledger.md`, `table3_tilt_law_by_model_sigma.md`

Each figure is available as .pdf, .png and .svg. Use the PDFs in LaTeX.

## Abstract (draft to refine; keep every number)
Random-perturbation post-training (RandOpt) samples thousands of Gaussian weight perturbations of a pretrained model,
keeps the best on a small selection set and majority-votes them, reportedly rivalling PPO and GRPO. We re-examine
what these "neural thickets" buy, using pre-registered experiments on three Qwen models. On chain-of-thought math
reasoning, a self-consistency baseline with no weight search matches or exceeds RandOpt's reported accuracy wherever
our base model reproduces the paper's (GSM8K: 79.8 vs 76.4 and 88.2 vs 87.1), and exceeds the paper's majority-vote
baseline by 6–11 points; weight perturbations flip the same questions as temperature sampling (ρ = 0.95). On
direct-answer tasks, a perturbation's effect is a first-order tilt of answer preferences that we predict from the
noise vector alone, without fitting (r = 0.92–0.94 across tasks and models; per item r = 0.88), localized to the
middle language layers. Selection picks tilts toward answer content that the selection labels reward, and a 50-model
vote averages them back to the base model on fresh matched data (+0.2 points). Selection inflation and a train/test
format mismatch can manufacture apparent "expert mirages", which we correct with matched splits. One reported gain,
on GQA, remains larger than self-consistency explains; we release all locks, data and code.

## Final checklist before submission
- [ ] Every number traceable to `RESULTS_MASTER.md`; every pre-registered test in Table 2.
- [ ] GQA exception, cross-paper caveat and Qwen-only scope stated in the abstract or introduction, and in
  Limitations.
- [ ] Prior work credited (format thickets, selection bias).
- [ ] No sentence implies "RandOpt is just self-consistency" without the qualifier "on comparable chain-of-thought rows".
- [ ] Code, plan locks and per-item outputs released.

# Behavioral diversity of the N = 5,000 weight-space neighbourhood

**CPU only. Post-hoc, descriptive.** Branch:
`research/expert-mirages-behavioral-diversity`, from `research/expert-mirages-paper`.

- **Inputs:** stored outputs only; no GPU, no new perturbations, no modified
  artifacts.
- **Script:** `paper/figures/scripts/behavioral_diversity.py`, seeded
  (`20261006`), with stages `all`, `direction` and `score_matched`.
- **Results:** `results/paper-analysis/behavioral-diversity/`.
- **Figures:** `paper/figures/behavioral-diversity/`.
- **Tables:** `paper/tables/table_behavioral_*`, `table_selection_diversity`,
  `table_cross_split_geometry`.

**Terminology.** Spectral effective rank is an **effective behavioral
dimension** of observed prediction variation. It is not a number of independent
models, and no "N candidates behave like X models" statement is made here
(see Part 11).

| Metric | Result |
|---|---:|
| Nominal candidates | 5,000 |
| SEARCH questions | 200 |
| Median answers changed vs base | **12 of 200** (IQR 7–23; σ-medians 5 / 9 / 16 / 31) |
| Answer-space entropy effective rank | 78.9 (rank ceiling 275) |
| Answer-space participation ratio | 39.5 |
| Correctness-space entropy effective rank | 43.5 (ceiling 123) |
| Standardized answer effective rank | 211.6 (ceiling 275) |
| Null answer effective rank (σ-stratified per-question permutation, 1,000 reps) | 82.9 [82.8, 83.0] |
| Observed/null ratio | **0.951** |
| PCs for 90% variance (answer, raw) | 78 (null 80) |
| SEARCH top-50 effective rank (answer, raw) | 31.9 (ceiling 49) |
| Top-50 percentile vs σ-matched / σ + score-matched random committees | 10.9 / 14.2 |
| SEARCH↔RERANK audit CKA (answer) | 0.534 (σ-stratified null 0.479 [0.469, 0.487]; within-σ excess 0.00–0.16) |
| Final classification | **`NO_EFFECTIVE_DIMENSION_RESULT`** |

## Part 0: input verification (all passed)

- **SEARCH population:** 5,000 unique SEARCH candidates in lock order, 200
  questions each, with no missing cells, and 1,250 per σ. Every candidate's
  SEARCH count equals the locked `correct_count`.
- **Base:** base SEARCH 72/200, identical to re-parsed raw base outputs.
- **Raw spot check:** 25 randomly chosen candidates (fixed seed) re-parsed from
  their raw `.json.gz` files are identical to the matrix.
- **Audit:** 500 audit candidates from the protocol's precommitted indices, 125
  per σ, identical to the master-table flag. All 500 have complete RERANK200
  predictions.
- **SEARCH top 50:** `ranked_ids[:50]` equals the lock's `top50` field. σ mix:
  27 × 0.002, 11 × 0.001, 12 × 0.0005, 0 × 0.00025. 7 members are also audit
  candidates.
- **TEST:** predictions exist for exactly those 50 candidates, 561 questions
  each.

## Part 2: how far does a nearby model move? (`table_behavioral_distance`)

A typical candidate changes **12 of 200 answers** (mean 15.5; P95 35; max 47).
Of those changes, 8 affect correctness, split almost evenly: 4 repairs and 4
regressions in the median (means 4.3 and 5.4). About 5 more are wrong→wrong
switches. Movement scales with σ:

| σ | Median answers changed | Correctness changes | Repairs / regressions (mean) |
|---:|---:|---:|---:|
| 0.00025 | 5 | 3 | 2.2 / 1.3 |
| 0.0005 | 9 | 6 | 2.9 / 2.8 |
| 0.001 | 16 | 10 | 4.6 / 6.0 |
| 0.002 | 31 | 19 | 7.6 / 11.5 |

Two candidates, both at σ = 0.00025, are answer-identical to the base.

## Parts 3–6: spectral effective dimension against an item-fragility null

**Null.** For every question and every σ stratum independently, candidate
identities are shuffled (1,000 replicates). This keeps:

- each question's per-σ answer distribution, and so its fragility;
- σ-specific behavior.

It destroys any persistent candidate signature across questions.

| Population | Representation | Raw r_ent | Raw r_PR | k90 | Null r_ent median [95%] | Obs/null | Std. r_ent (null) |
|---|---|---:|---:|---:|---:|---:|---:|
| All 5,000 | answer (800) | 78.9 | 39.5 | 78 | 82.9 [82.8, 83.0] | 0.951 | 211.6 (221.4) |
| All 5,000 | correctness (200) | 43.5 | 26.3 | 40 | 45.1 [45.0, 45.1] | 0.965 | 106.6 (110.1) |
| σ = 0.00025 | answer | 23.8 | 20.0 | 20 | 23.9 | 0.995 | 44.6 (44.7) |
| σ = 0.0005 | answer | 39.3 | 32.3 | 33 | 40.0 | 0.983 | 69.4 (70.0) |
| σ = 0.001 | answer | 67.2 | 53.1 | 56 | 70.1 | 0.959 | 121.9 (125.0) |
| σ = 0.002 | answer | 117.5 | 79.9 | 105 | 130.4 | 0.901 | 220.1 (236.4) |

Full rows for correctness, PR, k95 and ceilings are in
`table_behavioral_effective_dimension.tex`. Ceilings differ by representation
and by σ, so each row is compared only with its own null.

**Reading:**

1. **Low raw rank comes from item fragility.**
   - Variance is concentrated on a few fragile questions: PC1 holds 11.5% and
     10 PCs hold 34% for answers.
   - So the raw effective rank (79) is far below its ceiling (275).
   - The null keeps per-question fragility and reproduces 95% of that rank
     (82.9). That concentration is therefore a property of the *items*, not of
     coherent candidate structure.
2. **Standardization removes most of it.** After per-feature standardization the
   rank is 212 of 275. Answer changes are spread across many weakly coupled
   directions.
3. **Residual candidate coherence is real but small.**
   - At the whole-population level, the observed values lie below every one of
     1,000 null replicates (percentile 0), but only by 3–5%.
   - The excess grows with σ, from 0.995 at σ = 0.00025 to 0.90 at σ = 0.002.
     At σ = 0.002 the participation ratio gap is 0.77. That is consistent with
     the known shared directional answer shift at σ = 0.002.
4. **Higher σ adds directions.** It mostly creates *new behavioral directions*,
   because more questions move: effective rank 24 → 39 → 67 → 118, with the
   ceiling growing in step. It also adds a modest shared component on top.

## Part 7: does SEARCH selection compress diversity? (`fig_selection_diversity`, `table_selection_diversity`)

The selected top 50 was compared against two references:

- 10,000 random committees with the identical σ mix;
- 2,000 committees also matched member-by-member on SEARCH correct count (±1;
  mean gap −1.7 questions).

Selected-group percentiles:

| Metric | σ-matched | σ + score-matched |
|---|---:|---:|
| Answer effective rank (raw / std.) | 10.9 / 35.5 | 14.2 / 22.6 |
| Correctness effective rank (raw / std.) | 53.5 / 39.3 | 94.0 / 46.0 |
| Mean pairwise disagreement | **0.0** (0.140 vs 0.152) | **23.4** (0.140 vs 0.141) |
| Mean error Jaccard | 84.4 | 15.2 |
| Mean distance from base | 20.2 | 74.0 |

The selected committee agrees with itself more than σ-matched random groups do.
That is entirely explained by selecting *accurate* candidates on the same
questions: two high scorers must agree wherever both are right. Once score is
matched, every metric is typical. **Outcome B:** selection does not
concentrate the committee into a narrow behavioral region. The majority-vote
failure is not explained by compressed committee diversity.

## Part 8: `SELECTED_COMMITTEE_TEST_GEOMETRY` (top 50 on TEST561)

- **Effective rank.** Answer effective rank is 37.0 of a maximum of 49 (raw;
  PR 29.5; k90 33; PC1 9%); correctness 35.4. Within-committee permutation null:
  38.7 and 37.0. The committee's deviations span **most of the available
  dimensions**.
- **Size of deviations.** Each member changes a median of 55.5 of 561 answers
  (range 16–85). Pairwise disagreement has a median of 12.7% (range 3.6–20.7%).
- **Vote outcome.** The base answer is the majority on 96.6% of questions, and
  the vote scores 259/561, the same as base.

So the committee is not redundant in direction. Its members move in many
different directions, but each moves only a little. Since every member keeps
the base answer on about 90% of questions, scattered, mostly unshared
deviations are outvoted by the shared base answer. This is a descriptive
account for the vote, not a population-wide inference.

## Part 9: does candidate geometry persist from SEARCH to RERANK? (500 audit candidates)

| Statistic | Pooled | Within σ = 0.00025 / 0.0005 / 0.001 / 0.002 |
|---|---:|---|
| Distance to base, Spearman | 0.89 | −0.06 / −0.12 / 0.10 / **0.19** (p = 0.03) |
| Pairwise distance, Spearman (Mantel) | 0.872 (σ-stratified null 0.854; pct 100) | −0.05 (pct 11) / 0.02 (76) / **0.14** (100) / **0.27** (100) |
| Linear CKA, answer | 0.534 (σ-stratified null 0.479; pct 100) | 0.158 vs 0.155 (62) / 0.279 vs 0.219 / 0.397 vs 0.310 / 0.569 vs 0.411 (all pct 100) |
| Direction-only CKA, answer (unit-length deviations) | 0.238 | 0.151 vs 0.147 (67) / 0.268 vs 0.215 / 0.390 vs 0.309 / 0.567 vs 0.414 |

- **Pooled correlations are almost entirely σ.** How far a candidate moves is
  set by σ, and within σ it barely persists (only σ = 0.002, ρ = 0.19).
- **A weak candidate-specific geometry persists at σ ≥ 0.0005 and survives
  removing movement magnitude** (direction-only CKA). Pairwise distances
  correlate at r = 0.14–0.27 within σ = 0.001–0.002. Nothing persists at
  σ = 0.00025, where candidates differ from base by about 5 answers.
- **Interpretation.** Perturbations are not pure per-question noise. A given
  perturbation has a reproducible, if weak, behavioral direction on disjoint
  questions. That direction is not a coherent skill: compare the forensic
  audit, the RERANK→TEST transfer of r = −0.27, and the Part N prompt
  sensitivity.

## Part 10: unique behavioral patterns (descriptive)

| Pattern | Unique among 5,000 | Largest group |
|---|---:|---:|
| Full 200-answer vectors | 4,890 | 6 |
| Sets of changed questions | 4,872 | 6 |
| Correctness signatures {−1, 0, +1} | 4,172 | 62 (a 2-change pattern) |

Nearly every candidate is unique, but uniqueness is cheap here. The median
candidate's nearest neighbour differs by 6 answers, 14% have a neighbour within
1 answer, and the largest groups are 1–3-change patterns. Uniqueness is **not**
evidence of large functional diversity.

## Part 11: independent-equivalent candidate count: not reported

No defensible calibration exists here, and two observable statistics point in
opposite directions:

- **Spectral structure.** The item-marginal-preserving independent reference at
  the *same* N reproduces 95–99.5% of the observed effective rank. On this
  statistic the population behaves almost like independent candidates, so the
  implied M ≈ N.
- **Score maxima.** The paper analysis (Part H) found that 500 random
  candidates reach a RERANK maximum of +7.5 pp. An independent-flip reference
  with the same per-item rates gives only +5.25 [4.0, 7.0] at M = 500. Matching
  the observed maximum would need **more** than 500 independent candidates,
  because shared σ-driven shifts fatten the tail.

The equicorrelation formula was not used. **No effective candidate count is
reported.**

## Part 14: classification

**`NO_EFFECTIVE_DIMENSION_RESULT`**

- **Low-dimensionality is mostly item fragility.** Observed effective ranks sit
  within 1–10% of a null that keeps per-question fragility (overall ratio
  0.95). The apparent low-dimensionality relative to the ceiling is mostly
  fragility, not candidate redundancy.
- **The residual is small.** It is statistically clear (a narrow null band) but
  small in effect size, and largest at σ = 0.002, i.e. driven mainly by one σ.
- **No selection compression.** Once accuracy is matched, the SEARCH-selected
  top 50 shows typical diversity.
- **Not strong.** `STRONG` would need dimension "materially below" the null, or
  a compressed top 50. Neither holds.
- **Not moderate either, though it's close.** The compression is consistent
  across representations, but its size (≈ 5%) does not support any
  redundancy claim. A reader could argue for `MODERATE`. We do not, because no
  paper-level statement about redundancy would survive the effect size.

## Part 15: implications for the paper

1. **Are 5,000 candidates behaviorally low-dimensional?** Only in the sense
   that variance concentrates on a few fragile questions: raw rank 79 of 275,
   standardized 212 of 275.
2. **Stronger than per-question fragility alone?** Only by about 5%, clearest at
   σ = 0.002.
3. **Does effective dimension grow with σ?** Yes. Higher σ mainly moves more
   questions (new directions), plus a modest shared component.
4. **Does SEARCH selection compress diversity?** No, once member accuracy is
   matched.
5. **Does geometry persist SEARCH → RERANK?** Weakly, within σ ≥ 0.0005
   (CKA excess 0.06–0.16; pairwise r ≤ 0.27). Not at σ = 0.00025.
6. **Does the TEST committee occupy a small behavioral space?** No. It spans
   about 37 of 49 possible dimensions, but each member deviates on only about
   10% of answers, so the vote collapses to base.
7. **Defensible independent-equivalent count?** No. The statistics disagree.
8. **Does this materially strengthen the paper?** No new headline. It does
   rule out an alternative explanation for the committee failure, namely
   "redundant members", which helps.

**Recommendation: `DO_NOT_USE_EFFECTIVE_DIMENSION_IN_PAPER`** as a claim. Do
not add an effective-population-size or behavioral-redundancy thesis, and do
not rewrite the title (Part 16 is therefore not drafted). Optional appendix
material that is safe to use:

- the behavioral-distance table: "a typical neighbour changes 12 of 200
  answers";
- one sentence that the committee failure is *not* due to redundant members:
  diversity is typical once accuracy is matched, and on TEST members deviate in
  many directions but rarely;
- the weak cross-split persistence, as context for the RERANK→TEST
  non-transfer.

## Reproducibility

```
python paper/figures/scripts/behavioral_diversity.py all            # verification, Parts 2-10, figures, tables
python paper/figures/scripts/behavioral_diversity.py direction      # direction-only CKA (Part 9)
python paper/figures/scripts/behavioral_diversity.py score_matched  # σ + score-matched committees (Part 7)
python paper/figures/scripts/behavioral_diversity.py figures        # figures and tables from saved JSON
```

All random streams are seeded. Candidate IDs (search order, σ, top 50, audit
500, question IDs) are in `candidate_ids.json`, and all statistics are in
`behavioral_diversity_results.json`. The null samples for Part 7 are in
`selection_*_samples.npz`. Total CPU time is about 25 minutes.

# Improvement density vs cross-sample transfer

> **POST HOC, CPU only.** Branch: `research/expert-mirages-transfer-density`,
> from `0b83006`. It uses stored predictions only: no GPU, no new
> perturbations, no modified artifacts. TEST predictions appear only in
> Part 10, for seed 9504111 and the frozen top 50. No TEST label was used to
> design any analysis.
>
> Repeated split and replay trials describe the stability of **one observed
> prediction matrix**. They are not independent repetitions of search or
> training. All intervals are descriptive percentiles over splits or replays.
> There are no p-values and no confirmatory tests.

- **Script:** `paper/figures/scripts/transfer_density.py`, seed `20261007`,
  stages `all`, `figures` and `origin`.
- **Results:** `results/paper-analysis/transfer-density/`.
- **Figures:** `paper/figures/transfer-density/`.
- **Tables:** `paper/tables/table_transfer_density`, `table_selection_replay`,
  `table_replay_holdout_origin`, `table_repair_selectivity`,
  `table_sigma_transfer`, `table_rank_persistence`, `table_behavior_vs_task`.

**Terminology:**

- empirical improvement density: ρ;
- joint improvement density;
- empirical transfer rate: τ(A→B) = P(g_B ≥ m | g_A ≥ m);
- transfer lift: τ / ρ_B.

Nothing here estimates a population-level "solution density".

**Pooled estimates.** τ and lift are pooled over split trials: τ = mean joint
density / mean ρ_A. The per-trial ratio is undefined whenever a split has no
improvers, and averaging only the defined trials biases it. Lift is reported
only where the mean joint improver count is at least 2 candidates. Splitting a
*fixed* question pool makes ρ_A and ρ_B anticorrelated across splits, which
pushes pooled lift below 1 even with no candidate identity. **The reference
is therefore the item-fragility null, not 1.**

## Part 0: verification (all passed)

- **SEARCH population:** 5,000 unique candidates × 200 SEARCH questions, 1,250
  per σ. The candidate order is identical to the behavioral-diversity audit.
- **Audit:** 500 precommitted audit candidates (125 per σ), all complete on
  SEARCH200 and RERANK200 with the same identities.
- **Question pool:** 400 train-side questions with 400 unique IDs and **0
  duplicate image hashes**. Gold letters match the frozen JSONL files.
- **Strata:** SEARCH200 and RERANK200 have *identical* subtask × answer-letter
  strata (12 cells).

## Part 2: all 5,000 candidates, SEARCH100 vs SEARCH100 (5,000 stratified splits)

| Threshold (net questions) | ρ_A | ρ_B | Joint ρ | τ(A→B) | Lift | Null lift |
|---|---:|---:|---:|---:|---:|---:|
| > 0 (≥ 1) | 0.336 | 0.339 | 0.102 | 0.304 | 0.89 | 0.89 |
| ≥ +3 pp (≥ 3) | 0.053 | 0.055 | 0.0015 | 0.028 | 0.50 | 0.31 |
| ≥ +5 pp (≥ 5) | 0.0055 | 0.0057 | 0.0001 | 0.011 | -- (≈ 0.3 joint candidates) | -- |
| ≥ 2 net | 0.152 | 0.155 | 0.014 | 0.090 | 0.58 | 0.55 |

On 100 questions ≥ +1 pp equals > 0, so the "≥ +1 pp" row is that row.

- **Density versus joint.** A third of nearby models "improve" on any 100
  questions. At ≥ +3 pp about 5% of candidates look like improvers on one
  half, but only **0.15%** improve by ≥ +3 pp on both halves. **2.8% of A's
  ≥ +3 pp improvers repeat on B.** At > 0 the transfer equals the item-fragility
  null (lift 0.89 vs 0.89).
- **Gain correlation across halves.** Pooled r = 0.26 (null 0.17, which is
  nonzero because σ mean differences survive the null). Within σ:

  | σ | 0.00025 | 0.0005 | 0.001 | 0.002 |
  |---|---:|---:|---:|---:|
  | r | −0.01 | 0.02 | 0.09 | **0.22** |
  | null | 0.00 | 0.00 | 0.00 | 0.00 |

## Part 3: transfer as evaluation sets grow (500 audit candidates, 400-question pool, 2,000 splits per n)

| n | Gain r (null) | τ(>0) / ρ_B | τ(≥ +3 pp) / ρ_B [cutoff] | Lift ≥ +3 pp (null) |
|---:|---:|---:|---:|---:|
| 25 | 0.04 (−0.01) | 0.28 / 0.26 | 0.28 / 0.26 [≥ 1, same as > 0] | 1.06 (1.09) |
| 50 | 0.07 (−0.03) | 0.31 / 0.32 | 0.15 / 0.12 [≥ 2] | 1.23 (0.73) |
| 75 | 0.10 (−0.03) | 0.35 / 0.35 | 0.12 / 0.07 [≥ 3] | 1.72 (0.80) |
| 100 | 0.13 (−0.02) | 0.36 / 0.37 | 0.14 / 0.10 [≥ 3] | 1.47 (1.10) |
| 150 | 0.19 (−0.04) | 0.37 / 0.38 | (≥ 5: joint < 2 candidates) | -- |
| 200 | **0.24** (−0.04) | 0.38 / 0.39 | (≥ 6: joint < 2) | -- |

- **Identity stabilizes as n grows.** The gain correlation rises steadily from
  0.04 to 0.24 while the null stays at about −0.03.
- **But it stays weak.** At > 0, transfer never departs from ρ_B. At
  ≥ +3 pp, only 12–15% of A-improvers repeat on B.
- **Limited resolution.** With only 500 candidates, joint counts at ≥ +3 pp for
  n ≥ 150 fall below 2 candidates. That is the resolution limit of the audit
  population.

## Part 4: item-fragility null

The null shuffles candidate identities per question within σ, which keeps every
question's per-σ correctness distribution.

- **Above the null:** the observed transfer exceeds it at ≥ 2 net questions,
  in rank overlap, and in gain correlation, mainly at σ ≥ 0.001.
- **At the null:** at > 0 the observed transfer is indistinguishable from it.

Candidate identity carries a real but small, σ-concentrated task-advantage
signature beyond item fragility.

## Part 5: by σ (1,250 each, SEARCH100 vs SEARCH100)

| σ | ρ_A(> 0) | Joint | τ(> 0) | Lift (null) | ρ_A(≥ +3 pp) | τ(≥ +3 pp) (null τ) | Gain r (null) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00025 | 0.479 | 0.192 | 0.401 | 0.83 (0.85) | 0.037 | 0.000 (0.005) | −0.01 (0.00) |
| 0.0005 | 0.377 | 0.113 | 0.299 | 0.79 (0.82) | 0.051 | 0.019 (0.013) | 0.02 (0.00) |
| 0.001 | 0.287 | 0.064 | 0.223 | 0.77 (0.73) | 0.061 | 0.015 (0.017) | 0.09 (0.00) |
| 0.002 | 0.200 | 0.039 | 0.194 | **0.95 (0.52)** | 0.065 | **0.062 (0.012)** | **0.22 (0.00)** |

**Larger σ raises apparent density at strict thresholds** (ρ ≥ +3 pp goes
0.037 → 0.065) while lowering it at > 0. **Only σ = 0.002 carries an
above-null transferable component:** τ(≥ +3 pp) is 5× the null and r = 0.22.
Even there, 94% of its ≥ +3 pp improvers on one half are not ≥ +3 pp on the
other. Smaller σ produces apparent improvers whose identity does not persist
at all.

## Part 6 and Part 12: improver-set stability and rank persistence (all 5,000; `fig_improver_set_stability`, `table_rank_persistence`)

**Set overlap:**

- Jaccard of improver sets: 0.18 at ≥ 1 net (null 0.17) and 0.017 at ≥ 3
  (null 0.008).
- Top-1% overlap: 3.1% against 0.9% under the null and 1% at random.

**Top-K on A, scored on B:**

| Top K on A | Share also top K on B | Enrichment (null) | Mean percentile rank on B | Mean net gain on B (pool −0.54; null top-K −1.55 to −0.82) |
|---:|---:|---:|---:|---:|
| 10 | 3.1% | 15.4 (1.1) | 48.4 | −0.29 |
| 50 | 3.1% | 3.1 (0.9) | 50.2 | −0.50 |
| 250 | 5.8% | 1.16 (0.8) | 48.8 | −0.42 |

**Search ranking has a little predictive content for extreme identity** (3%
vs 0.2% for the top 10). It has **essentially none for typical rank**: the
mean percentile is about 50 on B, where 50 is random. The top-K on A do score
above what item fragility predicts on B, but still below base on average.

## Part 7: replay of the two-stage procedure (500 audit candidates; SEARCH100 → top 50 → RERANK100 winner → fresh HOLDOUT200; 5,000 trials)

| | Mean | 95% of trials |
|---|---:|---:|
| Winner SEARCH gain | +3.77 pp | |
| Winner RERANK gain | +5.72 pp | |
| Winner HOLDOUT gain | **+1.71 pp** | [−2.5, +5.5] |
| Optimism gap (RERANK − HOLDOUT) | **+4.01 pp** | |
| P(HOLDOUT > 0) / ≥ +3 pp / ≥ +5 pp | 0.74 / 0.35 / 0.06 | |
| Same-pool mean HOLDOUT gain | −0.22 pp | |
| Item-fragility null: winner RERANK / HOLDOUT / P(H > 0) | +4.25 / −0.53 / 0.33 | |

**Winners are concentrated:**

- 94% of winners are σ = 0.002.
- Only 100 distinct winners appear across 5,000 replays.
- The most frequent winner (seed 9500991) wins 30% of replays, and the top 5
  win 65%.

**Holdout gain by question origin.** The replay reproduces the identical
winners (`table_replay_holdout_origin`):

| HOLDOUT items from | Winner HOLDOUT gain | Pool mean | P(winner > 0) |
|---|---:|---:|---:|
| RERANK200 (≈ 100 items) | **+3.38 pp** | +0.04 | 0.78 |
| SEARCH200 (≈ 100 items) | **−0.15 pp** | −0.49 | 0.41 |

**The winner's fresh-holdout gain is almost entirely on RERANK-origin
questions.** On fresh SEARCH-origin questions, drawn from the same official
train split with the same strata, the replay winner sits at base (+0.3 pp
above the pool mean). The "transferable" improvement in this pool is
attached to a particular item subset, consistent with the forensic finding of
fragile, templated RERANK items tipped by σ = 0.002.

**Descriptive cross-check.** This is not used in design. Of the 10 most
frequent replay winners, 4 are in the frozen SEARCH top 50. Their stored TEST
gains are −1.07, −1.43, +1.07 and +1.07 pp, i.e. about base.

## Part 8: replay as the candidate pool grows (2,000 trials per M; `fig_selection_replay_vs_pool_size`, `table_selection_replay`)

| M | K | Winner RERANK gain | Winner HOLDOUT gain | Optimism gap | P(H > 0) | P(H ≥ +3 pp) | Null HOLDOUT |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 5 | +2.13 | +0.21 | 1.92 | 0.48 | 0.07 | −0.33 |
| 50 | 5 | +2.79 | +0.61 | 2.18 | 0.55 | 0.14 | −0.38 |
| 100 | 10 | +3.70 | +0.95 | 2.74 | 0.61 | 0.20 | −0.47 |
| 250 | 25 | +4.83 | +1.33 | 3.50 | 0.68 | 0.27 | −0.48 |
| 500 | 50 | +5.72 | +1.71 | 4.01 | 0.74 | 0.35 | −0.53 |

From M = 10 to 500:

- **The selected RERANK gain rises +3.6 pp; the fresh-holdout gain rises +1.5
  pp.** The apparent score grows about 2.4× faster than held-out performance,
  and the optimism gap doubles (1.9 → 4.0 pp).
- **The null shows pure selection inflation.** Under the item-fragility null,
  RERANK still rises (+1.8 → +4.3) but HOLDOUT stays near −0.4 to −0.5.
- **Sensitivity.** The fixed-fraction variant (K = 10% of M) gives the same
  pattern: RERANK +0.17 → +5.71, HOLDOUT +0.11 → +1.71.

## Part 10: repair selectivity (`fig_repair_selectivity_transfer`, `table_repair_selectivity`)

S = P(correct | base wrong) − P(wrong | base correct), recomputed from raw
predictions.

| Seed 9504111 | Repairs / base-wrong | Repair rate | Damage rate | S |
|---|---:|---:|---:|---:|
| SEARCH | 10 / 128 | 0.078 | 0.069 | +0.009 |
| RERANK | 20 / 121 | 0.165 | 0.051 | **+0.115** |
| TEST | 23 / 302 | 0.076 | 0.147 | **−0.071** |

- **Audit 500, SEARCH vs RERANK.**
  - Pooled r = 0.32 against a within-σ permutation null of 0.14.
  - Within σ: 0.11, 0.17, 0.26, 0.31, at null percentiles 88, 97, 99.9 and 100.
  - Repair selectivity persists modestly *within* the train-side
    distribution, increasingly with σ.
- **Frozen top 50, RERANK vs TEST.**
  - r = −0.15 (Spearman −0.05; permutation null 0.03; percentile 11). It does
    not transfer.
  - Mean S on TEST is −0.018, and 9504111 has the highest RERANK selectivity
    of the 50 and the most negative TEST value.

## Part 11: behavioral direction vs task advantage (audit 500, within σ; `table_behavior_vs_task`)

| σ | Answer CKA obs. / null | Gain r (null pct) | Selectivity r (null pct) |
|---:|---:|---:|---:|
| 0.00025 | 0.158 / 0.156 | 0.12 (91) | 0.11 (88) |
| 0.0005 | 0.279 / 0.219 | 0.20 (98) | 0.17 (97) |
| 0.001 | 0.397 / 0.311 | 0.27 (99.7) | 0.26 (99.9) |
| 0.002 | 0.569 / 0.410 | 0.30 (99.9) | 0.31 (100) |

**Within the train-side distribution the decoupling is *not* supported.**
Where behavioral geometry persists, at σ ≥ 0.0005, task gain and repair
selectivity persist to a similar, modest degree. Where geometry does not
persist (σ = 0.00025), neither does task advantage.

Decoupling appears only across the shift to TEST (selectivity r = −0.15; gain
r = −0.27 from the main analysis) and across item subsets (the RERANK-origin
vs SEARCH-origin holdout split above). Behavioral persistence on TEST cannot be
measured for random candidates, because only the top 50 have TEST outputs.
So the statement *"stable behavioral directions whose task utility is
unstable"* is **not established** and should not be used.

## Part 16: classification

**`MODERATE_TRANSFER_INSTABILITY`**

1. **Density versus joint density.** Finite-sample improvement density far
   exceeds joint density: about 5% vs 0.15% at ≥ +3 pp on 100 questions, and
   34% vs 10% at > 0. Transfer rates are low: 3% at ≥ +3 pp for all 5,000, and
   12–15% for the audit pool.
2. **Persistence, but small.** Candidate identity is partially persistent
   above the item-fragility null, mostly at σ = 0.002. Gain r is 0.22–0.30
   and grows with question count, from 0.04 at n = 25 to 0.24 at n = 200.
3. **Selection inflates the score.** Two-stage selection inflates the selected
   score by about 4 pp at M = 500, and inflation grows with M (RERANK rises
   2.4× faster than holdout).
4. **Why not `STRONG_TRANSFER_DENSITY_GAP`:** within the train-side pool the
   replay winner's fresh-holdout gain is positive (+1.7 pp; P > 0 = 0.74) and
   above the null (−0.5). That is partial, not near-null, transfer, even
   though the decomposition shows it sits on one item subset.
5. **Why not `COHERENT_TRANSFER_SIGNAL`:** improver sets barely overlap, joint
   density is a small fraction of marginal, and on fresh SEARCH-origin items
   the winner is at base.

## Part 17: paper decision

1. **What is empirical improvement density at each threshold?** On 100
   questions: > 0: 34%; ≥ +3 pp: 5.3%; ≥ +5 pp: 0.55%. It grows with σ at
   strict thresholds.
2. **How much survives jointly?** 10% at > 0, 0.15% at ≥ +3 pp, 0.01% at
   ≥ +5 pp.
3. **What is the empirical transfer rate?** τ is 0.30 at > 0 (the null gives
   0.28) and 0.028 at ≥ +3 pp (null 0.012).
4. **How does transfer change with set size?** Identity stabilizes slowly:
   gain r goes 0.04 → 0.24 from n = 25 to 200, and τ at > 0 stays at ρ_B.
5. **How does it change with σ?** Only σ = 0.002 shows above-null transfer
   (r = 0.22). Smaller σ creates apparent improvers with no persistent
   identity.
6. **Does the pool contain stable task-advantage identities?** Weakly, mostly
   at σ = 0.002, and largely tied to a RERANK item subset.
7. **Does two-stage selection inflate the apparent gain?** Yes. The optimism
   gap is +4.0 pp at M = 500 (+5.7 RERANK vs +1.7 holdout).
8. **Does inflation worsen with M?** Yes, from 1.9 to 4.0 pp over M = 10 → 500.
9. **Does candidate identity improve transfer relative to the null?** Yes, the
   holdout is +1.7 vs −0.5 under the null. But the gain is item-subset
   specific: +3.4 on RERANK-origin vs −0.15 on SEARCH-origin items.
10. **Does repair selectivity transfer?** Modestly within the train-side
    distribution (r up to 0.31). Not RERANK → TEST for the top 50 (r = −0.15).
11. **Can behavioral direction persist without task advantage?** Not within
    the train-side distribution: the two co-vary. The claim is unsupported
    there.
12. **Does this materially strengthen the paper?** Yes, as a supporting
    result. It turns the single-candidate anecdote into a procedure-level
    estimate:
    - selection optimism of about 4 pp that grows with pool size;
    - low joint density relative to marginal density;
    - "transferable" advantage that is item-subset specific.

**Recommendation: `USE_TRANSFER_DENSITY_AS_SECONDARY_RESULT`.** Suggested
placement:

- **Main text, one figure:** `fig_selection_replay_vs_pool_size`, with the
  origin table as a sentence or inset.
- **Main text, one sentence:** the density-versus-joint numbers.
- **Appendix:** the rest.

## Part 18: optional framing text (supported wording only)

This is a moderate result, so the current framing ("Expert Mirages") should
stay. If the authors want transfer density more prominent, this wording is
supported:

> *Thesis addition.* Random local perturbations make the pretrained VLM appear
> surrounded by task-improving neighbours on any finite question sample (about
> 5% of candidates gain ≥ +3 pp on 100 questions), but under 3% of those
> apparent improvers repeat on a disjoint sample. Replaying the precommitted
> two-stage selection on the train-side pool, the selected winner's score
> overstates its fresh-sample gain by about 4 pp at a pool of 500. The
> overstatement grows with pool size, and what does transfer is concentrated
> on one item subset.

An optional title in the requested style, only if the authors restructure
around this:

*Auditing Weight-Space Experts: Finite-Sample Density, Selection, and Transfer
in a Vision-Language Model*

**Do not claim:**

- that Neural Thickets' solution density is wrong;
- any "true" or non-empirical density;
- any generalization beyond Qwen3-VL-8B on OmniSpatial Perspective Taking with
  this population.

## Reproducibility

```
python paper/figures/scripts/transfer_density.py all      # verification, Parts 2-12, figures, tables (~20 min CPU)
python paper/figures/scripts/transfer_density.py origin   # holdout-by-origin decomposition (re-runs the primary replay; verifies identical winners)
python paper/figures/scripts/transfer_density.py figures  # figures and tables from saved JSON
```

Seeds are fixed. Saved artifacts:

- **Split indices:** `splits_primary_search100x100.npz`,
  `splits_size_curve.npz`.
- **Question pool:** `question_pool_400.csv`, with ID, phase, subtask, answer
  and image hash.
- **Candidates:** `candidate_ids.json`.
- **Per-trial primary replay:** `replay_primary_trials.csv`.
- **Per-candidate repair selectivity:** `audit_repair_selectivity.csv`,
  `top50_repair_selectivity.csv`.
- **All summaries and nulls:** `transfer_density_results.json`.

# Perturbation-radius audit: GO / NO-GO for "How Local Should Weight-Space Search Be?"

> **POST HOC, CPU only.** Branch: `research/perturbation-radius-audit`, from
> `research/expert-mirages-transfer-density`. It uses stored predictions only:
> no GPU, no new perturbations, no modified artifacts. Replay intervals describe
> one observed prediction matrix, not independent searches.

- **Script:** `paper/figures/scripts/radius_audit.py`, seed `20261008`.
- **Results:** `results/paper-analysis/radius-audit/`. This includes the
  per-trial replays, split indices, candidate IDs with σ, and the summary JSON.
- **Figures:** `paper/figures/radius-audit/`.
- **Tables:** `paper/tables/table_radius_*`.

## Verdict

**`NO_RADIUS_PAPER` → `ABANDON_RADIUS_FRAMING`.**

**The selection side is clean.** As σ grows:

- mean accuracy falls monotonically;
- variance, the selected winner's score, and selection optimism all rise
  monotonically.

**The transfer side is not.**

- **Holdout is flat, then steps.** The fresh-holdout gain of the selected
  winner is ≈ 0 at three of the four radii and positive only at σ = 0.002.
- **The positive point sits on one item subset.** At σ = 0.002 the gain lives
  entirely on RERANK-origin questions (+3.8 pp vs −0.1 pp on SEARCH-origin
  questions).
- **No mismatch to report.** The same radius maximizes both the selected score
  and the holdout, so the hoped-for "most impressive radius ≠ most
  transferable radius" result does not occur.
- **The evidence is thin.** Paired radius differences have replay intervals
  spanning 0. With four radius levels and 125 candidates each, there isn't
  enough to support a radius paper.

## Part 0: verification (all passed)

- **SEARCH population:** 5,000 candidates × 200 SEARCH questions, 1,250 per σ,
  base 72/200, 0 missing predictions.
- **Audit:** 500 audit candidates (125 per σ), complete on SEARCH200 and
  RERANK200.
- **Question pool:** 400 unique questions, 0 duplicate image hashes, pool order
  matches the transfer-density audit. σ is known for every candidate.

## Parts 1–2: what radius does to the SEARCH population (`fig_radius_search_distribution`, `table_radius_population`)

| σ | Mean gain | SD | P5 | Median | P95 | P99 | Max | Top-10 mean | > base | ≥ +3 pp | Median answers changed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00025 | +0.46 | 0.72 | −0.5 | +0.5 | **+1.5** | +2.0 | +2.5 | +2.20 | 61% | 0.0% | 5 |
| 0.0005 | +0.03 | 0.99 | −1.5 | 0.0 | **+1.5** | +2.25 | +3.0 | +2.70 | 42% | 0.3% | 9 |
| 0.001 | −0.70 | 1.40 | −3.0 | −0.5 | **+1.5** | +2.0 | +3.0 | +2.75 | 24% | 0.4% | 16 |
| 0.002 | −1.98 | 2.06 | −5.5 | −2.0 | **+1.5** | +3.0 | +5.5 | +3.70 | 14% | 1.3% | 31 |

- **Mean falls and spread widens.** The mean decreases monotonically (+0.46 →
  −1.98 pp) and the SD increases monotonically (0.72 → 2.06).
- **The widening is almost all on the left.** The 95th percentile is
  **identical (+1.5 pp) at every radius**. The 99th percentile is not
  monotonic. Only the extreme (max, top-10) grows, and clearly so only at
  σ = 0.002. The tail spread (q99 − median) grows from 1.5 to 5.0, mostly
  because the median falls.
- **Larger σ creates more variance, not better candidates.** It adds damage
  much faster than it adds a right tail.

## Part 3: same-procedure replay within each radius (`table_radius_transfer`)

**Procedure:**

- 125 candidates per σ → SEARCH100 top 25 → RERANK100 winner → fresh
  HOLDOUT200.
- 5,000 trials, with **identical stratified splits for all four radii in every
  trial**.

| σ | SEARCH | RERANK (selected) | HOLDOUT mean / median [95%] | P(> 0) | P(≥ 1) | P(≥ 3) | P(≥ 5) | Optimism | Transfer fraction | HOLDOUT on SEARCH- / RERANK-origin items | Null HOLDOUT |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---:|
| 0.00025 | +1.47 | +2.08 | −0.14 / 0.0 [−2.0, +1.5] | 0.32 | 0.13 | 0.00 | 0.00 | +2.22 | −0.07 | +0.42 / −0.70 | +0.09 |
| 0.0005 | +1.76 | +2.69 | +0.03 / 0.0 [−2.0, +2.0] | 0.42 | 0.26 | 0.01 | 0.00 | +2.65 | 0.01 | +0.07 / −0.01 | −0.10 |
| 0.001 | +2.11 | +3.39 | +0.03 / 0.0 [−3.0, +2.5] | 0.46 | 0.31 | 0.01 | 0.00 | +3.35 | 0.01 | −0.14 / +0.21 | −0.36 |
| 0.002 | +3.62 | +5.81 | **+1.85** / +2.0 [−2.5, +5.5] | 0.76 | 0.70 | 0.37 | 0.07 | +3.96 | 0.32 | **−0.06 / +3.76** | −0.50 |

**Paired ordering.** With the same splits for every σ:

- σ = 0.002 has the best holdout in 65% of trials and beats each smaller radius
  in 75–78% of trials.
- The three smaller radii are indistinguishable (pairwise wins 36–50%; mean
  differences ≤ 0.17 pp).
- Every pairwise mean difference has a replay interval spanning 0. For example,
  0.002 − 0.001 is +1.8 pp [−6.0, +2.5].

## Parts 4–6: selection versus transfer, best radius, optimism (`fig_radius_selection_vs_transfer`)

- **The selected score rises monotonically with σ** (+2.1 → +5.8 pp), as does
  **selection optimism** (+2.2 → +4.0 pp).
- **The fresh holdout is flat at ≈ 0 for σ ≤ 0.001, then jumps at σ = 0.002.**
  That is a step, not a smooth tradeoff curve.
- **The transfer fraction is ≈ 0 for the three smaller radii and 0.32 at
  σ = 0.002.**

**Best radius by metric:**

| Metric | Best σ |
|---|---:|
| Mean holdout | 0.002 |
| P(holdout > 0) | 0.002 |
| P(holdout ≥ +3 pp) | 0.002 |
| SEARCH maximum | 0.002 |
| Selected RERANK gain | 0.002 |
| Population mean | **0.00025** |

**Is the radius with the most impressive selected winner the one that
transfers best? Yes (both σ = 0.002).** The clean "mismatch" result does not
occur.

**But the σ = 0.002 holdout gain is not task-level transfer.**

- It is **+3.76 pp on RERANK-origin items and −0.06 pp on SEARCH-origin
  items**. Both sets come from the same official train split with identical
  strata.
- This is the RERANK-specific shift of fragile, templated items identified in
  the forensic and transfer-density audits.
- On the official TEST, the SEARCH top 50 (54% σ = 0.002) had a mean gain of
  −0.57 pp for their σ = 0.002 members. Their selected RERANK winner scored
  −2.67 pp.

## Part 7: density versus transfer by radius (recomputed from source; matches the transfer-density audit)

| σ | Density ≥ +3 pp on a 100-question half | Joint (both halves) | Joint count | Transfer rate | Gain r between halves |
|---:|---:|---:|---:|---:|---:|
| 0.00025 | 3.97% | 0.000% | 0.0 | 0.000 | −0.014 |
| 0.0005 | 5.32% | 0.091% | 1.1 | 0.017 | 0.019 |
| 0.001 | 6.37% | 0.095% | 1.2 | 0.015 | 0.088 |
| 0.002 | 6.73% | 0.403% | 5.0 | 0.060 | 0.220 |

Radius raises apparent ≥ +3 pp density modestly (4.0% → 6.7%) and raises joint
density from 0 to 0.4%. Transferable density stays at 0–6% of apparent
density.

## Part 8: repair versus damage (`fig_radius_repair_damage`, `table_radius_repair_damage`)

| σ | All 125, fresh HOLDOUT: repair / damage (%) | Selected winner, HOLDOUT: repair / damage (%) |
|---:|---:|---:|
| 0.00025 | 1.6 / 2.4 | 1.5 / 2.8 |
| 0.0005 | 2.3 / 4.1 | 2.6 / 4.2 |
| 0.001 | 4.1 / 7.6 | 4.5 / 7.4 |
| 0.002 | 7.6 / 13.8 | 9.4 / 10.6 |

- **Damage grows faster than repair as σ increases.** The population damage
  rate is 1.5–1.9× the repair rate at every radius, and the absolute gap grows
  from 0.8 to 6.2 points. This accounts for the falling mean.
- **Selection mainly filters damage.** At σ = 0.002 the selected winner has
  repair 9.4 vs damage 10.6. It cuts damage by 3.2 points relative to the
  population, rather than adding repairs.

## Part 9: does σ mainly scale movement? (appendix, `table_radius_behavior`)

| σ | Median answers changed | Answer effective rank / ceiling | Std. rank | Obs./null rank | Gain variance (pp²) |
|---:|---:|---:|---:|---:|---:|
| 0.00025 | 5 | 23.8 / 46 | 44.6 | 0.995 | 0.52 |
| 0.0005 | 9 | 39.3 / 73 | 69.4 | 0.983 | 0.98 |
| 0.001 | 16 | 67.2 / 136 | 121.9 | 0.959 | 1.96 |
| 0.002 | 31 | 117.5 / 275 | 220.1 | 0.901 | 4.24 |

Larger σ mostly moves the model further: changes, effective rank and its
ceiling all roughly double per step. It adds only a modest shared component
at σ = 0.002 (observed/null 0.90).

## Part 10: search budget within each radius (appendix, `fig_radius_budget`, `table_radius_budget`)

**For σ ≤ 0.001:**

- the selected RERANK gain rises with M (for example +1.9 → +3.4 at
  σ = 0.001);
- the holdout stays at 0.00 ± 0.17 for every M.

**At σ = 0.002** the holdout rises from +0.8 to +1.8 as M goes from 10 to 125,
with the same RERANK-origin caveat. More search inflates the selected score at
every radius, but buys fresh-sample gain only at the largest radius.

## Part 11: classification

**`NO_RADIUS_PAPER`.**

- **The holdout curve is a step at one radius,** not a peak or a monotonic
  rise. That step is confined to one item subset and contradicted on the
  official TEST.
- **The best radius depends on the metric:** 0.00025 for the population mean,
  0.002 for the selected score and holdout.
- **Paired holdout differences span 0.**
- **The selection-side regularities are clean but not new:**
  - optimism grows with σ;
  - P95 is flat while the mean falls and the left tail widens;
  - damage outpaces repair.

They are an extension of the existing expert-mirage findings, not a
standalone radius result.

## `STRICT_REVIEWER_VERDICT`

1. **Is mean performance monotonic with σ?** Yes. It decreases: +0.46, +0.03,
   −0.70, −1.98 pp.
2. **Is variance monotonic with σ?** Yes. SD goes 0.72, 0.99, 1.40, 2.06.
3. **Is the right tail monotonic with σ?** No. P95 is flat at +1.5 pp and P99
   is non-monotonic (2.0, 2.25, 2.0, 3.0). Only the extreme (max, top-10) grows,
   driven by σ = 0.002.
4. **Is fresh selected HOLDOUT performance monotonic?** No. It is flat at 0 for
   three radii, then +1.85 at σ = 0.002, and that is item-subset specific.
5. **Is there a clear optimal radius?** No. σ = 0.002 is "best" for holdout
   only through RERANK-origin items. The population mean is best at the
   smallest radius.
6. **Does selection optimism grow with σ?** Yes, monotonically (+2.2 → +4.0 pp).
7. **Does the repair/damage tradeoff explain the pattern?** It explains the
   falling mean: damage outpaces repair. It does not explain the holdout step.
8. **Is the pattern robust across 5,000 replay splits?** The selection-side
   trends are. The holdout ordering among the three smaller radii is a coin
   flip. σ = 0.002 wins 75–78% of pairings, but every paired interval spans 0.
9. **Could the result be summarized accurately in one sentence?** Only as a
   caveat-laden sentence: "Larger radii lower the mean, widen the left tail and
   inflate selected scores, while fresh-sample gains appear only at the largest
   radius and only on one item subset." That isn't a clean thesis.
10. **Would a skeptical reviewer consider the radius manipulation controlled?**
    Partly.
    - **For:** a fixed precommitted grid, equal counts, identical splits and
      selection rule.
    - **Against:** only four levels, one model and task, and 125 audit
      candidates per radius. Radius is confounded with movement magnitude and
      with a σ = 0.002-specific shared answer shift. There is also no fresh
      held-out set from a different distribution except TEST, where only the
      SEARCH top 50 were evaluated.

**Scores:**

| | /10 |
|---|---:|
| Clarity | 4 |
| Effect size | 4 |
| Robustness | 3 |
| Novelty | 3 |
| Paper-worthiness | 3 |

## Final

**`ABANDON_RADIUS_FRAMING`.**

Radius does not cleanly control a tradeoff between searchable score extremes
and transferable performance in this population. Part 13 (abstract skeleton)
is therefore not drafted.

**What can be reused, as one paragraph or appendix table in the existing
"Expert Mirages" paper:** within this population, larger σ

- lowers the mean;
- leaves the 95th percentile unchanged;
- raises damage faster than repair;
- monotonically raises selection optimism, from +2.2 to +4.0 pp under an
  identical replay.

All of these are descriptive and post hoc.

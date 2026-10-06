# Selection effects vs a shared RERANK-versus-TEST response (candidate 9504111)

> **POST HOC, CPU only. No GPU used.**
>
> - **Branch:** `research/perspective-selection-vs-shared-response`, from local
>   `c57bf9c`. The parent branches are not on the remote.
> - **Plan:** locked before analysis in commit `00aab64`
>   (`results/paper-analysis/selection-vs-shared-response/analysis_plan.md`,
>   with `source_hashes.json`).
> - **Code:** `scripts/selection_vs_shared.py`.
> - **Tests:** `tests/test_selection_vs_shared.py` (5 passing).
> - **Unchanged:** earlier reports. The margin standardization and the reserved
>   additivity check were not repeated or run.

## Plain-English answer

1. **Is the RERANK advantage shared?** Partly. The 50 SEARCH-selected
   candidates on average did a little better on RERANK (+1.37 pp) than on
   TEST (−0.35 pp). 9504111's advantage, though, was its own:
   - it was the best of all 50 on RERANK and the worst of all 50 on TEST;
   - its RERANK score is above every one of the 125 random σ = 0.002
     neighbours;
   - those random neighbours have no TEST results.
2. **Group-wide or winner-specific?** Of the −10.7 pp swing, about −1.7 pp
   accompanies a group-wide shift. About −9.0 pp reflects the winner becoming
   less exceptional relative to its peers. Within the σ = 0.002 members the
   split is −2.5 and −8.2.
3. **Does reselection on half of RERANK help on the other half?** Yes, inside
   RERANK. The half-selected winner gains +5.2 pp on the other half, against
   +1.4 for a random member. The increment is +3.8 pp, positive in 84% of
   splits. 9504111 is the winner in 65% of splits.
4. **What is missing?** TEST results for unselected σ = 0.002 neighbours, to
   learn whether ordinary perturbations also favour RERANK over TEST.
5. **GPU:** none.

**Labels:**

- `WINNER_EXCEPTIONALITY_IS_SELECTION_ASSOCIATED`
- `SHARED_SPLIT_RESPONSE_OBSERVED_IN_SELECTED_GROUP`
- `WITHIN_RERANK_SELECTION_HAS_PREDICTIVE_VALUE`
- `RANDOM_CONTROL_TEST_EVIDENCE_MISSING`
- `FUNCTIONAL_CAUSE_UNRESOLVED`

## Coverage (verified)

- **RERANK200:** 543 candidates (the SEARCH top 50 plus the 500 audit, with 7
  in both), all complete. Base 79/200.
- **TEST561:** the top 50 only, all complete. Base 259/561.
- **Audit TEST outputs:** only 7 of the 500 audit candidates have them (6 at
  σ = 0.002), solely because they are top-50 members. Audit TEST is reported
  as **not measured**; nothing is imputed.
- **RERANK rule:** RERANK count, then SEARCH count, then the frozen hash
  (`perspective.py::rank_validation`).

## Analysis A: how exceptional was the winner?

| Set | Selected how | Phase | n | Mean gain | Median | SD | Min / max | > base | Repairs / regressions (sum) | 9504111 rank (1 = best) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| A: 9504111 | SEARCH + RERANK rank 1 | RERANK | 1 | +8.00 | | | | 1 | 20 / 4 | |
| | | TEST | 1 | −2.67 | | | | 0 | 23 / 38 | |
| B: SEARCH top 50 (27 × 0.002, 11 × 0.001, 12 × 0.0005) | SEARCH-selected | RERANK | 50 | +1.37 | +1.0 | 1.96 | −2.5 / +8.0 | 34 | 471 / 334 | **1 of 50** |
| | | TEST | 50 | −0.35 | −0.36 | 0.96 | −2.67 / +1.25 | 17 | 921 / 1,019 | **50 of 50** |
| C: σ = 0.002 members of B | SEARCH-selected | RERANK | 27 | +1.89 | +1.5 | 2.22 | −1.5 / +8.0 | 18 | 348 / 246 | **1 of 27** |
| | | TEST | 27 | −0.57 | −0.89 | 1.08 | −2.67 / +1.25 | 7 | 661 / 748 | **27 of 27** |
| D: random audit, σ = 0.002 | outcome-independent | RERANK | 125 | +0.68 | +0.5 | 2.53 | −9.0 / +7.5 | 72 | 1,358 / 1,189 | above all 125 |
| | | TEST | — | **not measured** (6/125 only) | | | | | | |
| E: all 500 audit (mixed σ) | outcome-independent | RERANK | 500 | +0.05 | 0.0 | 1.63 | −9.0 / +7.5 | 201 | 2,611 / 2,560 | above all 500 |
| | | TEST | — | **not measured** (7/500 only) | | | | | | |

**Reading.** 9504111 is the single most favourable RERANK case of every
measured comparison set, selected or not. On TEST it is the least favourable
of the 50 candidates that were evaluated there. Its +8.0 is 7.3 pp above the
random σ = 0.002 mean (+0.68). That is a descriptive contrast, not an estimate
of selection bias.

## Analysis B: group-wide vs winner-specific accounting (exact identity, error 0)

g_T* − g_R* = (μ_T − μ_R) + [(g_T* − μ_T) − (g_R* − μ_R)], with the same
fixed membership in both phases.

| Reference group | Winner in mean? | μ_R → μ_T | Gap | Group-wide change | Winner-advantage change |
|---|---|---|---:|---:|---:|
| SEARCH top 50 | yes | +1.37 → −0.35 | −10.67 | **−1.72** | **−8.95** |
| SEARCH top 50 (leave winner out) | no (49) | +1.23 → −0.30 | −10.67 | −1.54 | −9.14 |
| σ = 0.002 members | yes (27) | +1.89 → −0.57 | −10.67 | **−2.46** | **−8.21** |
| σ = 0.002 members (leave winner out) | no (26) | +1.65 → −0.49 | −10.67 | −2.15 | −8.53 |

Of the observed −10.7 pp gap, about 1.5–2.5 pp accompanies a group-wide shift
among the same selected candidates. About 8.2–9.1 pp reflects the winner going
from the top to the bottom of its peer group.

**Per-candidate paired TEST − RERANK difference** (descriptive):

- **Top 50:** mean −1.72, median −1.47, SD 2.40; 13 of 50 do *better* on TEST.
- **The winner:** −10.67, the most negative of all 50.
- **σ = 0.002 members:** mean −2.46, median −2.39; the winner is again the
  most negative.
- **RERANK vs TEST gain correlation:** −0.27 across the top 50, −0.40 across
  the σ = 0.002 members.

## Analysis C: RERANK-only reselection (only the 200 RERANK questions)

**Setup:**

- 2,000 stratified 100/100 partitions of RERANK200 (subtask × gold letter, 12
  cells with sizes 36, 30, 27, 26, 17, 17, 16, 15, 5, 4, 4, 3; each splits
  ⌊n/2⌋ / ⌈n/2⌉ by alternation).
- Selection on one half uses the original rule; the same partitions serve both
  candidate sets.
- These are conditional replays of one already-inspected matrix, not
  independent datasets. There are no p-values.

| Candidate set | Winner gain, selection half | Same winner, held-out half [2.5–97.5%] | Uniform member, held-out | Increment over uniform [2.5–97.5%] | P(increment > 0) | Winner is 9504111 |
|---|---:|---:|---:|---:|---:|---:|
| SEARCH top 50 | +8.29 | **+5.19** [−2.0, +11.0] | +1.37 | **+3.81** [−3.0, +8.2] | 0.84 | 65% (20 distinct winners) |
| σ = 0.002 members (27) | +8.28 | +5.25 [−2.0, +11.0] | +1.90 | +3.35 [−3.3, +7.6] | 0.79 | 66% (17 distinct winners) |

- **Within the RERANK pool, selection is informative.** The half-selected
  winner keeps most of its advantage on the other RERANK half, losing about
  3.1 pp, and beats a uniformly chosen member by about 3.8 pp.
- **9504111's RERANK advantage is not a lucky draw of specific questions
  within RERANK.** It recurs across random halves.
- **What this cannot show:** whether that RERANK-specific advantage reflects a
  property of the RERANK item pool shared by ordinary neighbours, or something
  exceptional about this candidate. The audit candidates have no TEST outputs.

## Levels kept separate

| Level | Status |
|---|---|
| A. Answer balance | Measured: 20/4 on RERANK vs 23/38 on TEST. |
| B. Selection-associated behaviour | The winner is maximal on its selection set (RERANK) and minimal on TEST among the same 50. Its RERANK advantage is stable within RERANK. |
| C. Shared family response | A modest RERANK > TEST shift is observed among SEARCH-selected candidates (−1.7 pp on average). For unselected neighbours only RERANK is measured: random σ = 0.002 mean +0.68 there, TEST unknown. |
| D. Functional mechanism | Unresolved. Nothing here shows memorization, a template shortcut or a visual skill. The earlier "templated RERANK items" explanation is not established, and approximate additivity does not explain why shifts align with RERANK labels. |

## Missing comparison: proposal only (not launched; needs explicit approval)

**Estimand.** The distribution, across outcome-independent σ = 0.002
neighbours, of the paired phase difference (TEST gain − RERANK gain), with A–D
first-token scores. It can then be compared descriptively with the winner's
−10.67. This cannot be computed now, because TEST exists only for top-50
members.

**Sample.** The 12 smallest SHA256(`selection-vs-shared-control-v1:` +
candidate_id) among the 125 σ = 0.002 audit candidates. The rule was fixed in
the plan before any outcome was read.

| Seed | Manifest index | In SEARCH top 50 | TEST outputs exist |
|---|---:|---|---|
| 9500059 | 59 | no | no |
| 9503167 | 3167 | no | no |
| 9500127 | 127 | no | no |
| 9502563 | 2563 | no | no |
| 9502995 | 2995 | no | no |
| 9503539 | 3539 | no | no |
| 9501135 | 1135 | no | no |
| 9502423 | 2423 | no | no |
| 9500271 | 271 | **yes** (kept, not excluded) | yes (reuse; rerun for A–D scores) |
| 9501023 | 1023 | no | no |
| 9504179 | 4179 | no | no |
| 9504827 | 4827 | no | no |

Full candidate IDs are in `selection_vs_shared_results.json` →
`control_proposal`.

**Conditions:**

- the original model revision, engine (vLLM 0.11 in-process), prompt,
  preprocessing, parser and greedy 16-token decoding;
- each candidate realized exactly with the original
  `apply_perturbation(seed, 0.002)`, verified against its stored SEARCH and
  RERANK outputs;
- evaluation on RERANK200 and TEST561 with A–D scores, then a reset to base.

The RERANK reruns also serve as the fidelity check against the stored audit
outputs.

**Preliminary estimate**, from session-1 measurements:

| Step | Time |
|---|---:|
| Setup and load | 5.4 min |
| Identity hash | 0.5 min |
| 12 candidates × (≈ 2 min for a 761-question pass + ≈ 10 s apply/reset) | ≈ 26 min |
| Retrieval and stop | 1.5 min |
| **Total** | **≈ 33 min** |

At the last observed $3.49/h this is **≈ $1.95**. The proposed **hard cap is
$3.00**, with the pod guard stopping it at $2.90 and an in-script guard before
each candidate.

A 12-candidate sample is descriptive only. It cannot estimate the search's
extreme tail or establish a mechanism.

## Artifacts (`results/paper-analysis/selection-vs-shared-response/`)

| Artifact | Contents |
|---|---|
| `analysis_plan.md`, `source_hashes.json` | Plan lock and source hashes |
| `candidate_sets.json` | Set memberships, σ, overlaps |
| `selection_vs_shared_results.json` | Analyses A, B, C and the control proposal |
| `rerank_split_manifest.npz` | All 2,000 partitions, with question IDs |
| `rerank_reselection_trials_*.csv` | Per-trial winners and gains |
| `paper/figures/selection-vs-shared-response/` | `fig_winner_vs_group`, `fig_rerank_reselection` |

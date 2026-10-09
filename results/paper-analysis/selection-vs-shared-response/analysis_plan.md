# Selection vs shared response for 9504111: analysis plan lock

**Post-hoc plan lock** (not an external preregistration). It was committed
before any result below was computed.

- **Branch:** `research/perspective-selection-vs-shared-response`, from local
  commit `c57bf9c`. The parent branches are local only.
- **Unchanged:** all earlier reports and artifacts.
- **No GPU.**
- **Not repeated:** the margin standardization and the reserved additivity
  check.
- **Source hashes:** `source_hashes.json`.

## Data and coverage (verified before locking)

**Predictions** (master table, re-parsed raw outputs; correctness = first
non-space character equals gold):

- **RERANK200:** 543 candidates, i.e. the SEARCH top 50 plus 500 audit, with 7
  in both.
- **TEST561:** only the SEARCH top 50.
- **Audit candidates with TEST outputs:** only 7, and only because they are in
  the top 50; 6 of them are at σ = 0.002. All other audit TEST results are
  **not measured** and are never imputed.

**Candidate sets:**

| Set | Definition |
|---|---|
| A | 9504111 |
| B | the original SEARCH top 50 (`locks/search.json` `ranked_ids[:50]`) |
| C | the σ = 0.002 members of B |
| D | the 125 precommitted audit candidates at σ = 0.002 (protocol `density_audit.indices`) |
| E | all 500 audit candidates (secondary reference) |

**Original RERANK rule** (`src/thicket_runtime/perspective.py::rank_validation`):
RERANK correct count descending, then SEARCH200 correct count descending, then
SHA256(`visual-rank-v1:` + candidate_id) ascending.

## Analysis A: how exceptional was the winner?

**Per set and phase** (only where outputs exist; otherwise "not measured"):

- n;
- mean, median and SD of gain;
- minimum and maximum;
- count above base;
- total repairs and regressions.

**9504111's position:** its rank (1 = best, ties shared) and the share of the
set strictly below it, in sets B and C (RERANK and TEST) and in sets D and E
(RERANK only). D and E are outcome-independent; B and C are SEARCH-selected.
Their σ mixtures differ (B is 27 × 0.002, 11 × 0.001, 12 × 0.0005), which is
stated with each comparison.

## Analysis B: group-wide vs winner-specific accounting

**Identity:**

g_T* − g_R* = (μ_T − μ_R) + [(g_T* − μ_T) − (g_R* − μ_R)]

**Computed for:**

- the fixed set B, with the winner included in the mean;
- the fixed set C, with the winner included in the mean;
- one leave-winner-out sensitivity for each, with the winner excluded from
  μ.

No other reference groups. Reported in pp. It is an accounting identity, not
a causal decomposition.

## Analysis C: RERANK-only reselection (only the 200 RERANK questions)

**Candidates:** the fixed set B (50). The sensitivity uses the fixed set C.

**Splits:**

- 2,000 partitions of RERANK200 into 100 selection and 100 held-out
  questions, seed `20261011`.
- Stratified by subtask × correct letter: questions are sorted by (stratum,
  random key) with a random stratum order, then assigned alternately from a
  random offset. Each stratum therefore splits ⌊n/2⌋ / ⌈n/2⌉, with the odd
  member's side set by the alternation.
- The cell counts are recorded.
- The same partitions are used for B and C.

**Per trial:**

1. Rank all candidates on the selection half with the original rule (selection
   correct count, then SEARCH200 count, then hash).
2. Take the winner.
3. Compute the winner's held-out gain, and the mean held-out gain of all
   candidates in the set on the same half.

**Recorded per trial:**

- winner ID and σ;
- selection gain;
- held-out gain;
- held-out increment over the set mean;
- selection − held-out gap.

**Summaries:**

- mean selected held-out gain;
- mean uniform-member held-out gain;
- their difference;
- 2.5–97.5 percentiles over partitions;
- win frequencies.

There are no p-values. Partitions are conditional replays of one already
inspected matrix.

## Analysis D: missing-comparison proposal only (no GPU)

**Sample:** 12 candidates from set D, chosen as the 12 smallest
SHA256(`selection-vs-shared-control-v1:` + candidate_id). The rule never reads
any outcome. Top-50 members are not excluded; overlaps are recorded and their
existing outputs reused.

**Estimand:** the distribution over outcome-independent σ = 0.002 neighbours
of the paired phase difference, TEST gain − RERANK gain, together with A–D
first-token scores. This cannot be computed from existing data, because TEST
outputs exist only for top-50 members.

## Figures (at most two)

1. Winner versus group gains on RERANK and TEST.
2. RERANK-only selected selection score, selected held-out score, and the
   uniform-member reference.

## What this cannot establish

- The phase response of unselected neighbours on TEST.
- Any functional, visual or semantic cause.
- Memorization or template shortcuts.

Approximate additivity is not an explanation of label alignment.

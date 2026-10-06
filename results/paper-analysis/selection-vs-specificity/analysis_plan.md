# Selection vs candidate-specificity for 9504111: post-hoc plan lock

This plan was committed before any of these computations. CPU only, using
existing predictions:

- SEARCH for all 5,000 candidates;
- RERANK for 543 (top 50 + 500 audit);
- TEST for 61 (the SEARCH top 50, plus 11 random controls not in the top 50).

A–D scores exist only for BASE, 9504111 and the 12 controls. The frozen
candidate, prompts, parser and splits are unchanged. Every result here is
**post hoc**. The only frozen and precommitted evidence is the original
SEARCH→RERANK→TEST chain and the precommitted 500-candidate audit.

## Fixed definitions

- Gains are in pp over BASE (RERANK 79/200, TEST 259/561),
  with correctness from the generated answers.
- **Winner accounting refinement** (exact):
  gap* − μ_gap = (g_T* − μ_T) − (g_R* − μ_R), i.e. a TEST deficit minus a
  RERANK excess, relative to the 12 controls.

## A/B. Cross-fitted RERANK selection (only existing predictions)

- **Partitions:** reuse the frozen 2,000 RERANK 100/100 partitions from
  `selection-vs-shared-response/rerank_split_manifest.npz`, unchanged.
- **Selection rule:** the original rule (selection-half correct count, then
  SEARCH200 count, then the frozen hash) among the fixed SEARCH top 50.
- **Per trial:** the selected candidate's held-out RERANK gain and its TEST
  gain (TEST is never used for selection).
- **Reported:**
  - P(9504111 selected);
  - E[held-out | 9504111 selected] as its selection-corrected RERANK-pool
    gain;
  - its RERANK optimism = full RERANK gain − that value;
  - E[TEST of the selected candidate] against the uniform top-50 TEST mean
    and the random-control TEST mean;
  - E[TEST − held-out RERANK of the selected candidate].
- **Repetitions:** effective information is bounded by the 200 RERANK
  questions; the 2,000 partitions are not independent experiments.
- **Second pool:** the same cross-fitting among the 500 audit candidates on
  RERANK alone, which yields P(selected) and held-out gains. Audit TEST is not
  measured.

## C. Candidate-specificity conditional on RERANK strength

- **Pool:** the 60 TEST-measured candidates other than 9504111.
- **High-RERANK group:** g_R ≥ +4.0 pp (threshold fixed here). Their TEST
  gains are listed individually, with 9504111's rank.
- **Regression:** a descriptive OLS of g_T on g_R over the 60, with the
  prediction at g_R = 8 marked as an extrapolation (max other g_R noted). The
  residual of 9504111 is reported; no p-value.
- **Question-bootstrap:** 10,000 resamples of TEST questions (paired,
  seed `20261013`) of
  - (9504111 TEST gain − mean TEST gain of the 12 controls);
  - (9504111 TEST gain − mean TEST gain of the high-RERANK group).

  This conditions on the fixed candidates and resamples questions.
- **Score vectors:** limited to 12 controls + 9504111 (the only candidates
  with A–D scores); already reported, not extended.

## D/E. Decomposition table and verdict

Exactly one of `SELECTION_EXPLAINS_MOST`,
`CANDIDATE_SPECIFIC_EXPLAINS_MOST`, `BOTH`, `INSUFFICIENT_DATA`.

## Possible GPU follow-up (proposal only)

**Sample:** all audit candidates (not already TEST-measured) whose RERANK gain
is ≥ +5.0 pp. The rule is fixed here and uses RERANK only.

**Run:** TEST561 only, with A–D scores.

**Hypothesis:** extreme RERANK selection by itself (without SEARCH)
yields TEST failures of 9504111's size.

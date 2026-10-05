# Existing N=5,000: post-hoc Allocentric specialist search

**Post-hoc exploratory follow-up, not preregistered.** It reuses the completed
Perspective-Taking N=5,000 all-parameter population. No candidate was generated,
and no σ, mask or search inference was changed. It started from commit `0099a29`
on branch `research/existing-n5000-allocentric-specialists`. All outputs are new
files under
[`allocentric-followup/`](../results/perspective-taking-n5000-20261004/allocentric-followup/).

| Metric | Result |
|---|---:|
| Existing population | 5,000 |
| New candidates generated | 0 |
| Existing Allocentric SEARCH examples | 65 (base 22/65 = 33.8%) |
| CPU best SEARCH gain | +7.7 pp (27/65, seed 9503839, σ = 0.002) — the expected maximum under random flipping is +7.0 pp |
| Top M evaluated on STAGE2 | not run (CPU gate) |
| Fresh STAGE2 examples | not built: only 287 eligible fresh records exist, against 700 required |
| Frozen K | not reached |
| Fresh FINAL HOLDOUT | not built (see above) |
| Base final accuracy | — |
| Best individual final gain | — |
| Top-50 ensemble accuracy | — |
| Top-50 ensemble gain | — |
| Paired 95% CI | — |
| Allocentric thicket found? | **No** — `NO_ALLOCENTRIC_SIGNAL_IN_EXISTING_N5000` |
| Strong ≥ +5 pp thicket? | No |
| New H100-hours | **0** |

## Step 1: rescoring (CPU)

The 65 Allocentric questions of the original SEARCH200 were rescored from the
stored raw outputs of all 5,000 candidates. The parser is the independent
re-implementation of the frozen first-character rule from the forensic audit.

Candidates are ranked by Allocentric correct count, then by the existing hash.
For each candidate, the ranking file records:

- candidate ID, seed and σ;
- state fingerprint;
- Allocentric count, accuracy and gain;
- original mixed search rank;
- prediction distribution.

It is in
[`existing_n5000_allocentric_search_ranking.json`](../results/perspective-taking-n5000-20261004/allocentric-followup/existing_n5000_allocentric_search_ranking.json).

## Step 2: the full Allocentric search distribution

| | Value |
|---|---:|
| Base | 22/65 (33.8%); predicts C 27 times against 15 true C |
| Mean / median / SD (correct of 65) | 21.90 / 22 / 1.32 |
| One question | 1.54 pp |
| Best | 27/65 (+7.7 pp), seeds 9503839 and one other |
| Above base | 1,542 |
| ≥ +3 pp / ≥ +5 pp / ≥ +10 pp | 471 / 11 / 0 |
| Median candidate disagreement with base | **2 of 65 questions** |

Correct counts across the 5,000 candidates:

| Correct of 65 | 16 | 17 | 18 | 19 | 20 | 21 | **22 (base)** | 23 | 24 | 25 | 26 | 27 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Candidates | 3 | 13 | 42 | 144 | 432 | 1,088 | 1,736 | 1,071 | 368 | 92 | 9 | 2 |

| σ | Mean gain | SD | Max | ≥ +5 pp | ≥ +10 pp |
|---:|---:|---:|---:|---:|---:|
| 0.00025 | +0.08 | 1.25 | +4.6 | 0 | 0 |
| 0.0005 | +0.22 | 1.67 | +6.2 | 1 | 0 |
| 0.001 | −0.01 | 2.09 | +7.7 | 2 | 0 |
| 0.002 | −0.91 | 2.65 | +7.7 | 8 | 0 |

**Mixed versus Allocentric ranking.** Across all 5,000 candidates the Spearman
correlation is 0.46. The two rankings share 6 of their top-50 and 33 of their
top-200 members.

**Is there a tail beyond noise?** The fair reference keeps the population's
observed per-question, per-σ flip rates (about 7 fragile questions flip in ≥ 20%
of candidates) and makes each flip an independent random event for each
candidate. This
[structured null](../results/perspective-taking-n5000-20261004/allocentric-followup/structured_null.json)
gives:

| | Observed | Structured null, mean (p95) | p (null ≥ observed) |
|---|---:|---:|---:|
| ≥ +3 pp | 471 | 435 (466) | 0.03 |
| ≥ +5 pp | 11 | 9.1 (14) | 0.30 |
| ≥ +7 pp | 2 | 0.7 (2) | 0.15 |
| Maximum | +7.7 pp | +7.0 (+7.7) | 0.50 |

A naive fair-coin null, with every disagreement an independent win or loss, would
even predict a *heavier* tail: 99 candidates ≥ +5 pp and a maximum of +10.7 pp.

The observed right tail is fully explained by candidates independently tipping
the same few fragile questions. A slight excess appears only at +3 pp, which is
two questions. **The extreme tail shows no coherent Allocentric specialist.**

## Compute gate

The prespecified gate examples all apply:

- **The distribution is extremely compressed.** The median candidate changes 2 of
  65 answers relative to base, with an SD of 1.3 questions.
- **No meaningful top tail exists beyond random flipping:**
  - the maximum and the ≥ +5 pp and ≥ +7 pp counts all match the structured null;
  - nothing reaches +10 pp.
- **Top-M selection would be largely a lottery.** The top-200 cut-off falls inside
  a 368-way tie at +3.1 pp (24/65).

So no GPU stage was launched. **`NO_ALLOCENTRIC_SIGNAL_IN_EXISTING_N5000`.**

## Data feasibility of Steps 4–6

This is independent of the gate.

| Allocentric records in official train | Count |
|---|---:|
| Total | 996 |
| Not already used, with no image and no exact question text shared with SEARCH or VALIDATION | **287** (287 images, but only 197 distinct question texts) |
| With unique image and unique question text inside the new sets | 197 |

STAGE2 (200) plus a 500-example holdout needs 700 fresh records. Allocentric
questions are highly templated, so the exact-text exclusion removes most of the
pool. The plan as written cannot be built from the official train split. With
200 for STAGE2, at most 87 records, and 0 under within-set text uniqueness, would
remain for the holdout. Any future run would need a relaxed text-duplicate rule
or another Allocentric source, decided before inference.

## Answers

1. **Was there an Allocentric right tail inside the existing N=5,000
   population?** Only a noise-sized one. The best candidate is +5 of 65
   questions, matching the expected maximum from independent flips of a few
   fragile questions.
2. **Did it survive a fresh STAGE2 set?** Not tested. The CPU gate closed, and a
   200-question STAGE2 plus a holdout isn't available from official train.
3. **Did the frozen top 50 survive a fresh Allocentric holdout?** Not reached.
4. **Did the committee beat the base?** Not reached.
5. **Complementary specialists or answer bias?** There was no specialist signal
   to characterise. Movement is concentrated on about 7 fragile questions. The
   base over-predicts C (27 of 65 predictions, against 15 true C).
6. **Did any standalone Allocentric expert exist?** No evidence of one. +7.7 pp on
   65 questions is the noise maximum of 5,000 draws.
7. **How much additional GPU compute was required?** None: 0 H100-hours. All
   analysis used stored raw outputs.
8. **Final GO / NO-GO:** **NO-GO** — `NO_ALLOCENTRIC_SIGNAL_IN_EXISTING_N5000`.

Splitting the mixed Perspective-Taking objective into its Allocentric part
doesn't reveal hidden specialists in this population. That's consistent with the
forensic audit: the population's movement is a small set of borderline-question
flips, not coherent sub-capabilities. Per the brief, no new perturbations were
generated.

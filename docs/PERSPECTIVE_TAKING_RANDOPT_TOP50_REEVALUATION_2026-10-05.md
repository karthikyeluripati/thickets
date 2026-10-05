# Perspective-Taking: RandOpt top-50 ensemble re-evaluation

| Metric | Result |
|---|---:|
| Model | Qwen3-VL-8B-Instruct |
| Search candidates | 5,000 |
| Search examples | 200 |
| Top-K selection source | SEARCH ranking only |
| K | 50 |
| Official TEST examples | 561 |
| Base TEST accuracy | 46.17% (259/561) |
| Best individual SEARCH-top50 TEST | 47.42% (266/561, +1.25 pp; search rank 13) |
| Mean individual SEARCH-top50 TEST | 45.82% (−0.35 pp); median 45.81% |
| Top-5 ensemble TEST | 45.63% (−0.53 pp) |
| Top-10 ensemble TEST | 45.28% (−0.89 pp) |
| Top-25 ensemble TEST | 45.63% (−0.53 pp) |
| **Top-50 ensemble TEST** | **46.17% (259/561)** |
| **Top-50 gain over base** | **+0.00 pp**, paired 95% CI [−1.25, +1.25], McNemar p = 1.00 |
| Standalone expert result | **NO-GO** (`NO_GO_STANDALONE_PERSPECTIVE_EXPERT`) |
| RandOpt ensemble result | **NO-GO** (`NO_GO_RANDOPT_PERSPECTIVE_ENSEMBLE`) |

**The previous study rejected the stronger standalone-expert hypothesis. This
analysis evaluates the same N=5,000 search population under the original
RandOpt top-K ensemble protocol.** The SEARCH-selected top-50 majority vote
answers exactly as many test questions correctly as the base model does. It
gets 7 questions right that the base gets wrong, and 7 the other way round.

## 1. What did the previous experiment test?

It asked whether **one individually selected perturbation** becomes a
standalone, transferable +5 pp Perspective-Taking expert. The selection chain
was:

1. Search ranked all 5,000 candidates.
2. The top 50 went to validation.
3. The validation top 10 went to test.

A candidate counted as an expert only if, by itself, it met all of the
following:

- +5 pp on the official test;
- a positive paired-bootstrap lower bound;
- broad sub-task gains;
- no single-letter concentration of its gains.

No frozen candidate did, so the decision was
`NO_GO_VISUAL_NEURAL_THICKET_PERSPECTIVE_N5000`. That decision is unchanged and
is recorded here as `NO_GO_STANDALONE_PERSPECTIVE_EXPERT`.

## 2. Why was that stricter than RandOpt?

RandOpt's claim is about the **ensemble** of search winners, not about any
single perturbation being an expert. An ensemble can beat the base even when
every member is individually weak, as long as members err differently and the
vote cancels the errors. The standalone test also added a validation stage, a
+5 pp bar and per-candidate validity rules that RandOpt does not apply.

## 3. How does original RandOpt define and select top-K models?

The released code is `randopt.py` at RandOpt `4000d34`. It works as follows:

1. **Score:** every sampled `(seed, σ)` is scored on the training/search prompts.
2. **Rank:** candidates are sorted by that reward, highest first.
3. **Select:** the first K form the committee (`top_k_perturbs`).
4. **Evaluate:** each member is evaluated on test.
5. **Vote:** in `run_ensemble_evaluation`, for each question:
   - collect each member's extracted answer in rank order;
   - drop empty (invalid) answers;
   - take `Counter(answers).most_common(1)[0][0]`, which breaks ties in favour
     of the answer seen first in rank order;
   - count a question with no valid answer as wrong.

This re-evaluation applies those semantics. `tests/test_perspective_committee.py`
checks our vote against a literal transcription of the upstream loop on 300
random cases, including ties and all-invalid questions.

**Differences from the released code, disclosed up front:**

| Item | This study | Released RandOpt default |
|---|---|---|
| Ties at the K=50 boundary | resolved by the frozen hash rule in the committed ranking | resolved by sampling order |
| Parameters perturbed | all, including vision (`PERTURB_VISUAL=1`) | skips `visual.*` |
| Restoration | snapshot anchoring | sequential add/subtract |

> This ensemble test evaluates the already-generated all-parameter population.
> It matches the RandOpt top-K selection/ensemble protocol, but not necessarily
> the released worker's default VLM parameter mask. It is not a reproduction of
> the original GQA setup.

## 4. Does SEARCH-selected top-50 voting improve Perspective Taking on TEST?

**No.** The [addendum](../experiments/perspective_randopt_top50_addendum.json)
fixed the K = 50 endpoint and the decision rule before any new test inference.
The committee is the exact top 50 of the committed SEARCH ranking: 27 at
σ = 0.002, 11 at σ = 0.001, 12 at σ = 0.0005. It was frozen in commit `b43e47f`
at 12:29:28 UTC, before the committee test shards started at 12:38:50 UTC.

| Base vs top-50 ensemble on TEST561 | Value |
|---|---:|
| Accuracy | 46.17% vs 46.17% |
| Difference | +0.00 pp |
| Ensemble-only correct | 7 |
| Base-only correct | 7 |
| Both correct | 252 |
| Both wrong | 295 |
| Paired bootstrap 95% CI (10,000 resamples) | [−1.25, +1.25] pp |
| Exact McNemar p | 1.00 |

Top 50 does not strictly exceed the base, so the decision is
`NO_GO_RANDOPT_PERSPECTIVE_ENSEMBLE`.

## 5. How does ensemble performance change with K?

| K | Test accuracy | Gain over base | Paired 95% CI | Ensemble-only / base-only correct |
|---:|---:|---:|---|---:|
| 1 | 44.39% (249) | −1.78 pp | [−4.10, +0.53] | 17 / 27 |
| 5 | 45.63% (256) | −0.53 pp | [−2.67, +1.43] | 16 / 19 |
| 10 | 45.28% (254) | −0.89 pp | [−2.85, +0.89] | 12 / 17 |
| 25 | 45.63% (256) | −0.53 pp | [−2.14, +0.89] | 8 / 11 |
| **50** | **46.17% (259)** | **+0.00 pp** | [−1.25, +1.25] | 7 / 7 |

These are nested prefixes of the SEARCH ranking. The curve is descriptive, and
K was not chosen using TEST. The K = 1 member, the search winner, is below
base. As K grows, the vote does not climb above base; it converges toward it.
Disagreements with the base shrink from 44 questions at K = 1 to 14 at K = 50.
In other words, the committee's majority reverts to the base model's answers,
which is what one expects when perturbations add mostly uncorrelated noise
around the same underlying model.

## 6. Are individual models weak while the ensemble improves?

**Individual models are weak, and the ensemble does not improve.**

| Individual SEARCH-top-50 member on TEST | Value |
|---|---:|
| Best | 47.42% (+1.25 pp; search rank 13, σ = 0.002) |
| Mean / median | 45.82% / 45.81% (−0.35 pp mean gain) |
| Above base | 17 / 50 |
| ≥ +1 pp | 4 / 50 |
| ≥ +3 pp | 0 / 50 |
| ≥ +5 pp | 0 / 50 |

| σ | Members | Mean test gain | Best |
|---:|---:|---:|---:|
| 0.0005 | 12 | −0.09 pp | +0.89 |
| 0.001 | 11 | −0.08 pp | +0.71 |
| 0.002 | 27 | −0.57 pp | +1.25 |

No member produced an invalid answer. Of the 50 members, 10 reuse the audited
test outputs from the completed study (the validation top 10). The other 40 were
generated in this re-evaluation; each had its state fingerprint checked against
its search fingerprint before generation.

## 7. Is any improvement broad across Perspective-Taking sub-tasks?

There is no aggregate improvement to spread.

**TEST accuracy by sub-task:**

| | Allocentric (376) | Egocentric (102) | Hypothetical (83) |
|---|---:|---:|---:|
| Base | 37.5% | 77.5% | 47.0% |
| K = 1 | 36.4% | 73.5% | 44.6% |
| K = 5 | 36.4% | 76.5% | 49.4% |
| K = 10 | 37.0% | 76.5% | 44.6% |
| K = 25 | 37.2% | 76.5% | 45.8% |
| K = 50 | 37.5% | 78.4% | 45.8% |

**Prediction marginals (A / B / C / D):**

| | A | B | C | D |
|---|---:|---:|---:|---:|
| True answers | 160 | 137 | 143 | 121 |
| Base | 134 | 161 | 155 | 111 |
| K = 50 | 133 | 163 | 158 | 107 |

At K = 50 the sub-task changes (0.0, +1.0, −1.2 pp) are within a handful of
questions. The prediction marginals are essentially the base's, so the vote
neither improves nor reshuffles answers. As instructed, the 80% concentration
rule is not applied as a gate here.

## 8. Final conclusion

- **Standalone visual expert:** `NO_GO_STANDALONE_PERSPECTIVE_EXPERT`
  (unchanged). No individually frozen perturbation was a transferable +5 pp
  expert. Even among all 50 SEARCH-selected members, the best test gain is
  +1.25 pp.
- **RandOpt / Neural-Thickets ensemble:** `NO_GO_RANDOPT_PERSPECTIVE_ENSEMBLE`.
  The SEARCH-selected top-50 majority vote equals the base on the untouched
  official test: 259/561 each, +0.00 pp, 95% CI [−1.25, +1.25]. No prefix
  K ∈ {1, 5, 10, 25, 50} exceeds the base.

So, for Qwen3-VL-8B on OmniSpatial Perspective Taking, with this all-parameter
N = 5,000 population, the Neural-Thickets effect is absent under both the strict
standalone-expert test and the original RandOpt top-K ensemble standard.

## Controls and provenance

**Inputs.**

- **Reused, unchanged:** the SEARCH and VALIDATION results, all previous locks
  and the previous report. No search or validation run was repeated, and no
  existing file was overwritten.
- **New:** the [committee lock](../results/perspective-taking-n5000-20261004/locks/committee.json),
  generated from the committed search lock (`220ba16`).

**Committee test shards.** 8 shards of 5 members each ran on 8× H100. Every
shard passed all controls:

- zero perturbation reproduced the committed base TEST outputs exactly;
- the first member was reconstructed again, with identical state and outputs;
- every restoration was bitwise exact;
- images were freshly encoded, and preprocessing and prompt hashes matched the
  baseline;
- each member's fingerprint matched its search fingerprint before generation.

**Software.** Same as the parent study: pinned Qwen3-VL-8B `0c351dd`, BF16,
TP=1, greedy decoding, vLLM 0.11.0 in-process executor, RandOpt worker
`4000d34`.

**Analysis.** [`scripts/analyze_perspective_committee.py`](../scripts/analyze_perspective_committee.py)
fully audits the existing TEST phase and all committee shards, re-scores every
raw generation, and checks that the committee lock predates the committee
inference. Its output is in [randopt-top50/](../results/perspective-taking-n5000-20261004/randopt-top50/).

**Cost.** About 1.2 H100-hours of generation on 8 GPUs, plus about 30 minutes of
pod setup. The pod was stopped automatically after every shard and record had
been retrieved; no failed attempts occurred.

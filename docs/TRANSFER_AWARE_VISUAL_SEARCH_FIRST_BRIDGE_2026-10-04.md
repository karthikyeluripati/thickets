# Transfer-aware visual search: first bridge, 2026-10-04

| Metric | VANILLA | TRANSFER_MIN |
|---|---:|---:|
| Best search gain (pooled search300, rank 1) | +7.00 pp | +5.00 pp (transfer score +3 pp) |
| Best validation gain (top 20) | **+6.00 pp** | +4.00 pp |
| Top-10 validation ≥ +3 pp | **6 / 10** | 1 / 10 |
| Best final-test gain | not run: validation gate failed | not run: validation gate failed |
| Top-5 final-test mean gain | not run | not run |
| Genuine +5 pp visual expert found? | No (not established; test not run) | No |

**Decision: `NO_GO_TRANSFER_AWARE_VISUAL_SEARCH`, at the validation gate.** Neither
condition A nor condition B held. As the protocol requires, final-test1000 was
never evaluated, no other score was tried, and no BO or CMA-ES was introduced.

## 1. Does cross-fold consistency predict held-out quality better than pooled RandOpt score?

**No.** Across the 30 validated candidates (the union of both top-20 lists), the
two scores relate to validation gain in opposite directions:

| Score → validation gain | Pearson | Spearman |
|---|---:|---:|
| Pooled search gain (VANILLA) | +0.21 | +0.33 |
| Min fold gain (TRANSFER_MIN) | **−0.28** | **−0.33** |

Within each method's own top 20, the pattern is the same:

| Within own top 20 | Pearson | Spearman |
|---|---:|---:|
| VANILLA pooled search gain | +0.21 | +0.45 |
| TRANSFER_MIN score | −0.12 | −0.24 |

These are descriptive correlations on 30 candidates, which are selection-enriched
and share validation questions.

## 2. Does TRANSFER_MIN find better validation experts?

**No. VANILLA's top 20 transferred better on every predeclared summary.**

| Validation300 (base 81/300 = 27.00%) | VANILLA | TRANSFER_MIN |
|---|---:|---:|
| Best gain | +6.00 pp | +4.00 pp |
| Top-5 mean gain | +2.93 pp | +1.67 pp |
| Top-10 mean gain | +2.93 pp | +1.40 pp |
| Top-20 > base / ≥1 / ≥3 / ≥5 pp | 18 / 16 / 7 / 1 | 19 / 16 / 4 / 0 |
| Top-10 > base / ≥1 / ≥3 / ≥5 pp | 9 / 9 / 6 / 1 | 9 / 7 / 1 / 0 |

**Condition A failed.** The best TRANSFER_MIN gain (+4.00 pp) cleared +3 pp, but
it was 2.00 pp *below* VANILLA's best rather than 2 pp above. **Condition B
failed.** One TRANSFER_MIN top-10 candidate reached +3 pp (3 were required),
against six for VANILLA.

TRANSFER_MIN mostly chose low-sigma candidates whose search gains were spread
evenly but small (+2 to +4 pp on every fold). Those candidates drifted toward
answering "3" and transferred weakly.

## 3. Does that advantage survive final-test1000?

**Not applicable.** There was no validation advantage to test. The protocol
forbids final-test inference after a failed gate, and no test phase exists in
the evidence.

## 4. Did we find a genuine ≥ +5 pp visual expert?

**No.** No TRANSFER_MIN candidate reached +5 pp even on validation. No validated
candidate from either method had a paired bootstrap 95% lower bound above zero.
The closest, seed 7300238 at +4.00 pp, had a lower bound of exactly 0.00.

## 5. Did VANILLA find one?

**Not established.** VANILLA's best validation candidate, seed 7300351 at
σ=0.002, gained **+6.00 pp** (99/300 vs 81/300). However, it would have failed
two of the frozen expert checks on its own validation evidence:

- Its paired 95% interval was **[−1.0, +13.0] pp**, which includes zero.
- **96%** of its positive per-label net gains came from true label 4. It raised
  "4" predictions from 6 to 186 out of 300. It gained +54 correct on label 4 and
  lost 25 on label 2 and 13 on label 3.

Under the protocol it was never tested, because the gate concerns TRANSFER_MIN's
advantage. This is not a held-out expert claim.

## 6. Are gains spread across difficulty and answer classes?

**Across difficulty: often yes. Across answer classes: almost never.** On
validation, the base model almost never answers 1 or 4:

| | Label 1 | Label 2 | Label 3 | Label 4 |
|---|---:|---:|---:|---:|
| Base accuracy by true label | 2.7% | 49.3% | 53.3% | 2.7% |

Most validation "gains" are therefore output redistribution:

- **TRANSFER_MIN's picks** typically gain about 20–27 correct on label 3 while
  losing about 15 on label 2.
- **VANILLA's strongest picks** shift mass toward "4" or "1".

Only **3 of 30** validated candidates pass the < 80% single-label concentration
rule, and none of those reaches +5 pp. Difficulty spread looks better, but it is
not enough to qualify as an expert:

| Seed | Method | Easy / Medium / Hard | Label concentration |
|---|---|---|---:|
| 7300351 | VANILLA best | +8 / +4 / +6 pp | 96% |
| 7300238 | TRANSFER_MIN best | +5 / +1 / +6 pp | 91% |

This matches the redistribution failure mode recorded for the 3B v1 final
experiment.

## 7. Final decision: GO or NO-GO?

**`NO_GO_TRANSFER_AWARE_VISUAL_SEARCH`.** The answer to the bridge question —
whether transfer-aware ranking of a fixed weight-space population converts
search-set winners into transferable visual experts — is **no**, under this
model, task, population and budget. Min-fold-gain ranking selected candidates
that transferred *worse* than pooled ranking, and its score was negatively
correlated with held-out gain. This is a scientific negative, separate from
operations (all runtime controls passed). Per the protocol, this direction stops
here: no 3B/32B scaling, no new optimizer, no rescue score.

## Setup and search population

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen2.5-VL-7B-Instruct@cc594898137f460bfe9f0759e9844b3ce807cfb5`, BF16 |
| Execution | one H100 80GB per process, TP=1, greedy decoding |
| Worker | original RandOpt worker `4000d34`, `PERTURB_VISUAL=1`, exact snapshot anchoring |
| Software | vLLM 0.10.2, Ray 2.49.2, all 8 package versions pinned and verified |
| Data | fresh v1 line-tracing splits, committed in `dd89aae` before inference |
| Protocol | committed in `0d188d3` before any 7B output |

Base accuracies, with no capability gate:

| Split | Base accuracy | Notes |
|---|---:|---|
| Search300 | 24.33% (73) | folds A / B / C = 30% / 15% / 28% |
| Validation300 | 27.00% (81) | |
| Test1000 | 27.00% (270) | base only; no candidate ever touched test |

All 400 candidates ran on all 300 search images. Mean pooled gain rose with sigma:

| Sigma | Mean pooled gain | Max pooled gain | Candidates ≥ +3 pp |
|---:|---:|---:|---:|
| 0.00025 | +0.67 pp | +4.67 pp | 5 / 100 |
| 0.0005 | +1.08 pp | +4.33 pp | 20 / 100 |
| 0.001 | +1.58 pp | +6.00 pp | 26 / 100 |
| 0.002 | +2.21 pp | +7.00 pp | 31 / 100 |

The low base accuracy on fold B (15%) let VANILLA's top candidates gain up to
+18 pp on fold B while staying flat or negative on folds A and C. The two top-20
lists overlapped by 10, giving 30 validated candidates.

## Controls and integrity

**All runtime controls passed in every phase.** That means the baseline, the 8
search shards and validation:

- Zero perturbation reproduced the committed base outputs exactly, including on
  pod B's two GPUs.
- Each shard's and validation's first candidate was repeated with identical state
  fingerprint and outputs.
- Every snapshot restoration was bitwise exact.
- Every image was freshly encoded with an empty encoder cache.
- All 400 search fingerprints were distinct.
- Every validation candidate matched its search fingerprint before generation.

**Lock order was verified.** Lock commits precede the phases that consumed them:

| Lock | Commit | Committed (UTC) | Consuming phase started (UTC) |
|---|---|---|---|
| Baseline | `014cac9` | 12:26:44 | every used search shard; the earliest started 12:50:58 |
| Ranking | `006a981` | 14:50:50 | validation, 14:50:51.98 |
| Validation gate | `37076d3` | 15:17:25 | — (records NO-GO) |

[`scripts/verify_transfer_aware.py`](../scripts/verify_transfer_aware.py)
independently re-derives, from raw generations, every search record, both
top-20 lists, all 30 validation gains and the gate decision, and re-checks lock
timing. It reproduced every committed value locally.

**Operational notes.** None of these affected any result used.

1. **Duplicate shard 0 on pod A.** An orchestrator swap on pod A briefly ran two
   shard-0 processes. Both were killed and kept as `search-shard-00.failed-1`
   and `.failed-2`; shard 0 was rerun from scratch.
2. **Shards split across pods.** To save wall-clock time, shards 3–7 ran on a
   second pod (2× H100, each process TP=1). They were copied to pod A and
   re-audited there before ranking. Commit `92aea69` added this sharding option;
   it is operational only.
3. **Rendering is platform-specific.** Pillow output differs between Windows
   and Linux, so the committed PNGs, not regenerated ones, were used everywhere.

**Budget.**

| Phase | Requests |
|---|---:|
| Baseline | 4,800 |
| Search | 120,000 candidate + 4,800 control |
| Validation | 9,000 candidate + 600 control |
| Test | none |

Total GPU usage was roughly 6.5 H100-hours, and both pods were stopped right
after their results were retrieved.

[Evidence](../results/transfer-aware-visual-20261004/) includes the locks, raw
generations, recipes and fingerprints, controls, environment records and the
[analysis](../results/transfer-aware-visual-20261004/analysis/). The
[protocol](TRANSFER_AWARE_VISUAL_SEARCH_PROTOCOL.md) is unchanged. `main` and
all earlier experiments and artifacts are untouched.

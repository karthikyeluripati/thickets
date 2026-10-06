# Random-control transfer for candidate 9504111

**Summary.**

- **Controls.** Ordinary random controls showed a modest shared tilt: 10 of 12
  random σ = 0.002 neighbours did better on RERANK than on TEST (mean
  TEST − RERANK gap −1.65 pp, range −5.8 to +1.0).
- **The winner.** Candidate 9504111 differed by about −9.0 pp beyond that
  average. Its −10.67 pp gap lies outside every control. Its per-question
  score changes resemble the controls' average much less than a typical
  control does (cosine about 0.2 vs about 0.5).
- **What this supports:** both a shared phase response in this sampled
  candidate family and an additional, winner-specific reversal.
- **What it does not explain:** which visual or semantic computation produces
  either.
- **Actual GPU spend:** about $3.20 (approved cap $3.50).

> **POST HOC.** Branch: `research/perspective-random-control-transfer`, from
> local `e96e062` (the parent branches are not on the remote).
>
> - **Plan lock:** committed before any output was read (`42c9718`):
>   `results/paper-analysis/random-control-transfer/analysis_plan.md`,
>   `source_hashes.json`, `frozen_controls.json`.
> - **Code:** GPU `scripts/random_control_transfer.py` (pod runner
>   `scripts/pod_random_control.sh`, approved runner `runs/run_rc.sh`); CPU
>   `scripts/random_control_analysis.py`.

## Controls, fidelity and execution

**Controls.** The frozen 12 from `selection_vs_shared_results.json` →
`control_proposal`, unchanged: the 12 smallest
SHA256(`selection-vs-shared-control-v1:` + id) among the 125 σ = 0.002 audit
candidates. Seeds: 9500059, 9503167, 9500127, 9502563, 9502995, 9503539,
9501135, 9502423, **9500271** (a SEARCH top-50 member, kept), 9501023,
9504179 and 9504827. Each was verified against `locks/search.json`: seed,
σ = 0.002, all-parameter mask, audit index, and original
`candidate_state_id`.

**Engine.** The original path:

- vLLM 0.11 in-process (`uni`), eager, BF16;
- Qwen3-VL-8B-Instruct @ `0c351dd`;
- the pinned RandOpt `apply_perturbation` / `reset_to_base_weights`;
- the official direct prompt, greedy decoding, 16 tokens, first-character
  parser;
- first-token A–D log-probabilities.

**Fidelity.** All checks passed:

| Check | Result |
|---|---|
| BASE fingerprint (original `fast_state.fingerprint`, `state-sha256-tree-v1`) | = recorded base ID `5bc2499c…` |
| BASE generations vs stored | RERANK 200/200, TEST 561/561 text-identical |
| BASE A–D scores vs session 1 | 761/761 identical |
| Each control's fingerprint | = its recorded `candidate_state_id` (12/12) |
| Each control's RERANK generations vs stored audit outputs | 200/200 identical (12/12) |
| 9500271 TEST generations vs its stored committee-test outputs | 561/561 identical |
| Missing A–D scores | 0 |
| Exact BASE restoration after each control (tensor-by-tensor) | 0 mismatches (12/12) |

- **Ties.** Exact BF16 ties at the top score occur in 1–11 RERANK and 10–20
  TEST outputs per control (`per_control_metrics.csv`). Correctness uses the
  generated answer, which is always one of the tied maxima; no tie-break was
  substituted.
- **9504111.** Its outputs and scores were reused from the verified session 1
  and not re-run.

**Execution and cost** (`gpu/cost_log.json`, `pod/`):

| Event | Time (UTC) | Spend |
|---|---|---:|
| Pod start (rate $3.49/h, from the RunPod API) | 03:02:21 | |
| Setup only (no inference) under the original $3.00 guard | until 03:08 | |
| User approval of the $3.50 cap; old guard replaced (pod stop at $3.40) | 03:08:25 | |
| Model loaded | 03:09:12 | $0.40 |
| BASE fidelity done | 03:12:30 | $0.59 |
| 12 controls, ≈ 3.65 min each | 03:12–03:56 | |
| Done | 03:56:27 | $3.15 |
| Results retrieved; pod self-stopped | about 03:57 | **≈ $3.20** |

## Primary result: is the phase difference shared?

| Seed | Top 50? | RERANK correct (base 79) | TEST correct (base 259) | g_R (pp) | g_T (pp) | Gap g_T − g_R | RERANK repairs / regressions | TEST repairs / regressions |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 9500059 | no | 78 | 249 | −0.5 | −1.78 | −1.28 | 12 / 13 | 18 / 28 |
| 9503167 | no | 77 | 253 | −1.0 | −1.07 | −0.07 | 8 / 10 | 21 / 27 |
| 9500127 | no | 83 | 261 | +2.0 | +0.36 | −1.64 | 9 / 5 | 29 / 27 |
| 9502563 | no | 77 | 257 | −1.0 | −0.36 | +0.64 | 15 / 17 | 28 / 30 |
| 9502995 | no | 79 | 254 | 0.0 | −0.89 | −0.89 | 7 / 7 | 20 / 25 |
| 9503539 | no | 81 | 257 | +1.0 | −0.36 | −1.36 | 14 / 12 | 25 / 27 |
| 9501135 | no | 84 | 254 | +2.5 | −0.89 | −3.39 | 16 / 11 | 17 / 22 |
| 9502423 | no | 84 | 260 | +2.5 | +0.18 | −2.32 | 9 / 4 | 29 / 28 |
| 9500271 | **yes** | 84 | 253 | +2.5 | −1.07 | −3.57 | 18 / 13 | 28 / 34 |
| 9501023 | no | 80 | 256 | +0.5 | −0.53 | −1.03 | 10 / 9 | 29 / 32 |
| 9504179 | no | 91 | 260 | +6.0 | +0.18 | −5.82 | 16 / 4 | 26 / 25 |
| 9504827 | no | 76 | 256 | −1.5 | −0.53 | +0.97 | 6 / 9 | 27 / 30 |
| **9504111 (winner)** | yes | **95** | **244** | **+8.0** | **−2.67** | **−10.67** | 20 / 4 | 23 / 38 |

**Summary of the 12 controls:**

- **Gap:** mean −1.65 pp, median −1.32, SD 1.92, range −5.82 to +0.97.
- **Bootstrap:** the 95% interval of the mean gap is −2.71 to −0.67. This
  resamples the 12 candidates, conditional on the fixed questions; it is post
  hoc.
- **Direction:** 10 of 12 have RERANK gain > TEST gain. 7 of 12 are above base
  on RERANK and 3 of 12 on TEST.
- **Mean gains:** RERANK +1.08, TEST −0.56.

**Where 9504111 sits relative to the observed controls:**

- its RERANK gain (+8.0) is above all 12;
- its TEST gain (−2.67) is below all 12;
- its gap (−10.67) is beyond the most negative control (−5.82).

Twelve controls cannot estimate an extreme-tail probability among 5,000, so no
rarity figure is given.

**Variability across candidates versus across questions.**

- Across controls, RERANK gains vary much more (SD 2.14 pp) than TEST gains
  (SD 0.62 pp). TEST gains cluster near −0.6 pp.
- The gap therefore tracks the RERANK gain: r(g_R, gap) = −0.96 and
  r(g_R, g_T) = +0.49.
- The larger RERANK spread exceeds the √(561/200) ≈ 1.7× factor expected from
  set size alone.
- The control with the highest RERANK gain (9504179, +6.0) also has the
  largest drop (−5.8). This is descriptive and fits a high RERANK score being
  partly RERANK-specific even without selection.

**Selected comparison (stored, selected on SEARCH).** Across the SEARCH top
50 the mean gap is −1.72, median −1.47, with 37 of 50 negative. That is
similar to the random controls, but it is a selected sample.

## Winner-versus-control accounting (descriptive, exact identity)

g_T* − g_R* = (μ_T − μ_R) + [(g_T* − μ_T) − (g_R* − μ_R)], with μ taken over
the 12 controls.

| Term | pp |
|---|---:|
| Observed gap of 9504111 | −10.67 |
| Measured random-control mean change (μ_R = +1.08 → μ_T = −0.56) | **−1.65** |
| Change in the winner's advantage relative to those controls (+6.92 above them on RERANK, −2.11 below on TEST) | **−9.03** |

This is an accounting description, not a causal split. The reference mean
comes from 12 candidates; the individual rows above show how much it could
move.

## Answer-score comparison (same definitions as the margin follow-up)

**Contrast.** t = the change in [score(correct) − score(BASE's strongest wrong
option)], with that comparator held fixed from BASE. The standardized mean
uses the margin follow-up's frozen strata and weights.

| | 12 controls: TEST − RERANK (mean / median / range / n < 0) | 9504111 |
|---|---|---:|
| Standardized mean t | −0.19 / −0.20 / [−0.53, +0.38] / 10 | **−0.72** |
| BASE-wrong questions | −0.44 / −0.40 / [−1.60, +0.27] / 10 | −0.88 |
| BASE-correct questions | +0.07 / +0.06 / [−0.82, +1.54] / 5 | −0.39 |

**The controls also shift scores toward the correct answer more on RERANK than
on TEST,** but by about a quarter of the winner's difference.

**The two profiles differ in kind** (`fig_paired_score_shift`).

- *Controls* move BASE-wrong questions toward correct (mean t +0.7 to +1.7)
  and BASE-correct questions away (−0.1 to −1.5) in both phases. This is the
  pattern expected when a perturbation flattens the base's answer
  preferences.
- *9504111* instead
  - protects BASE-correct answers (t +0.58 on RERANK, the highest of all 13
    models; +0.19 on TEST);
  - moves BASE-wrong questions only slightly toward correct on RERANK (+0.16);
  - moves BASE-wrong questions away from correct on TEST (−0.72).

**Centred A–D score-change vectors versus the mean of the 12 controls**
(coefficient 1, nothing fitted):

| Phase | 9504111 vs control mean: pooled cosine / median per-question cosine / residual ratio | Typical control vs mean of the other 11 (median) |
|---|---|---|
| RERANK | 0.20 / 0.14 / 1.07 | 0.51 / 0.66 / 0.87 |
| TEST | 0.17 / 0.06 / 1.02 | 0.44 / 0.53 / 0.91 |

The winner's per-question score changes are *less* like the control average
than an ordinary control is. Most of its shift (residual about 1.0× its own
size) is not the shared response. The reference mean is an estimate from 12
candidates, not the population response.

## WHAT_IS_NOW_ESTABLISHED

- **Measured candidates and controls are exact.** The 12 controls are
  reproduced bit-exactly (fingerprints and stored generations). 9504111's
  numbers are unchanged.
- **A shared phase response exists in this sampled family.** Ordinary
  σ = 0.002 neighbours tend to do somewhat better on RERANK than on TEST
  (10 of 12; mean −1.65 pp, bootstrap over candidates −2.7 to −0.7). Their
  score shifts toward the correct answer are likewise more favourable on
  RERANK.
- **The winner's reversal is far larger than that average.** It lies outside
  every control on RERANK gain, TEST gain, gap and standardized score shift.
  Its score-change profile differs in kind from the controls' (it protects
  base-correct answers rather than flattening them), and it is weakly aligned
  with their average response.
- **Selected and outcome-independent samples agree on the group shift.** The
  random controls give a group-wide shift about the same size as the SEARCH
  top 50 (−1.65 vs −1.72 pp).

## WHAT_IS_STILL_UNRESOLVED

- **The functional reason:** what visual or semantic computation makes
  RERANK generally more favourable, and makes 9504111 in particular help
  RERANK while hurting TEST.
- **Rarity:** how often a random neighbour would show a reversal as large as
  9504111's. Twelve controls cannot estimate that tail.
- **The source of the residual:** whether the winner's extra reversal comes
  mainly from selection on RERANK (the winner is the maximum of correlated
  candidates on that set), from candidate-specific behaviour, or from both.
  These data describe the residual; they do not decompose it.

**Have we identified the functional reason for the reversal, or only
established how widespread the response is?** Only how widespread it is. A
modest RERANK-favouring response is shared across random neighbours. The
selected winner's much larger reversal is distinct from that average.
Neither result identifies the computation responsible.

## Artifacts (`results/paper-analysis/random-control-transfer/`)

| Artifact | Contents |
|---|---|
| `analysis_plan.md`, `source_hashes.json`, `frozen_controls.json` | Plan lock, hashes, frozen controls |
| `gpu/` | Raw generations and A–D scores for BASE and all 12 controls, plus the cost log |
| `pod/` | Setup, run and guard logs; pip freeze; GPU identity |
| `per_control_metrics.csv`, `random_control_results.json` | Per-control metrics, fidelity, accounting, score comparisons |
| `paper/figures/random-control-transfer/` | `fig_paired_gains`, `fig_paired_score_shift` |

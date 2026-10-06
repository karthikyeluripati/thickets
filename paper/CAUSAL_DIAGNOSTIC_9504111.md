# Causal diagnostic of candidate 9504111

**Plain-English answer.**

- *Question.* We tested whether candidate 9504111's answer changes can be
  traced to one identifiable part of the network.
- *Result.* Changing any single parameter group to the candidate's values (or
  back to the base's) moved only 0–36% of its changed answers. That covers
  vision, embeddings, any quarter of the language layers, and the final
  norm + head. The best groups were language layers 0–8 for repairs and vision
  for regressions, each about 0.3 against the pre-registered bar of 0.5.
- *Mechanism.* The answer-score shifts behind the changes were spread across
  vision and all four language quarters. They roughly summed across groups,
  sometimes with opposite signs in different cells.
- *Reserved examples.* Not tested: the pre-registered rule stopped the
  follow-up because no group qualified.
- *Share explained.* No share of the answer changes is accounted for by a
  specific account; we have a distributed description only.
- *Open.* We still do not know which function, if any, the perturbation
  changed, or why its net balance favoured RERANK over TEST.

**Status: `DISTRIBUTED_OR_INTERACTING_EFFECTS`.** This was decided by the
predeclared rule (§6); no targeted route was run. GPU spend was **$0.88** of
the authorized $3.50 (session 1 only).

This is a post-hoc investigation of one candidate. Hybrids of BASE and
CANDIDATE are diagnostic objects on already-inspected evaluation sets. They are
never promoted as experts, and nothing here explains the whole population or
committee.

## Results (session 1, 2026-10-06, 1× H100)

### R1. Exact reproduction and controls (`session1/cost_log.json`)

| Check | Result |
|---|---|
| BASE state SHA-256 | `2582817f…`, equal to the forensic hash |
| CANDIDATE state SHA-256 after `apply_perturbation(9504111, 0.002)` | `7d7ef38b…`, equal to the forensic hash |
| BASE vs stored, 761 RERANK + TEST | raw text 761/761 identical, parsed 761/761 identical |
| CANDIDATE vs stored, 761 | raw text 761/761 identical, parsed 761/761 identical. Scores reproduce 79/95 and 259/244, with 20/4 and 23/38 transitions. |
| First generated token is A–D | 1,522/1,522 |
| All four letters within the top 20 at the answer position | 1,522/1,522, so every contrast below is exact (no missing scores) |
| Repeated reconstruction vs first snapshot | 0 / 642 tensors differ |
| No-op control (candidate rebuilt through the copy path) | state 0 mismatches; LOCALIZATION text 83/83 identical |
| Reset-to-base control (copy path) | state 0 mismatches; LOCALIZATION text 83/83 identical |
| 14 hybrids: every tensor equals its intended source | 0 mismatches in every hybrid |
| Final restore to base | 0 mismatches |

Status `INSTRUMENTATION_MISMATCH` was not triggered.

**Measured perturbation by group** (`session1/group_norms.json`):

- the RMS of δ is 0.0020 in every group, i.e. σ, as expected;
- ‖δ‖ is 48–83 by group size;
- the relative ‖δ‖/‖W‖ is 0.036 for vision and 0.067–0.094 for language,
  embedding and head.

### R2. Were changed answers close contests under the base? (all 761, base letter scores)

| Phase | Median base top-2 log-prob gap, changed answers | Median gap, unchanged answers |
|---|---|---|
| RERANK | repairs 1.62; regressions 0.88; wrong→wrong 0.63 | both correct 5.25; both wrong 4.25 |
| TEST | repairs 1.00; regressions 1.25; wrong→wrong 0.38 | both correct 4.75; both wrong 4.00 |

Yes. The questions whose answer changed were ones where the base's top two
letters were within about 0.4–1.6 nats. Unchanged questions had gaps of
4–5 nats. This is a measured observation, not an explanation.

**Letter tendency (LOCALIZATION, centered candidate − base log-prob offsets).**
The offsets are A +0.30, B −0.64, C +0.12, D +0.22, with per-example SD about
1.5–2.0, much larger than the means. A fixed letter-offset model predicts the
candidate's answer on only 15 of 59 changed localization examples (25%). That
is below the predeclared 60% needed for Route A, so no consistent answer-letter
bias explains the changes.

### R3. Coarse interventions on LOCALIZATION (`session1_analysis.json`, `session1_localization_per_example.csv`, `fig_group_effects`)

**Predeclared summary.**

- R = the share of changed answers that revert to the base answer under
  REMOVAL;
- I = the share reproduced under INSERTION;
- S = (R + I) / 2;
- changed-answer n: repairs 22, regressions 21, wrong→wrong 16 (both phases
  pooled);
- the last column counts unchanged controls (n = 24) whose answer changed.

| Group | Params | Repairs R / I / S | Regressions R / I / S | Wrong→wrong R / I / S | Controls changed (removal / insertion) |
|---|---:|---|---|---|---|
| vision | 0.58 B | 0.23 / 0.23 / 0.23 | 0.29 / 0.33 / **0.31** | 0.38 / 0.31 / 0.34 | 1 / 3 |
| embed | 0.62 B | 0.05 / 0.00 / 0.02 | 0.05 / 0.05 / 0.05 | 0 / 0 / 0 | 1 / 1 |
| lm_q1 (layers 0–8) | 1.74 B | 0.36 / 0.27 / **0.32** | 0.29 / 0.24 / 0.26 | 0.38 / 0.50 / 0.44 | 1 / 0 |
| lm_q2 (9–17) | 1.74 B | 0.32 / 0.27 / 0.30 | 0.24 / 0.33 / 0.29 | 0.25 / 0.31 / 0.28 | 0 / 3 |
| lm_q3 (18–26) | 1.74 B | 0.18 / 0.23 / 0.20 | 0.14 / 0.29 / 0.21 | 0.19 / 0.19 / 0.19 | 1 / 1 |
| lm_q4 (27–35) | 1.74 B | 0.18 / 0.27 / 0.23 | 0.05 / 0.05 / 0.05 | 0.19 / 0.06 / 0.12 | 2 / 0 |
| final norm + head | 0.62 B | 0.00 / 0.09 / 0.05 | 0.05 / 0.14 / 0.10 | 0 / 0.06 / 0.03 | 1 / 0 |

**Predeclared rule outcome.** The highest S for repairs is lm_q1 at 0.32, and
for regressions vision at 0.31. **Neither reaches 0.5, so the rule classifies
`DISTRIBUTED_OR_INTERACTING_EFFECTS` and runs no targeted route** (A, B or C).
The MECHANISM-CHECK examples were therefore never evaluated under any
intervention.

**Score-level picture.** These are descriptive, post hoc and from LOCALIZATION
only. *m* is the share of the candidate's fixed-contrast shift reproduced by a
single-group insertion, as a mean per cell.

| Group | RERANK repairs (n = 10) | RERANK regressions (n = 2) | TEST repairs (n = 12) | TEST regressions (n = 19) |
|---|---:|---:|---:|---:|
| vision | −0.11 | +0.24 | +0.15 | +0.21 |
| embed | −0.09 | −0.11 | +0.01 | −0.01 |
| lm_q1 | +0.19 | +0.43 | +0.79 | +0.21 |
| lm_q2 | +0.40 | +0.41 | +0.52 | +0.30 |
| lm_q3 | +0.12 | 0.00 | −0.27 | +0.25 |
| lm_q4 | +0.35 | −0.34 | +0.16 | −0.03 |
| norm + head | −0.06 | +0.11 | −0.06 | +0.02 |

- **Additive in score space, thresholded in answer space.** Summed over the
  seven single-group insertions, the per-example share of the shift has mean
  1.04 and median 0.92 (IQR 0.67–1.18). The removal reversions sum to a mean
  of 0.93 and median of 1.00. 63% of changed answers are reproduced by at
  least one single-group insertion, and 22 of 59 by none.
- **No single owner.** The changes come from contributions spread over vision
  and all four language quarters. They add up roughly but flip an answer only
  when the base contest was close. Embeddings and the final norm + head
  contribute about nothing.
- **Partial differences between phases.**
  - Vision pushes *against* RERANK repairs (−0.11) but *toward* TEST
    regressions (+0.21).
  - lm_q4 contributes to RERANK repairs (+0.35) but not TEST regressions
    (−0.03).
  - lm_q1 and lm_q2 contribute to everything.

  These are small-n descriptive differences and not a tested claim. They agree
  in direction with the forensic Phase R mask diagnostic: vision alone gives
  RERANK +3 and TEST 0.
- **Group size matters.** Per billion parameters, vision's insertion share
  (0.21) is similar to lm_q1 (0.22) and lm_q2 (0.23). Nothing marks vision as
  intrinsically special.
- **Fragility baseline.** On the 18 changed RERANK localization items, a
  *random* σ = 0.002 audit candidate gives 9504111's answer 34% of the time
  (σ = 0.0005: 16%; σ = 0.00025: 11%). Single-group insertions reproduce it
  0–44% of the time (lm_q1 44%, vision 33%, lm_q2 33%). Partial reproduction
  by one group is no more specific than an unrelated perturbation of the same
  size.

### R4. Example traces (`fig_example_traces`; mechanically selected: the LOCALIZATION example with the median base top-2 gap in each changed cell)

| Cell | Example | Base → candidate (gold) | Single-group insertion reproducing the candidate answer |
|---|---|---|---|
| RERANK repair | `train:605_2` (Egocentric) | C → A (A) | none: no single group flips it, only the full candidate |
| RERANK regression | `train:466_0` (Allocentric) | A → B (A) | vision, lm_q2 (lm_q1 gives C) |
| TEST repair | `test:10_0` (Allocentric) | C → D (D) | none (lm_q1 reaches a tie) |
| TEST regression | `test:56_0` (Allocentric) | A → B (A) | vision only, with a larger contrast than the full candidate |

### R5. What the evidence supports

- **Repairs versus regressions.** Both draw on overlapping groups, mainly lm_q1
  and lm_q2. Vision contributes to regressions in both phases and opposes
  RERANK repairs. That asymmetry is descriptive and was not checked on reserved
  examples.
- **Not established:**
  - any functional mechanism, perceptual or answer-letter;
  - any reason for the RERANK/TEST balance beyond item closeness and spread
    contributions;
  - anything about other candidates.

## Execution and GPU cost log

| Item | Value |
|---|---|
| Pod | 1× H100 80GB at $3.49/h (rate from the RunPod API) |
| Pod start | 2026-10-06 00:30:16 UTC |
| Model loaded | 00:35:40 ($0.31) |
| Reproduction complete | 00:40:15 ($0.58) |
| 14 hybrids + controls | done by 00:44:33 ($0.83) |
| Results retrieved; pod self-stopped | about 00:45 |
| **Total** | **≈ $0.88** (cap $1.75; hard guard at $1.65) |
| Session 2 | not run: predeclared rule (§6) |
| Remaining authorized | ≈ $2.62, unspent |

# Preparation and frozen design (written before any GPU output)

## 1. Provenance and identities

| Item | Value |
|---|---|
| Branch | `research/perspective-causal-diagnostic` |
| Starting commit | `b0236c71c9addb27090e9f2438bce9d1e0ce5d62`. **Local only.** It sits on three unpushed branches (behavioral-diversity, transfer-density, radius-audit); `research/expert-mirages-paper` (`a5195e9`) is on GitHub. |
| BASE | `Qwen/Qwen3-VL-8B-Instruct` @ `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` (model, processor, tokenizer, template; from the protocol JSON) |
| CANDIDATE | candidate_id `25a60f0b…b843`, seed 9504111, σ 0.002, rng `randopt-per-tensor-v1`, sign +1, mask `all-parameters-including-vision`; study state fingerprint `7f0bdb3b62424fde…` (identical in its SEARCH, RERANK and TEST records); lock index 4111 |
| Engine state hashes (forensic Phase Q, vLLM 0.11 in-process) | base `2582817f…`, candidate `7d7ef38b…` |
| Reproduced from stored outputs (CPU) | RERANK 79 → 95 (20 repairs, 4 regressions, +16, +8.00 pp); TEST 259 → 244 (23 repairs, 38 regressions, −15, −2.67 pp). Exact match to the record. |

**Prior evidence (not repeated).** The forensic audit
(`docs/PERSPECTIVE_TAKING_N5000_FORENSIC_AUDIT_2026-10-05.md`) established:

- byte-identical clean-process reproduction (base, 9504111 twice, 4 others);
- in-place eager parameters with no derived weight copies;
- per-call re-encoding of images;
- per-tensor reseeding of the noise;
- no leakage and no parser issue (every output for both models is a bare
  letter).

Its single-seed Phase R mask diagnostic gave, as RERANK / TEST:

| Hybrid | RERANK | TEST |
|---|---:|---:|
| Language only | 91 | 255 |
| Vision only | 82 | 259 |
| All parameters | 95 | 244 |

Because the upstream worker reseeds per tensor, those masked runs equal
copied-value hybrids:

- language-only = insertion of every language group;
- vision-only = insertion of the vision group.

Session 1 re-measures all of this with copied values, per example and with
answer scores.

## 2. Reproduction strategy (no backend migration)

**Engine.** All work runs in the original engine: vLLM 0.11.0 in-process
(`uni` executor), BF16, eager, prefix caching off, multimodal processor cache
off, with the pinned upstream RandOpt `WorkerExtension` (`4000d34`). Prompt,
image processing, greedy decoding with 16 tokens, and the first-character
parser are as in the study. No Hugging Face re-implementation and no noise
regeneration.

**Candidate construction.** The candidate is created by the original
`apply_perturbation(9504111, 0.002)`, and its realized BF16 tensors (642 vLLM
tensors, packed qkv and gate_up) are snapshotted to host memory.

**Hybrids.** Every parameter is overwritten by `copy_` from either the stored
BASE weights or the stored CANDIDATE snapshot, then checked
tensor-by-tensor with `torch.equal` against its intended source. Rebuilding
all weights from sources before every run gives exact restoration.

**Controls (session 1):**

- base and candidate SHA-256 must equal the forensic hashes;
- base and candidate on all 761 RERANK + TEST questions must match the stored
  parsed answers on every question, and raw text is compared too;
- a repeated reconstruction must match the first snapshot exactly;
- a no-op control rebuilds the candidate through the copy path on
  LOCALIZATION, and must give identical outputs;
- a reset-to-base control rebuilds the base through the copy path, and must
  give identical outputs;
- a final restore to base is verified.

Any parsed-answer mismatch stops the run with `INSTRUMENTATION_MISMATCH`.

## 3. Answer-decision measurement

- **Single-token letters.** The pinned tokenizer encodes A, B, C and D as
  single tokens (IDs 32–35, no leading space). The generation prompt ends with
  `assistant\n`. All 1,522 stored base and candidate outputs on RERANK + TEST
  are bare letters.
- **What is recorded.** The answer decision is the first generated token.
  Session 1 records vLLM raw log-probabilities (top 20) at that position, so
  A–D scores come from the *same prefix* (the prompt) for every model and
  hybrid. Letters outside the top 20 are recorded as missing, with the top-20
  floor as a bound. Coverage is reported, and sequence scoring is not needed
  unless coverage fails.
- **Derived measures:**
  - the per-option scores;
  - the margin (gold minus the strongest wrong option);
  - for changed-answer examples, the fixed contrast score(original candidate
    answer) − score(original base answer), evaluated identically under every
    intervention.

## 4. Frozen example manifests (`results/paper-analysis/causal-diagnostic/example_manifest.json`)

Built before any intervention output exists, with seed `20261009`:

- **Changed-answer cells** (repair, regression, wrong→wrong-different) are
  halved per phase. The halving is stratified by subtask, using a
  stratum-sorted systematic split.
- **Unchanged cells** (both correct, both wrong same) contribute 6 examples per
  phase to each set, with the same stratification.

| Phase \| transition | Total | LOCALIZATION | MECHANISM-CHECK (reserved) |
|---|---:|---:|---:|
| RERANK repair | 20 | 10 | 10 |
| RERANK regression | **4** | **2** | **2** |
| RERANK wrong→wrong-different | 12 | 6 | 6 |
| RERANK both correct | 75 | 6 | 6 |
| RERANK both wrong, same answer | 89 | 6 | 6 |
| TEST repair | 23 | 12 | 11 |
| TEST regression | 38 | 19 | 19 |
| TEST wrong→wrong-different | 20 | 10 | 10 |
| TEST both correct | 221 | 6 | 6 |
| TEST both wrong, same answer | 259 | 6 | 6 |
| **Total** | 761 | **83** | **82** |

- **RERANK regressions are tiny** (2 + 2). No conclusion will be drawn from
  them alone.
- **Localization accuracy is not a dataset estimate.** Accuracy on these
  transition-balanced sets estimates nothing about full-dataset accuracy.
- **What the reserved check is.** MECHANISM-CHECK outputs are *unseen
  intervention outcomes*. The examples' original labels were already
  inspected, so this is a post-hoc reserved-example check, not a fresh
  confirmatory test.

## 5. Frozen parameter partition (`parameter_groups_hf.json`; exhaustive and disjoint, tested)

These are actual model boundaries from the pinned checkpoint:

- 27 vision blocks, with DeepStack taps at vision layers 8, 16 and 24;
- 36 language layers;
- `tie_word_embeddings = false`, so there is no aliasing between the
  embeddings and `lm_head`.

| Group | Contents | HF tensors | vLLM tensors | Parameters | Share | ≈ ‖δ‖₂ (σ√n, before BF16 rounding) |
|---|---|---:|---:|---:|---:|---:|
| `vision` | patch/pos embed, 27 blocks, merger, 3 DeepStack mergers | 351 | 351 | 576.4 M | 6.6% | 48 |
| `embed` | token embeddings | 1 | 1 | 622.3 M | 7.1% | 50 |
| `lm_q1` | language layers 0–8 (DeepStack injection sites) | 99 | 72 | 1,736.5 M | 19.8% | 83 |
| `lm_q2` | layers 9–17 | 99 | 72 | 1,736.5 M | 19.8% | 83 |
| `lm_q3` | layers 18–26 | 99 | 72 | 1,736.5 M | 19.8% | 83 |
| `lm_q4` | layers 27–35 | 99 | 72 | 1,736.5 M | 19.8% | 83 |
| `final_norm_head` | final norm and untied `lm_head` | 2 | 2 | 622.3 M | 7.1% | 50 |
| **Total** | | 750 | **642** (matches the forensic vLLM count) | 8,767.1 M | | |

Exact per-group ‖δ‖, relative ‖δ‖/‖W‖ and RMS are measured on the GPU from the
realized tensors. The language quarters hold three times the parameters of the
other groups, so effects will be read relative to size and ‖δ‖.

## 6. Session-1 interventions and the predeclared selection rule

For each of the 7 groups G, on LOCALIZATION only:

- **REMOVAL:** candidate everywhere, BASE in G.
- **INSERTION:** base everywhere, CANDIDATE in G.

That is 14 hybrids plus 2 controls.

**Reported separately** for RERANK repairs, RERANK regressions, TEST repairs,
TEST regressions, wrong→wrong changes, and unchanged controls:

- reversion to the base answer;
- retention of the candidate answer;
- movement of the fixed contrast;
- invalid outputs.

**Predeclared follow-up selection.** For each group, compute on LOCALIZATION
changed-answer examples:

- R_G: the fraction of candidate answer changes that revert to the base answer
  under removal;
- I_G: the fraction reproduced under insertion;
- S_G = (R_G + I_G) / 2, computed separately for repairs (both phases) and
  regressions (both phases).

The rules are:

1. Select the group with the highest S_G for repairs and the group with the
   highest S_G for regressions. That is at most two groups; ties go to the
   smaller group.
2. If no group reaches S_G ≥ 0.5 for either class, classify
   `DISTRIBUTED_OR_INTERACTING_EFFECTS` and do not run a targeted route.
3. Hybrid accuracy is never a selection criterion.

**Predeclared route choice.** It is applied only after localization, and only
one route runs.

- **Route B (visual path)** if a selected group is `vision`. Visual features,
  including the merger and the DeepStack features, depend only on vision
  weights and the image. So the four feature-exchange conditions are exactly:
  - base;
  - candidate;
  - REMOVAL(vision): the candidate language model receives base features;
  - INSERTION(vision): the base language model receives candidate features.

  The self-conditions are base and candidate themselves, and the feature
  caches are implicit and weight-keyed.
- **Route A (answer letter / option position)** if the selected groups are
  language-side **and** a letter-offset model fits LOCALIZATION. The model is
  the mean centered candidate-minus-base A–D log-probability offset, and it
  must predict ≥ 60% of the localization answer changes. The offsets are then
  frozen and tested on MECHANISM-CHECK, together with 4 cyclic option orders
  (order-sensitive option sets are flagged).
- **Route C (activation patching)** only if neither applies *and* budget
  remains, with at most one subdivision of the selected group. Otherwise stop
  at `LOCALIZED_PARAMETER_CONTRIBUTION`.

The explanation card (observation, hypothesis, prediction, falsification) will
be written and committed *before* any MECHANISM-CHECK intervention output is
generated.

## 7. Cost plan

Measured basis: the previous identical-engine run cost $3.49/h. Pod start to
model loaded took 4.3 min, and one 761-question direct-prompt pass took about
110 s.

| Session | Work | Estimated time | Estimated cost |
|---|---|---:|---:|
| 1 | setup and load (4.3 min); 2 SHA-256 identity checks (≈ 2 min); base + candidate on all 761 (≈ 4 min); snapshot and group norms (≈ 1 min); 2 controls + 14 hybrids × 83 examples (≈ 4 min); pull and stop (≈ 1.5 min) | ≈ 17 min | ≈ $1.00 |
| 2 (one route) | setup and load; candidate rebuild and snapshot; selected group hybrids, base and candidate on MECHANISM-CHECK; Route B ≈ +1 min, Route A ≈ +10 min (4 orders × ≤ 6 conditions × 165 examples) | 12–20 min | $0.70–1.20 |
| **Total** | | | **≈ $1.70–2.20** |

**Requested authorization: $3.50 total.** The hard caps would be:

- session 1: $1.75, stopping before any step that would exceed it;
- session 2: $1.75, with the same rule;
- the pod's own guard stops it $0.10 below each cap.

Partial results are written after every condition. No extra seeds, layers or
experiments are added automatically.

**Minimum GPU work if the budget is smaller.** Session 1 alone (about $1.00,
cap $1.75) delivers:

- the exact reproduction with answer scores;
- all controls;
- the 14 coarse hybrids.

That would support at most `LOCALIZED_PARAMETER_CONTRIBUTION`.

## 8. Artifacts so far

| Artifact | Contents |
|---|---|
| `scripts/causal_diag_9504111.py` | Partition, manifests, score contrasts and permutation helpers; GPU session 1 |
| `tests/test_causal_diag_9504111.py` | 6 tests, all passing: exhaustive partition on HF and expected vLLM names, manifest determinism and disjointness, exact hybrid copy and restore on a BF16 toy model, score contrasts, cyclic option permutations |
| `scripts/pod_causal_diag.sh` | Self-protecting pod runner: real-rate lookup, hard-cap guard, auto-stop |
| `results/paper-analysis/causal-diagnostic/` | Example manifest; HF tensor inventory at the pinned revision; group table |

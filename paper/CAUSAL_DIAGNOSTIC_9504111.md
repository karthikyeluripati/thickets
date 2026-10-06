# Causal diagnostic of candidate 9504111: preparation (CPU) and frozen design

**Status: preparation complete; no GPU work run yet. Awaiting an authorized
GPU budget.** If no GPU budget is authorized, the final status is
`INCONCLUSIVE_OR_BUDGET_LIMITED`; the minimum GPU work needed is given at
the end.

This is a post-hoc investigation of one candidate. Hybrids of BASE and
CANDIDATE are diagnostic objects on already-inspected evaluation sets. They are
never promoted as experts, and nothing here explains the whole population or
committee.

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

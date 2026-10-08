# Thickets or Tilts?

**A pre-registered re-examination of random weight perturbation as post-training.**

[Neural Thickets](https://arxiv.org/abs/2603.12228) (Gan & Isola, ICML 2026) proposes **RandOpt**: sample thousands
of Gaussian perturbations of a pretrained model's weights, keep the K best on a small selection set, and
majority-vote their answers. It reports gains that rival PPO and GRPO, and interprets them as evidence that diverse
task experts are dense around pretrained weights.

This repository asks two questions about that result:

1. **What does the weight search actually buy?**
2. **What do the selected perturbations change inside the model?**

Every confirmatory test was specified in a plan lock committed before its run; exploratory analyses are labelled
separately. All locks, per-item outputs and analysis code are in this repository.

> **Status (2026-10-08):** all planned experiments are complete; the paper is being written.

## Findings

### 1. Weight search pays only when selection finds a shared shift; otherwise self-consistency votes better

Three same-run comparisons at the paper's settings (N = 5000 perturbations, top K = 50), against self-consistency
with the same number of test-time generations (50 samples at T = 0.7, no weight search):

| Task / model | Base | RandOpt (paper) | Self-consistency | RandOpt − SC [95% CI] |
|---|---|---|---|---|
| GSM8K / Qwen2.5-1.5B | 60.3 | 77.2 (76.4) | 79.8 | **−2.65 [−4.32, −0.99]** |
| GSM8K / Qwen2.5-3B | 80.7 | 86.7 (87.1) | 88.2 | **−1.59 [−2.65, −0.53]** |
| GQA / Qwen2.5-VL-3B (1238 questions) | 53.4 | 63.5 (69.0*) | 60.0 | **+3.47 [+1.62, +5.41]** |

<sub>*The paper evaluates GQA on all of testdev with train-split selection; we use image-disjoint testdev splits.</sub>

On GSM8K, the selected models are individually only 0–4 pp better than the base model, so the gain is the vote, and
sampling the unperturbed model votes better. On GQA, the selected models are about 5 pp better individually. An
pre-registered follow-up shows they answer in about half as many tokens, and that fixing answers that never finish
explains only about a quarter of RandOpt's GQA lead: with a 4× token budget it shrinks from +3.6 to +2.7 pp but
persists.

### 2. A perturbation's effect on answers is first-order, within clear limits

A prediction built from the base model's gradient and each perturbation's noise, with **no fitted coefficients**,
tracks how small language-weight perturbations shift answer preferences:

| Setting | r (prediction vs measurement) |
|---|---|
| Qwen3-VL-8B, two OmniSpatial tasks (tilts) | 0.938, 0.915 |
| Qwen2.5-VL-7B (after a reliability-gate failure and a pre-registered remedy) | 0.925 |
| Qwen2.5-VL-3B, RandOpt's GQA setting, per question | 0.930 |
| **OLMo-2-1B / 7B, ARC-Challenge** (non-Qwen, text-only) | **0.790 / 0.787** |

The effect is localized to the middle language layers (block-level r = 0.978). The account has tested limits: it
fails for vision weights (r = 0.137), breaks down by σ = 0.005, and does not predict chain-of-thought correctness.

### 3. Selection favours label-aligned answer tilts, and that explains only part of a "winner"

In an N = 5000 search on OmniSpatial (Qwen3-VL-8B), the selected model's +8.0 pp selection gain sat on a question
format the test set lacked. On fresh, format-matched items it keeps **+2.67 pp [0.17, 4.93]**. Selection favours
perturbations that tilt answers toward content the selection labels reward. But removing that tilt still leaves
+2.33 of the +2.67 pp. We report this as a limit of the account, not as a solved case.

### Theory linking the three

Under the first-order law, a random perturbation shifts each question's answer margin by a Gaussian with standard
deviation σ‖∇margin‖. Four consequences follow (`paper/THEORY.md`):

- Flip probabilities are predictable per question (AUC 0.935, calibrated).
- An unselected vote returns the base model's answer (96/96 questions).
- σ‖∇‖ acts as a per-question sampling temperature.
- Top-K selection is a noisy first-order step along the selection set's gradient.

This regime covers direct answers at small σ, not chain-of-thought.

## Repository layout

```
paper/
  STORY.md                   the paper's argument: one question, three claims, where each experiment goes
  RESULTS_MASTER.md          every number the paper may use, with its lock commit and source file
  THEORY.md                  first-order derivations (T1–T4) and their checks
  WRITING_PROMPT.md          drafting rules (wording, what not to claim, credit to prior work)
  CORRECTION_SPLIT_MISMATCH.md, WHY_INVESTIGATION_9504111.md   the OmniSpatial case-study record
  figures/thickets-or-tilts/ generated figures and tables;  figures/scripts/thickets_or_tilts.py regenerates them
results/paper-analysis/<study>/
  plan_lock.md               the pre-registration (committed before the run)
  *_RESULT.md, *_results.json   outcome against the locked rule
  pod/                       raw per-item outputs pulled from the GPU runs
results/perspective-taking-n5000-20261004/   raw scores of the original N = 5000 OmniSpatial search
scripts/                     runners (GPU) and locked analyses (CPU), named by study: stage12_, stage3_, r1_, r2_,
                             i5_, m1_, c1_, p0_–p3_, s1_, c_, g1_, g2_, theory_check, geometry_*
src/thicket_runtime/         a small library for fast, verified weight-state handling (used by the runners)
tests/                       unit tests for the perturbation-folding and grouping code
examples/                    the frozen OmniSpatial item splits used by the case study
```

Study index (lock → result), in the order the paper uses them:

| Study | What it tests | Where |
|---|---|---|
| Stage 1+2, Stage 3, R1, R2/R2b, I5, P0, S1-A, S1-7B | the first-order law, localization, generalisation | `stage12/`, `stage3/`, `r1/`, `r2/`, `r2b/`, `i5/`, `p0/`, `s1/`, `s1-7b/` |
| GPU-A | whole-model and vision boundary | `geometry-gpu-a/` |
| M1, C1, τ-check, S1-B | the OmniSpatial winner on fresh data; calibration | `m1/`, `c1/`, `tau-check/`, `s1/` |
| P1, P2, P3 | chain-of-thought regime; SC vs published RandOpt | `p1/`, `p2/`, `p3/` |
| C, C3B, G2 | same-run RandOpt vs SC (GSM8K 1.5B, GSM8K 3B, GQA) | `c-sameRun/`, `c3b-sameRun/`, `g2-sameRun/` |
| Theory | T1/T2 checks | `theory/` |

The full ledger of every locked test and its outcome, including falsified and inconclusive ones, is in
`paper/RESULTS_MASTER.md` §R9.

## Reproducing

**CPU analyses and figures** (Python ≥ 3.10):

```bash
pip install -e '.[test]'
git clone https://github.com/sunrainyg/RandOpt third_party/RandOpt && git -C third_party/RandOpt checkout 4000d34
(cd third_party/RandOpt && python ../../scripts/c_prep_gsm8k.py)   # GSM8K in RandOpt's format (needs `datasets`)
python paper/figures/scripts/thickets_or_tilts.py          # figures and tables
python scripts/theory_check.py                              # theory checks (T1, T2, T4)
python scripts/s1_analysis.py --a-dir results/paper-analysis/s1/pod/out/A \
       --b-pred results/paper-analysis/s1/pod/out/B/B_pred.json
python scripts/c_analysis.py --upstream third_party/RandOpt --d results/paper-analysis/c-sameRun/pod6/c/out \
       --sc-dir results/paper-analysis/c-sameRun/pod/c/out --out /tmp/c_results.json
pytest -q
```

**GPU runs** used single- and multi-H100 pods with a pinned stack: vLLM 0.11.0, transformers 4.57.1,
torch 2.8.0, huggingface-hub 0.36.2, datasets 3.6.0. `scripts/pod_jobqueue.sh <cap_usd> <model> <revision>` sets up
the pod, pins packages, clones RandOpt @ 4000d34, enforces a hard spending cap and an idle stop, and runs queued
job scripts. Each study's `plan_lock.md` gives its exact commands, models and revisions.

Models (pinned revisions in each lock): Qwen3-VL-8B-Instruct, Qwen2.5-VL-7B/3B-Instruct, Qwen2.5-1.5B/3B-Instruct,
OLMo-2-0425-1B-Instruct, OLMo-2-1124-7B-Instruct. Data: OmniSpatial, GQA (testdev-balanced), GSM8K,
ARC-Challenge, MATH-500.

**A note on GQA:** RandOpt's released `randopt.py` (4000d34) builds text-only prompts and does not pass images to
vLLM, so it cannot run GQA with images. Our GQA run (`g2-sameRun/`) re-implements its loop on RandOpt's own
perturbation, scoring and voting components.

## History

This repository began as a broader exploration: runtime profiling, adaptive evaluation, complementarity selection,
visual line-tracing and an "expert mirage" framing of the OmniSpatial winner. Those directions were removed from the
working tree on 2026-10-07 because they are superseded or outside the paper's story. Everything is preserved at git
tag [`pre-cleanup`](../../tree/pre-cleanup).

## Credit

This work builds on and re-examines Neural Thickets and RandOpt (Gan & Isola; code at
[sunrainyg/RandOpt](https://github.com/sunrainyg/RandOpt)). Related observations we credit:
- the original paper's own analysis of format effects;
- selection bias toward the selection set (arXiv 2608.10867);
- the blog posts *A Thicket by Any Other Name* and *When does RandOpt work?*.

Self-consistency follows Wang et al. (2023).

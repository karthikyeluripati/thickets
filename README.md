# Thickets or Tilts?

**A pre-registered re-examination of random weight perturbation as post-training.**

[Neural Thickets](https://arxiv.org/abs/2603.12228) (Gan & Isola, ICML 2026) proposes **RandOpt**: sample thousands
of Gaussian perturbations of a pretrained model's weights, keep the K best on a small selection set, and
majority-vote their answers. It reports gains that rival PPO and GRPO, and interprets them as evidence that diverse
task experts are dense around pretrained weights.

This repository asks: **what does the weight search actually buy, and what do the selected perturbations change?**

Confirmatory tests were specified in plan locks committed before their runs; exploratory analyses are labelled
separately. All locks, per-item outputs and analysis code are in this repository.

> **Status (2026-10-09):** all confirmatory runs are complete; the paper is being written.

## Findings

### 1. Choosing a prompt on RandOpt's own selection data matches or beats its weight search

We ran RandOpt at the paper's settings (N = 5000 perturbations, top K = 50) in four rows, and compared it with
self-consistency: 50 samples of the unperturbed model, majority-voted, the same test-time budget. RandOpt's published
numbers reproduce on the Qwen GSM8K rows (77.2 vs 76.4; 86.7 vs 87.1).

The practical baseline: choose among three prompts by greedy accuracy on the same 200 selection questions RandOpt
uses (600 generations instead of RandOpt's 1,000,000), then sample and vote.

| Row | RandOpt | Self-consistency, prompt chosen on selection data | Difference [95% CI] |
|---|---|---|---|
| GSM8K / Qwen2.5-1.5B (boxed chosen) | 77.2 | 80.1 | **−2.96 [−4.62, −1.29]** |
| GSM8K / Qwen2.5-3B (boxed) | 86.7 | 87.0 | −0.38 [−1.67, +0.91], equivalent |
| GSM8K / OLMo-2-1B (boxed) | 52.5 | 76.1 | **−23.65 [−26.23, −21.08]** |
| GQA / Qwen2.5-VL-3B (direct) | 63.5 | 64.7 | −1.21 [−2.83, +0.40] |

**RandOpt is ahead in none of the four rows** (self-consistency ahead in two, no difference detected in two).

Why, row by row, under RandOpt's own prompts:

| Row | Base | Selected models, individually | RandOpt | Self-consistency | RandOpt − SC |
|---|---|---|---|---|---|
| GSM8K / Qwen2.5-1.5B | 60.3 | 64.3 | 77.2 | 79.8 | −2.65 [−4.32, −0.99] |
| GSM8K / Qwen2.5-3B | 80.7 | 80.9 | 86.7 | 88.2 | −1.59 [−2.65, −0.53] |
| GQA / Qwen2.5-VL-3B | 53.4 | 58.4 | 63.5 | 60.0 | +3.47 [+1.62, +5.41] (replicated: +3.63) |
| GSM8K / OLMo-2-1B | 35.3 | 40.3 | 52.5 | 43.7 | +8.72 [+6.75, +10.77] |

- **Where RandOpt loses**, the selected models are barely better than the base model on their own. The gain is the
  vote, and sampling the unperturbed model votes better.
- **Where RandOpt wins**, the selected models are about 5 points better on their own: selection found a shift
  shared across questions. In both rows RandOpt's own prompt damages the base model heavily, and the shift recovers
  part of that accuracy:
  - On **GQA**, the step-by-step prompt costs the model 11.3 points. The selected models mostly stop reasoning and
    answer in a few words. Told to answer directly, the base model scores 64.7% with **one** generation (RandOpt:
    63.5%, no difference detected), and search adds nothing measurable on top of that prompt.
  - On **OLMo**, the "output the final answer after ####" instruction halves the model's accuracy (33.7% vs 65.3%
    when simply asked the question).
- Damage is not the whole story: against the boxed prompt selection chose, Qwen-1.5B is damaged by 10 points too, yet
  RandOpt still lost there (`paper/figures/thickets-or-tilts/fig8_prompt_damage`).

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

### 3. Why selection finds shared shifts, and where that stops

Under the first-order law, a random perturbation shifts each question's answer margin by a Gaussian with standard
deviation σ‖∇margin‖ (`paper/THEORY.md`). So flip probabilities are predictable per question (AUC 0.935), an
unselected vote returns the base model's answer (96/96 questions), σ‖∇‖ acts like a per-question sampling
temperature, and top-K selection is a noisy step along the selection set's gradient: it favours shifts shared across
the selection questions. This covers direct answers at small σ, not chain-of-thought.

In an N = 5000 search on OmniSpatial (Qwen3-VL-8B), the selected model's +8.0 pp selection gain sat on a question
format the test set lacked; on fresh, format-matched items it keeps **+2.67 pp [0.17, 4.93]**. Selection favours
perturbations that tilt answers toward content the selection labels reward, but removing that tilt still leaves +2.33
of the +2.67 pp. We report this as a limit of the account.

**Scope.** Two tasks for the same-run comparisons, models up to 8B, one RandOpt search per GSM8K row (two on GQA),
three prompt candidates per task fixed in advance.

## Repository layout

```
paper/
  STORY.md                   the paper's argument: one question, three claims, where each experiment goes
  RESULTS_MASTER.md          every number the paper may use, with its lock commit and source file
  THEORY.md                  first-order derivations (T1–T4) and their checks
  WRITING_PROMPT.md          drafting rules for the paper (structure, wording, what not to claim, credit)
  CORRECTION_SPLIT_MISMATCH.md, WHY_INVESTIGATION_9504111.md   the OmniSpatial case-study record
  figures/thickets-or-tilts/ generated figures and tables;  figures/scripts/thickets_or_tilts.py regenerates them
results/README.md            index: every study's question, lock commit, result, verdict and role in the paper
results/paper-analysis/<study>/
  plan_lock.md               the pre-registration (committed before the run)
  *_RESULT.md, *_results.json   outcome against the locked rule
  pod/                       raw per-item outputs pulled from the GPU runs
results/perspective-taking-n5000-20261004/   raw scores of the original N = 5000 OmniSpatial search
scripts/                     runners (GPU) and locked analyses (CPU); scripts/README.md maps studies to scripts
                             (kept flat because plan locks cite these paths)
src/thicket_runtime/         a small library for fast, verified weight-state handling (used by the runners)
tests/                       unit tests for the perturbation-folding and grouping code
examples/                    the frozen OmniSpatial item splits used by the case study
```

The full ledger of every locked test and its outcome, including falsified and inconclusive ones, is in
`paper/RESULTS_MASTER.md` §R9.

## Reproducing

**CPU analyses and figures** (Python ≥ 3.10):

```bash
pip install -e '.[test]'
git clone https://github.com/sunrainyg/RandOpt third_party/RandOpt && git -C third_party/RandOpt checkout 4000d34
(cd third_party/RandOpt && python ../../scripts/c_prep_gsm8k.py)   # GSM8K in RandOpt's format (needs `datasets`)
python paper/figures/scripts/thickets_or_tilts.py          # all figures and tables
python scripts/ps_analysis.py --upstream third_party/RandOpt \
       --ps results/paper-analysis/ps-prompt-selection/pod/ps/out --out /tmp/ps_results.json   # the headline (PS)
python scripts/theory_check.py                              # theory checks (T1, T2, T4)
pytest -q
```

Each study's `plan_lock.md` gives its exact commands; `results/README.md` lists every study.

**GPU runs** used single- and multi-H100 pods with a pinned stack: vLLM 0.11.0, transformers 4.57.1,
torch 2.8.0, huggingface-hub 0.36.2, datasets 3.6.0. `scripts/pod_jobqueue.sh <cap_usd> <model> <revision>` sets up
the pod, pins packages, clones RandOpt @ 4000d34, enforces a hard spending cap and an idle stop, and runs queued
job scripts.

Models (pinned revisions in each lock): Qwen3-VL-8B-Instruct, Qwen2.5-VL-7B/3B-Instruct, Qwen2.5-1.5B/3B-Instruct,
OLMo-2-0425-1B-Instruct, OLMo-2-1124-7B-Instruct. Data: OmniSpatial, GQA (testdev-balanced), GSM8K,
ARC-Challenge, MATH-500.

**A note on GQA:** RandOpt's released `randopt.py` (4000d34) builds text-only prompts and does not pass images to
vLLM, so it cannot run GQA with images. Our GQA runs (`g2-sameRun/`, `g2r-seed/`) re-implement its loop on RandOpt's own
perturbation, scoring and voting components.

## History

This repository began as a broader exploration (runtime profiling, adaptive evaluation, complementarity selection,
visual line-tracing, an "expert mirage" framing). Those directions were removed from the working tree on 2026-10-07
and are preserved at git tag [`pre-cleanup`](../../tree/pre-cleanup). A smaller cleanup on 2026-10-09 (operational
logs) is preserved at [`pre-cleanup-2`](../../tree/pre-cleanup-2).

## Credit

This work builds on and re-examines Neural Thickets and RandOpt (Gan & Isola; code at
[sunrainyg/RandOpt](https://github.com/sunrainyg/RandOpt)). Related observations we credit:
- the original paper's own analysis of format effects;
- selection bias toward the selection set (arXiv 2608.10867);
- the blog posts *A Thicket by Any Other Name* and *When does RandOpt work?*.

Self-consistency follows Wang et al. (2023).

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

> **Status (2026-10-08):** the core experiments are complete, including the GQA replication (G2R) and the
> direct-answer prompt control (G4). Optional follow-ups (Countdown; a non-Qwen same-run row) are under consideration.
> The paper is being written.

## Findings

### 1. Choosing the prompt on RandOpt's own selection data matches or beats its weight search

Three same-run comparisons at the paper's settings (N = 5000 perturbations, top K = 50), against self-consistency
with the same number of test-time generations (50 samples at T = 0.7, no weight search):

| Task / model | Base | RandOpt (paper) | Self-consistency | RandOpt − SC [95% CI] |
|---|---|---|---|---|
| GSM8K / Qwen2.5-1.5B | 60.3 | 77.2 (76.4) | 79.8 | **−2.65 [−4.32, −0.99]** |
| GSM8K / Qwen2.5-3B | 80.7 | 86.7 (87.1) | 88.2 | **−1.59 [−2.65, −0.53]** |
| GQA / Qwen2.5-VL-3B (1238 questions) | 53.4 | 63.5 (69.0*) | 60.0 | **+3.47 [+1.62, +5.41]** |
| GQA, second RandOpt run (new perturbations) | 53.4 | 63.7 | 60.0 | **+3.63 [+1.78, +5.49]** |
| GSM8K / OLMo-2-1B (non-Qwen) | 35.3 | 52.5 (–) | 43.7 | **+8.72 [+6.75, +10.77]** |
| GSM8K / OLMo-2-1B, model asked just the question | **65.3** (one answer) | 52.5 | 74.7 | **−22.21 [−24.87, −19.56]** |
| GQA, base model told to answer directly | **64.7** (one answer) | 63.5 | 64.7 | **−1.21 [−2.83, +0.40]** |

<sub>*The paper evaluates GQA on all of testdev with train-split selection; we use image-disjoint testdev splits.</sub>

On GSM8K, the selected models are individually only 0–4 pp better than the base model, so the gain is the vote, and
sampling the unperturbed model votes better. On GQA, RandOpt's lead exists only under its own step-by-step prompt,
which costs this model 11 points. The selected perturbations mostly switch that reasoning off and answer in a few
words. Simply telling the base model to answer directly gives 64.7% from **one** generation with no search, at least as
good as RandOpt's 5000-model search plus 50-model vote, and search adds nothing on top of that prompt.
On a non-Qwen model (OLMo-2-1B), RandOpt beats self-consistency on GSM8K by 8.7 pp: like on GQA, its selected
models are individually better than base. But RandOpt's GSM8K instruction ("output the final answer after ####")
halves this model's accuracy: asked just the question, one generation scores 65.3% and self-consistency 74.7%, far
above RandOpt's 52.5%. In both settings where RandOpt beats self-consistency, the search is repairing damage done by
its own prompt, and repairing the prompt does better. On the Qwen rows RandOpt's prompt does not hurt (−7 to +5 points),
and there RandOpt loses to plain self-consistency. The plain prompt is not a universal fix (it is worse for Qwen), but
in all four rows a baseline without weight search matches or beats RandOpt.

The practical version: choose among three prompts by greedy accuracy on the same 200 selection questions RandOpt uses
(600 generations instead of RandOpt's 1,000,000), then sample 50 answers and vote. Against RandOpt's 50-model vote:

| Row | RandOpt | Prompt-selected self-consistency | Difference [95% CI] |
|---|---|---|---|
| GSM8K / Qwen2.5-1.5B (boxed chosen) | 77.2 | 80.1 | **−2.96 [−4.62, −1.29]** |
| GSM8K / Qwen2.5-3B (boxed) | 86.7 | 87.0 | −0.38 [−1.67, +0.91], equivalent |
| GSM8K / OLMo-2-1B (boxed) | 52.5 | 76.1 | **−23.65 [−26.23, −21.08]** |
| GQA / Qwen2.5-VL-3B (direct) | 63.5 | 64.7 | −1.21 [−2.83, +0.40] |

RandOpt is ahead in none of the four.

Prompt damage is part of the story but not all of it: against the boxed prompt that selection chose, Qwen-1.5B is
damaged by 10 points too, yet RandOpt still lost to self-consistency there (`paper/figures/thickets-or-tilts/fig8_prompt_damage`).

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
  WRITING_PROMPT.md          drafting rules (wording, what not to claim, credit to prior work); to be rewritten for G2R–G4
  CORRECTION_SPLIT_MISMATCH.md, WHY_INVESTIGATION_9504111.md   the OmniSpatial case-study record
  figures/thickets-or-tilts/ generated figures and tables;  figures/scripts/thickets_or_tilts.py regenerates them
results/README.md            index: every study's question, lock commit, result, verdict and role in the paper
results/paper-analysis/<study>/
  plan_lock.md               the pre-registration (committed before the run)
  *_RESULT.md, *_results.json   outcome against the locked rule
  pod/                       raw per-item outputs pulled from the GPU runs
results/perspective-taking-n5000-20261004/   raw scores of the original N = 5000 OmniSpatial search
scripts/                     runners (GPU) and locked analyses (CPU), named by study; scripts/README.md maps
                             each study to its scripts (kept flat because plan locks cite these paths)
src/thicket_runtime/         a small library for fast, verified weight-state handling (used by the runners)
tests/                       unit tests for the perturbation-folding and grouping code
examples/                    the frozen OmniSpatial item splits used by the case study
```

**Study index:** `results/README.md` lists every study (question → lock → result → verdict → role in the paper).

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
vLLM, so it cannot run GQA with images. Our GQA runs (`g2-sameRun/`, `g2r-seed/`) re-implement its loop on RandOpt's own
perturbation, scoring and voting components; `g3-termination/` and `g4-direct-prompt/` reuse those outputs.

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

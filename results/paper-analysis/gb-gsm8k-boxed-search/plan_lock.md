# GB plan lock: RandOpt's weight search under the selection-chosen (boxed) prompt on GSM8K

Committed before any GB output. User-provided H100 pod; session cap = the user's stated balance − $2 (recorded in
`pod/rate.txt` and an amendment if it differs from $48).

## Why
GD showed that on GQA, with the prompt held fixed, RandOpt's search is equivalent to sampling. PS chose the boxed prompt
for every GSM8K model, but no search was run under it, so the GSM8K rows still compare RandOpt under its own prompt
with SC under the chosen one. GB runs the search under the boxed prompt.

## Rows (in this order)
- **GB-1 (primary): OLMo-2-0425-1B-Instruct** @ 48d788eca847d4d7548f375ad03d3c9312f6139e.
- **GB-2 (conditional): Qwen2.5-1.5B-Instruct** @ 989aa7980e4cf806f80c7fef2b1adb7bc71aa306, run only if the budget rule
  below allows; otherwise reported as not run.

## Design
Runner `scripts/gsm_randopt_fast.py`: randopt.py's GSM8K loop re-implemented on RandOpt's components (WorkerExtension
`apply_perturbation`, GSM8KHandler `compute_reward` for selection, `extract_answer` / `is_answer_correct` for test and
vote). Population `np.random.default_rng(42)`: **the same 5000 perturbations as O1 (OLMo) and C (Qwen-1.5B)**
(checked: O1's reconstructed top 50 are in it with identical σ). Selection: first 200 GSM8K train rows; greedy
(temperature 0, seed 42), max_tokens 1024; ranking by reward, ties by index; K = 50 (10 secondary); test 1319 rows.
Prompt: **boxed** (`o2_eval.PROMPTS['boxed']`, identical to O2/Q2/PS). Deviations from randopt.py (stated): exact
weight reset from a stored base copy instead of subtract-to-restore; CUDA graphs; several workers per GPU. Texts are
saved for every test generation.

Engineering: selection with 3 workers per GPU, each next worker on a GPU launched only after the previous one prints
ENGINE_READY; test phase with 1 worker per GPU. Failed workers are re-run in parallel on free GPUs with identical
settings (resumable).

## Gates (per row)
- **Environment (blocking):** BASE greedy selection reward under RandOpt's prompt equals the value randopt.py printed
  (O1: 41.50; C: 73.00) within 1.0 pp, and under the boxed prompt equals PS's selection accuracy (OLMo 79.5; Qwen-1.5B
  80.5) within 1.0 pp; this run's boxed BASE test accuracy equals O2/Q2's boxed BASE (67.85; 70.20) within 1.0 pp.
  Otherwise the row is INVALID-ENV.
- **Fidelity (reported):** rewards of perturbations k = 0–23 under RandOpt's prompt vs those randopt.py logged in
  O1 / C; consistent if the mean |Δ| ≤ 0.03.
- **Budget:** smoke (boxed, 15 perturbations per worker) measures seconds per perturbation s. GB-1 runs if spent +
  5000/W × s × 1.10 × rate + test (51 passes / G × 25 s) × rate + $2 ≤ cap. GB-2 runs only if, after GB-1, spent +
  its own projection + $2 ≤ cap. Otherwise not run (reported; no silent reduction of N).

## Primary (per row; GB-1 is the primary test)
D = acc(RandOpt K = 50 vote, searched and voted under the boxed prompt) − acc(SC@50, boxed prompt; O2 for OLMo, PS for
Qwen-1.5B); paired item bootstrap (10,000, seed 0). **RANDOPT AHEAD** if CI lower > 0; **SC AHEAD** if CI upper < 0;
else **NO DIFFERENCE DETECTED**; plus **EQUIVALENT** if the CI lies within ±2 pp.

## Secondary
K = 10; members' mean accuracy vs the boxed BASE (paired CI); RandOpt (boxed) − RandOpt (its own prompt: O1 / C);
RandOpt (boxed) − one boxed BASE generation; top-50 overlap with O1 / C; σ of the top 50; selection gain vs test gain.

## How it is used (fixed now; no further GPU runs whatever the outcome)
RANDOPT AHEAD → with the prompt held fixed, weight search adds a measured amount on GSM8K for that model; the paper
reports it and states Claim 1 as "prompt selection captures most of RandOpt's gain; search adds X on top" for that row.
NO DIFFERENCE / SC AHEAD → with the prompt held fixed, weight search adds nothing measurable over sampling, as on GQA.

## Analysis
`scripts/gb_analysis.py`. Self-test with O1's RandOpt-prompt ensemble standing in for GB reproduces PS's OLMo row
(−23.65 [−26.23, −21.08]), top-50 overlap 50, own-prompt difference 0, members 40.33.

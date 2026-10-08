# O2 plan lock: is RandOpt's GSM8K win on OLMo-2-1B a prompt effect (as on GQA)?

Committed before any O2 output. 1× H100; session cap **$13** (the rest of the user's $60 for O1 + O2; pod guard).

## Why
O1 (e2e2400): on GSM8K / OLMo-2-1B, RandOpt K = 50 (52.46%) beats SC@50 (43.75%) by +8.72 [+6.75, +10.77]; base is
only 35% under RandOpt's prompt and the selected models are +5.1 pp better individually. On GQA, such a shared shift
was something a prompt also gives (G4). O2 tests whether the same holds here, with texts saved for every arm.

## Design
Model `allenai/OLMo-2-0425-1B-Instruct` @ 48d788eca847d4d7548f375ad03d3c9312f6139e; GSM8K test (1319; `openai/gsm8k`
main, as p2_sc.py); chat template; max_tokens 1024; greedy, and SC with 50 samples at T = 0.7, top_p 1, seed 20261007
(as p2_sc.py). Scorer: RandOpt's GSM8K handler (strict "####" extraction, else flexible last number) for every arm;
votes as O1 (most_common over non-empty extracted answers). Runner `scripts/o2_eval.py` (one vLLM engine, CUDA
graphs, RandOpt's WorkerExtension). Prompts, fixed now (user message):
- **randopt** (reference): question + ' Let\'s think step by step and output the final answer after "####".' (O1's prompt)
- **plain (primary):** the question only (the model's own answer format)
- **boxed (secondary):** question + "\nPlease reason step by step, and put your final answer within \boxed{}."

Selected models: O1's top 50, **reconstructed** from O1's log (RandOpt's seed/σ generator with global seed 42 and the
5000 logged rewards, stable sort as randopt.py; `top50_reconstructed.json`; matches all 10 rows randopt.py printed)
and rebuilt with `apply_perturbation` (same noise as randopt.py's `perturb_self_weights`, but from stored base weights).

Order (primary first): randopt base; plain base, plain SC; randopt members ranks 0–4 and 45–49 (fidelity); plain
members 0–49; boxed base, boxed SC.

## Validity
- Environment: randopt-prompt base greedy equals O1's greedy (same scorer) within 1.0 pp; else O2-1 is INVALID-ENV.
- Member fidelity (for the members secondary only): ranks 0–4 and 45–49 under the randopt prompt agree with O1's dumped
  answers on ≥ 90% of items on average and their accuracy differs by ≤ 2 pp on average (O1 ran eager vLLM with
  in-place subtract-to-restore; O2 uses CUDA graphs and exact restore). Else the members secondary is reported as
  not valid.

## Primary O2-1
D = acc(RandOpt K = 50 vote, O1, RandOpt's prompt) − acc(SC@50, **plain** prompt); paired item bootstrap (10,000, seed 0).
**RANDOPT AHEAD** if CI lower > 0; **PROMPT-SC AHEAD** if CI upper < 0; else **NO DIFFERENCE DETECTED**; plus
**EQUIVALENT** if the CI lies within ±2 pp.

## Secondary (reported regardless)
- Same D with the boxed prompt; SC@10.
- Plain/boxed base greedy vs randopt-prompt base; RandOpt (O1) − plain/boxed base greedy (one generation).
- Search on top of the prompt: O1's selected models evaluated with the plain prompt, K = 50 vote − SC@50 (plain), and
  their mean accuracy (valid only if the fidelity gate passes).
- Format per arm: mean tokens, share hitting max_tokens, share containing "####", share with \boxed.

## How it is used
RANDOPT AHEAD → OLMo's GSM8K win is not just RandOpt's prompt: weight search gives something a prompt change does not
(paper: RandOpt genuinely helps this model). EQUIVALENT or PROMPT-SC AHEAD → as on GQA, the shared shift is one a prompt
also gives (paper: both RandOpt wins are prompt mismatches). NO DIFFERENCE without equivalence → inconclusive.

## Analysis
`scripts/o2_analysis.py`. Self-test with every O2 arm replaced by O1's own data reproduces O1: D = +8.72 [+6.75, +10.77],
environment difference 0, fidelity 1.0, members 40.3%.

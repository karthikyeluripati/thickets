# GD plan lock: RandOpt's weight search under the selection-chosen (direct) prompt on GQA

Committed before any GD output. 6× H100 ($23.94/h); session cap **$25** (pod guard; user-approved).

## Why
PS showed prompt selection plus SC@50 matches RandOpt on GQA, but RandOpt ran under its own CoT prompt, and G4's
"search on top of the prompt" re-used perturbations selected under that CoT prompt. GD runs the search itself under
the prompt PS chose for GQA (direct), so prompt is held fixed between RandOpt and SC.

## Design: identical to G2 except the prompt
- Runner `scripts/g2_randopt_gqa.py` with `--prompt direct` (G4's direct prompt, verified identical) and
  `--pop-seed 42`: **the same 5000 perturbations as G2**; only the prompt differs. Same model revision, 200 selection
  questions, 1238 test questions, greedy, max_tokens 256, ranking, K = 50 (and 10), vote, scorer.
- Engineering (no effect on outputs): CUDA graphs, 2 workers per GPU (12 workers on 6 GPUs, gpu_memory_utilization
  0.33), vLLM multimodal preprocessing cache on (weight-independent image preprocessing; prefix caching stays off).
  Failed workers are re-run alone with identical settings (resumable).
- SC arm: G4's direct-prompt SC@50 (same questions, model, prompt, T = 0.7, seed 20261007); BASE reference: G4's
  direct-prompt greedy.

## Gates
- Smoke: 12 perturbations + 30-question test pass, all workers, separate directory.
- Projection: spent + 5000/12 × (mean smoke seconds per perturbation, slower 3/4 of workers) × 1.10 × rate + test
  (51 passes / 12 workers × 60 s) × rate + $2 ≤ **$24**, else the full run does not start (ask the user).
- Validity: this run's direct-prompt BASE greedy test accuracy equals G4's (64.70%) within 1.0 pp; else INVALID-ENV.

## Primary GD-1
D = acc(RandOpt K = 50 vote, direct prompt) − acc(SC@50, direct prompt), paired item bootstrap (10,000, seed 0).
**RANDOPT AHEAD** if CI lower > 0; **SC AHEAD** if CI upper < 0; else **NO DIFFERENCE DETECTED**; plus **EQUIVALENT**
if the CI lies within ±2 pp.

## Secondary (reported regardless)
K = 10; members' mean accuracy vs the direct-prompt base (paired CI); RandOpt (direct) − RandOpt (CoT, G2); RandOpt
(direct) − one direct-prompt base generation; overlap of the top 50 with G2's top 50; σ of the top 50; vote gain.

## How it is used
RANDOPT AHEAD → with the prompt held fixed, weight search adds a measured amount on GQA; the paper reports it and
qualifies "matches or beats" to "prompt selection captures most of RandOpt's gain; search adds X on top".
NO DIFFERENCE / SC AHEAD → with the prompt held fixed, weight search adds nothing measurable over sampling on GQA.

## Analysis
`scripts/gd_analysis.py`. Self-test with G2's CoT run standing in for GD reproduces G4's primary (−1.21 [−2.83,
+0.40]), top-50 overlap 50, CoT difference 0, and correctly flags INVALID-ENV (CoT base 53.39 ≠ direct 64.70).

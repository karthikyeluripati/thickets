# G2R plan lock: does RandOpt's GQA advantage over self-consistency replicate with a second perturbation population?

Committed before any G2R output. 2× H100 pod; session cap **$30** (hard stop by the pod guard).

## Why
G2 (lock d2b3d68 + amendment 1) found RandOpt K = 50 ahead of SC@50 on GQA / Qwen2.5-VL-3B: D = +3.47 [+1.62, +5.41].
That interval covers question sampling only, not which 5000 perturbations were drawn. G2 is the only setting in this
study where RandOpt beats SC; G2R tests whether that result depends on the draw.

## Design: identical to G2 except the population seed
- Runner `scripts/g2_randopt_gqa.py` (unchanged except a `--pop-seed` option, default 42 = G2); **G2R uses
  `--pop-seed 43`**: `np.random.default_rng(43)` → 5000 seeds and σ ∈ {0.0005, 0.001, 0.002}, same generator as
  randopt.py. 0 seeds shared with G2's population (checked).
- Same model revision, pinned RandOpt (4000d34), same prep (`scripts/g2_prep.py`; test items asserted identical to G2's
  1238), same 200 selection questions, prompt, greedy decoding (max_tokens 256), reward, ranking (ties by index),
  K = 50, vote.
- Same SC arm as G2: P2's GQA SC run (`p2/pod/gqa_vl3_T0.7.json.gz`, 50 samples at T = 0.7), mapped with G2's key.
- Engineering (no effect on data): "fast" configuration of G2 amendment 1 (CUDA graphs, 2 workers per GPU,
  gpu_memory_utilization 0.36) on 2 GPUs → **W = 4 workers**, worker w on GPU w mod 2, second wave 90 s later.
  Resumable runner; a worker that fails is re-run with the same settings and existing files are skipped (reported).

## Gates
- Smoke: 8 perturbations + 30-question test pass, all 4 workers, into a separate directory (asserts vision weights
  unchanged and exact restore, as G2).
- Projection: spent + 5000/4 × (mean smoke seconds per perturbation, slower 3/4 of workers) × 1.05 × rate + test
  estimate (51 passes / 4 workers × 120 s) × rate + $2 **≤ $28**, else the full run does not start (ask the user).
- Validity: G2R base greedy test accuracy equals G2's (53.39%) within 1.0 pp; otherwise reported as INVALID-ENV.

## Primary G2R-1
D₂ = acc(RandOpt K = 50 vote, seed 43) − acc(SC@50 vote), paired item bootstrap (10,000, seed 0), the G2 rule:
**REPLICATED** if CI lower > 0; **REVERSED** if CI upper < 0; else **NOT REPLICATED**.

## Secondary (reported regardless)
- D₂ − D₁ (paired bootstrap over items; describes how much the draw moves D); two-seed mean D with CI.
- K = 10 version of all of the above; member mean and range; σ of the top 50; selection-reward range; base reward.

## Analysis
`scripts/g2r_analysis.py --d1 <G2 dir> --d2 <G2R dir>`. Self-test with d1 = d2 = G2 reproduces G2 exactly
(D = +3.47 [+1.62, +5.41]; D₂ − D₁ = 0).

## How it is used
REPLICATED → the paper reports both seeds and the two-seed mean. NOT REPLICATED or REVERSED → the GQA claim is
weakened to "positive in one of two draws" and the paper says so; the GSM8K side is unaffected.

# O1 plan lock: same-run RandOpt (N = 5000) vs self-consistency on a non-Qwen model (GSM8K / OLMo-2-1B-Instruct)

Committed before any O1 output. 8× H100 pod, user-provided; session cap **$60** (pod guard).

## Why
All three same-run rows (C, C3B, G2/G2R) use Qwen models, and RL-for-reasoning gains are known to be Qwen-specific in
some settings. O1 asks whether Claim 1 (on GSM8K, RandOpt's gain is the vote and self-consistency votes better)
holds on a different model family. OLMo-2-1B is the non-Qwen model already used in Claim 2 (S1). The Neural Thickets
paper reports no number for it, so there is no published reference to reproduce.

## Design: identical to Session C / C3B except the items below
- Model `allenai/OLMo-2-0425-1B-Instruct` @ `48d788eca847d4d7548f375ad03d3c9312f6139e`; pinned stack as
  `scripts/pod_jobqueue.sh`; RandOpt @ 4000d34.
- RandOpt arm: RandOpt's own `randopt.py`, unmodified except the same one-line answer dump as C/C3B;
  `--dataset gsm8k --num_engines 8 --tp 1 --cuda_devices 0,1,2,3,4,5,6,7 --train_samples 200 --precision bfloat16
  --max_tokens 1024 --sigma_values 0.0005,0.001,0.002 --global_seed 42 --population_size 5000
  --top_k_ratios 0.01,0.002` (K = 50, 10). Data from `scripts/c_prep_gsm8k.py` (hashes train c6f812ae33c9159d,
  test 59ec1b7f9357c7a2, as C/C3B). RandOpt's engine launcher places one engine per GPU; 8 engines use all 8 GPUs.
- SC arm (same pod, after the RandOpt arm): `scripts/p2_sc.py --task gsm8k --model allenai/OLMo-2-0425-1B-Instruct
  --revision <pinned> --tag o1 --temps 0.7 --n 50` (greedy + 50 samples at T = 0.7, seed 20261007).

## Gates
- Smoke: 8 engines, 24 perturbations, 30 test items; measures seconds per batch b (from randopt's per-batch log).
- Projection for the primary: spent + 625·b·1.05/3600·rate + $4 (test ensemble + SC) + $2 ≤ **$57**, else the full run
  does not start (ask the user; no silent reduction of N).
- Validity: recomputed K = 50 vote equals randopt.py's printed count, else O1-1 is INVALID.

## Conditional second population seed (O1b; uses leftover budget, decided by rule now)
After the primary and SC arms, run the identical RandOpt arm with `--global_seed 43` **only if** spent +
625·b·1.05/3600·rate + $3 ≤ $58.5 (b from the primary run's own per-batch log). Otherwise O1b is not run (reported).

## Primary O1-1 (rules as C-1)
D = acc(RandOpt K = 50 vote, seed 42) − acc(SC@50, T = 0.7), 1319 items, paired item bootstrap (10,000, seed 0),
RandOpt's vote rule and scorer for both arms. **RANDOPT AHEAD** if CI lower > 0; **SC AHEAD** if CI upper < 0; else
**NO DIFFERENCE DETECTED**; plus **EQUIVALENT (±2 pp)** if the CI lies within [−2, 2].

## Secondary (reported regardless)
K = 10; base greedy (randopt.py and p2_sc); member mean and range; vote curve K = 1/5/10/20/50; SC single-sample mean;
if O1b runs: its D, D(s43) − D(s42), two-seed mean D (paired bootstrap).

## Reading fixed in advance
- SC AHEAD or NO DIFFERENCE DETECTED → Claim 1's GSM8K finding extends to a non-Qwen family (stated with the CI).
- RANDOPT AHEAD → Claim 1 is qualified as model-dependent on GSM8K; the member-vs-vote decomposition is reported to
  show whether the selected OLMo models are individually better (a shared shift) or the gain is the vote.

## Analysis
`scripts/o1_analysis.py`. Self-test on Session C's data (d1 = d2 = C) reproduces C: D = −2.65 [−4.32, −0.99],
1018 = printed, members 64.3%, vote curve 68.2/74.6/77.2/77.5/77.2, seed difference 0.

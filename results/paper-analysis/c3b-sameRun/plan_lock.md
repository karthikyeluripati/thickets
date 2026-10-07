# Session C3B plan lock: same-run RandOpt (N = 5000) vs self-consistency, GSM8K / Qwen2.5-3B-Instruct

Committed before any C3B output. 6× H100 pod (rate $20.94/h), user-provided; session cap **$120** shared with the
later GQA and 7B runs; this run's projection gate is **$72** (see below). Design identical to Session C
(`../c-sameRun/plan_lock.md` 1437d44 + `amendment_1.md` 6118222) except the model and the items below.

## Fixed setup
- Model `Qwen/Qwen2.5-3B-Instruct` @ `aa8e72537993ba99e69dfaafa59ed015b17504d1` (repo unchanged since 2024-09-25, so it
  is also the revision P3 ran as "main"). Paper (Neural Thickets) numbers for this row: Base 79.8, TT-MV 82.5,
  RandOpt 87.1.
- RandOpt arm: RandOpt's `randopt.py` @ 4000d34 with the same one-line answer dump; `--dataset gsm8k --num_engines 6
  --cuda_devices 0,1,2,3,4,5 --train_samples 200 --precision bfloat16 --max_tokens 1024
  --sigma_values 0.0005,0.001,0.002 --global_seed 42 --population_size 5000 --top_k_ratios 0.01,0.002` (K = 50, 10).
  Data from `scripts/c_prep_gsm8k.py` (identical to Session C).
- **Primary SC arm = P3's SC run** (`results/paper-analysis/p3/pod/gsm8k_q3_{greedy,T0.7}.json.gz`, same model
  revision, same pinned environment and `p2_sc.py`, H100), fixed now before any RandOpt output. A same-pod replicate
  (tag c3b) runs only if projected spend after it stays ≤ $72; it is secondary.
- Projection gate: a 6-engine smoke run (24 perturbations, 30 test items) measures seconds per batch b. Start the full
  run only if spent + (5000/6)·b·1.05/3600·rate + $3.0 ≤ $72. Otherwise do not start; ask the user (no silent N cut).

## Primary (C3B-1), rules identical to C-1
D = acc(RandOpt K = 50 vote) − acc(SC@50, T = 0.7), 1319 items, paired item bootstrap (10,000, seed 0), RandOpt's
vote rule and scorer for both arms; validity requires the recomputed K = 50 vote to equal randopt.py's printed count.
**RANDOPT AHEAD** if CI lower > 0; **SC AHEAD** if CI upper < 0; else **NO DIFFERENCE DETECTED**; plus
**EQUIVALENT (±2 pp)** if the CI lies within [−2, 2].

## Secondary
K = 10; base gate vs paper 79.8 (±2.0; reported, does not change C3B-1); RandOpt K = 50 vs paper 87.1; the
member-vs-vote decomposition (as in C).

## Reading fixed in advance
Combined with C (1.5B: SC AHEAD), the two same-run rows are reported separately; no pooled test.

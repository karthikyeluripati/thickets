# Session C amendment 1 (committed before any RandOpt-arm output; supersedes only the items below)

Reason: on the first pod (1× H100, $8 cap) the locked budget rule measured 8.2 s per perturbation, so no N ≥ 1000
fit; the RandOpt arm did not run (02-randopt rc=4, `N_DOES_NOT_FIT`). The SC arm ran there as locked.
The user provided a 6× H100 pod (rate $20.94/h); hard cap **$55**.

Changes:
1. **N = 5000 fixed** (the Neural Thickets paper's population), replacing the {2000, 1500, 1000} budget rule.
   `--num_engines 6 --cuda_devices 0,1,2,3,4,5`; `--top_k_ratios 0.01,0.002` (K = 50, 10). All other randopt.py
   arguments as locked (σ ∈ {0.0005, 0.001, 0.002}, seed 42, 200 train items, greedy, max_tokens 1024, full test).
   A 6-engine smoke run (12 perturbations, 30 test items) must pass first. If its projected total exceeds the cap,
   the full run is not started and the user is asked (no silent reduction of N).
2. **Primary SC arm = the SC run from the first pod** (`pod/c/out/gsm8k_c15_{greedy,T0.7}.json.gz`; H100, same
   model revision, same pinned environment, same `p2_sc.py`), fixed now, before any RandOpt-arm result exists.
   If budget remains after the RandOpt arm (projected spend + $5 ≤ $54), SC is re-run on this pod as a **secondary**
   same-pod replicate (tag c15b); it does not replace the primary.
3. Recovery: randopt.py writes nothing until the end. Its per-batch log lines (rewards in deterministic seed order,
   seeds from `default_rng(42)`) are kept so that, if the pod is stopped mid-search, the top-50 can be reconstructed
   and the ensemble evaluated later with `--resume_dir`; such a recovery would be reported.

Decision rules, vote rule, scorer, statistics and caveats are unchanged (`plan_lock.md`, commit 1437d44).

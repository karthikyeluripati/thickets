# Validation history — October 3, 2026

## Work 1 original-worker and vLLM/Ray closure

- **50 passed, 1 skipped** on H100; the skipped test is the CPU-only CUDA guard.
- Original-worker HF controls reproduce both previous correctness audits exactly,
  with identical candidate recipes and IDs (24 additional timed evaluations).
- Six complete native vLLM/Ray processes, counterbalanced by strategy order,
  validate 144 timed evaluations across fixed-32, fixed-128, and natural workloads.
- All native snapshot gates pass; all legacy restoration gates fail. Native
  candidate-state hashes, token equality, and reward equality are audited outside timing.
- A native noise-layout probe demonstrates that the original per-tensor RNG
  produces different perturbations on packed vLLM versus separate HF tensors.
- All 133 new artifact checksums and all 39 prior checksums verify. Exact measured
  source snapshots are preserved. No GPU profiling jobs remain running.

See [the closure report](WORK1_CLOSURE_2026-10-03.md) for measured fractions,
natural stopping lengths, and limits on broader or cross-backend closure claims.

## Subsequent H100 real-LLM session

Python 3.12.3 / PyTorch 2.8.0+cu128 / Transformers 4.56.2 on one H100 80GB HBM3:

- GPU suite: **43 passed, 1 skipped** (the CPU-only unavailable-CUDA guard).
- Five completed Qwen2.5-0.5B BF16 runs: fixed 32/128 tokens, batch 1/4, plus
  a separate 32-token batch-4 wall-only control.
- All snapshot-copy exact-semantics gates pass; all legacy gates fail. Real
  output and score disagreements occur in the legacy 128-token batch-4 audit.
- Apply/restore is 0.57–2.24% of snapshot wall time in the four event-enabled cases.
- Raw artifacts verified against 39 file checksums; remote runtime source hashes
  match the local implementation after LF normalization.
- Reproduction shell syntax, Python code compilation, and analysis on all five
  reports were checked. The full packaged runner was not rerun.
- All ten derived strategy summaries match the originals; all 120 timed rows
  preserve the shared candidate trace and expected repetitions.

See [the LLM pilot report](LLM_PILOT_2026-10-03.md) and
[raw artifacts](../results/llm-bounded-20261003/). This validates real HF baseline
execution; it does not establish a custom-kernel or vLLM/Ray speedup.

## Initial local CPU session (historical)

Environment actually exercised: **Python 3.13.5, PyTorch 2.10.0+cpu**.
CUDA was unavailable. This is not a GPU benchmark report.

## Executed successfully

- `python -m pytest -q`: **43 passed, 1 skipped** (CUDA-event smoke test).
- Editable installation with the available dependencies and the installed
  `thicket-profile` console entry point.
- Synthetic float32 coarse smoke; synthetic bfloat16 installed-CLI smoke.
- Synthetic CPU diagnostic profiler run and Chrome trace export.
- Python bytecode compilation and shell-script syntax check.

Tests cover candidate validation/identity, native-dtype arithmetic for float32/
float16/bfloat16, versioned RNG behavior, invalid-base rejection, exact snapshot
reset, order independence, exception recovery, finite-score enforcement,
nonpersistent-buffer hashing/audits, tied parameters, JSON artifacts, refusal to
overwrite runs, and absence of silent CUDA-to-CPU fallback.

The synthetic tests detect nonzero legacy add/subtract drift and exact
snapshot-copy resets. These establish that the harness can detect the numerical
issue; they do not quantify drift or performance on an LLM.

## Not executed / not established

- CUDA timing, memory measurements, and performance claims.
- Hugging Face model loading or generation (optional dependency/models unavailable).
- The original-worker adapter on a complete pinned RandOpt checkout.
- Full vLLM/Ray replication, tensor parallelism or multi-GPU behavior.
- Any optimized kernel, novel mechanism, expert-quality improvement or paper result.
- GitHub CI: a workflow file exists but was not run remotely.

The package dependency ranges and the proposed Python 3.11 CI environment are not
a claim of validation on all those combinations. Pin the actual GPU environment
when measurements begin.

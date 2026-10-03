# Initial local validation — October 3, 2026

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

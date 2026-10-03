# Thickets — candidate runtime research baseline

**Status:** measurement infrastructure, not an optimized kernel and not a demonstrated
speedup. This is a fresh direction; no prior visual/forecasting work is reused.

Research boundary:

```
search policy -> CandidateSpec -> realize weights -> inference -> score
                                  \-> reset candidate state
```

The initial question is whether candidate state manipulation consumes enough of
real search wall time to justify a kernel intervention. We preserve candidate
semantics before claiming any reduction in execution cost.

## What is implemented

- Versioned candidate recipes bound to a hash of actual base parameters/buffers.
- Two **existing** single-device state strategies: native-dtype add/regenerate/
  subtract, and snapshot copy/apply/reset. Neither is a novel method.
- Separate apply, inference, restore and scoring wall timings, optional CUDA-event
  intervals, CUDA allocated/reserved memory, fixed candidate traces, warmup and repeats.
- Separate sequential-drift and snapshot-reference output/reward audits.
- Optional Chrome/Perfetto PyTorch traces with noise, scale, add and reset labels.
- Synthetic CPU/GPU smoke workload and optional Hugging Face causal-LM generation.
- Opt-in hash-pinned adapter to the original RandOpt worker weight operations.

**Not implemented:** a custom CUDA/Triton kernel, vLLM/Ray end-to-end instrumentation,
TP-invariant noise, multi-GPU scheduling, BO/ES integrations, or a production runtime.
The original-worker adapter profiles original weight operations against a PyTorch
model; it does **not** turn this into the original vLLM execution stack.

## CPU smoke test

From this repository root, with PyTorch installed:

```bash
pip install -e '.[test]'
pytest -q
thicket-profile --device cpu --candidates 4 --repeats 2 --out runs/cpu-smoke
```

The synthetic workload is a small MLP. Its score is a checksum-like scalar, not
accuracy. CPU timings are **not evidence** about GPU or LLM throughput.

## First CUDA smoke

Use a CUDA-enabled PyTorch environment. Do not replace its wheel with a CPU wheel.

```bash
bash scripts/run_gpu_pilot.sh runs/gpu-pilot
```

This runs CPU/CUDA tests and **synthetic** lifecycle measurements. A real LLM run
requires the optional dependencies, a model/revision and representative inputs:

```bash
pip install -e '.[hf,test]'
thicket-profile --workload hf --device cuda:0 --dtype bfloat16 \
  --model YOUR_MODEL_ID --revision FULL_40_CHARACTER_MODEL_COMMIT \
  --data examples/arithmetic_smoke.jsonl --max-new-tokens 32 \
  --candidates 8 --repeats 3 --out runs/llm-short
```

The four included arithmetic prompts only validate plumbing. They do not constitute
a research benchmark or establish expert quality. Use a representative, frozen
scoring set before drawing end-to-end conclusions. Remote models require an
immutable revision; local model directories are also accepted and weights hashed.

## Original RandOpt weight-operation baseline

No upstream source is copied into this project. On a machine with network access:

```bash
git clone https://github.com/sunrainyg/RandOpt.git /workspace/RandOpt
git -C /workspace/RandOpt checkout 4000d34fb5b69a3121cf1d2c564aa0be5a6a41ca
thicket-profile --upstream-root /workspace/RandOpt \
  --strategy legacy-add-subtract --device cuda:0 --dtype bfloat16 \
  --width 2048 --depth 8 --batch 8 --candidates 8 --repeats 3 \
  --out runs/upstream-weight-ops
```

The worker file must match Git blob `3b672d6636364845474be02e01cce764b6e18af3`
before it is imported. Combine this option with `--workload hf` for real LLM
inference with the pinned weight operations. The backend remains Hugging Face,
not vLLM. The adapter is opt-in and was not executed in the CPU-only development
session because the full upstream checkout was unavailable locally.

## Outputs

Each run creates a new directory; overwriting an existing run is refused:

```
manifest.json               configuration, versions, workload hash, status
candidates.json             full candidate recipes and IDs
legacy-add-subtract.json    raw phase timings, memory, drift and output audit
snapshot-copy.json          same for the existing copy baseline
summary.json                within-run descriptive statistics
*.trace.json                optional diagnostic profiles
```

A completed report does not imply exact equivalence. Inspect `exact_semantics_gate`.
The legacy path can fail that gate due to finite-precision add/subtract drift.
CUDA-event intervals and host wall time must not be added together. Diagnostic
`--trace` runs must not be used as headline throughput measurements.

See [the research protocol](docs/PROTOCOL.md), [RunPod runbook](docs/RUNPOD.md),
[upstream notes](docs/UPSTREAM.md), and [validation status](docs/VALIDATION.md).

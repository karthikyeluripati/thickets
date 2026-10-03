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

The [October 3 H100 real-LLM pilot](docs/LLM_PILOT_2026-10-03.md) is complete:
Qwen2.5-0.5B BF16 with HF eager generation spends 0.57–2.24% of snapshot-path
wall time in apply/restore across the tested 32/128-token, batch-1/4 cases.
Snapshot passes the measured exactness gates; add/subtract fails and can change
generated outputs and scores. These bounded arithmetic microbenchmarks do not
establish research-task quality or optimized-engine performance.

The [Work 1 upstream closure study](docs/WORK1_CLOSURE_2026-10-03.md) now validates
the original RandOpt worker and actual single-engine vLLM/Ray flow. The original
worker reproduces the HF correctness failures. Native vLLM/Ray state overhead is
11.4–14.5% at 32 fixed tokens, 2.8–4.6% at 128 tokens, and 6.8–8.8% on a small
natural-stopping workload. Inference dominates, but the HF numerical range does
not generalize. Packed native tensors change the perturbation associated with a
seed, so cross-backend candidate identities are explicitly distinguished.

The [shared-base speculative feasibility study](docs/SHARED_SPECULATIVE_FEASIBILITY_2026-10-03.md)
is also complete on `research/shared-speculative-feasibility`: 300 independently
generated, snapshot-anchored candidates across three sigmas and 40 frozen prompts.
Short word problems share substantial prefixes, but the GSM8K subset supports only
1.1–8.1% ideal round reduction at block size 16 before real costs. The frozen gate
failed; no speculative verifier, optimized kernel or measured speedup is claimed.

The [adaptive-evaluation feasibility study](docs/ADAPTIVE_EVALUATION_FEASIBILITY_2026-10-03.md)
is complete on `research/adaptive-evaluation-feasibility`: 504 snapshot-anchored
candidates, 200 frozen selection prompts and 40 disjoint ensemble-test prompts.
The locked simple race removes about 60% of selection pairs on average with about
93% top-10 recall, but meets the joint cost/quality gate on only 40% and 38% of
held-out prompt orders (90% required). More conservative existing racing removes
about 34-35% with strong retention. The frozen gate fails; novel-method development
stops. These are offline budget reductions, not measured wall-clock speedups.

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
- Actual pinned RandOpt launcher/worker profiling through single-engine vLLM/Ray,
  including natural stopping, independent process runs, and native-state audits.

**Not implemented:** a custom CUDA/Triton kernel, full-paper RandOpt replication,
TP-invariant noise, multi-GPU scheduling, BO/ES integrations, or a production runtime.
The original-worker adapter profiles original weight operations against a PyTorch
model. The separate `thicket-profile-vllm` path uses the actual vLLM/Ray stack.

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

To reproduce the bounded Qwen pilot (four candidates, three repeats per strategy,
32/128 fixed tokens, batch 1/4, plus a wall-only control):

```bash
bash scripts/run_llm_pilot.sh runs/llm-pilot
python scripts/summarize_llm_pilot.py runs/llm-pilot
```

Raw reports from the completed session are preserved in
[`results/llm-bounded-20261003`](results/llm-bounded-20261003/).

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
not vLLM. The adapter is opt-in and was subsequently validated on the H100.
For actual vLLM/Ray execution and the isolated dependency setup, use the
[Work 1 reproduction instructions](docs/WORK1_CLOSURE_2026-10-03.md#reproduction).

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

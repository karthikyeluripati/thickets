# Phase I: characterize before optimizing

## Research hypothesis and non-claims

H1: In a specified, practically relevant weight-space-search workload, candidate
realization/reset has a material end-to-end cost, and some of it may be removable
without changing candidate semantics. H0: its removal would yield negligible
end-to-end benefit in that workload, or candidate equivalence cannot be maintained.

A small seed is a compact recipe, not evidence of cheap evaluation. A dense delta
still contains one value per perturbed parameter. No asymptotic shortcut follows
from its short description. This repository currently implements no new kernel.

## Correctness contract

A candidate is identified by actual base weights, seed, scale, sign and RNG scheme.
The report additionally identifies tensor layout, dtype, backend, framework version,
workload, decoding policy and input data. Candidate IDs alone are NOT a promise of
cross-backend/cross-GPU bitwise equality. PyTorch does not guarantee RNG/results
are identical across devices, releases or platforms.

`randopt-per-tensor-v1` intentionally restarts a generator with the same seed for
each named parameter, following the pinned upstream worker. Identically shaped
parameters receive identical noise. That is not equivalent to drawing independent
noise for every coordinate of a globally flattened parameter vector.

`named-tensor-v1` is a different, explicitly versioned noise recipe. It must never
be substituted into an alleged same-candidate speed comparison. It also is NOT a
shard-invariant counter-based RNG. This pilot is single-device and dense/eval-only.
Tied Parameter objects are deduplicated; unusual overlapping-storage parameters
are rejected. Quantization and changing model structure are out of scope.

Native floating-point `(W + delta) - delta` is not generally equal to `W`.
The legacy baseline deliberately retains this drift within each candidate trace.
Every repetition starts from the exact base. Correctness audits compare each
sequential candidate output/reward against an independently snapshot-anchored
candidate, and separately measure reset drift. A failed gate is reported, not
silently repaired inside the timed candidate loop.

For HF generation, output agreement means generated token IDs agree. It does not
establish bitwise equality of hidden activations/logits. A later fused kernel must
also pass parameter/logit-level tolerance tests with preregistered tolerances.

## Current measurement scope

1. Reference PyTorch state operations, with a synthetic forward-pass workload.
2. Same operations with optional real Hugging Face greedy generation.
3. Optional original, pinned RandOpt worker operations against the loaded PyTorch
   model. This isolates actual upstream state operations but excludes Ray/vLLM.

Full end-to-end RandOpt/vLLM replication is a **remaining requirement**, not a
completed result. Inference speed changes the state-overhead fraction; HF numbers
must not be passed off as vLLM numbers.

## Timings and memory

Each candidate has synchronized wall-time spans for apply, inference, restore and
score, plus a total wall time. CUDA-event intervals describe elapsed device timeline
intervals; they can include host-launch gaps and are NOT sums of pure kernel time.
Never add host time to device-event time. Stage wall spans are serial in this pilot.

Weight/RNG generation and scaling use per-parameter temporaries; the upstream does
not keep a full-model delta tensor. Its base snapshot is model-sized and persists.
The experiment therefore must not claim to newly eliminate a full delta allocation
that the reference never made.

Startup, loading, snapshot creation, full-weight hashing and correctness audits are
excluded from steady-state candidate latency. Warmups are excluded from rows.
Per-repeat peak CUDA allocated/reserved memory is captured BEFORE the drift audit.
Audit scratch memory is chunked. CPU memory and measured HBM bytes are not provided.
Parameter/snapshot byte counts are logical counts, not hardware memory-traffic
measurements. Use Nsight for actual DRAM throughput/bytes and launch analysis.

`--trace` enables diagnostic operator attribution. The trace includes warmup and
unscoped housekeeping; inspect the `thicket/apply`, `thicket/inference`,
`thicket/restore` regions. Nested inclusive profiler times must not be summed.
The run is flagged diagnostic, not headline throughput evidence.

The wall-only `--no-cuda-events` run measures instrumentation sensitivity. Run in
separate processes. Host synchronization itself can affect timing; this harness is
a serial baseline, not a substitute for an uninstrumented original-driver run.

## Pilot and analysis order

A. CPU tests and small synthetic CUDA smoke: correctness/instrumentation only.
B. One non-quantized text LLM, frozen inputs and revisions, single GPU. Start with
   8 candidates, 2 warmups, 3 repeated traces; do not launch 1,000 candidates yet.
C. Repeat short and long output regimes and small/larger scoring batches. Both
   natural greedy lengths and fixed-token-budget microbenchmarks are useful, but
   must be labeled separately. Never compare unequal work as a kernel speedup.
D. Compare original worker operations and the reference implementation on identical
   model layouts. Profile real vLLM/Ray before claiming an original-RandOpt speedup.
E. Only then expand to another model size/GPU and design an intervention.

Strategies run in a fixed order for smoke convenience. Publication measurements
must counterbalance order via separate invocations, repeat entire runs, record GPU
clock/power/driver state and estimate uncertainty across independent repetitions.
Candidate timings within a repetition are not independent experiment replicates.

For each measured workload compute:

  f = (apply_wall + restore_wall) / total_wall
  ideal_speedup_bound = 1 / (1 - f)

This is an upper bound for eliminating ONLY those stages while keeping other work
fixed, not a predicted achieved speedup. Increasing candidate count N scales total
cost, but does not itself increase f or the percentage speedup. Larger models can
increase inference and setup costs together; model size alone predicts no benefit.

A small timing fraction does not rule out a memory-capacity benefit. Record both
memory and latency before accepting/rejecting a specific target. There is no
universal pass threshold in this exploratory baseline. Define a practical target
and confirmatory workloads before evaluating a proposed kernel.

## Candidate mechanisms, only after evidence

- Reduced kernel launches/temporary traffic in RNG-scale-apply operations.
- Exact base-anchored rebuilding vs regenerated subtract, with explicit memory cost.
- Counter-based shard-local generation with a new, validated RNG contract.
- Candidate-aware execution only where a measured cost model justifies it.

Fusing perturbation generation into every forward pass can regenerate dense noise
on every decode step and LOSE the amortization of once-per-candidate materialization.
Likewise `W*x + delta*x` may add another dense multiplication; algebraic equivalence
is not a speedup. Avoid low-rank/Rademacher substitutions in same-candidate tests.
KV/prefix caches normally depend on weights: do not reuse them across perturbed
candidates without a proof or exact candidate-aware isolation. Search early stopping
and reward-dependent pruning alter the search policy and are out of this runtime
baseline. Optimizer policy owns selection, stopping, and any distillation/update.

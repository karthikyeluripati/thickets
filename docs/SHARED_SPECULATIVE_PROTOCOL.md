# Shared-base speculative feasibility: frozen protocol

This branch starts at completed Work 1 commit
`3d9e029f6231d3d07ac904c77c945617ebe3f136`. It studies independent candidate
outputs before deciding whether to implement a verifier. Work 1 data and code
remain unchanged. Search policies and weight-operation optimizations are outside
this experiment.

The machine-readable plan is
[`experiments/shared_speculative_protocol.json`](../experiments/shared_speculative_protocol.json).
It is committed before GPU trace collection, together with the frozen prompts.
Use Qwen2.5-0.5B revision `060db6499f32faf8b98477b0a26969ef7d8b9987`, BF16,
TP=1, the pinned original RandOpt snapshot/apply/reset methods, and the same
single-engine Ray/vLLM eager stack as Work 1. Native base, layout and candidate
hashes are authoritative; HF-layout candidate IDs are not interchangeable.

## Frozen scope

- Seeds 42 through 141, paired across sigma 0.0005, 0.001 and 0.002: 300 planned
  candidates and 12,000 candidate/prompt pairs. There is no adaptive selection.
- All eight existing natural word problems, preserving their generation/scorer
  settings, and the first 32 GSM8K main/train examples in dataset order. Prompt
  selection precedes measurement and uses no agreement/reward information.
- GSM8K uses the official [OpenAI dataset](https://huggingface.co/datasets/openai/gsm8k/tree/740312add88f781978c0658806c59bc2815b9866)
  and the pinned RandOpt preprocessing, non-instruct prompt format and
  strict-then-flexible reward. Its cap is 512 new tokens versus the upstream
  default 1024; capped trajectories remain included and explicitly marked.
  Dataset provenance and original reference solutions are retained.
- A 45-minute sweep budget, excluding setup and controls. Finish each paired
  seed across all three sigmas. Before another seed, allow 1.5 times the slowest
  completed seed duration; if it would exceed the budget, stop with all completed
  seeds. A 60-minute outer timeout includes setup/controls and leaves interrupted
  runs invalid, rather than treating partial data as a negative result.

Independent base and candidate generations are saved with full token IDs,
prompt token IDs, text, termination metadata and per-prompt rewards. Candidate
state fingerprints and full bitwise reset audits are separate from descriptive
lifecycle timings. Zero-sigma equality, base output repeats before/after the
sweep, and a repeated nonzero candidate check deterministic output and anchoring.
Prefix caching is disabled; every request completes before weights change.

## Offline model and gate

For block sizes 2, 4, 8 and 16, model one ideal candidate verification round per
base block, accepting through its first incorrect token. Rejection emits the
candidate's corrected token. **After the first rejection, decode every remaining
token ordinarily.** The saved base suffix was generated at a different prefix;
matching aligned suffixes provide no evidence for further shared drafting.
No free bonus token is counted. Short terminal blocks are allowed. Both ordinary
and simulated output-round counts exclude prompt prefill, treating the first
output token as a round. This is a cost abstraction, not actual vLLM kernel counts.

The trace-supported round saving still assumes a block verification costs one
round and excludes base drafting, prefill, cache maintenance and state changes.
It is an optimistic bound for this initial-prefix-only policy, not wall-clock
speedup and not a bound on all possible speculative algorithms. A separate
all-accepted oracle ceiling is reported to avoid confusing it with observed
agreement. Full-block acceptance is reported both for the initial block and for
full blocks attempted along the shared prefix, with denominators. Short equal
responses do not count as matching unavailable longer blocks.

Proceed to a minimal exact verifier only if at least one common `(sigma, k)`,
with k=4 or 8, has on **both** workloads: at least 32 candidates per sigma, median
prefix at least 4 tokens, initial full-block match at least 50%, and token-weighted
round saving at least 20%. Controls must pass and the bounded sweep must complete.
These are predeclared engineering thresholds, not a statistical significance
test or proof of practical benefit. Failure stops phase 4; missing/invalid data
is inconclusive. The threshold applies equally to every sigma and does not select
which results to publish.

Report distributions and candidate-level aggregates, cap rates, immediate
divergence, exact observed equality and uncensored equality. Reward association
uses workload mean reward, Spearman correlation with mean prefix, all candidates
above base reward, and the top reward quartile including cutoff ties. This small
frozen set does not establish expert status or causal relationships. Prompt/seed
pairs are dependent; pooled quantiles are descriptive, without iid confidence
intervals. A failing gate is a bounded result for this model/layout/workload,
not a general impossibility theorem.

## Reproduction

In the Work 1 environment (vLLM 0.10.2, Ray 2.49.2, Transformers 4.56.2,
PyTorch 2.8.0+cu128), with the upstream checkout at its pinned commit:

```bash
python -m pytest -q
bash scripts/run_shared_feasibility.sh /workspace/RandOpt-feasibility \
  results/shared-speculative-20261003/run-01
```

The runner refuses an existing destination. Each run captures source snapshots,
commands, commits, package/GPU information, hashes, controls, raw candidate traces,
summary and gate. `scripts/freeze_gsm8k.py` (PyArrow required) reproduces the
committed input from the pinned dataset and refuses to overwrite it. Do not rerun
the freeze script in a checkout where the frozen input already exists.

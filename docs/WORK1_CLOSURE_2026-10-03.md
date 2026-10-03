# Work 1: original-worker and vLLM/Ray closure study

The original RandOpt worker reproduces the HF pilot's drift and correctness
failures. In the actual single-engine vLLM/Ray lifecycle, inference still dominates,
but the earlier **0.6–3.1% state-overhead range does not hold generally**: the
32-token case spends 11.4–14.5% in apply/restore, and the natural workload spends
6.8–8.8%. No kernel was designed or implemented, and no speedup is claimed.

The bounded upstream-native characterization is complete. A stricter claim that
the *identical HF weight perturbations* were evaluated in vLLM remains unestablished:
the original worker's tensor-local RNG changes its perturbation when vLLM packs
parameters. Native identities and HF source identities are recorded separately.
This limitation prevents calling the measurements an exact cross-backend
same-candidate comparison; it does not invalidate either within-backend audit.

## What actually ran

One H100 80GB HBM3, driver 580.126.09; Qwen2.5-0.5B BF16 at revision
`060db6499f32faf8b98477b0a26969ef7d8b9987`. The HF controls retained the original
Python 3.12.3 / PyTorch 2.8.0+cu128 / Transformers 4.56.2 environment. A separate
environment added **vLLM 0.10.2 and Ray 2.49.2**, retaining the same PyTorch and
Transformers versions. Full dependency lists are in the artifacts.

The upstream checkout was pinned to
`4000d34fb5b69a3121cf1d2c564aa0be5a6a41ca`. These files were verified before loading:

| Upstream file | Git blob |
|---|---|
| `utils/worker_extn.py` | `3b672d6636364845474be02e01cce764b6e18af3` |
| `core/engine.py` | `6603321e9c89f67e0b922313c1bc2d0581d7beec` |
| `randopt.py` | `134c627940b54516177614108d77aa59a0b5fc91` |

The vLLM profiler calls the **unchanged upstream `launch_engines`** and
`WorkerExtension`, using its Ray placement group, `RandOptNcclLLM`, Ray distributed
executor, BF16 model, and `enforce_eager=True`. It executes the actual
`collective_rpc.remote` / `generate.remote` / `ray.get` sequence from `run_sampling`:

```
worker perturb RPC -> vLLM generation -> worker restore RPC -> host scoring
```

Legacy invokes `perturb_self_weights` and `restore_self_weights`. Snapshot invokes
the original `apply_perturbation` and `reset_to_base_weights`. These are upstream
implementations, not the reference weight-operator reimplementation. Snapshot's
upstream reset does not call `empty_cache`; this difference is retained and is
one reason its timings must not be conflated with the HF reference implementation.

Only policy selection was replaced by replay of the frozen four-candidate trace:
seeds 42–45, sigma 0.001, positive sign, `randopt-per-tensor-v1`. There is no adaptive
selection, pruning, ensemble evaluation, update, or distillation. The host scorer
is the same last-signed-integer exact-match scorer as the HF pilot, not the paper's
full dataset-handler suite. This is an actual upstream runtime path, not a
reproduction of every stage or accuracy result of the RandOpt paper.

Each of three workloads ran in **two independent whole processes**, first
legacy-before-snapshot and then snapshot-before-legacy. Each process created its
own local Ray cluster. Each strategy had two warmups and three four-candidate
traces: 144 timed candidate evaluations across the six matrix processes. The
separate one-candidate startup smoke is preserved and excluded from these results.
Two original-worker HF controls contributed another 24 timed evaluations.

The launcher used one GPU, TP=1, one host thread, prefix caching disabled, and
GPU-memory utilization 0.25. Every generation call completed all requests before
the next weight change; prefix/KV state was not reused across candidates. The
KV allocation pool remained resident. The initial model, snapshots, loading,
profiling initialization, warmup, and audits are excluded from candidate latency.

## Original worker on the same HF candidates

The hash-pinned adapter ran the original worker on the original HF model layout
with the same arithmetic batch, 32/128 fixed-token budgets, and **identical full
candidate recipes and IDs**. Both complete correctness-audit objects match the
earlier reference-implementation audits exactly; see
[hf-validation.json](../results/work1-closure-20261003/hf-validation.json).

| Fixed tokens / batch | Output matches | Score matches | Exact restoration |
|---|---:|---:|---|
| 32 / 4 | 1/4 | 4/4 | Fails |
| 128 / 4 | 1/4 | 1/4 | Fails |

After each four-candidate trace, **99,117,501 / 494,032,768 parameter values** differ
from the base (20.06294%); maximum absolute drift is **0.00390625**. This reproduces
the failure in the original worker, rather than only in our reimplementation.

## Actual vLLM/Ray timings

Ranges below span the two independent process means for each strategy/workload.
They are descriptive ranges, not confidence intervals. State fraction is
`sum(apply_wall + restore_wall) / sum(total_wall)`; inference uses the same
denominator. All timings are synchronized **driver wall time including RPC and
serialization**. They are not pure CUDA kernel timings.

| Workload | Strategy | State fraction | Inference fraction | Candidate ms | Candidates/s |
|---|---|---:|---:|---:|---:|
| Fixed 32, batch 4 | Legacy | 12.71–14.47% | 85.48–87.24% | 258.47–262.64 | 3.808–3.869 |
| Fixed 32, batch 4 | Snapshot | 11.36–11.65% | 88.30–88.59% | 249.46–260.40 | 3.840–4.009 |
| Fixed 128, batch 4 | Legacy | 2.80–4.57% | 95.40–97.17% | 880.44–902.28 | 1.108–1.136 |
| Fixed 128, batch 4 | Snapshot | 3.18–3.47% | 96.50–96.79% | 853.30–962.14 | 1.039–1.172 |
| Natural, batch 8 | Legacy | 6.82–8.84% | 91.13–93.17% | 405.66–420.13 | 2.380–2.465 |
| Natural, batch 8 | Snapshot | 6.85–7.13% | 92.83–93.14% | 406.66–414.57 | 2.412–2.459 |

Across the matrix, process-mean apply costs span 11.42–21.73 ms, restore costs
8.11–22.01 ms, and scoring costs 0.035–0.250 ms. Every individual apply, inference,
restore, score, and total span is retained in the raw JSON. Per-process and
within-process repeat summaries are in
[closure-analysis.json](../results/work1-closure-20261003/closure-analysis.json)
and each strategy report.

Inference dominates every measured regime, but the HF numerical range does not
survive the upstream stack, especially for shorter generation. A substantial
portion of the measured state phase may be RPC/host overhead; this study does
not separately attribute that time to GPU weight kernels. Fixed-token output
counts were validated at exactly 32/128 for every request.

No throughput ratio is interpreted as a speedup. There are only two independent
processes per workload; process/order variation is visible, and the legacy path
fails the semantic gate. Natural stopping also produces slightly unequal work
between the two state strategies. Counterbalancing alone does not remove these
limitations.

## Natural search-like evaluation

The [frozen eight-question set](../examples/search_word_problems.jsonl) was authored
for this study before observing vLLM rewards. It uses two worked examples per
prompt, multi-step integer word problems, greedy generation, EOS or `\nQuestion:`
stopping, and a 128-token safety cap. All eight questions are scored for every
candidate; there is no reward-dependent selection or early rejection. The
[provenance record](../examples/search_word_problems.provenance.json) identifies
this as a small lifecycle workload, not a representative accuracy benchmark.

Legacy responses contain 20–75 tokens (mean 33.0); snapshot responses contain
20–81 tokens (mean 33.1875). **None of the 384 timed responses hit the cap**.
This is a natural-length evaluation, not a relabeled fixed-token workload.

State handling remains a minority cost at 6.8–8.8%, with inference at 91.1–93.2%.
It is larger than the earlier HF overhead range and should not be described as
universally negligible. The small authored set does not establish how much state
cost matters for a production search distribution.

## Native correctness and identity

Results below reproduce in both independent processes. Counts are per four
audited candidates; matching state refers to a full parameter/buffer hash before
generation. Output equality compares complete generated token sequences, including
variable lengths. It does not promise bitwise-equal hidden logits.

| Workload | Legacy state matches | Legacy output matches | Legacy score matches | Snapshot state/output/score matches |
|---|---:|---:|---:|---|
| Fixed 32 | 1/4 | 2/4 | 3/4 | All 4/4 |
| Fixed 128 | 1/4 | 1/4 | 4/4 | All 4/4 |
| Natural | 1/4 | 2/4 | 3/4 | All 4/4 |

All six legacy restoration gates fail; all six snapshot gates pass. Native legacy
traces change **99,116,495 / 494,032,768 values** (20.06274%), with maximum absolute
drift 0.00390625 and zero changed buffers. Snapshot has zero measured drift.
Audits execute after timing, compare each sequential candidate against an
independently snapshot-anchored candidate, and also check the reference reset.
The new native reset audit compares bytes, including signed-zero differences.

All six native runs have the same base ID:
`991345c343a27a7aa28528a633c876aa4a0ea7d52f3f6291d6240702058e622f`.
The HF source base ID is
`e8b4d10c775c43fba465ffa51ffd0efb5772e7769fa579c986dbb214479c2737`.
Native candidate IDs bind to the native base; `source_candidate_id` preserves the
original HF recipe identity without claiming equivalence.

The audit explicitly compares the seed-42 native packed draws with concatenated
HF-shaped tensor-local draws. They differ at 229,174 of 1,032,192 QKV noise values
and 4,354,214 of 8,716,288 gate/up noise values. The worker is unchanged in both
cases: packing itself changes the RNG contract's tensor boundaries. Full layouts
and these probes are recorded in every matrix run. Thus model revision and seed
alone cannot support an exact HF-to-vLLM candidate comparison.

## Memory, validation, and preservation

Both native strategies retain a 988,065,536-byte (0.92021 GiB) base snapshot.
Peak worker CUDA allocation is approximately **15.6875 GiB**, including vLLM's
resident KV pool; peak reserved counters are also retained per repeat. Memory is
captured before correctness audits. These are worker allocator peaks, not a
continuously sampled whole-device peak or HBM-traffic measurement. Boundary
device-memory readings and `nvidia-smi -q` snapshots provide additional context.
The HF and vLLM memory numbers have different cache-allocation policies.

The GPU suite passed **50 tests, one CPU-only guard skipped**. New tests cover
identity rebinding, exception recovery, timing/audit separation, sequential drift,
output/score audits, and signed-zero bitwise detection. The one-candidate startup
run passed before the matrix. The summarizer verifies all six complete processes,
counterbalanced orders, native/source identities, 144 timed rows, and fixed token
counts. Original-worker HF validation verifies exact prior recipe identity and
equality of the complete correctness reports.

All 133 files listed in the new `sha256.json` were verified after transfer, and
all 39 earlier HF artifacts still match their original checksums. Exact measured
runtime source files are included under `measured-source/`; local runtime files
matched them byte-for-byte before commit. Manifests retain the remote checkout's
baseline commit, working-tree status, and source-file hashes, so the uncommitted
measurement source is fully recoverable despite the remote checkout starting at
`c5cc037`. Each run records upstream/model revisions, environment, commands,
recipes, input hashes, output tokens/text, rewards, timings, and memory.

The SSH gateway disconnected during the matrix, but its driver and measurement
processes continued; no case was restarted or overwritten. All six processes
completed, and the GPU was idle after cleanup. Original and startup artifacts
remain intact. No changes were made to `main`.

## Reproduction

Use a separate Linux CUDA environment; preserve the original HF environment for
the adapter controls. The full freezes in the result directories are the tested
environments; package ranges alone are not an environment lock.

```bash
git clone https://github.com/sunrainyg/RandOpt.git /workspace/RandOpt-work1
git -C /workspace/RandOpt-work1 checkout 4000d34fb5b69a3121cf1d2c564aa0be5a6a41ca
export HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1

# In the original HF environment; use fresh output paths.
for tokens in 32 128; do
  python -m thicket_runtime.cli --workload hf --device cuda:0 --dtype bfloat16 \
    --upstream-root /workspace/RandOpt-work1 --strategy legacy-add-subtract \
    --model Qwen/Qwen2.5-0.5B --revision 060db6499f32faf8b98477b0a26969ef7d8b9987 \
    --data examples/arithmetic_smoke.jsonl --max-new-tokens "$tokens" --fixed-length \
    --candidates 4 --repeats 3 --warmup 2 --out "runs/work1-new/original-hf-fixed-$tokens"
done

# In a separate vLLM environment with the compatible CUDA PyTorch wheel.
pip install -e '.[vllm,test]'
python -m pytest -q
python scripts/run_work1_suite.py --upstream-root /workspace/RandOpt-work1 \
  --out runs/work1-new/vllm-matrix
python scripts/summarize_work1.py runs/work1-new
```

The suite gives every process a 900-second bound and refuses existing output
destinations. Every actual command is saved in the artifacts. Raw closure data
are under [results/work1-closure-20261003](../results/work1-closure-20261003/);
the original pod copies are under `/workspace/thickets-work1/runs/` and the HF
controls are additionally preserved under `/workspace/thickets-runtime-pilot/runs/`.

## Remaining limits on closure

Work 1 now answers the original-worker correctness question and measures the
actual upstream single-engine lifecycle. Its conclusion is **inference-dominated,
with workload-dependent state overhead**, not the earlier universal 0.6–3.1%
range. Exact cross-backend HF candidate equality remains blocked by the native
packed-tensor RNG contract. The scope also excludes multi-engine/TP execution,
the full upstream dataset-handler/search/ensemble pipeline, hidden-logit equality,
and a representative large scoring distribution. Those restrictions prevent a
broader paper-level or strict same-candidate cross-stack closure claim. They do
not require a new kernel to state the bounded findings accurately.

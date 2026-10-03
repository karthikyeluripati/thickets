# Shared-base speculative feasibility: bounded negative result

**Decision: stop before phase 4.** Shared prefixes are substantial on the eight
short word problems, but cover too little of the longer GSM8K generations to pass
the predeclared gate. No verifier or kernel was implemented, and no wall-clock
speedup was measured.

All 300 candidates completed: 100 paired seeds (42–141) at each sigma, 40 frozen
prompts, **12,000 candidate/prompt pairs**. The H100 sweep took 1,467.7 seconds
(24.46 minutes), below its 45-minute bound. This uses Qwen2.5-0.5B BF16, TP=1,
greedy independent generation through the original RandOpt worker and Ray/vLLM
eager executor, with snapshot anchoring. Work 1 is preserved unchanged.

1. **How long do trajectories agree?** At sigma 0.001, the median common prefix
   is **15 tokens on natural8 and 3 on GSM8K32**. The mean prefixes are 14.76 and
   11.11 tokens; the means conceal a long right tail. The table gives pooled
   descriptive distributions, not independent statistical samples.

   | Workload | Sigma | Prefix p10 / median / p90 | Mean prefix fraction | Fully equal observed outputs | Immediate divergence |
   |---|---:|---:|---:|---:|---:|
   | natural8 | 0.0005 | 0 / 20 / 35 | 69.88% | 58.50% | 11.50% |
   | natural8 | 0.001 | 0 / 15 / 31 | 53.90% | 38.62% | 17.38% |
   | natural8 | 0.002 | 0 / 3 / 20 | 26.71% | 10.62% | 22.00% |
   | GSM8K32 | 0.0005 | 1 / 17 / 64 | 9.28% | 0.72% | 7.03% |
   | GSM8K32 | 0.001 | 0 / 3 / 30 | 3.70% | 0.00% | 14.09% |
   | GSM8K32 | 0.002 | 0 / 1 / 13 | 1.27% | 0.00% | 35.03% |

2. **How does sigma affect agreement?** Increasing sigma reduces agreement on
   both workloads across the same seed set. Small sigma does yield nontrivial
   GSM8K prefixes: median 17 tokens at 0.0005. However, mean candidate lengths are
   323–337 tokens on GSM8K versus 34–37 on natural8. The useful shared fraction is
   therefore much smaller on the benchmark; this is not a universal “0–1 tokens”
   result.

3. **What fraction of draft blocks are accepted?** These are *initial full-block*
   match rates over all candidate/prompt pairs; shorter outputs do not count as
   accepting unavailable full blocks.

   | Workload | Sigma | k=2 | k=4 | k=8 | k=16 |
   |---|---:|---:|---:|---:|---:|
   | natural8 | 0.0005 | 87.88% | 83.25% | 78.75% | 61.88% |
   | natural8 | 0.001 | 80.50% | 71.38% | 64.50% | 41.38% |
   | natural8 | 0.002 | 70.12% | 47.38% | 32.00% | 12.50% |
   | GSM8K32 | 0.0005 | 75.72% | 68.97% | 65.62% | 53.81% |
   | GSM8K32 | 0.001 | 59.09% | 49.69% | 41.28% | 28.41% |
   | GSM8K32 | 0.002 | 34.19% | 24.72% | 18.47% | 7.16% |

   Conditional acceptance among full blocks actually attempted along the shared
   prefix is a different denominator. At sigma 0.001 it is 92.09/84.40/69.22/43.08%
   on natural8 and 84.07/71.27/52.55/30.06% on GSM8K for k=2/4/8/16. High conditional
   acceptance can coexist with little whole-response saving. All k=1/2/4/8/16/32
   initial rates and eligible denominators are in the raw summary.

4. **What is the conservative theoretical reduction?** The trace-supported
   simulation shares only the initial base trajectory, emits the candidate's
   corrected token at rejection, then uses ordinary decoding for the remainder.
   It never treats aligned suffix matches as valid drafts at a corrected prefix.
   Token-weighted ideal output-round reductions are:

   | Workload | Sigma | k=2 | k=4 | k=8 | k=16 |
   |---|---:|---:|---:|---:|---:|
   | natural8 | 0.0005 | 29.36% | 44.26% | 51.37% | 55.10% |
   | natural8 | 0.001 | 21.58% | 32.49% | 37.71% | 40.41% |
   | natural8 | 0.002 | 10.09% | 15.11% | 17.39% | 18.43% |
   | GSM8K32 | 0.0005 | 4.33% | 6.48% | 7.54% | 8.08% |
   | GSM8K32 | 0.001 | 1.81% | 2.67% | 3.10% | 3.31% |
   | GSM8K32 | 0.002 | 0.63% | 0.91% | 1.03% | 1.10% |

   This is conservative about *reuse*, but optimistic about *cost*: a block is
   assigned one ideal output round, with no bonus token and no drafting,
   verification-width, prefill, state-switching or cache-maintenance cost. Counts
   treat each ordinary output token as a decision round, including the first
   output; they are not physical vLLM decode-kernel counts. They do not model batch
   scheduling or establish latency savings. A separate all-accepted oracle bound
   is retained in JSON. At sigma 0.001/k=16, mean positions verified after rejection
   are 5.29 on natural8 and 10.77 on GSM8K per pair. Divergence positions are
   zero-based; equal observed traces have no observed divergence position.

5. **Does agreement persist for higher-reward candidates?** The relationship
   depends on workload. On GSM8K, reward/mean-prefix Spearman correlations are
   +0.239/+0.286/+0.175 across increasing sigma. The top reward quartile, including
   ties, has median prefixes 19/5/1 and k=16 simulated reductions only
   **8.66%/3.60%/1.23%**. At sigma 0.001, all 34 candidates above base reward are
   this same higher-reward group; their advantage does not rescue the gate.
   Natural8 correlations are −0.508/−0.111/+0.226, so better rewards do not imply
   greater similarity to base. Above-base counts are 89/69/13 on natural8 and
   61/34/0 on GSM8K. These are descriptive rankings on frozen prompts, not evidence
   of held-out expert quality or causality.

6. **Does a proof of concept preserve exact outputs?** No proof of concept was
   built because the gate failed. The independent reference experiment passed
   every exact restoration audit, zero-sigma state/output equality, base output
   repeats, and repeated nonzero candidate state/output equality. The four
   overlapping Work 1 candidates also reproduce their native state hashes and
   natural-workload token outputs exactly.

7. **Does measured end-to-end speedup survive full costs?** **Unmeasured.** No
   speculative executor ran, so there is no verifier latency, population
   amortization curve or actual speedup to report. Recorded ordinary lifecycle
   timings are descriptive only. Peak allocated/reserved GPU memory was
   16,844,292,608 / 16,959,668,224 bytes. No percent-level performance comparison is
   claimed from this one process.

8. **Is this enough to justify a systems/kernel paper?** **No; stop this proposed
   implementation at the feasibility gate.** The preregistered screen required
   one common sigma and k=4 or 8 to reach median prefix ≥4, initial acceptance ≥50%
   and token-weighted round saving ≥20% on *both* workloads. No configuration
   passed. GSM8K stays below 8.1% even at k=16 before any real costs; its top reward
   quartiles stay below 8.7%. The natural8 result alone is insufficient to justify
   a population runtime. This closes the bounded initial-shared-trajectory study,
   not all possible speculative methods or other model/proposal distributions.

The main limitations are one base model and native packed-tensor RNG contract,
one independent GPU process, eight authored prompts and the first 32 GSM8K
training examples, and bounded generation. GSM8K uses the upstream prompt/reward
rules but a 512-token cap rather than its default 1024; candidate cap rates are
9.34/12.44/27.53%, versus 6.25% for base. Natural8 cap rates are 0/0.38/5.12%.
Cap-limited equality is distinguished from complete natural termination in the
artifacts. Prompt/seed dependence precludes treating 12,000 pairs as iid samples.
These limitations restrict generalization; they do not invalidate the failed
predeclared gate for the measured scope.

The [frozen protocol](SHARED_SPECULATIVE_PROTOCOL.md) was committed before outputs
were collected (`aeb1877`, identity safeguards `0c392f8`). Measured source is commit
`0c392f8dcb159af6a5cf765b7f46f9d6895d28bb`, upstream is
`4000d34fb5b69a3121cf1d2c564aa0be5a6a41ca`, and the model revision is
`060db6499f32faf8b98477b0a26969ef7d8b9987`. The native base is
`991345c343a27a7aa28528a633c876aa4a0ea7d52f3f6291d6240702058e622f`.

Evidence: [raw run and environment](../results/shared-speculative-20261003/run-01/),
[full distributions](../results/shared-speculative-20261003/run-01/summary.json),
[independent reanalysis](../results/shared-speculative-20261003/reanalysis/audit.json),
[exact setup commands](../results/shared-speculative-20261003/execution-notes.json),
and [distribution figure](../results/shared-speculative-20261003/figures/agreement-distributions.png).
Reanalysis verified all 333 run checksums, every candidate identity and reward,
all token-derived metrics, summaries and the gate. CPU and H100 test suites each
passed 57 tests with one environment-appropriate skip. All 172 checksummed Work 1
artifacts remain unchanged.

To audit and render again, use fresh output directories:

```bash
PYTHONPATH=src python scripts/analyze_shared_feasibility.py \
  results/shared-speculative-20261003/run-01 \
  --upstream-root /path/to/pinned/RandOpt --out runs/agreement-audit-new
python scripts/plot_shared_feasibility.py \
  results/shared-speculative-20261003/run-01 --out runs/agreement-figures-new
```

The audit needs exact upstream source bytes (LF, including on Windows).
The figure script uses Matplotlib; neither reanalysis command requires a GPU.

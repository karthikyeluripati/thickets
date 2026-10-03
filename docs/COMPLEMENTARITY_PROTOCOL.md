# Complementarity selection: frozen two-test protocol

Branch: `research/complementarity-selection-feasibility`, from adaptive-study
commit `07d2d22b2157cc728c0425af9e1cd90244cea1a9`. Prior evidence and `main` remain
unchanged. The authoritative configuration is
[`experiments/complementarity_protocol.json`](../experiments/complementarity_protocol.json).

Test 1 reuses only the first 200 selection columns of the existing three
168-candidate populations. It does not regenerate candidates, use the previous
40 test columns, or change search. Standard selection ranks individual reward
with the existing candidate-ID hash tie break. Greedy selection starts with the
best individual, then appends the unused candidate maximizing exact committee
voting accuracy. Accuracy ties prefer higher individual reward, then a fixed
hash. Continue to each requested K even when the best marginal change is zero
or negative. Committee order is preserved because upstream vote ties are stable
insertion-order ties. Empty answers do not vote; extraction and formatted-answer
scoring use the pinned upstream code.

The rule is committed a priori rather than selected using test performance.
Development is processed first. Tie seed 0 is primary; seeds 1 and 2 are
diagnostics and cannot replace it after testing. Offline K is 3/5/10/20/50.
The optional disagreement/lambda heuristic is omitted: this study tests one
proposed selector with exact-quality random controls, not another tuning search.

Random controls use seeds 101/102/103. Each position samples without replacement
from candidates with exactly the same individual correct-count as the primary
greedy member at that position. Every prefix thus matches the entire individual
score multiset, not merely its range. These are distinct random experts without
an error-diversity objective; actual disagreement, correlation and the fraction
of constrained/singleton choices are reported.

Test 2 uses 300 fresh official GSM8K main/test prompts, indices 40-339, at dataset
revision `740312add88f781978c0658806c59bc2815b9866`. All earlier example inputs
were checked for exact formatted-prompt overlap; none was found. Freeze their
JSONL hash, indices and provenance before any new GPU output. The old 200-prompt
matrix determines all committees and the exact selected union. Commit the
committee lock and union before execution. Collect only that union, including
the required random and tie controls; do not collect all 504 candidates again.
Standard GPU K is 3/5/10/20/50 (K=3 requires no additional expert beyond top-50);
greedy and random GPU K is 3/5/10/20. Greedy K=50 is selection-only.

Use the same pinned Qwen2.5-0.5B revision, original RandOpt worker, BF16 TP=1,
greedy eager Ray/vLLM and snapshot anchoring. Retain the previous 512-token cap,
report cap rates, and do not equate it with upstream's 1024-token default. Disable
prefix caching, drain all requests before changing weights, and compare every
candidate fingerprint to its existing matrix fingerprint **before generation**.
Audit exact restoration outside timing. Require zero-sigma equality, repeated
base equality, and repeated candidate outputs for one frozen expert per
population. An outer 105-minute bound protects the run; incomplete/invalid data
is an operational failure, not a scientific NO.

The frozen primary gate uses tie seed 0 only. A **single identical comparison**
must pass on both validation populations:

- Efficiency: greedy K=5 or 10 is no more than one percentage point below
  standard K=20 or 50.
- Accuracy: at the same K in 3/5/10/20, greedy gains at least two absolute
  percentage points over standard selection.

Either condition yields GO under the requested point-estimate gate; otherwise
NO-GO. Development never qualifies the gate. Diagnostic seeds, controls and
alternative K choices cannot be substituted opportunistically. All listed K
comparisons are declared here in advance. No threshold or selector changes
follow held-out inspection. Selection gains that do not yield a qualifying
held-out quality/efficiency result are overfitting and a NO. Also show whether
standard small-K already matches large-K: satisfying the literal efficiency
gate alone must not be presented as incremental selection benefit in that case.

Report selection gain -> held-out gain, paired improved/worsened questions,
per-question vote margins, expert overlap, individual quality, pairwise error
correlation and answer disagreement. Report descriptive paired question-bootstrap
95% intervals using seed 20261003 and 10,000 repetitions; they do not change the
gate. Shared prompts and paired candidates are dependent, and passing a finite
300-question margin is not a statistical proof of population equivalence.

Deployment cost is K experts, K full-test generation RPCs and 300K generated
request sequences, with exact token totals from saved traces. Report union
collection/control costs separately. Expert-pass reductions are exact logical
counts, not measured wall-clock speedups. Stop after the GO/NO-GO report; no
subsequent algorithm, lambda search or kernel work is authorized by this study.

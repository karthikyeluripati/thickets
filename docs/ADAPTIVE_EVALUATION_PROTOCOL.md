# Adaptive evaluation feasibility: frozen protocol

This study starts from `92ef27d7a03abb66e928cdb7807b5bf87f6ec213` on
`research/adaptive-evaluation-feasibility`. It does not continue kernel or
speculative-execution work. Previous code and evidence remain unchanged.

The authoritative plan is
[`experiments/adaptive_evaluation_protocol.json`](../experiments/adaptive_evaluation_protocol.json).
It is committed before matrix collection. Gate thresholds, seed splits and
analysis rules are frozen before any validation rankings are inspected.

## Complete reference matrix

Use the pinned Work 1 Qwen2.5-0.5B revision, original RandOpt worker, BF16 TP=1,
one H100, Ray/vLLM eager, temperature zero, disabled prefix caching, and exact
snapshot anchoring. Each candidate is fully evaluated on **all** prompts.
Original tensor-local RNG and perturbation semantics are retained; only the
external experiment supplies a fixed seed/sigma population.

Target 504 candidates: 56 seeds in each of three populations, each paired across
sigma 0.0005, 0.001 and 0.002. Development starts at seed 10000; validation A at
20000; validation B at 30000. No seed crosses a population boundary. At each seed
offset collect all nine population/sigma combinations before another offset.
Stop before a new block if elapsed time plus 1.5 times the slowest completed block
would exceed 90 minutes, excluding model setup and controls. At least 34 completed
offsets (306 candidates, 102 per population) are required for a conclusive study.
No decision depends on observed rewards. An outer 105-minute timeout protects
against hangs; incomplete/invalid runs are not negative scientific findings.

Selection uses GSM8K main/train indices 32–231 (200 prompts); these exclude the
previous study's first 32 training prompts. Ensemble evaluation uses main/test
indices 0–39 (40 disjoint prompts). Both use the same pinned dataset revision and
upstream preprocessing, non-instruct prompt formatting, reward extraction and
majority-vote answer rules. Retain the 512-token cap used previously and report
censoring, rather than silently equating it with upstream's 1024-token default.
All input rows and source checksums are frozen without outcome filtering.

Save native base/candidate identities, per-candidate state fingerprints, every
reward, generated text/token IDs, termination metadata, token counts, workload
latency and memory. Compact JSON is gzip-compressed losslessly per candidate;
input tokens are stored once, with equality checked on each candidate. Complete
population reward matrices are also retained as ordinary JSON. Audit restoration
outside lifecycle timing; require zero-sigma equality, repeated base outputs,
repeated nonzero candidate state/output equality and exact final base identity.

## Offline policies and ranking

Reference ranking is descending full selection-set mean, with a fixed SHA256
candidate-identity tie break independent of rewards. Partial rankings use the same
tie rule. **Strict top-10 identity recall is the primary endpoint.** Also report
tie-aware recall and boundary ties so equal-score alternatives are not mistaken
for quality loss, without changing the gate to rescue a result.

Evaluate fixed dataset prefixes and nested random subsets of 4/8/16/32/64/80/100/
200 prompts. Development uses 20 frozen random-order seeds. Validation uses 50
different order seeds for each independent candidate population. The fixed prefix
is descriptive only and cannot qualify the gate.

Standard Successive Halving doubles the per-survivor prompt budget while halving
the survivor count, retaining at least ten; finalists continue to the full set.
Test starting budgets 4/8/16. A union-bounded Hoeffding race is the conservative
baseline: eliminate only when a candidate's upper bound is strictly below the
active top-10 lower-bound frontier. Observations are without replacement, and
bounds become exact at the full finite set. Discarded candidates are never
resurrected or ranked using their hidden full scores.

Only if development random subsets of at most 64 prompts have mean Spearman ≥0.5
and mean top-10 miss rate outside partial top-50 ≤0.1, explore one simple Wilson
bound race at the predeclared z values. These are pointwise heuristic intervals,
**not anytime confidence guarantees**. No new selection algorithm or learned
prompt allocation is introduced. Final ranking lists survivors first, then
discarded groups in reverse elimination order, ordered by last observed means.
Report that discarded candidates have unequal sample counts; rank correlations
use last observed scores, not hidden full scores.

For every policy retain observed evaluation masks, selected candidate identities,
evaluation counts, exact trace-derived token counts, top-1 recovery, top-5/10/20
recall and Jaccard, Spearman/Kendall correlation, selected best full reward, top-K
mean full reward, regret and false eliminations. Full scores are available only
to the offline evaluator, never to the policy. Token/pair savings are theoretical
selection-stage reductions, not measured wall-clock speedups; early execution
would change batching and scheduling.

## Development lock and held-out gate

A single order succeeds only when **all** hold: pair fraction ≤40%, strict
top-10 recall ≥90%, and selected-best full-score regret ≤0.01 (one percentage
point, two questions out of 200). Development qualification requires success on
at least 90% of its random orders. Among qualifying policies select the lowest
mean pair cost, then highest recall, lowest regret and lexicographic ID. If none
qualifies, lock one diagnostic policy using the predeclared development-only
ranking among policies costing at most 40% on average. Such a fallback does not
satisfy the development qualification gate.

The development command reads only development candidate rows. Write and **commit
the policy lock before running validation**. Do not tune thresholds, budgets or
the gate after inspecting validation. The selected policy must qualify on
development and achieve joint success on at least 90% of orders in **each**
validation population. Other validation baselines remain descriptive and cannot
replace the locked policy. A failed gate stops novel-method development.

## Diagnostics and optional ensemble endpoint

Report partial/full score correlation, top-K inclusion, misses outside top-2K and
top-5K, score error/uncertainty, ranking reversals and top-K churn as budgets grow.
Plot fixed-rule examples of top, middle, poor, false-positive early leaders and
false-negative late winners; use a predeclared development prompt order.

Prompt statistics include difficulty/base correctness, between-candidate variance,
correlation with the full score and with the score excluding that prompt, and
top-10 enrichment. Analyze pooled and per-sigma populations so simple sigma
separation is not confused with fine ranking. Define specialist signals by unique
or rare correctness (≤10% of the population), original-order late prompts and
partial misses of full top candidates. These do not establish semantic skill
categories or causal specialization.

After policy locking, compare full-top-K and partial-selected majority-vote
ensembles on the 40 held-out test prompts using upstream answer extraction and
nonempty-answer voting. Vote ties follow first occurrence in candidate rank order,
matching upstream Counter behavior. Test rewards never select the policy. This
small held-out ensemble result is separate from finite-selection-set top-K
identity and best-candidate regret.

Shared prompts, paired sigmas and repeated orders create dependence. Order
repetitions estimate conditional subset sensitivity, not independent candidate or
population samples. Full-set rankings are ground truth for this finite benchmark,
not perfect population-accuracy rankings. Report these limits even if the gate
passes. Raw data, scripts, source commits, commands, environment, splits and
checksums must allow independent offline reproduction.

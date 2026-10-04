# Final v1 weight-space expert experiment

This protocol asks whether a nearby random weight perturbation of the pinned
Qwen2.5-VL-3B-Instruct transfers a material selection gain to held-out visual
line tracing. It supersedes earlier scientific stopping rules for this new run;
all earlier protocols and artifacts remain unchanged. There is no capability,
calibration, population-mean, or diagnostic-score gate before the final search.

The original 650 v1 images, labels, prompt and parser are used byte-for-byte from
`49bb56e160691e83c786fc7bac40be290c0e8f75`. All three difficulties count. Selection
contains 150 images and held-out contains 500, with disjoint IDs, image hashes and
generation seeds. The old 35/150 and 133/500 base results are references only.

The [machine-readable protocol](../experiments/visual_line_tracing_final_protocol.json)
and [sigma lock](../results/visual-line-tracing-final-20261003/locks/sigma.json)
freeze sigma **0.0005** by explicit user instruction, based on the already
recorded v1 selection right tail. The existing best calibration candidate,
seed **3100000**, scored 50/150. Its exact recipe and state fingerprint are
frozen for one diagnostic 500-image held-out evaluation. That result cannot
affect the final search, ranking or GO decision.

The decisive search uses exactly **300 new candidates**, seeds **6100000–6100299**,
each evaluated on all selection150. The frozen protocol records previous seeds
and excludes even the older unexecuted reserved search range. Every candidate
starts from the same bitwise base snapshot. The original hash-pinned RandOpt
worker at `4000d34fb5b69a3121cf1d2c564aa0be5a6a41ca` applies the unchanged native
per-tensor RNG with `PERTURB_VISUAL=1`; all parameters are perturbed. No method,
model, geometry, prompt, parser or inference setting changes.

Model/processor revision is `66285546d2b821cf421d4f5eb2576359d3770cd3`, BF16,
TP1, greedy, 16 output tokens, 448² image pixels, eager vLLM 0.10.2/Ray 2.49.2.
All eight relevant package versions must match the prior run exactly. The
unchanged executor clears encoder state and verifies fresh image encoding;
prefix and multimodal preprocessor caches remain disabled.

Ranking accepts only audited selection150 records: correct count descending,
then SHA256(`visual-rank-v1:<candidate_id>`) ascending. The complete ranking,
rank-1, top-5 and top-10 recipes, selection scores and fingerprints are committed
before any final-search candidate held-out generation. The ranking entrypoint
has no diagnostic/held-out candidate-results input. Only those frozen ten
candidates are then reconstructed for all held-out500 images. Their fingerprints
must match search before generation.

The primary endpoint is the frozen selection rank-1. Primary GO requires all:
gain ≥5 pp (at least 158/500 versus base 133/500), paired bootstrap 95% lower bound
strictly above zero, strictly positive gains in at least two difficulty levels,
no true-answer class contributing ≥80% of the sum of positive per-class net
correct-count gains, and all runtime controls. The 80% operational definition
was explicitly approved by the user before execution. Confusion matrices and
prediction marginals remain visible; this concentration rule is a limited bias
diagnostic, not proof of the internal reasoning mechanism.

If primary GO fails, replication GO requires at least three top-10 candidates
with gain ≥3 pp, mean top-10 individual gain ≥3 pp, and at least one gain ≥5 pp.
These are final held-out decision criteria, never early population gates.
Otherwise record `NO_GO_VISUAL_THICKET_LINE_TRACING`, explicitly flagging any
≥5 pp rank-1 that fails the other primary checks, as approved by the user.
Operational/control failures are separate and never scientific NO-GO.

Top-5/top-10 majority voting uses the original invalid-answer and deterministic
rank-order tie rules. A ≥7 pp ensemble improvement is an additional result,
not another GO route. Top-10 counts above base and at ≥1/3/5 pp are reported;
they describe a selection-enriched group, not random-neighborhood density.
The experts share selection and held-out questions, so these are not independent
statistical replications or ten untouched primary tests.

Paired statistics use 10,000 common question-index bootstrap resamples with
PCG64 seed 20261003, int64 indices, and the 2.5/97.5 percentile interval (linear
quantiles). Base and candidate are resampled together; see the
[paired bootstrap definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).
Report paired wins/losses, both correct/wrong, and the exact two-sided McNemar
binomial p-value on discordant pairs, using the
[two-sided binomial test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html).
Secondary intervals/p-values are descriptive and unadjusted. P-values never
replace the +5 pp effect-size requirement. Previously inspected base and
diagnostic held-out outcomes are disclosed; neither selects final candidates.

Compute budget: 45,000 search image requests +5,000 selected held-out requests
+500 optional diagnostic requests. Necessary controls add 1,300 requests:
zero selection150 and heldout500, first-search repeat150 and rank-1 heldout
repeat500. The zero held-out call verifies the previous committed base output
identity and exact native base state; there is no new baseline or sigma sweep.
All 311 distinct nonzero recipes and control calls are preserved. Stop after
the final held-out answer, with no rescue experiments.

Run on the existing pinned environment with a fresh result root:

```bash
bash scripts/run_visual_final_study.sh \
  results/visual-line-tracing-final-20261003 /workspace/RandOpt-visual
```

Every phase refuses an existing output directory. Raw generations, scores,
fingerprints, restoration/cache evidence, GPU peaks, versions, exact commands,
source snapshots and checksums are recorded. Historical artifacts, v2 images
and `main` are preserved.

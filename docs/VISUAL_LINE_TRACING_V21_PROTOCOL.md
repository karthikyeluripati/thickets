# Visual line tracing v2.1: frozen primary endpoint

Continue on `research/visual-thickets-line-tracing` from `6d30f60`. Every v2
dataset byte stays identical to that commit. v1/v2 files and previous results
remain unchanged. The authoritative [configuration](../experiments/visual_line_tracing_v21_protocol.json)
is committed before any new VLM output.

Medium+Hard are the only scientific endpoint: selection100 (50+50) and heldout333
(167+166). Exact membership and population hashes are frozen in the configuration.
Easy is diagnostic only. Base inference covers all 650 images. Base selection
uses separate primary100 and easy50 calls, so primary base outputs have the same
batch membership/order as calibration and search. Held-out generation always
uses all 500 images in their existing order, with primary333 scored separately.
Calibration and search never generate easy or held-out candidate outputs.

Model/tokenizer/processor remain
`Qwen/Qwen2.5-VL-3B-Instruct@66285546d2b821cf421d4f5eb2576359d3770cd3`.
The unchanged executor uses BF16, TP=1, greedy decoding, seed0, 16 output tokens,
448×448 preprocessing, eager vLLM 0.10.2 / Ray 2.49.2 / Transformers 4.56.2 /
PyTorch 2.8.0+cu128 and Pillow 12.3.0. Required package versions are checked before
inference. Prompts, parser, voting, snapshot anchoring, upstream worker commit
`4000d34fb5b69a3121cf1d2c564aa0be5a6a41ca`, `PERTURB_VISUAL=1`, and the original
per-tensor RNG contract are unchanged. Prefix and multimodal preprocessing caches
are disabled; every image must be freshly encoded under the current weights.

## Capability gate

Proceed only if base Medium+Hard held-out accuracy is inclusively 30–70%, and at
least one of Medium or Hard individually reaches 30%. With 333 primary examples,
the combined condition requires 100–233 correct. Otherwise stop before RandOpt,
report the measured baseline and `STOP_benchmark_capability_gate`. This is a
NO-GO to proceeding on this benchmark, not a test of the weight-space hypothesis.
No automatic v3, relabeling, model replacement or prompt change follows.

## Calibration and search

Calibration evaluates exactly 20 seeds at each of 0.0005, 0.001 and 0.002 on
primary selection100. Preserve the previously declared recipe ranges:
3100000–3100019, 3100020–3100039, 3100040–3100059, interleaved by offset and sigma.
Repeat the first candidate per sigma for state/output controls.

Report mean/best primary accuracy, fraction beating primary base, fraction at
least 3 points above base and catastrophic fraction. Catastrophic retains the
original definition: accuracy <= max(15%, primary base−20pp), or invalid answers
>=50%. Select highest best primary correct count, then highest fraction strictly
beating primary base, then smaller sigma.

**Scientific calibration stop:** if every sigma's best is <= base+3 points AND
its mean is <= base+1 point, report `NO_GO_no_search_signal` and stop. The user
explicitly selected the 1-point tolerance before inference. If this joint stop
condition does not hold, freeze one sigma and continue; this also resolves the
case where a mean exceeds that tolerance without a best exceeding +3 points.

Search uses exactly 300 new seeds, 4100000–4100299, disjoint from calibration.
Every candidate sees only primary selection100. Rank primary correct count
descending, then SHA256(`visual-rank-v1:<candidate_id>`) ascending. Commit best,
top5, top10, state fingerprints and all 300 ranked records before any nonzero
candidate held-out generation. No calibration candidate enters the ranking.

## Held-out gate

Generate base and only the frozen top10 union on all 500 held-out images. Report
primary333, Medium, Hard, Easy diagnostic and overall accuracy, including top5/
top10 individual means and majority votes. Use unchanged invalid-answer and
voting tie rules. Never reselect on held-out results.

GO requires all correctness/disjointness controls, plus either:

- Frozen best gains >=5 points on primary333 and strictly improves Medium or Hard.
- Frozen top5 or top10 vote gains >=7 points on primary333, and either both Medium
  and Hard strictly improve, or one gains >=5 points while the other regresses
  by at most 1 point. The user explicitly chose >=5 points for “substantial.”

Easy and overall accuracy never affect a decision. If neither condition passes
after a valid 300-candidate experiment, report `NO_GO_close_model_task`. Stop
after this result; no extra candidates, sigmas, geometry changes or optimizer.
Optional paired primary-question bootstrap intervals (10,000 draws, seed20261003)
are descriptive and do not change the frozen point-estimate gate.

## Controls and evidence

Require zero == base outputs on every baseline evaluation set; exact snapshot
restoration; identical repeated candidate states and outputs; base equality
within/across phases; selected fingerprints checked before held-out inference;
no easy/held-out membership in selection records; and dataset equality to all
654 files at `6d30f60`. Repeat one search candidate and one selected held-out
candidate. Invalid controls are operational failure, never scientific NO-GO.

Fresh directories preserve raw generations, recipes, state/population hashes,
commands, versions, GPU details, source copies and committed phase locks. Final
report: `docs/VISUAL_THICKETS_LINE_TRACING_V2_RESULT_2026-10-03.md`.

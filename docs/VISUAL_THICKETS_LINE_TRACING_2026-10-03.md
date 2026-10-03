# Visual Neural Thickets line tracing — 2026-10-03

**Decision: STOP — operational scale failure.** All three calibration scales met
the criterion frozen before inference. No sigma was selected, so the 300-candidate
search and candidate held-out evaluation were not run. This is neither a GO nor
a scientific NO-GO; the held-out expert hypothesis remains untested. The study
ends at its prescribed calibration stop, without additional scales or rescue tuning.

The hypothesis was: **nearby vanilla RandOpt weight perturbations of pretrained
Qwen2.5-VL-3B-Instruct materially improve unseen visual line tracing without
gradient training.** Branch: `research/visual-thickets-line-tracing`, created from
completed research commit `a7e435a5ddd3b843c8b30b2ea83b81d475834895`.

## Scientific answers

1. **Does the base struggle?** Yes, on this frozen benchmark: 23.33% selection
   accuracy and 26.60% held-out accuracy, near the four-choice chance rate of 25%.
   Every answer had valid format; errors were wrong endpoints.
2. **Is there a meaningful right tail of visual experts?** Not established.
   Calibration means were 24.93–25.60%. The best calibration candidate scored
   33.33% on selection, 10 points above the below-chance selection baseline, but
   it has no held-out result. This selection maximum does not establish transfer.
3. **Does the best selection expert transfer?** Unmeasured: calibration stopped
   the study before the final search or candidate held-out evaluation.
4. **Do ensembles improve further?** Unmeasured; no committees were selected.
5. **Are improvements visible across difficulty levels?** No held-out candidate
   evidence exists to answer this. Base accuracy is near chance at every level.
6. **Does the frozen GO gate pass?** It was not reached. The earlier operational
   scale gate stopped execution; a scientific rejection would overstate the evidence.

## Frozen benchmark and baseline

The [dataset](../examples/visual-line-tracing-v1/) contains 150 selection and 500
held-out 448×448 RGB images. Four black paths connect left starts to right endpoint
digits 1–4; a red dot identifies the target start. White underpass gaps distinguish
crossings from junctions. Exact topology supplies the answer. Easy/medium/hard use
4/7/10 adjacent path swaps, different spacing and curvature, and minimum target-path
crossing counts. Labels require only one digit.

Selection seeds are 1100000–1100149; held-out seeds are 2200000–2200499. All 650
image hashes are unique, and both seeds and hashes are disjoint across splits.
Start/end joint cells are approximately balanced within each difficulty.
The opposite-endpoint heuristic scores 25.33% on selection and 24.60% on held-out.
PNG bytes, prompts, answers, graph metadata, generator and Pillow 12.3.0 provenance
were frozen at `49bb56e160691e83c786fc7bac40be290c0e8f75`, before baseline inference.
Topology tests independently replay every image's answer.

| Difficulty | Selection correct / total | Selection accuracy | Held-out correct / total | Held-out accuracy |
|---|---:|---:|---:|---:|
| Easy | 13 / 50 | 26.00% | 45 / 167 | 26.95% |
| Medium | 10 / 50 | 20.00% | 44 / 167 | 26.35% |
| Hard | 12 / 50 | 24.00% | 44 / 166 | 26.51% |
| Overall | 35 / 150 | 23.33% | 133 / 500 | 26.60% |

There were 115 selection and 367 held-out wrong-endpoint errors, zero invalid
answers and zero token-capped outputs. Both splits met the frozen 15–90%
acceptance bounds. Accuracy fell below the desired rough 30–75% regime; this is
a material limitation. The baseline was accepted without changing the benchmark,
and its [lock](../results/visual-line-tracing-20261003/locks/baseline.json) was
committed at `9646eb8bded36b32738b68d0df85c417469ddbb5` before calibration.
The requested base-only held-out inspection did not enter sigma selection.

## Fixed calibration and stop

Twenty distinct seeds per sigma were evaluated on all 150 selection examples:
3100000–3100019 at 0.0005, 3100020–3100039 at 0.001, and 3100040–3100059 at 0.002.
Execution interleaved the scales. Three additional evaluations repeated the
first candidate at each scale for the frozen reproducibility controls.

| Sigma | Candidates | Mean accuracy | Best accuracy | Fraction beating base | Catastrophic fraction | Scale failed |
|---|---:|---:|---:|---:|---:|---|
| 0.0005 | 20 | 25.37% | 33.33% (50/150) | 70% | 0% | Yes |
| 0.001 | 20 | 24.93% | 30.67% (46/150) | 60% | 0% | Yes |
| 0.002 | 20 | 25.60% | 30.67% (46/150) | 90% | 0% | Yes |

The [predeclared rule](VISUAL_LINE_TRACING_PROTOCOL.md) classifies a scale as failed
when mean accuracy is at most 27.5% and best accuracy is at most 40%, or at least
80% of candidates are catastrophic. Catastrophic means accuracy at most
max(15%, base−20pp), or at least 50% invalid answers. Every scale met the first
condition. All 9,000 primary calibration answers were valid and uncapped.

The [calibration decision](../results/visual-line-tracing-20261003/locks/sigma.json)
was committed at `b0b2ee778558ef9eae85303ec091289247f9faf2`: `scale_valid=false`,
`sigma=null`. No final-search candidates, selected committees or nonzero held-out
candidate outputs exist. The requested 300-candidate histogram, expert densities
at +1/+3/+5 points, and held-out best/top5/top10 comparisons are **not estimated**.
The 60 calibration candidates are not substituted for that gated-off experiment.

Had calibration passed, the unchanged protocol required exactly 300 new seeds,
selection-only ranking, committed best/top5/top10 identities, then only their
top10 union on heldout500. The scientific GO thresholds remain +5 points for the
frozen best expert or +7 points for top5/top10 voting, with the same method
improving at least two difficulty levels including medium or hard, plus all
controls. None of these held-out comparisons was performed.

## Model, execution and controls

Model, tokenizer and processor were pinned to
`Qwen/Qwen2.5-VL-3B-Instruct@66285546d2b821cf421d4f5eb2576359d3770cd3`.
Execution used one H100 80GB HBM3, BF16, TP=1, greedy eager inference, sampling
seed 0 and a 16-token cap. The stack was Python 3.12.3, PyTorch 2.8.0+cu128,
vLLM 0.10.2, Ray 2.49.2, Transformers 4.56.2, NumPy 2.1.2, tokenizers 0.22.2,
huggingface-hub 0.36.2 and Pillow 12.3.0. Full package and GPU records accompany
each phase, together with engine arguments and the native parameter layout.

The actual RandOpt Ray engine and worker were hash-checked against upstream
`4000d34fb5b69a3121cf1d2c564aa0be5a6a41ca`. Its existing `PERTURB_VISUAL=1` option
perturbed all vision and language parameters. Every candidate used exact snapshot
anchoring and restoration. The unchanged `randopt-per-tensor-v1` contract reseeds
per tensor; equal-shape tensors therefore have correlated noise within a candidate.
Distinct candidate seeds do not imply independent noise for every scalar weight.

Native base state SHA256:
`d51b308f7a01a09732301ec836200002d082333d0765330b6c2809fe6332fc20`.
Each raw candidate records its recipe, candidate ID, state fingerprint, parameter
mask hash, outputs, token IDs, scores, restoration evidence and peak memory.

All required controls for the executed phases passed: zero perturbation exactly
matched base state and outputs on both splits; every snapshot restoration was
bitwise exact; all three repeated nonzero candidates reproduced their states
and outputs; base outputs repeated within and across processes. Prefix and
multimodal preprocessing caches were disabled. Encoder caches were cleared and
unique image UUIDs used; a forward hook verified fresh encoding of every image.
Calibration generated selection outputs only.

Both the CPU and H100 suites passed **83 tests**, with one environment-specific
skip each. The independent terminal audit checked every phase checksum, rescored
all raw generations, verified tokenized prompts, exact recipe lists, parameter
scope, repeats, split isolation and the committed decisions. Its
[result](../results/visual-line-tracing-20261003/analysis/gate.json) reproduces the
operational stop. Tests and audits do not establish held-out expert quality.

## Evidence and reproduction

All evidence is in [results/visual-line-tracing-20261003](../results/visual-line-tracing-20261003/):
`baseline-v1/`, `calibration-01/`, committed `locks/`, derived `analysis/`, full
logs, environment/install records, transfer hashes, commands and test output.
Each phase preserves measured source bytes and its exact source commit.
The baseline ran from `49bb56e`; calibration ran from `9646eb8`, which adds only
the baseline lock. The analysis source is `4154fc7872988e18bd8679d397ef08b41417a3bf`.

The baseline phase ran 23:15:08–23:17:37 UTC and calibration 23:18:30–23:38:30 UTC.
The [phase script](../results/visual-line-tracing-20261003/complete-phases.sh) and
run manifests preserve the GPU commands. To reproduce the offline decision:

```bash
PYTHONPATH=src python scripts/analyze_visual_line_tracing.py \
  --baseline results/visual-line-tracing-20261003/baseline-v1 \
  --calibration results/visual-line-tracing-20261003/calibration-01 \
  --locks results/visual-line-tracing-20261003/locks \
  --calibration-stop --out runs/visual-stop-recheck
```

Use a new output directory; existing evidence is never overwritten. The final
[integrity record](../results/visual-line-tracing-20261003/integrity.json) verifies
all phase files and measured sources, preserves all 1,391 previously indexed
research artifacts, and confirms `main` remains
`deb414253139cc2559d19cdfe7e6b4786e7c40db`. The GPU was idle after cleanup.
No new optimizer, kernel, localization, sigma, model or prompt change followed.

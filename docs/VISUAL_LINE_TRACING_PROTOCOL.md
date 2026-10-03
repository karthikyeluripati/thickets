# Visual Neural Thickets: frozen feasibility protocol

This is a new model/task study on `research/visual-thickets-line-tracing`, from
completed commit `a7e435a`. Previous directions are closed and their artifacts
are preserved. The authoritative configuration is
[`visual_line_tracing_protocol.json`](../experiments/visual_line_tracing_protocol.json).

Question: does Qwen2.5-VL-3B-Instruct have nearby random weight-space experts that
materially improve unseen visual line tracing without gradient training?

The synthetic dataset has 150 selection and 500 held-out images, generated from
disjoint seed ranges. Each 448x448 image has four black cables, one red starting
dot, and four large single-digit endpoint labels. Crossings use explicit white
underpass gaps; cables do not join. Easy/medium/hard use 4/7/10 adjacent cable
swaps, different separation and curvature, and minimum target-path crossings.
Start/end joint cells are approximately balanced within each level, so endpoint
position does not provide a shortcut. Exact path topology provides the answer;
there is no model-generated label. Metadata stores every swap and drawing seed.
PNG bytes, prompts, answers, split files, generator and renderer provenance are
frozen before baseline evaluation. No learned or generated raster assets are used.

Evaluate the base on all 650 images first, including difficulty and detectable
format/cap failures. If either split is outside 15-90% accuracy, repair/freeze a
new dataset before any candidates; preserve every version. The desired range is
roughly 30-75%. The requested base-only held-out evaluation is an explicit
exception to held-out blindness; it never selects a sigma or candidate. Once
calibration starts, no dataset, prompt, parser, geometry or model changes.

Model, tokenizer and processor revision:
`Qwen/Qwen2.5-VL-3B-Instruct@66285546d2b821cf421d4f5eb2576359d3770cd3`.
One H100, BF16, TP=1, greedy decoding, 16 output tokens, 448x448 images, eager
vLLM 0.10.2 / Ray 2.49.2 / Transformers 4.56.2 / PyTorch 2.8.0+cu128.
Full installed versions, engine arguments, actual model-state hash and source
commit are recorded for every phase. The model is supported by the
[pinned vLLM release](https://docs.vllm.ai/en/v0.10.2/models/supported_models.html).

Use unchanged RandOpt worker commit `4000d34fb5b69a3121cf1d2c564aa0be5a6a41ca`
and its snapshot `apply_perturbation`/`reset_to_base_weights` methods. Set its
existing `PERTURB_VISUAL=1` option: perturb all parameters, including vision,
without localization. Preserve the upstream per-tensor reseeding contract;
equal-shape tensors have correlated noise within a candidate. Final candidates
have separate seeds, disjoint from calibration seeds. No custom optimizer or
change in perturbation geometry is introduced.

Prefix caching and multimodal preprocessing caching are disabled. vLLM's
model-dependent encoder cache is cleared after requests drain; image UUIDs are
unique across generation calls. A vision forward hook verifies the exact number
of newly encoded images on every evaluation, preventing stale vision embeddings
from hiding weight changes. Image content and prompt tokens remain fixed.

Calibration: 20 distinct candidates at each sigma 0.0005, 0.001 and 0.002, on
selection150 only. Choose highest best correct count, then largest fraction
beating base, then smaller sigma. Report mean/best, fraction beating base,
catastrophic fraction and repeat controls. Catastrophic means accuracy at or
below max(15%, base-20pp), or at least 50% invalid answers. A scale fails if mean
accuracy is at most 27.5% and best at most 40%, or catastrophic fraction is at
least 80%. If all scales fail, stop operationally; do not call it scientific NO
or add scales. Otherwise commit exactly one chosen sigma.

Search: exactly 300 new seeds at that sigma, every candidate on all selection150.
Rank correct count descending, then SHA256 of `visual-rank-v1:<candidate_id>`.
Commit exact best/top5/top10 identities and state fingerprints before candidate
held-out generation. Evaluate only that top10 union and base on heldout500;
calibration candidates cannot enter this ranking. No candidate is reselected.

Score only a single digit 1-4, optionally followed by a period/exclamation and
surrounding whitespace. Other outputs are invalid and incorrect. Ensemble voting
ignores invalid answers; Counter ties follow frozen rank order; all-invalid is
incorrect. Preserve text, token IDs, finish reasons, parsed answers and votes.

The frozen gate is GO if best gains at least 5 absolute points, OR top5/top10
majority gains at least 7 points, over held-out base. The same qualifying method
must improve in at least two difficulty levels including medium or hard. All
state/repeat controls and dataset disjointness must pass. Otherwise close this
model/task setup with NO-GO; invalid/incomplete execution is operational failure.
Diagnostic paired bootstrap intervals (10,000 samples, seed 20261003) do not
change these point-estimate thresholds. Report top5/top10 mean individual quality
separately. Selection expert densities at +1/+3/+5pp are descriptive, not held-out
density estimates.

Controls: zero equals base on both splits; repeated state/output identity;
bitwise base restoration; repeated calibration candidate at each sigma;
repeated search candidate; pre-inference fingerprint checks for selected experts;
one held-out repeat; base equality across phases; no held-out records accepted by
ranking code. Exact commands and immutable raw artifacts accompany each phase.

Stop after the fixed GO/NO-GO report. No larger population, extra sigma, prompt
tuning, model change, optimizer comparison, architecture localization or kernel
work follows this study.

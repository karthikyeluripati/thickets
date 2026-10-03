# Visual line-tracing feasibility: frozen, awaiting GPU execution

**No scientific result exists yet.** The hypothesis is that nearby vanilla
RandOpt perturbations of Qwen2.5-VL-3B-Instruct improve genuine held-out visual
line tracing without gradient training. This new study does not continue any
previous direction.

Branch: `research/visual-thickets-line-tracing`, from completed `a7e435a`.
Dataset, model revision, gate and complete phase runner were frozen at
`49bb56e160691e83c786fc7bac40be290c0e8f75` before any baseline or candidate output.
See the [protocol](VISUAL_LINE_TRACING_PROTOCOL.md) and
[machine-readable configuration](../experiments/visual_line_tracing_protocol.json).

The frozen benchmark contains 150 selection and 500 held-out images. Four black
cables, a red starting dot and simple endpoint digits isolate visual connectivity;
white underpass gaps disambiguate crossings. Difficulty changes crossings,
spacing and curvature. Exact topology supplies labels. Start/end joint cells
are approximately balanced within each level; the opposite-endpoint heuristic
scores 25.33% on selection and 24.60% on held-out. Seeds and image hashes are
disjoint. The [dataset](../examples/visual-line-tracing-v1/) includes all PNGs,
prompts, answers, graph metadata, seeds, hashes and Pillow 12.3.0 provenance.

Model/tokenizer/processor revision is
`Qwen/Qwen2.5-VL-3B-Instruct@66285546d2b821cf421d4f5eb2576359d3770cd3`.
The prepared stack is BF16, greedy eager vLLM 0.10.2, Ray 2.49.2,
Transformers 4.56.2 and PyTorch 2.8.0+cu128 on one H100. The pinned original
RandOpt worker uses its existing `PERTURB_VISUAL=1` option to perturb all vision
and language parameters, with exact snapshot anchoring. Every generation is
required to re-encode every image; stale vision embeddings cannot be reused.

All **83 CPU tests passed**, with one CUDA skip. Tests cover independently
replayed topology labels, exact image regeneration, split disjointness,
single-digit scoring, voting ties, the frozen difficulty-aware gate, scale
failure, selection-only ranking and pre-generation state rejection.

The H100 is reachable and the pinned model is downloaded. Automatic approval
review blocked transfer of the new code/data bundle because explicit export
authorization is required. The audited incremental bundle is 21,436,133 bytes,
SHA256 `14cce22e326b0f025b697267665d52d06ed60d5353191a2abdaacbd07d725ed3`;
it contains only new study code, protocol, test log and synthetic dataset, without
prior raw generations. Approval was requested; no new study payload has been
transferred and no baseline or candidate inference has run.

After authorization, the fixed sequence is: full base evaluation and benchmark
acceptance; at most 60 selection-only calibration candidates; commit one sigma;
300 fresh selection-only candidates; commit best/top5/top10; evaluate only those
ten experts and base on heldout500; apply the unchanged +5pp single / +7pp ensemble
gate with improvement in multiple difficulty levels. No rescue tuning follows.

Prior artifacts and `main` are unchanged. Base accuracy, calibration, candidate
density, held-out performance and GO/NO-GO remain unmeasured.

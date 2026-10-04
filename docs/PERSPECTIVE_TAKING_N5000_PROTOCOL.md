# Perspective-Taking Visual Neural Thickets, N = 5,000: protocol

**Question.** At full Neural-Thickets/RandOpt scale, does Qwen3-VL-8B contain a
nearby model with materially and transferably stronger visual perspective-taking?

The [machine-readable protocol](../experiments/perspective_taking_n5000_protocol.json)
is authoritative. It was committed before any Qwen3-VL inference.

## Fixed setup

| Item | Pin |
|---|---|
| Model | `Qwen/Qwen3-VL-8B-Instruct@0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` (model, tokenizer, processor and chat template) |
| Precision and decoding | BF16, TP=1 per H100, greedy (temperature 0, which overrides the checkpoint's sampling defaults) |
| Software | vLLM 0.11.0, transformers 4.57.1, torch 2.8.0, Ray 2.49.2 |
| Weight perturbation | original RandOpt worker `4000d34` with `PERTURB_VISUAL=1`: all parameters, exact snapshot anchoring and restoration |
| Benchmark | OmniSpatial `208bac2`, dataset `6691f32`, `Perspective_Taking` only |

**Data.** The [frozen splits](../examples/omnispatial-perspective-taking/) were
committed in `faf8f54` and contain no images. Every image is verified by sha256
on the GPU host.

- **SEARCH200 and VALIDATION200** come from official train, one QA pair per
  image, with no image shared between splits. Each has the same composition:
  Allocentric 65, Egocentric 119, Hypothetical 16, and answers A/B/C/D =
  51/56/46/47. The remaining 2,679 train records are unused.
- **FINAL TEST** is all 561 official test QA pairs: Allocentric 376,
  Egocentric 102, Hypothetical 83.

**Evaluation.** The official OmniSpatial direct path: the vendored official
prompt and the first-character scorer, with no fallback letter. Images are
converted to RGB, as in the official `fetch_image`, and the pinned processor
resizes them. RGB pixel and prompt hashes are recorded per image and must match
in every phase.

## Population and selection

- **Population.** 5,000 candidates: seed `9500000+i`, sigma
  `[0.00025, 0.0005, 0.001, 0.002][i % 4]`, so exactly 1,250 per sigma. They run
  in 50 shards of 100 candidates. 1,730 prior seeds are excluded.
- **Density audit.** 500 manifest indices, 125 per sigma, were drawn with seed
  20261006 and committed before any output exists.
- **Search.** All 5,000 candidates are evaluated on SEARCH200 and ranked by
  correct count, with ties broken by `SHA256('visual-rank-v1:'+id)`. The full
  ranking and the **top 50** are committed before validation.
- **Validation.** The top 50 and the density audit are evaluated on
  VALIDATION200. The top 50 are ranked by validation correct count, then search
  correct count, then the hash. Rank 1, the top 5 and the top 10 are committed
  before any test inference.
- **Final test.** Only the frozen top 10 are evaluated on all 561 test
  questions.

## Decision

A frozen candidate is a **TRANSFERABLE_PERSPECTIVE_EXPERT** only if all of these
hold:

- test gain ≥ +5 pp;
- paired bootstrap 95% lower bound > 0;
- positive gain on ≥ 2 of the 3 sub-tasks;
- concentration < 0.80, where concentration is the largest positive net gain on
  a single true-answer letter divided by the sum of positive net gains;
- all runtime controls pass.

The outcomes are:

- **Primary GO.** Validation rank 1 is an expert.
- **Replicated GO.** All of the following hold:
  - ≥ 3 of the top 10 gain ≥ +3 pp;
  - ≥ 1 gains ≥ +5 pp;
  - the mean top-10 gain is ≥ +3 pp;
  - a +5 pp candidate passes the expert checks.
- **NO-GO.** `NO_GO_VISUAL_NEURAL_THICKET_PERSPECTIVE_N5000` otherwise. There is
  no rescue.

Top-5 and top-10 majority votes are reported as secondary results only.

## Controls (operational, never scientific)

**Every phase:**

- Zero perturbation reproduces the base outputs.
- Every reset is bitwise exact.
- Images are freshly encoded, with the encoder cache cleared and caches disabled.

**Reconstruction:**

- The first candidate of each shard and of test is repeated, with identical
  state and outputs.
- Validation and test reject any candidate whose fingerprint differs from its
  search fingerprint.

**Faster state checks.** The baseline phase proves them equivalent to the
originals before any candidate runs.

- **Drift check:** a bitwise comparison on the GPU, falling back to the original
  value-by-value comparison when tensors differ.
- **Fingerprint:** a per-tensor SHA256 tree (`state-sha256-tree-v1`).

**Locks and retries:**

- Every lock is committed before the phase that consumes it.
- A failed shard is kept as `.failed-N` and rerun entirely, so no candidate is
  double-counted.

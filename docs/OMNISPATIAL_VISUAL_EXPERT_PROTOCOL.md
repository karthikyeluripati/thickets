# OmniSpatial visual-expert existence test: protocol

**Question.** Does the local weight neighborhood of Qwen2.5-VL-7B contain a
genuinely better, transferable visual-spatial expert on OmniSpatial Complex
Spatial Logic?

The [machine-readable protocol](../experiments/omnispatial_visual_expert_protocol.json)
is authoritative. It was committed before any OmniSpatial inference.

## Benchmark

| Item | Pin |
|---|---|
| Repository | `qizekun/OmniSpatial@208bac258d8671c6857b3be2834997936ba7fc51` |
| Dataset | `qizekun/OmniSpatial@6691f3288bb1ff207d6ead4d841b505de08a6fd8` |
| Train zip LFS sha256 | `6f3878c1…f1e198` |
| Test zip LFS sha256 | `6c2cc57a…c988` |
| Dimension | every `Complex_Logic` QA pair; no sub-task selection |

[Frozen splits](../examples/omnispatial-complex-logic/), committed in `443a423`,
contain IDs, split membership and every image's sha256. Images are not
committed; the GPU host extracts them from the pinned zips and verifies each one.

| Split | Source | QA pairs | Sub-tasks | Answer letters A / B / C / D |
|---|---|---:|---|---|
| SEARCH | official train (60%) | 705 | Geometric_Reasoning | 187 / 193 / 200 / 125 |
| VALIDATION | official train (40%) | 475 | Geometric_Reasoning | 126 / 131 / 136 / 82 |
| FINAL TEST | official test (100%) | 252 | Geometric 155, Pattern 97 | 66 / 67 / 56 / 63 |

**How train is split.** The 60/40 split is stratified by sub-task and answer
letter, with seed 20261004. Byte-identical images (official train reuses some
under different IDs) are kept within one split.

**Audit findings, disclosed before any inference.**

- **Train and test differ in kind.** Official train Complex Logic contains only
  Geometric_Reasoning: templated cube, net and arrow puzzles with text options.
  Official test is IQ-test style, 38% Pattern_Recognition, and 229 of its 252
  items carry their options inside the image. The final test therefore measures
  real distribution transfer.
- **Train images are large.** At official resolution, the median train image is
  about 4,400 visual tokens; the median test image is about 180.

## Evaluation (frozen)

The deterministic multiple-choice path is the official
`vlms_eval/qwenvl_eval.py` with `--prompt_type none --eval_type direct`.

- **Prompt.** The official default system text and direct format, from a
  vendored copy of `system_prompts.py` verified by git blob `38fa3eb`. Then come
  the question and the lettered options. The user message is the image followed
  by the text, under the Qwen chat template.
- **Image preprocessing.** Official `qwen_vl_utils==0.0.8` `fetch_image`: RGB
  conversion plus smart_resize to multiples of 28, between 3,136 and 12.8M
  pixels. The resized pixels are hashed per image, and every phase must match
  the baseline.
- **Scorer.** The official direct scorer takes the first character of the
  stripped, upper-cased response. There is no fallback letter; any other first
  character is invalid and wrong.
- **Decoding.** Greedy decoding, temperature 0, `max_tokens` 16. The direct
  scorer reads only the first non-space character, so the cap cannot change a
  score.

**Model and execution.**

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen2.5-VL-7B-Instruct@cc594898…`, BF16 |
| Parallelism | TP=1 on one H100 80GB |
| Worker | unchanged RandOpt worker `4000d34`, `PERTURB_VISUAL=1` (all parameters, including vision) |
| Snapshot | exact snapshot anchoring and restoration |
| Software | vLLM 0.10.2 / Ray 2.49.2, pinned versions |

## Population, search, validation and test

1. **Population.** 400 candidates: seed `9100000+i`, sigma
   `[0.00025, 0.0005, 0.001, 0.002][i % 4]`, so exactly 100 per sigma. There is
   no sigma sweep, and 1,330 prior seeds are excluded.
2. **Search.** All 400 candidates are evaluated on SEARCH. They are ranked by
   SEARCH correct count, with ties broken by `SHA256('visual-rank-v1:'+id)`. The
   records, ranking and **top 30** are committed before validation.
3. **Validation.** The top 30 are evaluated on VALIDATION and ranked by
   validation correct count, with ties broken by search rank. Rank 1, the top 5
   and their fingerprints are committed before any test candidate inference.
4. **Final test.** Only the frozen top 5 are evaluated on the 252 official test
   questions.

**`TRANSFERABLE_OMNISPATIAL_EXPERT`.** A frozen candidate qualifies only if all
of these hold:

- final-test gain ≥ +5 pp;
- paired 10,000-resample bootstrap 95% lower bound > 0;
- strictly positive gain on both test sub-tasks (Geometric_Reasoning and
  Pattern_Recognition);
- no true-answer letter supplies ≥ 80% of positive net correctness gains;
- all runtime controls pass.

**Decision.** `GO_OMNISPATIAL_VISUAL_EXPERT` if any frozen top-5 candidate
qualifies, otherwise `NO_GO_OMNISPATIAL_VISUAL_EXPERT`. The answer-bias rule is
a validity check only; it never reranks candidates. Top-30 validation counts are
reported as validation-enriched, not as expert density.

## Controls (operational, never scientific)

**Every phase:**

- Zero perturbation reproduces the base outputs exactly.
- Restoration is bitwise exact.
- Images are freshly encoded under unique UUIDs, with the encoder cache cleared
  and prefix and multimodal caches disabled.
- Preprocessing and prompt hashes equal the baseline.

**Reconstruction:**

- Each phase repeats its first candidate, with identical state and outputs.
- Validation and test reject a candidate unless its fingerprint equals its
  search fingerprint.

**Locks.** Every lock is committed before the phase that consumes it.

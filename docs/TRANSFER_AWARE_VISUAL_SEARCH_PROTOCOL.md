# Transfer-aware visual search: first-bridge protocol

**Question.** Given the exact same population of nearby VLM weight perturbations,
can a transfer-aware ranking rule find candidates that generalize better than
standard RandOpt ranking?

The machine-readable [protocol](../experiments/transfer_aware_visual_protocol.json)
is authoritative. It was committed before any 7B inference. Previous experiments
and artifacts are unchanged.

## Fixed setup

- **Model.** `Qwen/Qwen2.5-VL-7B-Instruct@cc594898137f460bfe9f0759e9844b3ce807cfb5`
  (model, tokenizer and processor at the same revision). BF16, TP=1, one H100 80GB,
  greedy decoding, 16 output tokens, 448² pixels.
- **Software.** Unchanged vLLM 0.10.2 / Ray 2.49.2 stack. Original RandOpt worker at
  `4000d34`, with `PERTURB_VISUAL=1`, exact snapshot anchoring/restoration and fresh
  image encoding. No quantization, LoRA or model change.
- **Data.** [Fresh v1 line-tracing splits](../examples/visual-line-tracing-v1-transfer-aware/),
  committed in `dd89aae` before inference: search 300, validation 300, test 1000.
  - The v1 geometry, renderer, prompt, parser and labels are unchanged.
  - Seeds start at 8100000, 8200000 and 8300000. They are seed- and image-hash-disjoint
    from each other and from v1/v2.
  - Labels are exactly balanced; difficulty levels differ by at most one.
  - Search folds A/B/C have 100 examples each, are stratified by difficulty × label,
    and have 25 examples per label.
- **Candidates.** 400 candidates: seed `7300000+i`, sigma `[0.00025, 0.0005, 0.001, 0.002][i % 4]`,
  so exactly 100 at each sigma. All seeds are fresh (930 prior seeds excluded).
  Every candidate is reconstructed from the same exact base. They run in eight
  operational shards of 50.
- **No base gate.** The base is evaluated on all three splits; nothing is gated on it.

## Rankings over the identical 400 audited search records

- **VANILLA.** Pooled search300 correct, descending.
- **TRANSFER_MIN.** `min(fold gain A, B, C)` descending, then mean fold gain
  descending. A fold gain is the candidate's fold accuracy minus the base's fold
  accuracy.
- **Ties.** Both rules break remaining ties with `SHA256('visual-rank-v1:' + candidate_id)`
  ascending.

There is no lambda, no variance term and no alternative score.

## Validation gate

Each method's top 20 are committed before any validation output exists. Their
union is then evaluated on validation300.

The gate passes if **A or B** holds. The request says "both", then gives A OR B;
its later text, which treats either condition as sufficient, is applied.

- **A.** The best TRANSFER_MIN top-20 validation gain is ≥ +3 pp, and at least
  2 pp above the best VANILLA top-20 gain.
- **B.** At least 3 of the TRANSFER_MIN top 10 (in search order) reach ≥ +3 pp,
  and that count exceeds VANILLA's top-10 count by at least 2.

If neither holds, record `NO_GO_TRANSFER_AWARE_VISUAL_SEARCH` and stop. There is
no final test.

## Final test (only if the gate passes)

**Final candidate set.** For each method, order its top 20 by validation gain,
breaking ties by that method's search rank. Its best and top 5 are taken. The
deduplicated union is committed, and then evaluated once on test1000.

**Genuine visual expert.** A candidate counts only if all of these hold:

- test gain ≥ +5 pp;
- paired 10,000-resample bootstrap 95% lower bound > 0;
- positive gain on ≥ 2 of easy/medium/hard;
- no true-answer class supplies ≥ 80% of the positive per-class net gains.

**Decision rules.**

- **`GO_TRANSFER_AWARE_VISUAL_SEARCH`.** The TRANSFER_MIN primary (its best
  validation candidate) is a genuine expert, all controls pass, and either
  condition holds:
  - the VANILLA primary is not a genuine expert; or
  - the TRANSFER_MIN primary's test gain exceeds the VANILLA primary's by ≥ 2 pp.
- **`PROMISING_BUT_NOT_EXPERT`.** Strong GO fails, but TRANSFER_MIN still
  dominates:
  - ≥ 3 of the TRANSFER_MIN top 5 reach ≥ +3 pp;
  - the TRANSFER_MIN top-5 mean is ≥ +3 pp;
  - VANILLA does not meet both of those conditions.
- **`NO_GO_TRANSFER_AWARE_VISUAL_SEARCH`.** Otherwise.

## Controls (operational, never scientific)

**Every phase:**

- Zero perturbation reproduces the base outputs.
- Each candidate's snapshot reset is bitwise exact.
- The encoder cache is cleared and every image is freshly encoded.
- Prefix and multimodal caches are disabled.

**Reconstruction:**

- Each search shard, validation and test repeats its first candidate, with
  identical state and outputs.
- Validation and test reject a candidate before generation unless its state
  fingerprint equals its search fingerprint.

**Locks:**

- The protocol, baseline, ranking and validation locks are each git-committed
  before the phase that consumes them.
- The analysis checks the commit times against phase start times and rescores
  every raw generation.

## Run

```bash
bash scripts/pod_transfer_aware.sh results/transfer-aware-visual-20261004 /workspace/RandOpt-visual
```

The study is resumable. Completed phases are skipped. A failed phase is kept
under `.failed-N` and rerun.

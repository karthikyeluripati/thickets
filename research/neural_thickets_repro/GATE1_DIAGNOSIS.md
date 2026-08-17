# Gate 1 HARD FAIL — diagnosis (baseline 17.94% → 54.19% reconstructed vs 56.60% published)

Status: **Image-blindness confirmed at full scale.** Full 12,578-example image-aware
reconstruction: **54.19% march-era scoring** vs published 56.60% = **−2.41pp**, inside the
pre-agreed 1–3pp investigation band. No prompts, scoring, data prep, or model settings have
been changed toward 56.6%. RandOpt / Gate 2 remains blocked.

## Residual −2.41pp gap investigation (this pass)

Five items requested, three resolved from static/network research (no GPU needed), two
still need the pod:

| # | Item | Status | Finding |
|---|---|---|---|
| 1 | Independent lmms-eval GQA control | **Still outstanding** | Not yet run/reported (was already owed from the prior round too) |
| 2 | GQA parquet vs canonical source | **RULED OUT** | Full 12,578-row audit against `lmms-lab-encoder/GQA` `testdev_balanced_instructions`: exact row count match, order matches source exactly, IDs unique, zero null/empty answers, zero rows needing case/whitespace normalization our construction didn't already apply, no filtering anywhere. Our parquet is a faithful, unmodified copy of the source. |
| 3a | March-era vLLM version | **Real, documented gap — candidate contributor** | `docker/Dockerfile_vllm` pins `FROM vllm/vllm-openai:v0.11.0` at the March 2026 commit (`60061828e3`) — and this line is **byte-identical** at the pinned HEAD commit today, i.e. the authors' own Docker recipe has never moved off v0.11.0. Our pod reports vLLM **0.27.1** — 16+ minor versions ahead. `requirements.txt` only floors `vllm>=0.10.0` (both then and now), so pip-based installs were never pinned exactly, but the Docker tag is the best actual evidence of the authors' real environment. |
| 3b | March-era transformers version | **Unresolvable exactly** | `requirements.txt` says bare `transformers`, no version bound, unchanged March→HEAD. No other artifact in the repo pins it. Genuinely not recoverable from the repo alone. |
| 3c | Qwen2.5-VL image preprocessing (min/max_pixels, patch/merge size) | **RULED OUT — stable since before the paper** | Fetched `preprocessor_config.json` at the commit where it was last touched (`1b989f2c63`, 2025-02-15) and at our pinned model revision (`66285546d2`, 2025-04-06): **byte-identical** (`min_pixels=3136, max_pixels=12845056, patch_size=14, merge_size=2, Qwen2VLImageProcessor`). Both predate the paper's March 2026 date by over a year with zero changes since. No drift possible here regardless of which revision the paper actually used. |
| 4 | March-era Qwen2.5-VL-3B-Instruct checkpoint revision | **RULED OUT — no drift possible** | Full HF commit history for the checkpoint: last commit `66285546d2` on **2025-04-06** ("Update tokenizer_config.json") — nothing since, at any point through today. That commit **is** our pinned revision. The checkpoint has been frozen for ~11 months before the paper published and remains frozen now — whatever revision the paper used in March 2026, it can only have been this one. |
| 5 | 200-example paired classification of remaining failures | **Tooling ready, needs the pod** | `diagnostics/residual_gap_audit.py` — reuses the *already-generated* `results/base_image_aware/predictions.jsonl` (no new GPU generation), classifies march-scoring-incorrect examples into `extraction_or_scoring_failure` / `model_wrong_yes_no` / `empty_or_degenerate_response` / `model_wrong_other`. |

### Ranked hypothesis for the remaining −2.41pp (pending items 1 and 5)

1. **Extraction/scoring imperfection** (not a model-content miss) — the 100-example audit
   already showed this failure mode exists even in the fully image-aware path; at n=12,578
   even a few-percent residual extraction-miss rate plausibly accounts for a meaningful
   slice of 2.41pp. `residual_gap_audit.py` will quantify this directly.
2. **vLLM version gap (v0.11.0 vs 0.27.1)** — real and documented, plausible contributor via
   differences in multimodal image handling maturity or generation-kernel numerics between
   major vLLM versions, but not yet isolated or measured.
3. **Generation numerics / determinism across library versions** — greedy (temperature=0)
   decoding is deterministic *within* a fixed stack, but different vLLM/CUDA/torch versions
   can produce different token-level outputs on identical inputs; this is a known,
   generally irreducible class of reproducibility gap, not a bug to fix.
4. **transformers version drift** — plausible but unresolvable exactly (item 3b).

### Preliminary read (will sharpen, not necessarily reverse, once items 1 and 5 land)

Both **fully controllable, "our fault" candidates are now ruled out** (data construction,
model revision, image preprocessing config). What remains — vLLM/transformers version
drift and generation-numerics differences — are environmental factors the *original authors
themselves* did not pin exactly either (`requirements.txt` has only floors, never exact
versions, in both the March and current commits). Current best assessment: **likely
"explain the gap but not fully eliminate it"** rather than reproduce-to-≤1pp, pending the
lmms-eval cross-check and the failure classification. Do not treat this as final.

## What to run on the pod

```bash
cd /workspace/thickets/research/neural_thickets_repro
git pull origin neural-thickets-repro-gate1-prep
pip install -e .

# Item 5: classify existing predictions (fast -- no new generation, reuses the full run)
python -m neural_thickets_repro.diagnostics.residual_gap_audit \
    --predictions results/base_image_aware/predictions.jsonl --sample-size 200 --scoring march

# Item 1: if not already running/done, the outstanding independent control
python -m lmms_eval --model qwen2_5_vl \
    --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
    --tasks gqa --batch_size 1 --log_samples \
    --output_path results/gate1_diagnosis/lmms_eval_gqa
```

## Then: the final controlled test (vLLM 0.27.1 vs 0.11.0)

This is the last planned diagnostic step for the residual gap — see
**`VLLM_VERSION_CONTROL_SETUP.md`** for the full procedure (official `vllm/vllm-openai:v0.11.0`
Docker image, matching the RandOpt repo's documented environment; does not touch the
working 0.27.1 environment). It compares the two vLLM versions on an identical fixed
200-example sample and applies a pre-agreed statistical decision rule (McNemar exact test,
not an arbitrary percentage threshold) that resolves directly to either "run the full
12,578 baseline under 0.11.0" or "stop and accept Gate 1 as a paper-faithful reconstruction
with a documented, unrecoverable runtime-version discrepancy." No further open-ended
debugging beyond this step is planned.

Bring back `results/gate1_diagnosis/residual_gap_audit.json` (classification counts +
overall accuracy) and the lmms-eval GQA number. Do not start RandOpt / Gate 2.

---

## Confirmed results (100-example paired audit, seed=42)

| Path | Current (HEAD) scoring | March/paper-era scoring |
|---|---|---|
| A — text-only (upstream-replica) | 15% | 22% |
| B — with actual images | 59% | 59% |

- Paired comparison (head scoring): **45 examples flipped wrong→correct** when the image
  was added; only **1** flipped the other way.
- **20/20** question↔image pairs independently verified against the source HF dataset.
- Proposed verdict from the pre-registered threshold (≥20pp delta ⇒ CONFIRMED): **CONFIRMED**
  — delta is +44pp, more than double the threshold.
- Independent lmms-eval control: **still pending** — not yet run/reported. The evidence
  above (44pp swing, 45:1 flip ratio, Path B already close to 56.6% on just 100 examples,
  20/20 pair integrity) is strong enough on its own to proceed to the full baseline while
  lmms-eval runs in parallel, but the lmms-eval number is still owed before this is called
  fully closed.

**Root cause: the released RandOpt GQA generation path never supplies image data to the
VLM — confirmed, not hypothesized.**

## Finding 1 (primary suspect): the released code never passes images to the model

Static-analysis chain, each step verified directly:

1. `data_handlers/gqa.py` builds multimodal messages containing `{"type": "image", "image": <path>}`.
2. `randopt.py` formats these through `tokenizer.apply_chat_template(..., tokenize=False)`
   into a **plain text string**. Verified locally against the real pinned tokenizer: the
   template inserts `<|vision_start|><|image_pad|><|vision_end|>` placeholder tokens into
   the text — but that is all they are, placeholders.
3. `randopt.py` then calls `engine.generate(text_prompts, sampling_params)` with those bare
   strings. vLLM's `generate()` only attaches images when the caller passes
   `multi_modal_data` — which **no code path in the repository ever constructs**. Verified
   by grep across the full repo at both the pinned HEAD and the March 2026 commit: zero
   occurrences of `multi_modal_data`; `AutoProcessor` is never used (only `AutoTokenizer`).
4. Therefore the model receives an image placeholder with no image behind it and answers
   **blind**. Gate 1's observed 17.94% is consistent with a blind model + strict answer
   formatting on GQA; the run completing without error is consistent with vLLM tolerating
   an unfilled placeholder token in a text-only request.

This also retroactively answers the long-open "multimodal engine flag" question: the missing
`multimodal=True` in `launch_engines` is real but secondary — `limit_mm_per_prompt` only
matters once images are actually passed, and they never are.

**Confirmation experiment (pod, `diagnostics/gate1_failure_audit.py`): 100-example seeded
sample, generation both ways** — Path A replicates upstream's text-only call (expected ≈18%),
Path B passes the identical prompts WITH `multi_modal_data` (if accuracy jumps toward
50-60%, root cause confirmed).

## Finding 2: base-eval scoring changed after the paper (May 2026, PR #4)

Git history of `sunrainyg/RandOpt` (full listing inspected this session):

- `60061828e3` (2026-03-01) — "initial commit, basic setup": **the closest commit to the
  paper's March 2026 experiment.** All earlier history is a README-only stub (`17ecfc8c9c`,
  2026-02-18).
- `cb108d44d6` (2026-05-08) — "Fix base model test evaluation." (external contributor PR #4):
  changed base test accuracy from `postprocess_outputs` (mean of `compute_reward` — the
  lenient path, which also whole-word-scans the first raw response line for the GT) to the
  stricter extract-answer→validate→check path used today.
- Verified by direct diff (March vs pinned HEAD): `data_handlers/gqa.py` and
  `utils/reward_score/gqa.py` are **byte-identical** apart from a one-line `true`→`True`
  typo fix — the extraction/normalization/matching logic did NOT change. `randopt.py`'s
  only behavioral change is the base-eval scoring path above.

Implication: the paper's 56.6% base was computed under the March (reward-based, more
lenient) scoring. Our 17.94% was computed under the HEAD (stricter) scoring. This is a real
evaluator mismatch worth quantifying — the audit scores every response BOTH ways — but it
cannot plausibly explain a 3x gap on its own; a blind model can.

## Finding 3: model revision was not reaching vLLM — fixed

Upstream `randopt.py` has no `--revision` argument, so passing the hub name let vLLM
resolve `revision=None` (whatever `main` currently is), ignoring our configured pin.
Fixed without touching upstream: `eval_base.py`/`run_randopt.py` now resolve the pinned
revision to a local snapshot via `huggingface_hub.snapshot_download(revision=...)`
(`vlm_adapter.resolve_model_snapshot`) and pass the local path as `--model_name`.
Regression-tested. Note: the currently-pinned revision IS current `main`, so this was a
correctness/pinning bug, not a plausible contributor to the 17.94%.

## What to run on the pod (diagnosis only — no RandOpt, no Gate 2)

```bash
cd /workspace/thickets/research/neural_thickets_repro
git pull origin neural-thickets-repro-gate1-prep
pip install -e .

# 1. The failure audit: 100-example sample, both generation paths, both scorings,
#    failure classification, paired A-vs-B comparison, proposed verdict, and 20
#    question<->image pair verification.
python -m neural_thickets_repro.diagnostics.gate1_failure_audit \
    --config configs/gqa_repro.yaml --sample-size 100 --verify-pairs 20
# Outputs: results/gate1_diagnosis/gate1_failure_audit.json
#            -> paths.A_upstream_replica_text_only.{accuracy_head_scoring,accuracy_march_scoring}
#            -> paths.B_multimodal_control.{accuracy_head_scoring,accuracy_march_scoring}
#            -> paired_comparison_head_scoring.a_wrong_b_correct
#            -> pair_verification.{pairs_ok,pairs_checked,all_ok}
#            -> proposed_verdict.verdict  (mechanical threshold, see "Interpretation guide" below)
#          results/gate1_diagnosis/sample_A_upstream_replica_text_only.jsonl
#          results/gate1_diagnosis/sample_B_multimodal_control.jsonl
#          (same question_id in both files, in the same order -- directly pairable)

# 2. Independent control: standard lmms-eval GQA baseline on the same checkpoint.
pip install lmms-eval
python -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
    --tasks gqa \
    --batch_size 1 \
    --log_samples \
    --output_path results/gate1_diagnosis/lmms_eval_gqa
# If the qwen2_5_vl model type is unavailable in the installed lmms-eval version, check
# `python -m lmms_eval --help` / their model registry for the current name (it has been
# renamed across releases: qwen2_5_vl / qwen2_vl).
```

## Next: full 12,578-example reconstructed baseline (still no RandOpt)

Confirmed by the 100-example audit above. Run the full testdev-balanced set through the
same minimal image-aware adapter (`src/neural_thickets_repro/vlm_adapter.py:
generate_with_images` — identical data/prompt/decoding/scoring to upstream, only change is
that the image is actually attached to the request):

```bash
cd /workspace/thickets/research/neural_thickets_repro
git pull origin neural-thickets-repro-gate1-prep
pip install -e .

python -m neural_thickets_repro.eval_base_image_aware --config configs/gqa_repro.yaml
```

Outputs, matching the original Step-2 spec:
- `results/base_image_aware/predictions.jsonl` — one record per example: `example_id`,
  `image_id`, `question`, `reference_answer`, `raw_prediction`, `normalized_prediction`,
  `correct_head_scoring`, `correct_march_scoring`.
- `results/base_image_aware/metrics.json` — `accuracy_head_scoring`,
  `accuracy_march_scoring`, both diffed against `published_base_accuracy` (0.566) in
  percentage points, plus the Gate 1 threshold rule from the config for reference.
- `results/base_image_aware/run_metadata.json` — model name/pinned revision/resolved
  snapshot path, dataset name/revision/split, our repo's git commit, external RandOpt's
  pinned commit, Python/package versions, seed, decoding config, prompt template, image
  handling note, example counts, elapsed time, exact command.

**Report both `accuracy_head_scoring` and `accuracy_march_scoring` from `metrics.json`.**
The paper-era (march) scoring is the fairer direct comparison to the published 56.6%, since
that's the evaluator that was actually in place in March 2026 — but bring back both, since
the head-scoring number is what any future Gate 2 run will actually use. Apply the agreed
Gate 1 threshold rule (≤1pp proceed & document / 1-3pp investigate / >3pp hard stop) to the
march-era number specifically.

If lmms-eval hasn't finished yet, run it in parallel with this — it doesn't block the full
baseline, only the final "fully closed" call on Finding 1.

## Interpretation guide (decided in advance, not after seeing numbers)

| Audit outcome | Conclusion |
|---|---|
| Path A ≈ 18%, Path B ≈ 50-60%, lmms-eval ≈ 55-62% | Root cause confirmed: released code is image-blind; paper must have run a variant that actually passed images. Reproduction needs a documented image-passing adapter (a deviation from the released code, recorded as such — NOT tuning toward 56.6). |
| Path A ≈ Path B (both ≈ 18%) | Images are not the (only) problem — suspect data prep or scoring; dig into the failure-classification counts and pair-verification results next. |
| Path B high but lmms-eval also ≈ 18% | Something wrong with our checkpoint/environment, not the pipeline. |
| Large head-vs-march scoring gap on the same path | Evaluator mismatch (Finding 2) is a material contributor; quantify and report both numbers. |

## What to bring back

- **lmms-eval GQA result** — still outstanding, from `results/gate1_diagnosis/lmms_eval_gqa/`.
- **Full-baseline `results/base_image_aware/metrics.json`**: `accuracy_head_scoring`,
  `accuracy_march_scoring`, and both diffs against 56.6%.
- Whether the march-era diff falls in the ≤1pp / 1-3pp / >3pp band, and therefore whether
  Gate 1 (as reconstructed) is a PASS to consider for eventual Gate 2 review, or needs
  further investigation.

## Explicitly not done

- No prompt, scoring, dataset, or model-setting changes toward 56.6%.
- No RandOpt / Gate 2 execution. `eval_base_image_aware.py` cannot start RandOpt -- that
  code path doesn't exist in this script.
- No extension of image-awareness into the RandOpt candidate-sampling/ensemble loops
  (`core/engine.py`, `utils/worker_extn.py`) -- out of scope until Gate 2 is authorized.
- Full 12,578-example baseline not yet run (needs the pod) -- results above are from the
  100-example audit only.
- `MULTIMODAL_FIX_NOTES` in `vlm_adapter.py` stays unset until the pod confirms the hypothesis.

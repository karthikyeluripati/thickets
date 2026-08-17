# Gate 1 HARD FAIL — diagnosis (baseline 17.94% vs published 56.6%)

Status: **root-cause hypothesis identified from static analysis; pod runs required to
confirm.** No prompts, scoring, data prep, or model settings have been changed. RandOpt /
Gate 2 remains blocked.

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
#    failure classification, and 20 question<->image pair verification.
python -m neural_thickets_repro.diagnostics.gate1_failure_audit \
    --config configs/gqa_repro.yaml --sample-size 100 --verify-pairs 20
# Outputs: results/gate1_diagnosis/gate1_failure_audit.json
#          results/gate1_diagnosis/sample_A_upstream_replica_text_only.jsonl
#          results/gate1_diagnosis/sample_B_multimodal_control.jsonl

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

## Interpretation guide (decided in advance, not after seeing numbers)

| Audit outcome | Conclusion |
|---|---|
| Path A ≈ 18%, Path B ≈ 50-60%, lmms-eval ≈ 55-62% | Root cause confirmed: released code is image-blind; paper must have run a variant that actually passed images. Reproduction needs a documented image-passing adapter (a deviation from the released code, recorded as such — NOT tuning toward 56.6). |
| Path A ≈ Path B (both ≈ 18%) | Images are not the (only) problem — suspect data prep or scoring; dig into the failure-classification counts and pair-verification results next. |
| Path B high but lmms-eval also ≈ 18% | Something wrong with our checkpoint/environment, not the pipeline. |
| Large head-vs-march scoring gap on the same path | Evaluator mismatch (Finding 2) is a material contributor; quantify and report both numbers. |

## Explicitly not done

- No prompt, scoring, dataset, or model-setting changes toward 56.6%.
- No RandOpt / Gate 2 execution.
- `MULTIMODAL_FIX_NOTES` in `vlm_adapter.py` stays unset until the pod confirms the hypothesis.

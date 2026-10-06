# GPU-A run 2 (amendment A1) — status: S2c incomplete

Lock: `plan_lock_A1.md` (+ unchanged parts of `plan_lock.md`). Run-2 hard cap **$5.00**; spent ≈ **$2.98**
(pod at $3.49/h, 2026-10-06 15:04–≈15:56Z). Pod stopped.

## Completed (frozen / pre-registered gates)
| Gate | Result |
|---|---|
| V0 (score-definition verification) + S2a | passed (see `pod2/run.log`, step `V0_S2a`) |
| S2b exact candidate bytes (9504111) | passed, 642/642 packed tensors SHA-identical to the vLLM stream check |
| F1 measured side | HF-internal fp32 contrasts for 9504111 + 12 frozen controls on RERANK+TEST (`f1_candidate_contrasts.json`) |

Outputs pulled to `pod2/repro/results/paper-analysis/geometry-gpu-a/gpu2/`: `base_contrast_fp32.json`,
`f1_candidate_contrasts.json`, `cost_log.json` (overwritten by the resume attempt; full step log is in `pod2/run.log`).
`pred_*.npy` there are invalid partials from the OOM'd attempts and are not used.

## Not completed
S2c (folded gradients + per-candidate first-order predictions + set projections). **No F1–F4 / S2d decision exists yet.**

1. First attempt: CUDA OOM in the fold (2.32 GiB fp32 temporary + fragmentation).
2. Resume: CUDA OOM at the first backward (78.1 GB allocated): all bf16 grads (17.5 GB) held at once + setF (7.5 GB) +
   block buffer + activations.

## Fix (engineering only; no change to any locked definition)
- `geometry_lib.StreamingFolder`: a post-accumulate-grad hook folds each parameter's gradient into the example's
  stream row and frees it immediately; per-group partials for LOCALIZATION examples come from the same hook.
  Unit-tested against the explicit fold (`tests/test_geometry_lib.py`, 5/5 pass).
- Gradient checkpointing (train mode; Qwen3-VL has no dropout). Safety check: the train-mode fp32 contrast must
  reproduce the stored eval-mode base contrast (abort if median |Δ| > 0.05 after the first block); reported per phase.
- Set-level sums on CPU during the loop; BLOCK 8; order RERANK → TEST → SEARCH so the locked primary analyses
  (F1, F2, F3-RERANK) complete first.
- `scripts/pod_geometry_a2_s2c.sh`: HF-only setup; guard stops 20 min after success, 5 min after failure.

Estimated S2c continuation ≈ 30 min incl. setup ≈ $1.75 at $3.49/h; remaining under the run-2 cap ≈ $2.02.

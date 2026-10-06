# GPU-A amendment A1: lock (approved by the user; committed before any gradient exists)

Supersedes `plan_lock.md` only where stated here. Run 1 (`NO_GO_S2a` under the
original lock) stays recorded unchanged. Run 2 uses
`scripts/geometry_gpu_a2.py`.

## Score definition (A1.1)

- **h_j:** the final post-norm hidden state at the last prompt position
  (`model.model(...).last_hidden_state[0, -1]`).
- **Decision contrast:** s_j = h_j·(W_head[y] − W_head[r_B]), computed in
  **fp32** (h and the two `lm_head` rows upcast). The normalizer cancels in a
  contrast. This is the gradient target and the HF-side measurement.
- **r_B** follows the original lock: the stored vLLM BASE scores for RERANK
  and TEST; for SEARCH, the HF BASE fp32 letter logits (flagged).

## V0 score-definition verification (gate; runs first, on BASE, all 961)

| Check | Pass condition |
|---|---|
| V0a, identity | \|s_j(fp32 logit contrast) − [logP(y) − logP(r_B)] from a full-vocabulary fp32 log-softmax of h_j·W_headᵀ\| ≤ 1e-3 nats for every example |
| V0b, consistency with the model's BF16 output | \|s_j(fp32) − (BF16 log-softmax contrast from `model(...).logits`)\|: median ≤ 0.125 and p99 ≤ 0.5 nats (BF16 output resolution is 0.125–0.25) |
| V0c, cross-engine | s_j(fp32) against the stored vLLM contrast: **reported only** (median, p90, p99, argmax agreement). Not a gate (A1.3). |

Any V0a or V0b failure → stop.

## Other gates

| Check | Pass condition |
|---|---|
| S2a | Prompt token IDs identical for ≥ 99% of 961 |
| S2b | All 642 packed-tensor byte hashes of the HF-built 9504111 equal the vLLM-realized ones (run 1, `vllm_stream_check.json`) |

**S2b behaviour:** HF answer agreement with the stored vLLM answers is
reported, not gated (A1.3).

## Primary F1 (A1.2): HF-internal first-order validity

- **Candidates:** 9504111 + the 12 frozen random controls.
- **Measured:** t_HF[j] = s_j(HF candidate) − s_j(HF base), fp32 contrast, on
  RERANK + TEST (761). Each candidate is built in HF from ε by the exact BF16
  add, then BASE is restored from a CPU copy.
- **Predicted:** t̂[j] = σ⟨fold(∇s_j), ε⟩.
- **Statistic:** pooled Pearson r over 13 × 761.
- **Decision:**
  - r ≥ 0.6 supports;
  - r < 0.3 **falsifies**.
- **Also reported:** the slope and r per candidate.

## Unchanged from `plan_lock.md`

- candidate lists;
- F2 (predicted flips against **vLLM-observed** changed answers of 9504111;
  < 50% falsifies);
- F3 (predicted against **vLLM-observed** accuracy gains; RERANK r < 0.3
  falsifies, ≥ 0.5 supports);
- F4;
- S2d (group partials against the session-1 measured insertion shifts, which
  are vLLM and quantized at 0.125; r < 0.5 falsifies).

## Budget

- Estimate ≈ 64 min ≈ $3.7 at $3.49/h. **Hard cap $5.00** (this run).
- The pod guard stops it at $4.90; the in-script guard stops it before any
  block that would exceed the cap.

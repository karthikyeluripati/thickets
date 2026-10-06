# GPU-A run 1: `NO_GO_S2a` under the lock, and proposed amendment A1

## Run 1 outcome (lock respected)

**Execution.** 1× H100 at $3.49/h, 14:45–14:56 UTC, about **$0.67**. The pod
was stopped after retrieval.

| Check | Result |
|---|---|
| S1b (vLLM) | **Pass.** All 642 realized tensors equal bf16(base + bf16(σ·ε[:n].view(shape))) for 9504111, 9500059 and 9504179. Each fingerprint equals its recorded state ID (9504111 → `7f0bdb3b…`). The single-stream structure is confirmed on the real tensor list. |
| S2a, prompt tokens | 961 / 961 identical (SHA-256 against the stored vLLM prompt IDs) |
| S2a, argmax = stored answer | SEARCH 197/200, RERANK 198/200, TEST 547/561: **942 / 961 = 98.0%** (lock: ≥ 99%) |
| S2a, median per-example max \|ΔlogP_{A–D}\| | RERANK 0.249, TEST 0.245 nats (lock: ≤ 0.10) |
| Decision | **`NO_GO_S2a`.** The run stopped as locked. S2b and S2c did not run, and **no gradients were computed**. |

## Diagnosis (post hoc, CPU)

**Quantization.** Every pairwise A–D score difference in **both** engines is
an exact multiple of 0.125 nats; 67% of the vLLM ones are multiples of 0.25.
The scores come from BF16 logits, whose spacing at typical magnitudes (16–64)
is 0.125–0.25.

**Size of the gap.** The HF–vLLM difference in the decision contrast s_j
(gold − BASE strongest wrong) has median **0.25**, p90 0.375 and p99 0.75
nats. That is about one or two BF16 steps, consistent with different kernel
accumulation orders. A common offset is small (median 0.07). For comparison,
9504111's candidate effect |t| has median 1.5 nats.

**Conclusion.** The locked 0.10-nat criterion was below the measurement
resolution of both engines' outputs. That was an error in the plan lock. The
run-1 decision is nevertheless recorded and kept as `NO_GO_S2a`.

## Proposed amendment A1 (needs explicit approval; would be locked before any gradient exists)

1. **Score precision.** Compute the HF decision contrast at full precision:
   s_j = (h_last · (W_head[y] − W_head[r_B])) in fp32. The normalizer cancels
   in a contrast. This removes BF16 output quantization from the gradient
   target and from HF-side measurements.
2. **Primary F1 moves inside HF.** The first-order theory is a claim about the
   model's function, so it is tested where both sides share numerics:
   - prediction t̂ = σ⟨fold(∇s_j), ε⟩;
   - target t_HF = s_j(HF candidate) − s_j(HF base), both in fp32 contrast;
   - scope: 9504111 + the 12 random controls on RERANK + TEST (13 HF candidate
     forward passes × 761);
   - thresholds unchanged: r ≥ 0.6 supports, r < 0.3 falsifies.
3. **Cross-engine anchoring stays, as behavioural tests.**
   - S2b: byte-identical candidate tensors across engines, which is exact by
     construction and checked by hash.
   - F2 and F3 still predict the **vLLM-observed** answers and accuracy
     gains, with thresholds unchanged.
   - Cross-engine score agreement becomes a **reported limitation**, not a
     gate: run-1 values plus the fp32-contrast values. It is no longer
     re-thresholded after the fact.
4. **Unchanged:** everything else in `plan_lock.md` (lists, F2, F3, F4, S2d).

**Cost.** Setup and load ≈ 7 min; S2b ≈ 3 min; 13 HF candidate passes
≈ 26 min; S2c ≈ 20 min; stop ≈ 2 min. That is about 58 min ≈ **$3.4**, with a
**hard cap of $5.00** for this run (separate from run 1's $0.67).

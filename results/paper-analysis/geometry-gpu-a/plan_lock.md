# GPU-A plan lock: folded-gradient geometry for 9504111

Committed before any GPU-A output exists. **Post hoc** with respect to the
original study. The definitions below are frozen; the analysis may not change
them after seeing results. "Implicit gradient step" is a **hypothesis label
only** until S2c and C1–C3 support it.

**Target claim (only if supported):** search-induced experts can be
quantitatively predicted as projections onto task-specific folded gradients,
and their apparent specialization and transfer failure follow from cross-set
gradient geometry.

## Fixed objects

- **Model:** Qwen3-VL-8B-Instruct @ `0c351dd`, BF16. HF Transformers 4.57.1
  for gradients (SDPA attention). vLLM 0.11 + pinned RandOpt `4000d34` for the
  realized candidates.
- **Stream:** ε_seed = `randn(622,329,856, bf16, cuda, Generator.manual_seed(seed))`.
  A candidate is δ_t = σ · ε[:numel_t].view(vLLM shape_t), applied as a BF16
  in-place add.
- **Contrast:** s_j = logP(gold) − logP(r_B) at the first answer position.
  r_B is the BASE strongest wrong option (tie → lowest letter). For RERANK and
  TEST, r_B comes from the stored vLLM base scores. SEARCH has no stored scores,
  so r_B comes from the HF base scores, and this is flagged.
- **Prediction:** t̂_k[j] = σ_k ⟨fold(∇s_j at base), ε_k⟩ (first order; it
  ignores BF16 rounding of the realized add).
- **Candidate lists:** frozen in `candidate_lists.json`:
  - RERANK: 543 candidates;
  - TEST: 61;
  - SEARCH: 1,000 (top 50 + 12 controls + a hash-filled 250 per σ; no outcomes
    read);
  - set-level projections: all 5,000.

## Fidelity falsifiers (stop if triggered)

| Check | Pass condition |
|---|---|
| S1b (vLLM) | For 9504111 and controls 9500059 and 9504179, every one of the 642 realized tensors equals `bf16(base + bf16(σ·ε[:n].view(shape)))` |
| S2a (HF base) | Prompt token IDs identical (SHA-256) for ≥ 99% of 961 examples; argmax equals the stored answer for ≥ 99%; median of per-example max \|ΔlogP_{A–D}\| vs the stored vLLM scores ≤ **0.10 nats** |
| S2b (HF candidate) | All 642 packed-tensor byte hashes equal the vLLM-realized ones; ≥ 99% of the 761 answers agree with the stored candidate |

**Threshold change, recorded before any data.** The research plan said
0.05 nats for S2a. This lock uses **0.10 nats**. Reason: HF SDPA and vLLM
kernels differ in BF16 accumulation, and the predicted quantity t is
typically 1–3 nats, so 0.10 nats is still under 10% of it.

## Primary hypothesis tests (S2c / S2d; computed on CPU after the run)

**F1. Per-example validity.**

- **Data:** the 13 candidates with stored A–D scores (9504111 + 12 random
  controls), on RERANK + TEST.
- **Statistic:** pooled Pearson r between t̂ and the observed
  t = (L_C[y] − L_C[r_B]) − (L_B[y] − L_B[r_B]).
- **Decision:**
  - r ≥ 0.6 supports;
  - r < 0.3 **falsifies** H★.
- **Also reported:** the slope (it should be ≈ 1) and r per candidate.

**F2. Changed answers of 9504111** (85 = RERANK 20 + 4, TEST 23 + 38).

- **Prediction:** a flip when sign(m_B + t̂) ≠ sign(m_B), with m_B the
  base margin against r_B.
- **Decision:** agreement < 50% **falsifies**.

**F3. Across candidates.**

- **Statistic:** Pearson r between predicted accuracy gain (predicted correct
  iff m_B + t̂ > 0) and observed accuracy gain.
- **Decision on RERANK** (543 candidates; the primary test):
  - r < 0.3 **falsifies**;
  - r ≥ 0.5 supports.
- **Also reported:** SEARCH (1,000) and TEST (61).

**F4. Geometry.**

- cos(F_SEARCH, F_RERANK), cos(F_SEARCH, F_TEST) and cos(F_RERANK, F_TEST).
- 9504111's percentile in ⟨F_RERANK, ε⟩ and ⟨F_TEST, ε⟩ among the 5,000.
- Predicted vs observed TEST for 9504111.

These are reported descriptively; the support criteria are in the research
plan, §11.

**S2d. Group attribution.**

- **Statistic:** on the 83 LOCALIZATION examples, Pearson r between the
  per-group first-order predictions σ⟨fold_G(∇s), ε⟩ and the measured
  session-1 single-group insertion shifts
  t_{I_G} = (L_{I_G}[y] − L_{I_G}[r_B]) − (L_B[y] − L_B[r_B]).
- **Decision:** r < 0.5 **falsifies** H★ at group resolution.

**Next steps depend on the outcome:**

- If F1, F2, F3 and S2d all pass: request GPU-B.
- If any of them falsifies: write a NO-GO for the mechanism.

## Budget

- Estimate ≈ $3.5 at $3.49/h. **Hard cap $5.00.**
- The pod guard stops it at $4.90.
- The in-script guard stops before any block that would cross the cap.
- Partial outputs are saved per phase and per block.

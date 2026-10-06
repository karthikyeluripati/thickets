# R2 plan lock: does the Stage-2 law replicate on a second model?

Locked before any R2 tilt measurement (job 01, a stream/inventory check with no tilt outcome, may run first).
User-approved cap: $6.00 for this pod session.

## What changes vs Stage 2 (everything else identical)
| | Stage 2 | R2 |
|---|---|---|
| Model | Qwen3-VL-8B-Instruct (36 layers, DeepStack vision) | **Qwen2.5-VL-7B-Instruct** (28 layers, different vision tower, no DeepStack); revision recorded on the pod |
| Probe / class | P96, "front" | same P96, same class |
| Perturbations | 80 frozen seeds, σ = 0.002 | the same 80 seeds and σ, applied to the new model (new noise realizations per tensor shape) |
| Groups | 6-layer blocks | 4-layer blocks (L00–03 … L24–27), embed, final_norm_head, vision |

## Validity gates (stop if failed)
- **V-stream** (job 01): for 2 seeds, every vLLM tensor after `apply_perturbation` equals
  bf16(base + σ·ε[:numel]) exactly.
- **V-map** (job 02): seed 9504111 rebuilt in HF through the vLLM↔HF mapping is byte-identical to vLLM, for every
  vLLM tensor.
- **V-per-perturbation** (job 03): every perturbation's tensors are byte-exact against the stream; vision reset and
  restore are exact; missing letters ≤ 1%.
- S1a reliability: split-half Spearman–Brown of T_FULL ≥ 0.7.
- P1 content: the letter-position-implied variance share of T_NOV must be < 0.20.

## Decisions
| Test | Statistic | Rule |
|---|---|---|
| **R2 primary** | r(pred_NOV, T_NOV) over the 80 | **≥ 0.6 PASS (REPLICATED); < 0.3 FALSIFIED (NOT REPLICATED); else INCONCLUSIVE**; n < 40 → INCONCLUSIVE |
| L2 localization (secondary) | predicted-variance share (1000 share seeds + 80) of the middle blocks L04–19 (relative depth 0.14–0.71; Stage 3 found 0.17–0.67) | ≥ 0.5 supports; reported either way |
| reported | r(T_NOV, T_FULL)², r(pred_NOV, T_FULL), r(pred_NOV + pred_V, T_FULL), r(pred_NOV, content part), slope | – |

**Interpretable** requires all validity gates. No re-tuning.

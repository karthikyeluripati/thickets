# P0 pilot result: GO on the pre-registered primary; the link to RandOpt's CoT accuracy is ABSENT

Lock: `plan_lock.md` (committed in d5152f7 before output). Qwen2.5-VL-3B @ 66285546, language-model-only perturbation (RandOpt
PERTURB_VISUAL=0); 96 GQA held-out items; 24 candidates (σ 0.001 / 0.002 / 0.005, 8 each).
- Pod spend ≈ $1 of the $10 cap. The first attempt failed at once (my bug: offline mode set before the image download);
  it was fixed and re-queued.

## Gates
- Stream: 580/580 tensors exact for 2 seeds (vision untouched).
- Mapping: 580/580 byte-identical (tied embeddings).
- Base contrast recompute: max diff 6e-6. Restore: exact.
- Base CoT accuracy on the 96 items: **56.25%**. RandOpt's reported 3B GQA testdev baseline is 56.6%, so the setup
  reproduces it.

## Primary (pre-registered): per-question single-token answer contrast is first-order → **GO**
| σ | r(pred, measured Δc) | mean \|Δc\| |
|---|---|---|
| 0.001 | 0.977 | 0.38 |
| 0.002 | 0.915 | 0.66 |
| **pooled σ ≤ 0.002 (1536 pairs)** | **0.930**, slope 0.95 | – |
| 0.005 | 0.306 (breaks down) | 2.26 |

Per-item median r = 0.97; 96/96 items have r ≥ 0.5.

## Secondary (pre-registered, reported) and a post-hoc follow-up
- **Per-candidate CoT gains (pp, 96 items):**
  - σ = 0.001: +1.0, −2.1, −11.5, +8.3, 0, −2.1, +3.1, 0
  - σ = 0.002: −5.2, 0, 0, +3.1, 0, +2.1, −10.4, +7.3
  - σ = 0.005: −56.2, −6.2, −16.7, −12.5, −28.1, −15.6, −55.2, −20.8 (destructive)
- **r(pred Δc, CoT correctness change)** over all pairs = −0.01. Post-hoc, restricted to σ ≤ 0.002:
  - r(pred Δc, CoT change) = −0.04;
  - r(measured Δc, CoT change) = −0.01;
  - sign agreement on changed pairs = 0.52 (chance).
- **Candidate level** (σ ≤ 0.002, n = 16): r(mean pred Δc, CoT gain) = 0.12.
- **Instability:** at σ ≤ 0.002, **22% of item × candidate pairs flip CoT correctness** (338/1536), while the
  single-token contrast moves smoothly by about 0.4–0.7 nats.

## Reading
1. The first-order law holds for the model's **single-token answer scores** in RandOpt's own GQA setting (third model,
   third task; r = 0.93).
2. RandOpt's **actual metric (256-token chain-of-thought accuracy) is not governed by that first-order signal.**
   Tiny perturbations re-route reasoning trajectories, flipping about a fifth of answers with no relation to the
   first-order direction.
3. **"A first-order theory of neural thickets" is therefore NOT supported for CoT-scored tasks.** At most it applies
   to single-token / short-answer evaluation.
4. **New hypothesis (not established):** in CoT settings, a small weight perturbation acts like a random re-sampling of
   the reasoning path. Selecting top-K on 200 items picks lucky re-samples, and majority voting over K perturbed
   models resembles self-consistency (voting over sampled chains). Testable: the per-item flip statistics of
   perturbations vs temperature samples, and RandOpt's top-K vote vs base-model self-consistency at matched K.

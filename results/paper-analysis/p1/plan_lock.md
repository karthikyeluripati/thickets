# P1 pilot lock: are RandOpt perturbations on CoT tasks just re-sampling of the reasoning path?

Locked before any P1 output. User-approved pod, cap **$10**. Motivation: P0 (`../p0/P0_RESULT.md`). Tiny language-model
perturbations flip about 22% of CoT answers, with no relation to the first-order signal.

## Setting
- Qwen2.5-VL-3B @ 66285546, GQA, RandOpt CoT prompt, greedy, max_tokens 256, RandOpt GQAHandler scorer.
- **Items:** selection S = the G1 selection set (200); held-out H = the first 200 of the G1 held-out set. Image-disjoint.
- **Perturbations P:** 64 language-model-only candidates (seed 9710000+k, σ 0.001/0.002 alternating), each answering
  every item greedily.
- **Sampling:** the unperturbed base model at T ∈ {0.3, 0.5, 0.7, 1.0}, n = 32 samples per item (top_p 1, fixed seed).
- **Answer identity** for flips and votes: RandOpt's own normalization (GQAHandler._normalize_answer of the extracted
  answer). Correctness uses RandOpt's compute_reward / _match_answer.
- **Matched temperature T\*** (mechanical): the T whose mean per-item rate of disagreeing with the greedy base answer
  is closest to the perturbations' mean flip rate, over S ∪ H.

## Tests
| Test | Statistic | Reading |
|---|---|---|
| **P1-A: same items flip?** | Spearman over the 400 items between f_P(j) (share of the 64 perturbed models whose answer differs from greedy base) and f_T\*(j) (share of T\* samples that differ) | **≥ 0.6 SUPPORTS re-sampling; < 0.3 AGAINST**; else MIXED |
| **P1-B: persistent expertise?** | r(selection gain, held-out gain) across the 64 perturbed models, Fisher CI. Reference: the same r for 32 pseudo-models built from T\* samples (pseudo-model s answers each item with sample s) | **PERSISTENT** if r ≥ 0.3 and the CI excludes 0; **RESAMPLE-LIKE** if the CI includes 0 and \|r\| < 0.15; else INTERMEDIATE |
| **P1-C: does RandOpt beat self-consistency?** | RandOpt-style top-K by S accuracy (ties → lower index), majority vote of answers on H, vs self-consistency (majority vote of the first K samples at T\*, and at T = 0.7) on H; K = 8 primary (16 and 32 reported). Item-bootstrap CI (2000, seed 20261007) of the difference | **NO ADVANTAGE** if the CI of (RandOpt − SC@T\*) includes 0 or is entirely negative; **ADVANTAGE** if it is entirely positive |
| reported | base greedy accuracy on S and H; vote of K random (unselected) perturbations (mean over 200 random subsets); per-T accuracy and flip rate; σ split | – |

**Overall reading (pre-stated):**
- **"Neural thickets on CoT ≈ self-consistency in weight space"** is supported only if P1-A SUPPORTS, P1-B is not
  PERSISTENT, and P1-C is NO ADVANTAGE.
- If P1-B is PERSISTENT or P1-C is ADVANTAGE, perturbations carry something beyond re-sampling.

No re-tuning of the items, temperatures, K or thresholds.

# G1 plan lock: GQA in the RandOpt / Neural Thickets setting, with matched splits

Locked before any GQA model output. The user authorized the GQA study ("if it's best for the paper, go ahead").
**Proposed hard cap: $30** for one 8×H100 pod session; the cap needs the user's go when the pod is requested.

## Setting (faithful to the Neural Thickets GQA experiment wherever possible)
- **Model:** Qwen/Qwen2.5-VL-3B-Instruct @ 66285546d2b821cf421d4f5eb2576359d3770cd3 (the earlier reproduction branch's pin).
- **Perturbation:** the pinned RandOpt WorkerExtension (4000d34) with **PERTURB_VISUAL=0**, i.e. language model only,
  vision frozen (the paper's GQA setting).
- **σ:** RandOpt's default mixture [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01].
- **Candidates:** 480 total: seed 9600000+i, σ = mixture[i mod 6], 80 per σ. This is a reduced population (the
  paper uses 5000).
- **Prompt and scoring:** RandOpt's GQA chain-of-thought prompt ("…reason step by step, and put your final answer
  within \boxed{}"), greedy, max_tokens 256, scored by **RandOpt's own GQAHandler.compute_reward**.
- **Direct probe (mechanism only):** "Answer with a single word." on yes/no items. It records first-token log-probs of
  "Yes" (9454) and "No" (2753).
- **Data:** GQA testdev-balanced (lmms-lab-encoder/GQA @ a6e72d6e…), **split by image**.
  - **Selection:** 200 questions, 132 images (yes 36 / no 38).
  - **Held-out (HO):** 400 questions, 170 images (yes 86 / no 61).
  - The two sets share no images. Rule in `frozen_items.json`.
- **Deviation from the paper (deliberate):** the paper selects on train[:200] and tests on all of testdev. We draw
  selection and held-out from the same split, image-disjoint, so that transfer is measured within distribution.

## Validity gates
| Gate | Rule |
|---|---|
| V-stream (GPU0, g1_check) | for 2 seeds, every language-model tensor equals bf16(base + σ·ε[:numel]) and every vision tensor equals base exactly |
| V-map (g1_gradient) | seed 9600003 rebuilt in HF is byte-identical for every vLLM tensor, else the τ tests are void (evaluation unaffected) |
| V-fid | every candidate has vision unchanged and an exact base restore; any failure stops that worker |
| V-det | BASE correctness identical across all GPU workers on ≥ 99% of items |

## Locked tests
Gains are 100·(acc_cand − acc_base); CIs are image-cluster bootstrap (2000, seed 20261006).

| Test | Statistic | Reading |
|---|---|---|
| **G1-A inflation (primary)** | Winner = argmax selection gain (ties → lowest index). Its HO gain with CI. | **INFLATED** if HO gain ≤ 0.5 × selection gain and the CI upper bound < selection gain; **TRANSFERS** if HO gain ≥ 0.8 × selection gain and the CI lower bound > 0; else PARTIAL. Also reported: top-10 mean selection vs HO; r(selection gain, HO gain) over all 480 and per σ. |
| **G1-B tilt law** | T = mean over HO yes/no items of Δ[lp(Yes) − lp(No)] (direct probe; missing → top-20 floor) vs τ_HO = Σ_groups σ⟨F_g, ε⟩, over candidates with σ ≤ 0.002 (n = 320) | **≥ 0.6 PASS; < 0.3 FALSIFIED; else INCONCLUSIVE**. Per-σ r reported, including 0.005 and 0.01. |
| **G1-C prior shift** | yes_shift = Δ(share of HO yes/no items whose extracted CoT answer is "yes"). Across all 480: R² of HO gain on yes_shift; r(yes_shift, gain on gold-yes items); r(yes_shift, gain on gold-no items) | **PRIOR-DOMINATED** if R² ≥ 0.30; **MINOR** if R² < 0.10; else MODERATE. Opposite-sign gold-yes / gold-no correlations are the prior signature. |
| G1-D | r(τ_SEL, HO gain); r(τ_SEL, yes_shift) | reported |
| G1-E | majority vote of the top-10 by selection (extracted answers) on HO vs base | reported |
| reported | base yes-rate vs gold yes-rate on HO; per-σ mean and SD of gains; tokens generated | – |

No re-tuning of the items, candidates, prompts, scoring or thresholds. If the run stops early under the budget guard,
every test uses the candidates measured. G1-B needs ≥ 100 candidates with σ ≤ 0.002, and G1-C needs ≥ 200, else
INCONCLUSIVE.

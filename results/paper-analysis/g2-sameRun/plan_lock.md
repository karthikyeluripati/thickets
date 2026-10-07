# G2 plan lock: same-run RandOpt (N = 5000) vs self-consistency on GQA / Qwen2.5-VL-3B-Instruct

Committed before any G2 output. Runs on the 6× H100 pod after C3B (session cap $120; G2 projection gate **$45 of
spend for G2 itself**, see below).

## Why not RandOpt's own script
RandOpt's released `randopt.py` (4000d34 = upstream HEAD on 2026-10-07) builds text-only prompts and never passes
image data or `multimodal=True` to vLLM, so it cannot run GQA with images. G2 therefore re-implements the
`randopt.py` loop on the pinned RandOpt components: `utils.worker_extn.WorkerExtension` (perturbation; PERTURB_VISUAL=0,
the paper's GQA setting), `data_handlers.gqa.GQAHandler` (prompt text, reward, vote key, vote correctness). Runner:
`scripts/g2_randopt_gqa.py`; data prep `scripts/g2_prep.py`; analysis `scripts/g2_analysis.py`.

## Faithful to randopt.py
- Seeds and σ: `rng = np.random.default_rng(42)`; seeds = rng.choice(2**31, 5000, replace=False); σ =
  rng.choice([0.0005, 0.001, 0.002], 5000) (randopt.py's generator; the σ set of its paper run script, as in C/C3B).
- Selection reward = mean `compute_reward` over 200 selection questions, greedy (temperature 0, seed 42),
  max_tokens 256. Ranking = sort by reward descending, ties in seed-generation order. Top K = 50 (and 10).
- Test: each top-50 model greedy on the test questions; vote = `Counter.most_common` over non-empty
  `extract_answer_for_voting`; correct iff `is_voted_answer_correct(vote, gt)`.
- Deviations (stated): weights reset exactly from a stored base copy (`apply_perturbation`/`reset_to_base_weights`)
  rather than randopt.py's subtract-to-restore; in-process vLLM per GPU (uni executor), not Ray.

## Data (lmms-lab-encoder/GQA @ a6e72d6e…, testdev-balanced)
- Model Qwen/Qwen2.5-VL-3B-Instruct @ 66285546d2b821cf421d4f5eb2576359d3770cd3.
- Selection: the 200 frozen G1 selection questions (`g1/frozen_items.json`). (The paper selects on GQA train[:200];
  deviation as in G1.)
- Test: P2's 2000-question hash sample (SHA256('p2-gqa-v1:'+id), first 2000) **minus every question whose image is
  used by a selection question**; the resulting count is reported.
- Prompt: RandOpt's GQA CoT text, Qwen2.5-VL chat template with one image (identical to P2's).

## Arms
- RandOpt arm: as above.
- **Primary SC arm = P2's GQA SC run** (`p2/pod/gqa_vl3_T0.7.json.gz`, same model revision, same pinned environment,
  same prompt, 50 samples at T = 0.7), fixed now before any G2 output; its stored answers are mapped to RandOpt's vote
  key by the handler's own normalization (extract_answer_for_voting = normalize + singularize + canonicalize of
  extract_answer), restricted to the test questions.

## Gates
Smoke (24 perturbations, 6 workers; also asserts vision tensors unchanged and exact restore). Projection: spent so
far for G2 + (5000/6)·(sec per perturbation per worker)·1.05·rate + $3 ≤ $45, else do not start (ask the user).
Validity: base greedy test accuracy from G2 equals P2's greedy on the test questions within 1.0 pp (same software);
otherwise G2-1 is reported as INVALID-ENV.

## Primary (G2-1), same rules as C-1
D = acc(RandOpt K = 50 vote) − acc(SC@50 vote), paired item bootstrap (10,000, seed 0) over the test questions.
**RANDOPT AHEAD** if CI lower > 0; **SC AHEAD** if CI upper < 0; else **NO DIFFERENCE DETECTED**; plus **EQUIVALENT
(±2 pp)** if the CI lies within [−2, 2]. Image-cluster bootstrap CI reported as secondary.

## Secondary
K = 10; RandOpt's gain over base vs the paper's +12.4 (56.6 → 69.0; different test set and selection set: descriptive);
member-vs-vote decomposition; base selection reward; σ of the top 50.

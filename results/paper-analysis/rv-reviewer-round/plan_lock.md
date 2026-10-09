# RV plan lock: the reviewer round (six items, one pod session)

Committed before any RV output. One 4× H100 pod; session cap **$115** (pod guard; `pod/rate.txt`). Jobs run in the
order below (`jobs/01`–`07`); the cheap items first so an early stop keeps them. Runner changes are additive:
`o2_eval.py` and `g3_eval.py` gain templates and `--temperature` (defaults unchanged); `gsm_randopt_fast.py` unchanged;
`pod_rv_common.sh` is the launcher earlier sessions kept on the pod only.

## Why
A strict reviewer of the Section 3 draft can still write: (1) the three prompt candidates were chosen after seeing
results; (2) "prompt held fixed" is shown in 2 of 4 rows; (3) one search per GSM8K row, the second OLMo seed skipped;
(4) GB-2's gate failure has no established cause; (5) SC@50 was run at one temperature. RV answers each with a test
whose outcome is fixed here.

## RV-1: public templates, chosen on the selection questions (candidate independence)
- **New candidates (verbatim public evaluation templates, never run before this lock):** GSM8K `harness_cot`
  (lm-evaluation-harness `gsm8k_cot_zeroshot`: "Q: {q}\nA: Let's think step by step."), `harness_plain`
  (lm-evaluation-harness `gsm8k`, zero-shot: "Question: {q}\nAnswer:"), `simple_evals` (openai/simple-evals math
  template). GQA `llava` (LLaVA-1.5 / lmms-eval: "{q}\nAnswer the question using a single word or phrase."), `blip`
  (BLIP-2 / InstructBLIP: "Question: {q} Short answer:"). Exact strings: `o2_eval.PROMPTS`, `g3_eval.PROMPTS`.
- **Rule (primary):** per row, chosen = argmax greedy BASE accuracy on RandOpt's 200 selection questions **among the
  new templates only**; ties by listed order (`rv_choose.py`, run on the pod). RandOpt's scorers (GSM8K: lenient
  extract; GQA: `is_answer_correct` on the text, max_tokens 256), exactly PS.
- **Then** SC@50 (T = 0.7, seed 20261007, as PS) and one greedy generation under the chosen template on test.
- **Primary (per row):** D = acc(RandOpt K = 50, same run, its own prompt) − acc(SC@50, chosen new template); paired
  item bootstrap (10,000, seed 0). RANDOPT AHEAD / SC AHEAD / NO DIFFERENCE DETECTED; EQUIVALENT if the CI lies within
  ±2 pp.
- **Secondary:** the pooled rule over all candidates (three PS + new) and its D; K = 10; RandOpt − one generation;
  selection accuracies of every new template.
- **How it is used:** SC AHEAD or NO DIFFERENCE in a row → the headline for that row does not depend on our candidate
  choice (stated as "a template set fixed from public harnesses before the run gives the same ordering"). RANDOPT
  AHEAD in a row → reported as such: in that row the headline rests on the PS candidates and the paper says so.

## RV-2a / RV-2b: RandOpt's search under the boxed prompt, Qwen2.5-3B (2a) and Qwen2.5-1.5B (2b)
Design exactly GB (`gsm_randopt_fast.py`, population seed 42 = C3B's / C's perturbations, boxed prompt, K = 50, 10
secondary; 3 selection workers per GPU, 1 test worker per GPU). Comparators: SC@50 boxed (PS `q3/boxed_sc.json`,
`q15/boxed_sc.json`), boxed BASE (Q2), RandOpt-prompt ensembles (C3B, C) and their top-50 seeds (`top_k_seeds.json`).
- **Gates (per row; `rv_gates.py`):** ENV (blocking): this runner's greedy BASE selection accuracy under RandOpt's
  prompt within 1.0 pp of PS's independent measurement (84.0; 68.5), under the boxed prompt within 1.0 pp of PS (90.0;
  80.5), and `o2_eval.py`'s boxed BASE test accuracy within 1.0 pp of Q2 (82.34; 70.20). FIDELITY (reported): rewards
  of perturbations k < 24 under RandOpt's prompt vs randopt.py's C3B / C logs, consistent if mean |Δ| ≤ 0.03.
  BUDGET (blocking): after a 180-perturbation smoke, spent + remaining/W × s × 1.10 × rate + 51/G × 60 s × rate + $2
  ≤ cap, else not run (no silent reduction of N).
- **Why the ENV gate no longer uses randopt.py's single printed base reward:** GB's gate compared our runner with the
  one base number randopt.py printed in C (73.00) and failed (68.00; PS 68.5). The perturbed rewards of the same 24
  perturbations agree between the two implementations with no offset (signed mean −0.001, mean |Δ| 0.018; OLMo
  −0.005, 0.0135), and C3B's print (85.50) also sits 1.5 pp above PS's 84.0 while OLMo's matched exactly. The
  faithful-implementation test is the perturbed-reward fidelity against randopt.py's own log; the printed base number's
  reproducibility is what RV-4 measures. GB-2 stays INVALID-ENV in the ledger as recorded.
- **Primary (per row):** D = acc(RandOpt K = 50, searched and voted under boxed) − acc(SC@50 boxed); the GB rule.
- **Secondary:** as GB (members vs boxed BASE; RandOpt boxed − RandOpt own prompt; top-50 overlap; selection gain vs
  test gain; K = 10). Analysis `gb_analysis.py` (self-test recorded in the GB lock), one call per row.
- **How it is used:** NO DIFFERENCE / SC AHEAD → "with the prompt held fixed, weight search adds nothing measurable" is
  shown in 3 (4) of 4 rows. RANDOPT AHEAD → stated for that row as the measured amount weight search adds on top of the
  prompt; Section 3.5 and the abstract change accordingly.

## RV-3: a second OLMo-2-1B search under RandOpt's own prompt (population seed 43, as G2R)
Fast runner (fidelity to randopt.py established in GB: mean |Δ| 0.0135 on this model); exact weight reset instead of
randopt.py's subtract-to-restore (stated). ENV gate: BASE selection accuracy under RandOpt's prompt within 1.0 pp of
41.5 (PS; randopt.py printed 41.50), `o2_eval.py` BASE test within 1.0 pp of O2's 33.66. Budget rule as above.
- **Primary:** D = acc(RandOpt K = 50, seed 43) − acc(SC@50, RandOpt's prompt, O1's samples); the O1 rule.
- **Secondary:** seed-to-seed difference vs O1's search (paired CI); members vs BASE; K = 10; selection gain vs test gain.
- **How it is used:** RANDOPT AHEAD → the OLMo row's "RandOpt ahead under its own prompt" is replicated (two seeds,
  two implementations); otherwise the row is reported as seed-dependent and Section 3.1 says so.

## RV-4: does randopt.py reproduce its own printed base reward for Qwen2.5-1.5B?
randopt.py @ 4000d34, exactly C's smoke invocation (population 24, 200 selection rows, 30 test rows), twice on this pod:
A with 4 engines, B with 1 engine; plus our runner's base under RandOpt's prompt (k < 24 pass, reused by RV-2b).
- **Reading (fixed):** if A and B are within 1.0 pp of our runner (≈ 68) → C's 73.00 was not reproducible by
  randopt.py itself; the exploratory statements that use it as the base (R8b.1 "population mean below base", the
  10.7% share above base, R8b.7's Qwen-1.5B selection gain) are recomputed with the measured base and corrected. If A
  and B are within 1.0 pp of 73.00 → randopt.py and our runner disagree on this model's base; reported as a fidelity
  limitation for Qwen-1.5B; RV-2b still runs under its own gates (above) because its comparison does not use that
  number. Otherwise → both reported; nothing corrected.

## RV-5: SC@50 temperature (the two no-difference rows)
SC@50 at T = 0.5 and T = 1.0 (top_p 1, seed 20261007, max_tokens as PS) under the PS-chosen prompt: Qwen2.5-3B boxed
(test) and GQA direct (256 tokens). D vs the same-run RandOpt K = 50 per T, the PS rule.
- **How it is used (fixed):** T = 0.7 remains the locked comparison in every row. If RandOpt is AHEAD at some T, the
  row is reported as temperature-dependent in Section 3.4; if SC is AHEAD at some T, the headline is **not** upgraded
  (appendix only).

## Validity and reporting
`pod/` keeps every log, `gates.txt`, `rate.txt`, `gpu.txt`, `pip-freeze.txt`. Every item is reported in the ledger
with its outcome, including gate failures and budget skips. Analysis: `scripts/rv_analysis.py` (RV-1/3/4/5) and
`scripts/gb_analysis.py` (RV-2a/2b). Self-test (`--selftest`, PS's candidates standing in for the new templates and
GB's OLMo run for RV-3) reproduces PS's four rows (−2.96 [−4.62, −1.29]; −0.38 [−1.67, +0.91]; −23.65 [−26.23,
−21.08]; −1.21 [−2.83, +0.40]) and GB's RandOpt (boxed) − RandOpt (own prompt) +21.53 [+18.88, +24.18], members
67.05: `selftest/rv_selftest.json`.

## Cost (estimates from GB's and GB-2's measured rates; the smoke projections decide)
RV-1 + RV-5 ≈ $7; RV-4 ≈ $3; RV-2a ≈ $21–30; RV-3 ≈ $13; RV-2b ≈ $13–15 (GB-2's own projection); prep ≈ $4.

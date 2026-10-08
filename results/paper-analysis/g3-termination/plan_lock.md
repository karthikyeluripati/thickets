# G3 plan lock: is RandOpt's GQA advantage a termination / answer-format repair?

Committed before any G3 output. 1× H100 ($3.49/h), user-provided pod, hard cap **$12**. Follows G2
(`../g2-sameRun/G2_RESULT.md`): RandOpt (N = 5000, K = 50) beat SC@50 on GQA by +3.47 [+1.62, +5.41]; an
EXPLORATORY keyword heuristic suggested the gain is on questions where the base model ends without a usable answer.
G3 tests that directly.

## Fixed setup (identical to G2 except the token budget and what is saved)
- Qwen/Qwen2.5-VL-3B-Instruct @ 66285546; G2's 1238 test questions (`g2-sameRun/pod/g2/items.json`); RandOpt GQA CoT
  prompt; RandOpt `GQAHandler` (reward, vote key, vote correctness); PERTURB_VISUAL=0; vLLM 0.11 in-process, CUDA
  graphs, 1 worker.
- Arms: **BASE** greedy; **MEMBERS** = G2's top-50 perturbations (`g2-sameRun/pod/g2/out/topk.json`, by rank), each
  greedy; **SC** = 50 samples at T = 0.7, top_p 1.0, seed 20261007 (P2's sampling settings).
- **Budgets:** max_tokens **256** (G2's) and **1024**.
- Saved per generation: vote key, correctness, finish reason, generated tokens, whether the text contains `\boxed{`.
  Texts saved (gzipped) for BASE and MEMBERS.
- Runner `scripts/g3_eval.py`; analysis `scripts/g3_analysis.py`.

## Definitions
- **Non-terminated** generation: finish reason = "length" (hit max_tokens) **or** no `\boxed{` in the text.
- Votes: RandOpt's rule for both RandOpt (over the 50 members) and SC (over its 50 samples), as in G2.
- D_b = acc(RandOpt K = 50 vote) − acc(SC@50 vote) at budget b, over the 1238 questions.

## Primary G3-1: does a longer budget remove RandOpt's advantage?
Paired item bootstrap (10,000, seed 0) of Δ = D_256 − D_1024 and of D_1024.
- **TERMINATION ACCOUNT SUPPORTED** if Δ's CI lower bound > 0 **and** D_1024's CI includes 0 or lies below 0.
- **NOT SUPPORTED** if Δ's CI includes 0 (the advantage does not change with budget).
- **PARTIAL** otherwise (the advantage shrinks but persists).

## Secondary G3-2: do selected perturbations terminate more often at 256?
Non-terminated rate of BASE minus mean non-terminated rate of MEMBERS at 256, item bootstrap CI.
**Supported** if the CI lower bound > 0. Reported: the same at 1024; the SC rate at both budgets.

## Reported (no rule)
Accuracies of BASE, member mean, RandOpt vote and SC vote at both budgets; K = 10; D_b split by whether BASE
terminated at 256; mean generated tokens per arm and budget; image-cluster CIs.

## Validity
G3's BASE at 256 must reproduce G2's BASE accuracy within 0.5 pp and G3's RandOpt vote at 256 G2's within 1.0 pp
(same software); otherwise G3-1 is reported as INVALID-ENV. A smoke run (5 questions × BASE + 2 MEMBERS + SC, both
budgets) must pass first; the full run starts only if its projected cost keeps the session ≤ $11.50.

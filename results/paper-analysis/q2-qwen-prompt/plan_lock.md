# Q2 plan lock: the plain-prompt control on the two Qwen GSM8K rows (where RandOpt lost to SC)

Committed before any Q2 output. 1× H100; session cap **$6.50** (pod guard).

## Why
Prompt controls so far ran only where RandOpt beat SC (G4 GQA, O2 OLMo), where RandOpt's prompt cost the base model
11–31 points. A one-sided control invites the question whether a better prompt changes the Qwen rows too. Q2 runs the
same control on Qwen2.5-1.5B (C) and 3B (C3B).

## Design (as O2)
`scripts/o2_eval.py` unchanged: GSM8K test (1319), chat template, max_tokens 1024, greedy, and SC with 50 samples at
T = 0.7, seed 20261007; RandOpt's GSM8K scorer for every arm (the analysis rescores saved answers; the runner's own
`c` field is not used); texts saved. Models: `Qwen/Qwen2.5-1.5B-Instruct` @ 989aa7980e4cf806f80c7fef2b1adb7bc71aa306
(tag q15) and `Qwen/Qwen2.5-3B-Instruct` @ aa8e72537993ba99e69dfaafa59ed015b17504d1 (tag q3). Per model, in this order:
randopt-prompt BASE (environment gate), plain BASE, plain SC@50, boxed BASE. Prompts as O2.
RandOpt K = 50 for each row: the same-run dumps of C (`c-sameRun/pod6/c/out`) and C3B (`c3b-sameRun/pod/c3b/out`).

## Validity (per row)
Randopt-prompt BASE greedy equals the same-run greedy run (C: `gsm8k_c15_greedy`, C3B: `gsm8k_c3b_greedy`; both rescored
with RandOpt's scorer) within 1.0 pp; else that row is INVALID-ENV.

## Primary (one test per row: Q2-1 = 1.5B, Q2-2 = 3B; reported separately, no pooling)
D = acc(RandOpt K = 50, same run, RandOpt's prompt) − acc(SC@50, **plain** prompt); paired item bootstrap (10,000,
seed 0). **RANDOPT AHEAD** if CI lower > 0; **PROMPT-SC AHEAD** if CI upper < 0; else **NO DIFFERENCE DETECTED**; plus
**EQUIVALENT** if the CI lies within ±2 pp.

## Secondary (reported regardless)
Prompt damage = plain BASE − RandOpt-prompt BASE (paired CI); boxed BASE; RandOpt − one plain BASE generation;
SC@50 (plain) − SC@50 (RandOpt's prompt); SC@10; format per arm (tokens, "####", \boxed, max_tokens).

## Prediction stated in advance (descriptive, across the four same-run rows)
"RandOpt beats SC under its own prompt only where that prompt damages the base model": on GQA (damage +11.3) and
OLMo (+31.6) RandOpt was ahead; on the Qwen GSM8K rows it was behind. Q2 reports the Qwen rows' damage; the prediction
is supported if both are small (< 5 pp) and contradicted if either is ≥ 10 pp. Reported either way.

## Analysis
`scripts/q2_analysis.py`. Self-test with every Q2 arm replaced by the same-run SC/greedy data reproduces C (−2.65
[−4.32, −0.99]) and C3B (−1.59 [−2.65, −0.53]), environment difference 0.

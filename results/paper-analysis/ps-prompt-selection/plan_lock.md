# PS plan lock: prompt selection on RandOpt's own selection set vs RandOpt's weight search (all four same-run rows)

Committed before any PS output. 1× H100; session cap **$5** (pod guard).

## Why
Q2 showed the better prompt is model-dependent (plain helps OLMo and GQA, hurts Qwen). The fair practical baseline is
to choose the prompt the way RandOpt chooses perturbations: on its own 200 selection questions. PS asks whether
"prompt search" (3 candidates, 600 greedy generations) matches RandOpt's weight search (5000 perturbations, 1,000,000
greedy generations) in each row.

## Rule (fixed now)
Per row, the chosen prompt = argmax of greedy BASE accuracy on RandOpt's selection set; ties broken by candidate
order. GSM8K rows (Qwen2.5-1.5B, Qwen2.5-3B, OLMo-2-1B): candidates (randopt, plain, boxed) as O2/Q2; selection set =
the first 200 GSM8K train rows (randopt.py's `train_samples`; verified equal to its parquet), RandOpt's scorer. GQA row
(Qwen2.5-VL-3B): candidates (cot, direct, short) as G4; selection set = G2's 200 selection questions; G3's scorer
(`is_answer_correct`), max_tokens 256.

## Runs
Selection-set greedy BASE for every candidate and row (`o2_eval.py --split select`, `g3_eval.py --split selection`;
runners otherwise unchanged; texts saved). Test-set SC@50 already exists for every candidate except boxed on the Qwen
rows; boxed SC@50 on test for a Qwen row is run **only if** boxed is chosen for that row (same rule, computed on the
pod), then the analysis runs.

## Primary (one per row, reported separately; no pooling)
D = acc(RandOpt K = 50, same run, RandOpt's prompt) − acc(SC@50 under the chosen prompt), test set, paired item
bootstrap (10,000, seed 0). **RANDOPT AHEAD** if CI lower > 0; **PROMPT-SELECTED SC AHEAD** if CI upper < 0; else
**NO DIFFERENCE DETECTED**; plus **EQUIVALENT** if the CI lies within ±2 pp.
Test files: C/C3B/O1 SC and RandOpt dumps; Q2/O2 plain/boxed outputs; G3 (CoT) and G4 (direct, short) outputs.

## Secondary
Selection accuracy per candidate; RandOpt − one chosen-prompt BASE generation on test; generation budgets.

## How it is used
If prompt-selected SC is ahead or not different in every row → "choosing a prompt on RandOpt's own selection data, at
1/1600 of its selection compute, matches or beats its weight search in all four rows". Any RANDOPT AHEAD row is reported
as such and qualifies that claim.

## Analysis
`scripts/ps_analysis.py`. Self-tests: when the default prompts win selection it reproduces C (−2.65 [−4.32, −0.99]),
C3B (−1.59 [−2.65, −0.53]), O1 (+8.72 [+6.75, +10.77]) and G3 (+3.63 [+1.70, +5.57]); when the alternatives win, Q2
(+2.58, +7.05), O2 (−22.21 [−24.87, −19.56]) and G4 (−1.21 [−2.83, +0.40]).

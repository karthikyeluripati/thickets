# Q2 result: on the Qwen GSM8K rows, RandOpt's prompt does not hurt the model; the damage prediction holds on all four rows

Lock 2a55ccc (before any Q2 output). Analysis `scripts/q2_analysis.py` → `pod/q2/q2_results.json`. 1× H100, ≈ $3.3.
Engineering: a CDN read timeout during the 3B download was retried automatically. The runner's stored `c` field is
not used (as O2); the analysis rescores saved answers.

## Validity
RandOpt-prompt BASE vs the same-run greedy (RandOpt's scorer): 1.5B 60.05 vs 59.97, 3B 80.14 vs 80.36 → both **VALID**.

## Primary (per row; RandOpt K = 50, same run, RandOpt's prompt − SC@50, plain prompt)
- **Q2-1, Qwen2.5-1.5B: +2.58 [+0.53, +4.62] → RANDOPT AHEAD**
- **Q2-2, Qwen2.5-3B: +7.05 [+5.23, +8.95] → RANDOPT AHEAD**

## Secondary
| GSM8K | BASE, RandOpt prompt | BASE, plain | BASE, boxed | SC@50, RandOpt prompt (C/C3B) | SC@50, plain | RandOpt K = 50 |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | 60.05 | 64.75 | 70.20 | **79.83** | 74.60 | 77.18 |
| Qwen2.5-3B | 80.14 | 72.93 | 82.34 | **88.25** | 79.61 | 86.66 |

- Prompt damage (plain − RandOpt-prompt BASE): 1.5B **+4.70 [+1.97, +7.43]**, 3B **−7.20 [−9.55, −4.78]** (RandOpt's
  prompt helps the 3B model).
- SC@50 plain − SC@50 RandOpt prompt: 1.5B −5.23 [−7.05, −3.41], 3B −8.64 [−10.46, −6.90]: for Qwen, sampling votes
  better under RandOpt's prompt.
- RandOpt − one plain BASE generation: +12.43 (1.5B), +13.72 (3B).
- Qwen mostly does not follow the "####" instruction at 1.5B (15.9% of answers; 28.3% use \boxed) and partly at 3B (82.7%).

## Prediction stated in advance: SUPPORTED
"RandOpt beats SC under its own prompt only where that prompt damages the base model." Damage on the Qwen rows is
+4.7 and −7.2 (both < 5 pp, the locked "supported" criterion) and RandOpt lost to SC there; on GQA (+11.3) and OLMo
(+31.6) it won. 4 of 4 same-run rows fit.

## Reading
The plain prompt is not a universal fix: for Qwen it is worse than RandOpt's own prompt, and plain-prompt SC falls
below RandOpt. What holds across all four rows is (a) the damage pattern above and (b) that in every row a baseline
without weight search, SC under one of the two prompts fixed in advance (RandOpt's or a plain/direct one), matches or
beats RandOpt. Which prompt is better is model-dependent and was not selected on test data here; a practical
"choose the prompt on the selection set" baseline is the natural next control (not run).

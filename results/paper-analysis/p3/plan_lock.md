# P3 lock: does self-consistency match RandOpt on more of the paper's Table 4 rows?

Locked before any P3 output. User-approved pod, cap **$5**. Same protocol as P2 (`../p2/plan_lock.md`): RandOpt's
prompts and scorers, greedy, plus SC with n = 50 at **T = 0.7** (T = 0.3 dropped to fit the budget), top_p 1, fixed seed.

| Row (priority order) | paper Base | paper TT-MV | **paper RandOpt** |
|---|---|---|---|
| GSM8K / Qwen2.5-3B-Inst (1319) | 79.8 | 82.5 | **87.1** |
| MATH-500 / Qwen2.5-1.5B-Inst (500) | 43.2 | 50.0 | **59.7** |
| MATH-500 / Qwen2.5-3B-Inst (500) | 58.6 | 60.8 | **68.7** |

**MATH-500 protocol:** RandOpt's math500 handler instruction (problem + "\n\n" + "Let's think step by step and output
the final answer after ####"), max_tokens 2048, MATH500Handler.compute_reward (strict then flexible), used unchanged.

**Vote:** majority over handler.extract_answer strings.
- GSM8K: the voted answer is correct iff it equals the gold number (as P2).
- MATH-500: the voted answer is correct iff the first sample carrying that extracted answer was scored correct.

**Gate and decisions are identical to P2:**
- The gate is greedy within ±2.0 pp of the paper's Base, else the row is NOT COMPARABLE.
- **MATCHES** if SC@50 ≥ RandOpt − 1.0; **RANDOPT AHEAD** if ≤ RandOpt − 3.0; else **CLOSE**.

Rows are run in the order above. Any row not finished under the cap is reported as not run.

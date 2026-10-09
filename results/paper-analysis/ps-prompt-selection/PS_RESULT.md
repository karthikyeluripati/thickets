# PS result: choosing the prompt on RandOpt's own selection data matches or beats its weight search in all four rows

Lock a1caf88 (before any PS output). Analysis `scripts/ps_analysis.py` (run locally on the pulled selection outputs and
the committed test outputs of C, C3B, O1, O2, Q2, G3, G4) → `ps_results.json`. 1× H100, ≈ $3.6. No engineering events.
Prep checks: GSM8K hashes as C/C3B; GQA selection questions identical to G2's (200).

## Prompt chosen on the selection set (greedy BASE accuracy, %)
| Row | RandOpt's prompt | plain / direct | boxed / short | chosen |
|---|---|---|---|---|
| GSM8K Qwen2.5-1.5B | 68.5 | 73.0 | **80.5** | boxed |
| GSM8K Qwen2.5-3B | 84.0 | 85.0 | **90.0** | boxed |
| GSM8K OLMo-2-1B | 41.5 | 77.0 | **79.5** | boxed |
| GQA Qwen2.5-VL-3B | 53.5 (CoT) | **66.0** (direct) | 62.0 (short) | direct |
Boxed SC@50 on test was run for both Qwen rows (required by the rule).

## Primary (per row): RandOpt K = 50 − SC@50 under the chosen prompt, test set
| Row | RandOpt | SC@50, chosen prompt | D [95% CI] | outcome |
|---|---|---|---|---|
| GSM8K Qwen2.5-1.5B | 77.18 | 80.14 | **−2.96 [−4.62, −1.29]** | **PROMPT-SELECTED SC AHEAD** |
| GSM8K Qwen2.5-3B | 86.66 | 87.04 | **−0.38 [−1.67, +0.91]** | NO DIFFERENCE DETECTED (**EQUIVALENT** within 2 pp) |
| GSM8K OLMo-2-1B | 52.46 | 76.12 | **−23.65 [−26.23, −21.08]** | **PROMPT-SELECTED SC AHEAD** |
| GQA Qwen2.5-VL-3B | 63.49 | 64.70 | **−1.21 [−2.83, +0.40]** | NO DIFFERENCE DETECTED |

**No row is RANDOPT AHEAD.** Per the lock: choosing a prompt on RandOpt's own selection data (3 candidates × 200 = 600
greedy generations, vs 1,000,000 for RandOpt's search) matches or beats its weight search in all four rows.

## Secondary: RandOpt vs ONE greedy generation under the chosen prompt
Qwen-1.5B +6.97 [+4.85, +9.10]; Qwen-3B +4.32 [+2.65, +6.07]; OLMo −15.39 [−18.20, −12.43]; GQA −1.21 [−2.83, +0.48].
On the Qwen rows RandOpt's 50-model vote still beats a single generation; the comparison that matches its 50
test-time generations is SC@50 (primary).

## Reading
The prompt is the dominant lever in every row: the selection set prefers a non-default prompt for all four models, and
with it, sampling the unperturbed model matches or beats RandOpt. Weight search's measurable advantages over SC (GQA,
OLMo) disappear once the prompt is chosen on the same data RandOpt selects perturbations on.

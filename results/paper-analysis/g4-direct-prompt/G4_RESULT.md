# G4 result: on GQA, one direct-answer generation from the base model matches RandOpt

Lock 2ad4e46 (before any G4 output). Analysis `scripts/g4_analysis.py` → `pod/g4/g4_results.json`. 1× H100, ≈ $3.
No engineering events.

## Validity
G4's CoT-prompt BASE equals G3's (53.39%, Δ 0.00) → **VALID**.

## Primary G4-1
D = RandOpt K = 50 (CoT prompt, seed 42) − SC@50 (direct prompt) = **−1.21 pp [−2.83, +0.40] → NO DIFFERENCE
DETECTED** (not EQUIVALENT: lower bound past −2). Point estimate favours the prompt.

## Secondary
| GQA, 1238 questions, max_tokens 256 | CoT prompt (RandOpt's) | direct prompt | short prompt |
|---|---|---|---|
| BASE greedy (1 generation) | 53.39 | **64.70** | 64.78 |
| SC@50 | 59.85 | 64.70 | 64.05 |
| G2 members, mean of 50 | 58.39 | 62.80 | – |
| RandOpt K = 50 vote (seed 42) | 63.49 | 64.62 | – |
| mean tokens, BASE | 147 | 3.7 | – |

- Prompt effect on BASE: +11.31 pp [+8.48, +14.14] (direct − CoT); on SC@50: +4.85 [+2.67, +7.03].
- RandOpt (CoT) − one direct BASE generation: −1.21 [−2.83, +0.48]; short prompt −1.29 [−2.99, +0.48].
- Seed-43 RandOpt − SC@50 (direct): −1.05 [−2.75, +0.65]. K = 10: −0.57 [−2.18, +1.05]. Short prompt (K = 50):
  −0.57 [−2.34, +1.29]. All NO DIFFERENCE DETECTED.
- **Search on top of the prompt:** the selected members evaluated with the direct prompt vote to 64.62 vs SC@50 (direct)
  64.70: D = −0.08 [−0.97, +0.81]. Individually the members are 1.9 pp below the direct-prompt base (62.80 vs 64.70).
- Form: under the direct prompt every arm answers in about 4 tokens; a `\boxed{}` is present in 31–43% (the handler
  extracts unboxed short answers; same scorer for all arms).

## Reading
RandOpt's GQA advantage over self-consistency (G2/G2R, +3.5 pp) exists only under RandOpt's own chain-of-thought
prompt, which costs this model 11 points. Asked to answer directly, the base model with **one** greedy generation and
no search scores 64.7, at least as high as RandOpt's 5000-perturbation search plus 50-model vote (63.5). Weight search
recovers what the prompt gives (the selected perturbations mostly turn off step-by-step reasoning, SHIFT_RESULT.md) and
adds nothing measurable on top of it (−0.08 [−0.97, +0.81]).

| Row | Base (RandOpt prompt) | RandOpt K=50 | RandOpt − SC@50, RandOpt prompt | Prompt chosen on selection set | SC@50, chosen prompt | RandOpt − SC@50, chosen prompt |
|---|---|---|---|---|---|---|
| GSM8K / Qwen2.5-1.5B | 60.0 | 77.2 | -2.65 [-4.32, -0.99] | boxed | 80.1 | -2.96 [-4.62, -1.29] |
| GSM8K / Qwen2.5-3B | 80.1 | 86.7 | -1.59 [-2.65, -0.53] | boxed | 87.0 | -0.38 [-1.67, +0.91] |
| GQA / Qwen2.5-VL-3B | 53.4 | 63.5 | +3.47 [+1.62, +5.41] | direct | 64.7 | -1.21 [-2.83, +0.40] |
| GSM8K / OLMo-2-1B | 33.7 | 52.5 | +8.72 [+6.75, +10.77] | boxed | 76.1 | -23.65 [-26.23, -21.08] |

Sources: C, C3B, G2, O1 (same-run RandOpt vs SC); PS (prompt selection). All locked; see results/README.md.

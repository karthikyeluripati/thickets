# G3 result: RandOpt's GQA advantage is only partly a termination repair

Lock dc19fd4 (before any G3 output). Analysis `scripts/g3_analysis.py` → `pod/g3_results.json`. 1× H100, ≈ $6.
Per-item outputs: `pod/g3_out_nontext.tgz` (votes, correctness, finish reasons, token counts) and `pod/g3_texts.tgz`
(BASE and MEMBER texts).

Engineering note: the first analysis call crashed on JSON serialization of numpy scalars after all generations had
finished; the save line was fixed (`default=lambda o: o.item()`, no change to any statistic) and the analysis re-run.

## Validity
G3 at 256 tokens reproduces G2 exactly: BASE 53.39% (Δ 0.00) and RandOpt K = 50 vote 63.49% (Δ 0.00) → **VALID**.

## Primary G3-1: does a 1024-token budget remove RandOpt's advantage?
| | 256 tokens | 1024 tokens |
|---|---|---|
| BASE | 53.39 | 53.72 |
| member mean (single model) | 58.39 | 58.60 |
| RandOpt K = 50 vote | 63.49 | 63.17 |
| SC@50 vote | 59.85 | 60.50 |
| **D = RandOpt − SC** | **+3.63 [+1.70, +5.57]** | **+2.67 [+0.73, +4.52]** |

Δ = D_256 − D_1024 = **+0.97 [+0.32, +1.70]**; D_1024's CI excludes 0 → **PARTIAL**: the longer budget shrinks the
advantage by about a quarter, but RandOpt stays ahead.

## Secondary G3-2: termination at 256
Non-terminated (hit max_tokens or no `\boxed{`): BASE 11.9%, MEMBERS 10.0%, SC samples 16.6%; BASE − MEMBERS
+1.90 [+0.31, +3.48] → **SUPPORTED** (small). At 1024: 10.2 / 9.2 / 14.2; difference +0.97 [−0.46, +2.49].
Caveat fixed in advance of seeing data (smoke run): some selected models answer directly without `\boxed{}`, which
this locked definition counts as non-terminated.

## Reported
- Mean generated tokens: BASE 147, MEMBERS **66**, SC 143 (256 budget; 148 / 69 / 144 at 1024). Selected perturbations
  roughly halve answer length.
- The BASE model rarely hits the token limit (BASE pass time 41.3 s at 256 vs 41.6 s at 1024); its "non-answers" are
  mostly missing final boxes, not truncation.
- D split by whether BASE terminated at 256: terminated questions (n = 1091) +3.57 (256) → +2.84 (1024);
  non-terminated (n = 147) +4.08 → +1.36.
- K = 10: D +4.28 [+2.10, +6.46] at 256; +3.72 [+1.70, +5.82] at 1024.

## Reading (replaces the exploratory "format/termination repair" reading of G2)
Termination accounts for about 1 pp of RandOpt's ~3.6 pp GQA advantage. The rest persists with a 4× budget and is
spread over questions the base model answers properly. The clearest measured change is answer **form**: selected
models answer in about half as many tokens. Whether conciseness itself, or a shift in which answers are given, carries
the remaining advantage is not resolved by G3.

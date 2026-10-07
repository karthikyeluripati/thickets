# P3 result: self-consistency matches RandOpt on GSM8K-3B; MATH-500 rows not comparable by the gate (similar gains)

Lock: `plan_lock.md` (c60b600, committed before output). Pod ≈ $2.7 of the $5 cap; pod stopped after a verified pull.
- The analysis output filename was corrected after the run (copy-paste from P2); no logic changed.

| Row | our greedy | paper Base | gate | **SC@50 T = 0.7** [95% CI] | paper TT-MV | **paper RandOpt** | verdict | SC gain over our base | paper RandOpt gain over its base |
|---|---|---|---|---|---|---|---|---|---|
| **GSM8K / Qwen2.5-3B** | 80.3 | 79.8 | pass | **88.2** [86.5, 90.1] | 82.5 | 87.1 | **MATCHES** (+1.1) | +8.0 | +7.3 |
| MATH-500 / Qwen2.5-1.5B | 49.6 | 43.2 | fail (+6.4) | 62.4 [58.2, 66.6] | 50.0 | 59.7 | NOT COMPARABLE | +12.8 | +16.5 |
| MATH-500 / Qwen2.5-3B | 64.4 | 58.6 | fail (+5.8) | 73.8 [70.0, 77.6] | 60.8 | 68.7 | NOT COMPARABLE | +9.4 | +10.1 |

SC@K curves (K = 1, 5, 10, 20, 50):
- GSM8K-3B: 78.7, 85.3, 87.0, 88.0, 88.2
- MATH-1.5B: 43.6, 53.0, 57.6, 60.2, 62.4
- MATH-3B: 63.6, 69.0, 71.6, 73.8, 73.8

## Combined with P2 (all rows, same protocol)
| Row | gate | SC@50 vs paper RandOpt | gain comparison |
|---|---|---|---|
| GSM8K-1.5B | pass | **79.8 vs 76.4: MATCHES** | +20.5 vs +17.6 |
| GSM8K-3B | pass | **88.2 vs 87.1: MATCHES** | +8.0 vs +7.3 |
| GSM8K-0.5B | fail | 56.0 vs 54.1 | +12.7 vs +14.2 |
| MATH-500-1.5B | fail | 62.4 vs 59.7 | +12.8 vs +16.5 |
| MATH-500-3B | fail | 73.8 vs 68.7 | +9.4 vs +10.1 |
| **GQA-VL-3B** | fail | **59.2 vs 69.0** | **+5.2 vs +12.4** |

## Reading
- On the two rows where our base reproduces the paper's (GSM8K 1.5B and 3B), properly sampled self-consistency (no
  weight search) **matches or exceeds** RandOpt's reported accuracy. It also exceeds the paper's TT-MV by 5.7–10.7 pp.
- On the text rows that fail the gate, SC's gain over our base is **similar to or somewhat smaller than** RandOpt's
  reported gain over its base (by 0.7–3.7 pp). These comparisons are descriptive only.
- **GQA is the clear exception:** RandOpt's reported gain is more than twice SC's. That is unexplained without
  running RandOpt at scale.
- **Caveat:** these are cross-paper comparisons (3-run means vs one run; different hardware). Same-run head-to-heads
  would be stronger.

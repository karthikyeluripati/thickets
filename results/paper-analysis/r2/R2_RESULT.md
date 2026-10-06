# R2 result (Qwen2.5-VL-7B): locked verdict INCONCLUSIVE (reliability gate failed); primary r = 0.920

Lock: `plan_lock.md` (committed in 47d2102 before tilt measurement). Model revision cc594898137f460bfe9f0759e9844b3ce807cfb5.
Pod session ≈ $2.1 (pod clock) of the $6.00 cap; pod stopped after a verified pull.

## Validity gates
| Gate | Result | Verdict |
|---|---|---|
| V-stream (job 01) | 525/525 tensors exact, 2 seeds | pass |
| V-map (job 02) | 525/525 packed tensors byte-identical HF vs vLLM | pass |
| V-per-perturbation | 80/80 byte-exact; vision reset and restore exact; 0 missing letters | pass |
| **S1a reliability** (split-half odd/even, Spearman–Brown) | **0.518** | **FAIL (≥ 0.7)** |
| P1 content | position-implied share 0.156 | pass |

## Locked decisions
| Test | Result | Rule |
|---|---|---|
| R2 primary r(pred_NOV, T_NOV) | **0.920**, slope 0.84 | PASS bar ≥ 0.6 |
| L2 localization | middle blocks L04–19 carry 0.71 (peak L12–15: 0.30; head 0.11) | supported |
| **Replication verdict** | S1a failed, so not interpretable | **INCONCLUSIVE** |

Reported: r(T_NOV, T_FULL)² = 0.94; r(pred_NOV, T_FULL) = 0.91; r(pred_NOV + pred_V, T_FULL) = 0.73;
r(pred_NOV, content part) = 0.82.

## Post-hoc diagnosis (does not change the verdict)
- **Small signal.** The tilt's spread across perturbations is about 5× smaller than on Qwen3-VL (sd 0.11 vs 0.59).
  - Random-split reliability: median 0.68, 90% range 0.01–0.88 (Qwen3-VL: 0.92, range 0.87–0.94).
  - Items tracking the total tilt (r > 0.3): 29/96 (Qwen3-VL: 50/96).
  - The split-half gate assumes the item halves are parallel; at this signal level they are not.
- **The primary correlation is robust.**
  - Spearman 0.891.
  - Pearson after dropping the 3/5/10 most extreme predictions: 0.893 / 0.878 / 0.867.
  - Stage 2 and R1 show the same robustness pattern.
- **Reading.** On Qwen2.5-VL the first-order law fits as well as on Qwen3-VL, but the measured tilt itself is weak.
  The locked rule correctly withholds "replicated".
- **A confirmatory R2b** would need its own lock, with a pre-specified remedy chosen *before* seeing new data, for
  example more probe items (all 363 eligible) to raise reliability. It must not tune σ or the class.

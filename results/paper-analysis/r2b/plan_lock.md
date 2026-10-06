# R2b plan lock: confirmatory second-model test with the reliability remedy fixed in advance

Locked before any R2b measurement. User-approved pod (cap **$5.00**). It follows R2 (`../r2/R2_RESULT.md`):
r = 0.920, but INCONCLUSIVE because split-half reliability was 0.518 (< 0.7). The post-hoc diagnosis was a weak tilt
signal on this model, too few items for a stable split-half estimate.

## The one pre-specified remedy
- **Probe: all 363** eligible SEARCH+RERANK items (≥ 1 front/forward option and ≥ 1 without), instead of 96.
- Spearman–Brown projection from R2: 0.52 → about 0.80 at 363 items. This is an expectation, not a gate.
- **Perturbations: 80 new** σ = 0.002 candidates, none used in Stage 2, R1 or R2.

**To fit the budget, only the NOV condition is measured** (vision reset to BASE, verified). NOV is the primary
test's target, and in R2 r(T_NOV, T_FULL)² was 0.94. FULL-condition statistics are therefore not reported.

## Unchanged from R2 / Stage 2
Model and revision (cc594898…; job 02 re-verifies the HF↔vLLM mapping byte-exactly against the R2 job-01 hashes).
Also unchanged: the "front" class, metric, engine, prompts, σ, groups, predictor (HF fp32 at BASE, no fitted
parameters), and per-perturbation byte-exact stream verification.

## Gates and decisions (thresholds unchanged)
| Test | Rule |
|---|---|
| V-map | 525/525 byte-identical, else stop |
| V-per-perturbation | byte-exact stream, exact vision reset and restore, else stop; missing letters ≤ 1% |
| **S1a reliability** | split-half (odd/even) Spearman–Brown of **T_NOV** ≥ 0.7 |
| P1 content | position-implied variance share of T_NOV < 0.20 |
| **R2b primary** | r(pred_NOV, T_NOV) over the measured perturbations: **≥ 0.6 PASS → REPLICATED; < 0.3 FALSIFIED → NOT REPLICATED; else INCONCLUSIVE**; n < 40 → INCONCLUSIVE |
| reported | Spearman, content-part r, SDs, group shares (80 seeds) |

**Interpretable** requires V-map, V-per-perturbation, S1a, P1 and the missing-letter rule.

**If S1a fails again:** R2b is reported INCONCLUSIVE, and the second-model claim is **dropped from the paper**. There is
no further remedy run.

## Budget
- Estimate ≈ $4.0: setup about $0.5, gradient on 363 items about $0.4, 80 × about 40 s about $3.1.
- The in-script guard stops before the $5 cap, and partial results are kept.

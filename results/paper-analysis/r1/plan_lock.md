# R1 plan lock: does the Stage-2 law replicate on a different task and content class?

Locked before any R1 measurement. GPU budget: **pending explicit user approval** (proposed cap $5.00, one pod session).
R1 is a replication of Stage 2 (`../stage12/plan_lock.md`, PASS r = 0.938) and Stage 3B's localization, on new data
and new perturbations. Same model and engine.

## What changes vs Stage 2 (everything else identical)
| | Stage 2 | R1 |
|---|---|---|
| Task / images | OmniSpatial Perspective_Taking | **OmniSpatial Complex_Logic (Geometric_Reasoning)**, never used in this project's pipeline |
| Content class | option contains "front"/"forward" | option contains the **word** "left" |
| Probe | 96 items | **all 98** 4-option search+validation items with ≥ 1 "left" option and ≥ 1 without |
| Perturbations | 80 (winner, controls, hash sample) | **80 new** σ = 0.002 candidates, none in the Stage-2 set, by smallest SHA256('r1-pert-v1:'+id) |

**Class selection rule (stated before any measurement):** the most frequent content word present in some but not
all options of an item, among words with a maximum letter-position share ≤ 0.40.
- "left" qualifies: 98 items, position shares 0.34 / 0.22 / 0.17 / 0.26.
- "none of the above" was excluded: 86% of its occurrences are at position D, and it is never gold.

## Metric, conditions, predictor
These are exactly as in Stage 2: T = mean over items of Δ[mean centered z(class options) − mean z(other options)].
The conditions are FULL and NOV (vision reset to BASE, verified), with exact restore. The predictor is
pred_NOV = σ Σ_{g ≠ vision} ⟨F^g, ε⟩, computed in HF fp32 at BASE with no fitted parameters. The 9 groups are those
of Stage 3.

## Gates and decisions
| Test | Statistic | Rule |
|---|---|---|
| S1a reliability | split-half r of T_FULL, Spearman–Brown | ≥ 0.7 |
| S1b fidelity | fingerprints, vision reset, restore exact; missing letters ≤ 1% | all |
| **P1 content (pre-registered this time)** | variance share of the letter-position-implied part of T_NOV | **< 0.20**, else the result is a position effect and R1 is uninterpretable |
| **R1 primary** | r(pred_NOV, T_NOV) over the 80 | **≥ 0.6 PASS (REPLICATED); < 0.3 FALSIFIED (NOT REPLICATED); else INCONCLUSIVE** |
| L1 localization (secondary) | share of predicted-tilt variance (all 5000 candidates) in layers 6–23 | ≥ 0.5 supports the Stage-3 localization; reported either way |
| reported | r(T_NOV, T_FULL)², r(pred_NOV, T_FULL), r(pred_NOV + pred_V, T_FULL), r(pred_NOV, content part) | – |
| minimum n | | < 40 perturbations → INCONCLUSIVE |

**Interpretable** requires S1a, S1b and P1. No re-tuning of the class, items, metric, thresholds or perturbations.

## Budget
- Estimate ≈ $3.5: setup about 6 min, gradient and projections about 5 min, 80 perturbations × about 31 s.
- Proposed hard cap $5.00, enforced by the pod guard and an in-script guard. Partial results are kept.

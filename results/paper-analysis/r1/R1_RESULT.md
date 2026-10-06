# R1 result: REPLICATED. The tilt law holds on a new task, a new content class and new perturbations

Lock: `plan_lock.md` (committed in 00155cc before measurement). 80 new σ = 0.002 perturbations × 98 OmniSpatial
Complex_Logic items; content class: the word "left". Spend $2.73 of the $5.00 cap; pod stopped after a verified pull.

| Locked test | Result | Rule | Verdict |
|---|---|---|---|
| S1a reliability | 0.866 | ≥ 0.7 | pass |
| S1b fidelity | 80/80 fingerprints; vision reset and restore exact; 0 missing letters | all | pass |
| P1 content, not position | position-implied variance share 0.184 | < 0.20 | pass (closer to the bar than Stage 2's 0.01) |
| **R1 primary** r(pred_NOV, T_NOV) | **0.915**, slope 0.91 | ≥ 0.6 | **PASS → REPLICATED** |
| L1 localization | layers 6–23 carry 0.73 of predicted variance (L06–11 0.37, L12–17 0.15, L18–23 0.21) | ≥ 0.5 | supported |

Reported:
- r(T_NOV, T_FULL)² = 0.91.
- r(pred_NOV, T_FULL) = 0.86.
- r(pred_NOV + pred_V, T_FULL) = 0.77.
- r(pred_NOV, content-only part) = 0.84.

Compared with Stage 2 ("front", Perspective_Taking): r = 0.938 vs 0.915; layers 6–23 carry 0.89 vs 0.73 of the
variance. On Complex_Logic the later layers and the head carry more (0.17 and 0.05).

Caveat: P1 passed, but letter position explains 18% of this tilt's variance (vs 1% in Stage 2). The prediction still
tracks the content-only part at r = 0.84.

Scope: same model (Qwen3-VL-8B), same OmniSpatial family and noise construction. A second model (R2) has not been
tested.

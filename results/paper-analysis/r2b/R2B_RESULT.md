# R2b result (Qwen2.5-VL-7B): REPLICATED

Lock: `plan_lock.md` (committed in 1cb3252 before measurement). The remedy for R2's failed reliability gate, more probe
items (363 vs 96), was fixed in advance; thresholds were unchanged. 80 new perturbations. Pod ≈ $3.2 of the $5.00 cap;
pod stopped after a verified pull.

| Gate / test | Result | Rule | Verdict |
|---|---|---|---|
| V-map | 525/525 byte-identical (model revision cc594898… re-verified) | all | pass |
| V-per-perturbation | 80/80 byte-exact against the stream; vision reset and restore exact; 0 missing letters | all | pass |
| S1a reliability (split-half Spearman–Brown of T_NOV) | **0.949** | ≥ 0.7 | pass |
| P1 content | position-implied share 0.073 | < 0.20 | pass |
| **R2b primary** r(pred_NOV, T_NOV) | **0.925** (slope 0.92; Spearman 0.91) | ≥ 0.6 | **PASS → REPLICATED** |

Reported:
- r(pred_NOV, content part) = 0.90.
- Middle blocks L04–19 (relative depth 0.14–0.71) carry 0.77 of the predicted variance, with the peak at L12–15
  (0.34). The final norm and head carry 0.05.

**Status of the law across all pre-registered tests:**

| | Model | Task / class | r |
|---|---|---|---|
| Stage 2 | Qwen3-VL-8B | Perspective_Taking / front | 0.938 |
| R1 | Qwen3-VL-8B | Complex_Logic / left | 0.915 |
| R2b | Qwen2.5-VL-7B | Perspective_Taking / front | 0.925 |
| I5 (item level) | Qwen3-VL-8B | Perspective_Taking / front | 0.771 pooled, 0.88 median per item |

The tilt is 5–6× smaller in magnitude on Qwen2.5-VL than on Qwen3-VL (sd 0.10 vs 0.59 nats) at the same σ. Both
models are Qwen-family; a non-Qwen model remains untested.

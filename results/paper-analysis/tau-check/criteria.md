# τ-predictor artifact check: criteria written before computing (post-hoc analysis of M1 data)

**Question:** is r(τ, fresh-item gain) = 0.61 (M1-3, n = 59) a real predictive relation or a selection/group artifact?

**"τ holds"** requires all three:
1. **Within the SEARCH-top-50 group alone (n = 47):** r(τ, H gain) ≥ 0.30.
2. **Partial r(τ, H gain | RERANK gain, SEARCH gain)** over all 59 ≥ 0.20.
3. **The sign is preserved among the 12 controls.** That subgroup is too small for a threshold; only a reversal (r < 0)
   fails it.

**Reported:**
- r(τ, measured H answer-content tilt);
- whether τ predicts H gain beyond the measured H tilt (partial r);
- bootstrap CIs.

**If τ fails:** drop the predictor framing (step 2 is not run).

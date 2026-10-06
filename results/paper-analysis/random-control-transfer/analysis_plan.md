# Random-control transfer for 9504111: analysis plan lock

**Post-hoc plan lock.** It was written and committed while the GPU run was in
progress, before any control output had been read.

- **Branch:** `research/perspective-random-control-transfer`, from local
  `e96e062`.
- **Controls:** the frozen 12 in `frozen_controls.json`, copied unchanged from
  `selection_vs_shared_results.json` → `control_proposal`. The seeds are
  9500059, 9503167, 9500127, 9502563, 9502995, 9503539, 9501135, 9502423,
  9500271 (a top-50 member, kept), 9501023, 9504179 and 9504827.
- **Verified on CPU:** each control's seed, σ = 0.002, all-parameter mask,
  audit membership, and original `candidate_state_id` against
  `locks/search.json` and its raw files.
- **Authorization:** GPU approved by the user with a $3.50 total-pod cap; the
  pod stops at $3.40. Rate $3.49/h, from the RunPod API.

## Primary analysis (per control, all 12 shown)

- **Per candidate:** g_R and g_T (pp over BASE, with BASE correctness from the
  generated answers: 79/200 and 259/561), gap = g_T − g_R, correct counts,
  repairs and regressions, and top-50 membership.
- **Summary:** mean and median gap, range, count with g_R > g_T, count above
  base in each phase, and 9504111's position relative to the observed
  controls.
- **Uncertainty:** a bootstrap over the 12 controls (resampling unit:
  candidates; conditional on the fixed questions), 10,000 replicates, seed
  `20261012`, for the control mean gap only. Questions are not resampled.
- **Selected comparison:** the stored SEARCH top-50 distribution is shown
  separately and labelled as selected.

## Accounting (descriptive)

g_T* − g_R* = (μ_T − μ_R) + [(g_T* − μ_T) − (g_R* − μ_R)], with μ taken over
the 12 controls. The two terms are the measured random-control mean change
and the change in the winner's advantage relative to those controls.

## Score comparison (same definitions as the margin follow-up)

- **Comparator and contrast:** r_B = BASE strongest wrong option (tie → lowest
  letter), held fixed. For each candidate k and question j,
  t_k[j] = (L_k[y] − L_k[r_B]) − (L_B[y] − L_B[r_B]).
- **Per candidate:** mean t by phase, separately for BASE-correct and
  BASE-wrong. A standardized mean t uses the margin follow-up's frozen strata
  (BASE correctness × |m_B| bins) and reference weights,
  w = (w_R + w_T)/2 over its 9 common-support strata, unchanged.
- **Centred A–D score changes:** Δz_k[j] = z_k[j] − z_B[j], with
  z = L − mean_{A–D}.
- **Reference:** the mean of the 12 controls' Δz, with coefficient 1 and
  nothing fitted.
- **Target vs reference, per phase:**
  - pooled cosine similarity of Δz* with the reference;
  - median per-question cosine;
  - pooled residual ratio ‖Δz* − ref‖ / ‖Δz*‖.
- **Baseline for those numbers:** the same metrics for each control against
  the mean of the other 11 (leave-one-out). This is how similar an ordinary
  control is to the rest.
- **Rules:** no TEST-label fitting; no projections.

## Figures (at most two)

1. Paired RERANK → TEST gains: 12 controls and 9504111.
2. Paired standardized mean t by phase: 12 controls and 9504111.

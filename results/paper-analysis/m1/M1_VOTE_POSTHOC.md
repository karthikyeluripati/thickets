# Post-hoc (CPU, M1 data): RandOpt-style majority votes on the fresh matched compass hold-out

**Not pre-registered.** Computed from the M1 outputs after M1's locked analysis.
- 600 fresh compass-format OmniSpatial-train items, 484 images.
- First-token answers (direct-letter protocol); ties broken alphabetically.
- CIs are image-cluster bootstrap (2000).
- Base accuracy 34.7%.

| Ensemble / model | gain vs base | 95% CI |
|---|---|---|
| **Majority vote of all measured SEARCH-top-50 members (n = 47; RandOpt-style K ≈ 50)** | **+0.2 pp** | [−1.0, +1.3] |
| Vote of the top-10 by RERANK gain | +2.8 | [+1.0, +4.7] |
| Vote of the top-10 by SEARCH gain | +2.0 | [+0.2, +4.0] |
| Vote of the 12 random controls | −0.3 | [−2.2, +1.5] |
| Single winner 9504111 | +2.7 | [+0.3, +4.9] |

Mean single-member gain: top-50 +0.28 pp; controls −0.88 pp.

**Reading.**
- In the direct-answer regime, the K ≈ 50 vote **returns the base model's accuracy** on fresh same-format data. The
  original "vote = base" observation therefore survives the split-format correction.
- The transferable gain (about +2–3 pp) sits in the few best candidates.
- This is consistent with the first-order picture: votes over many weakly selected first-order tilts average toward the
  base.

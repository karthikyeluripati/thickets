# I5 result: ITEM-LEVEL LAW. The first-order law holds per item, not only for the 96-item average

Lock: `plan_lock.md` (committed in a1c2a7a before computation). Pod about $0.6 of the $2.50 cap; pod stopped after a
verified pull.

| Test | Result | Rule / verdict |
|---|---|---|
| Gate: mean of per-item predictions vs Stage-2 pred_NOV | r = 0.99999 | ≥ 0.99, pass |
| **I5-1** pooled r(pred_j(c), ΔB_j(c)), 96 items × 80 perturbations = 7680 pairs, NOV | **0.771**, slope 0.74 | ≥ 0.6, **ITEM-LEVEL LAW** |
| I5-2 per-item r over 80 perturbations | median 0.877 (IQR 0.843–0.893); 95/96 items ≥ 0.6 | reported |
| I5-3 Spearman(per-item r, \|base contrast\|) | −0.007 | reported: no dependence on closeness to a tie |

**Reading.**
- For σ = 0.002 perturbations with vision at BASE, each item's answer-content contrast changes as a first-order
  function of the noise stream.
- Stage 2's aggregate r = 0.94 is therefore not an artifact of averaging.
- The pooled r (0.77) is below the per-item median (0.88) because per-item slopes differ. That is a scale
  heterogeneity, not a failure of linearity.
- **Contrast with H★.** GPU-A falsified first-order prediction of the *whole-model, gold-vs-strongest-wrong* contrast
  (F1 r = 0.14). The failing component there was the vision tower (S2d: vision r = 0.05; language groups 0.71–0.96).
  I5 is consistent with that: the language-model side is first-order at the item level.

# S1-7B result: the first-order law at 7B (OLMo-2-1124-7B-Instruct, ARC-Challenge)

Lock d64a1ad. Analysis `scripts/s1_analysis.py --a-dir pod/s17b/A` → `s1_7b_results.json`. Smoke passed first.

- Validity: restored exactly; base accuracy 0.68 (≥ 0.30) → valid.
- **7B-1 (primary): r = 0.787 [0.758, 0.820] at σ ≤ 0.002 (1600 pairs) → GO.**
- Per σ: 0.914 (0.001), 0.755 (0.002), 0.249 (0.005, breaks, as at 1B and in P0). Slope 0.77; 24/24 perturbations
  with per-perturbation r > 0.
- Same items and perturbations as S1-A (1B: 0.790): the law's fit does not degrade from 1B to 7B.

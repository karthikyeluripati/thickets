# S1-7B plan lock: the first-order law at 7B on a non-Qwen text model

Committed before any S1-7B output. Runs as job 09 on the 6× H100 pod after G2 (one GPU, ~10 min; session cap $120).
Identical to S1-A (`../s1/plan_lock.md`, 797fa96) in items, perturbations, contrast, prediction, gates and rules;
only the model changes.

- Model `allenai/OLMo-2-1124-7B-Instruct` @ `470b1fba1ae01581f270116362ee4aa1b97f4c84` (letters A–D = token ids
  32–35, verified; same chat format as the 1B).
- Runner `scripts/s1_a_olmo.py --model … --rev …` (the S1-A runner, model now a parameter; no other change).
- Inputs `../s1/frozen_A.json` (the same 100 ARC-Challenge items and 24 perturbations).
- Gates: restored_exact; base accuracy ≥ 0.30; smoke (5 items × 3 perturbations) passes first.
- **Primary 7B-1:** pooled r(pred, meas) at σ ≤ 0.002 (1600 pairs). **GO if ≥ 0.5; NO-GO if < 0.3**; else
  inconclusive. Item-cluster bootstrap CI reported. Per-σ r, slope, fraction of perturbations with r > 0 reported.
- Analysis: `scripts/s1_analysis.py --a-dir <7B output>` (the locked S1 analysis, A part).

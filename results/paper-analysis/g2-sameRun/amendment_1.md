# G2 amendment 1 (committed before any G2 output; engineering only, no change to items, population, rules or analysis)

Reason: GPU utilization. Same perturbations (randopt.py's seed/σ generator), prompts, greedy decoding, scorer, ranking
and vote; only how the work is spread over the 6 GPUs changes.

1. **Preferred configuration ("fast"):** vLLM with CUDA graphs (`enforce_eager=False`, vLLM's default and the mode P2's
   SC arm ran in) and **12 workers, 2 per GPU** (`gpu_memory_utilization=0.36` each, since each worker also keeps a
   base-weight copy outside vLLM's budget). Worker w runs on GPU w mod 6; the second wave starts 90 s after the first.
   Perturbation k goes to worker k mod W (W = 12); test rank r to worker r mod W.
2. **Automatic fallback ("fallback"):** if the fast configuration's smoke run (24 perturbations + a 30-question test
   pass, all workers) does not complete, the smoke is re-run with the locked configuration (6 workers, eager,
   0.85), and the full run uses whichever configuration passed (`/workspace/g2/config.env`, reported).
3. Projection gate unchanged in spirit: spent on G2 + 5000/W × (mean of the slower 3/4 of smoke per-perturbation
   seconds) × 1.05 × rate + $3 ≤ $45.

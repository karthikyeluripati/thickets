# G2R result: RandOpt's GQA advantage over self-consistency replicates with a second perturbation population

Lock 9e84c4f + amendment 1 (0754376, budget only), both before any full-run output. Analysis `scripts/g2r_analysis.py`
→ `pod/g2r/g2r_results.json`. 2× H100, ≈ $31.2 (cap $36).

## Engineering events (no effect on data)
- Smoke: second-wave workers 2 and 3 failed vLLM's memory-profiling check (workers 0/1 finished their 2 smoke
  perturbations during it); re-run alone, as locked. Smoke projection $29.00 > $28 gate → full run stopped as locked;
  user raised the cap to $36 (amendment 1).
- Full run: worker 1 hit CUDA OOM at 4561 (1141/1250 done) in selection and worker 3 in testing (vision-encoder
  activations outside vLLM's memory budget with 2 workers per GPU; same failure as G2's worker 6). Both re-run alone
  with identical settings; the resumable runner skipped existing results. All 5000 rewards and all 50 ranks + BASE
  present before analysis.

## Validity
Same 1238 test / 200 selection questions as G2 (asserted). G2R base greedy 53.39% = G2's 53.39% → **VALID**.
Populations disjoint (0 shared seeds; 0 shared among the top 50).

## Primary G2R-1 (K = 50, paired item bootstrap, 10,000 resamples)
| | seed 42 (G2) | seed 43 (G2R) |
|---|---|---|
| RandOpt K = 50 vote | 63.49 | 63.65 |
| SC@50 (same P2 run) | 60.02 | 60.02 |
| **D** | **+3.47 [+1.62, +5.41]** | **+3.63 [+1.78, +5.49]** |

→ **REPLICATED** (D₂ CI lower > 0).

## Secondary
- D₂ − D₁ = +0.16 [−0.97, +1.29]: the perturbation draw moves D by well under 1.5 pp. Two-seed mean D = **+3.55
  [+1.78, +5.37]**.
- K = 10: D₁ +4.36 [+2.26, +6.54], D₂ +4.68 [+2.50, +6.95]; difference +0.32 [−0.97, +1.70]; mean +4.52 [+2.50, +6.58].
- Members: mean 57.8% (range 52.6–63.4) vs 58.4% (52.5–63.1) in G2; base 53.4%.
- σ of the top 50: 44 × 0.002, 5 × 0.001, 1 × 0.0005 (G2: 46 / 3 / 1). Selection reward of the top 50: 0.57–0.65
  (G2: 0.58–0.635); base 0.535.

## Reading
The GQA result is not a lucky draw: a disjoint population of 5000 perturbations gives the same advantage over SC
(+3.6 vs +3.5 pp) and the same member-level gain. The paper reports both seeds and the two-seed mean.

# Session C result: same-run RandOpt (N = 5000) vs self-consistency, GSM8K / Qwen2.5-1.5B-Instruct

Lock 1437d44 + amendment 1 (6118222), both committed before any RandOpt-arm output. Analysis: `scripts/c_analysis.py`
→ `c_results.json` (primary), `c_results_samepod.json` (secondary). Pods: 1× H100 (SC arm; RandOpt arm did not fit
the $8 cap, rule-skipped), then 6× H100 (RandOpt arm, N = 5000; same-pod SC replicate). Total ≈ $46.

## Validity
- RandOpt's own `randopt.py` @ 4000d34, one logging line added (`pod6/c/out/randopt_patch.diff`).
- Recomputed K = 50 vote from the dumped answers = randopt.py's printed 1018/1319 exactly → **C-1 VALID**.
- Base: randopt.py base test 60.27%, p2_sc greedy 59.29% (paper 58.8; both within ±2.0 pp gate).
- **RandOpt reproduces the paper:** K = 50 = **77.18%** vs paper 76.4% (one run vs their 3-run mean).
- SC arm: the same-pod replicate (6× H100 pod) produced bit-identical samples on all 1319 items to the pod-1 run,
  so primary and secondary SC are the same data (SC@50 = 79.83%, as in P2).

## Primary C-1 (K = 50, paired item bootstrap, 10,000 resamples)
**D = RandOpt − SC@50 = −2.65 pp [−4.32, −0.99] → SC AHEAD.** Discordant items: 81 SC-only correct, 46 RandOpt-only.
Not equivalent within ±2 pp.

## Secondary
- K = 10: RandOpt 77.18% vs SC@10 75.21%, D = +1.97 [0.00, 3.87] → NO DIFFERENCE DETECTED (lower bound at 0).
  (K = 10 and K = 50 give the same total, 1018, with 47 items swapping each way; checked from the dump.)
- Selection (descriptive): top-50 train reward 0.780 (range 0.770–0.815) vs population mean 0.676 and base 0.730 on
  the 200 train items; σ of the top 50: 33 at 0.001, 9 at 0.0005, 8 at 0.002. "Best σ" by mean = 0.0005.
- Exploratory decomposition (not pre-registered): the 50 selected models individually score 64.3% on test on
  average (59.4–68.9%; base 60.3%); the vote of 50 gives 77.2%. SC single samples average 57.3%, and their vote
  gives 79.8%. Vote curve RandOpt K = 1/5/10/20/50: 68.2/74.6/77.2/77.5/77.2; SC: 57.5/69.9/75.2/78.7/79.8.

## Reading (for RESULTS_MASTER)
At the paper's own population size, with the paper's number reproduced, RandOpt's K = 50 ensemble is **below**
temperature-sampled self-consistency with the same number (50) of test-time generations, with a CI excluding zero;
this does not charge RandOpt's 5000 × 200 search generations. At K = 10, RandOpt's point estimate is higher, CI
touches zero. Single run, one model (1.5B), one task; the paper averages 3 runs.

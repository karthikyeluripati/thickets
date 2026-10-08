# C3B result: same-run RandOpt (N = 5000) vs self-consistency, GSM8K / Qwen2.5-3B-Instruct

Lock e78cfd0 (before any C3B output). Analysis `scripts/c_analysis.py --sc-dir results/paper-analysis/p3/pod --sc-tag q3`
→ `c3b_results.json`. Pod: 6× H100; smoke projection $51.91 (GO).

## Validity
- RandOpt's own `randopt.py` @ 4000d34 (one logging line added; `pod/c3b/out/randopt_patch.diff`), N = 5000, K = 50.
- Recomputed K = 50 vote = printed 1143/1319 exactly → **C3B-1 VALID**.
- Base: randopt.py base test 80.67%, P3 greedy 80.29% (paper 79.8; gate pass).
- **RandOpt reproduces the paper:** K = 50 = **86.66%** vs paper 87.1.

## Primary C3B-1 (K = 50, paired item bootstrap, 10,000 resamples)
RandOpt 86.66% vs SC@50 (T = 0.7, P3 run) 88.25%: **D = −1.59 pp [−2.65, −0.53] → SC AHEAD.** Discordant: 36 SC-only
vs 15 RandOpt-only correct. Not equivalent within ±2 pp.

## Secondary
- K = 10: RandOpt 85.90 vs SC@10 86.96, D = −1.06 [−2.35, +0.15] → NO DIFFERENCE DETECTED.
- Exploratory decomposition: the 50 selected models score 80.9% each on average (77.9–84.3; base 80.7); vote curve
  K = 1/5/10/20/50: 81.7/85.0/85.9/86.1/86.7. SC single samples average 78.0%; SC@50 votes to 88.2.
- Selection: top-50 train reward 0.910 (0.905–0.940) vs base 0.855 on the 200 train items; σ of the top 50:
  34 × 0.001, 15 × 0.0005, 1 × 0.002.
- Same-pod SC replicate (tag c3b): reported separately when complete (secondary).

## Reading
Second same-run row, at 2× scale, with the published RandOpt number reproduced: SC@50 is again ahead with a CI
excluding zero (C: −2.65 at 1.5B; C3B: −1.59 at 3B). Here selected members are on average only ~0.2 pp above base,
so essentially all of RandOpt's gain over base (+6.0) is the vote.

# GD result: with the prompt held fixed, RandOpt's weight search adds nothing on GQA

Lock 0bb89eb (before any GD output). Analysis `scripts/gd_analysis.py` → `pod2/gd/gd_results.json`. Cost ≈ $21.4
(6× H100 ≈ $20.5 + 1× H100 ≈ $0.85).

## Engineering events (no effect on outputs)
- Smoke projection $16.56 (GO). During the test phase all four second-wave workers (8–11) failed vLLM's startup memory
  check and were being re-run when the first pod stopped at ~05:08 UTC, before its $25 guard, most likely because the
  account balance ran out. All 5000 selection rewards, the BASE test and 36/50 test ranks had been pulled
  (`pod/`, commit 4f294fc).
- Finished on a second 1× H100 pod with identical settings and the resumable runner (ranks recomputed from the same
  5000 rewards): the 14 missing ranks (workers 8–11, run one at a time), then the locked analysis.
- **Cross-pod check:** worker 0's files (BASE and ranks 0, 12, 24, 36) recomputed on the second pod are **identical on
  1238/1238 answers** for all five files (`pod2/gd/crosspod_check.txt`).

## Validity
Direct-prompt BASE greedy in this run 64.70 = G4's 64.70 → **VALID**.

## Primary GD-1 (K = 50, paired item bootstrap, 10,000 resamples)
RandOpt (searched and voted under the direct prompt) 64.38 vs SC@50 (direct) 64.70: **D = −0.32 pp [−1.29, +0.65] →
NO DIFFERENCE DETECTED, EQUIVALENT within ±2 pp.** Discordant: 18 RandOpt-only vs 22 SC-only.

## Secondary
- K = 10: RandOpt 64.70 vs SC@10 63.89, D = +0.81 [−0.48, +2.10] (n.d.).
- Selected members individually: mean 64.10 (range 62.68–65.02) vs the direct BASE 64.70: −0.60 [−1.27, +0.05]. The
  vote adds +0.28 over the members.
- RandOpt (direct) − RandOpt (CoT, G2): +0.89 [−0.65, +2.42]. RandOpt (direct) − one direct BASE generation: −0.32
  [−1.05, +0.32].
- Selection: top-50 selection reward 0.680–0.695 (direct BASE on the selection set: 66.0%, PS). Top-50 overlap with
  G2's CoT-prompt top 50: **0** (same population). σ of the top 50: 20 × 0.0005, 26 × 0.001, 4 × 0.002 (CoT search:
  46 × 0.002).

## Reading
Under the direct prompt, the perturbations selection prefers score 1.4–2.9 points above base on the 200 selection
questions but are not better on test, and their vote equals sampling the unperturbed model (and one greedy
generation). The shared shift RandOpt found under its CoT prompt (switching reasoning off) has nothing left to find
once the prompt already does it. On GQA, the decomposition is complete: RandOpt's advantage under its own prompt is
the prompt effect plus the vote, and with the prompt held fixed weight search adds nothing measurable.

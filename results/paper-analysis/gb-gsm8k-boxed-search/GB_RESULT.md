# GB result: with the boxed prompt held fixed, self-consistency beats RandOpt's search on OLMo; Qwen-1.5B gated out

Lock 77332dd (before any GB output). Analysis `scripts/gb_analysis.py` → `pod/gb/olmo/gb_results.json`. 4× H100
($15.96/h), ≈ $18.95 (cap $48). Engineering: no worker failures (layered ENGINE_READY launch; 1 test worker per GPU).

## GB-1, OLMo-2-0425-1B-Instruct (primary)
**Gates:** environment PASS (BASE selection reward under RandOpt's prompt 41.50 = randopt.py's 41.50; boxed 79.50 = PS
79.5; boxed BASE test 67.85 = O2 67.85). Fidelity CONSISTENT (k = 0–23 under RandOpt's prompt vs O1's log: mean |Δ|
0.0135). Projection $16.62 (GO).

**Primary:** RandOpt (searched and voted under the boxed prompt) 74.00 vs SC@50 (boxed, O2) 76.12:
**D = −2.12 pp [−3.49, −0.83] → SC AHEAD** (not equivalent). Discordant: 25 RandOpt-only vs 53 SC-only.

**Secondary**
- K = 10: RandOpt 72.63 vs SC@10 74.68, D = −2.05 [−3.64, −0.45] (SC ahead).
- Selected members individually: mean 67.05 (64.75–68.84) vs the boxed BASE 67.85: −0.81 [−2.19, +0.57]. The vote adds
  +6.95 over the members.
- Selection: top-50 selection reward 84.0–86.0 vs BASE 79.5; it does not transfer (members not above base).
- RandOpt (boxed) − RandOpt (its own prompt, O1): **+21.53 [+18.88, +24.18]**. RandOpt (boxed) − one boxed BASE
  generation: +6.14 [+4.40, +7.81].
- Top-50 overlap with O1's top 50 (same population): 0. σ of the top 50: 42 × 0.001, 8 × 0.0005.

## GB-2, Qwen2.5-1.5B-Instruct: INVALID-ENV, not run (locked gate)
Gates (`pod/gb/q15/gates.txt`): BASE selection reward under RandOpt's prompt **68.00 vs 73.00 printed by randopt.py
in C → environment FAIL** (limit 1.0 pp); boxed 80.50 = PS 80.5 (pass); fidelity CONSISTENT (mean |Δ| 0.0181 vs C's
log). The job stopped at the gate as locked, before any search output beyond the gates; the user chose not to amend
the gate. Diagnosis (not used to override the gate): PS's independent run measured the same quantity at 68.5, and
the perturbed rewards agree with C's log, so the mismatch is confined to randopt.py's single printed base reward; its
cause is not established.

## Reading
With the prompt held fixed (boxed), RandOpt's selected OLMo models are no better than the base model on test, the
vote is its whole gain, and sampling the unperturbed model votes better (+2.1 pp). Together with GD (GQA, equivalent),
in both rows where the prompt-held-fixed search was run, weight search adds nothing over sampling.

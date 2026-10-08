# O1 result: on a non-Qwen model (GSM8K / OLMo-2-1B), RandOpt is clearly ahead of self-consistency

Lock e2e2400 (before any O1 output). Analysis `scripts/o1_analysis.py` → `pod/o1/o1_results.json`. 8× H100
($31.92/h), ≈ $46.6 (cap $60). No engineering events.

## Validity
RandOpt's own `randopt.py` @ 4000d34 with the C/C3B one-line answer dump (`pod/o1/out/randopt_patch.diff`); data hashes
equal C/C3B. Recomputed K = 50 vote = printed 692/1319 → **VALID**. Smoke projection $50.82 (GO).
**O1b (second seed) not run by the locked rule:** projected $87.06 > $58.5 (`pod/o1/o1b_decision.txt`).

## Primary O1-1 (K = 50, paired item bootstrap, 10,000 resamples)
RandOpt 52.46% vs SC@50 43.75%: **D = +8.72 pp [+6.75, +10.77] → RANDOPT AHEAD.** 153 RandOpt-only vs 38 SC-only correct.

## Secondary
- K = 10: RandOpt 50.11 vs SC@10 42.53, D = +7.58 [+5.31, +9.93] → RANDOPT AHEAD.
- Base greedy: randopt.py 35.25%, p2_sc 33.36%. SC single samples 33.7% on average; SC vote gains only +10 pp.
- **Members:** the 50 selected models score 40.3% individually on average (35.9–44.7), **+5.1 pp above base** — like
  GQA (+5.0), unlike GSM8K/Qwen (+4.0 at 1.5B, +0.2 at 3B). Vote curve K = 1/5/10/20/50: 40.2/49.0/50.1/53.0/52.5.
- Best σ by mean reward: 0.0005.

## Reading (fixed in advance: RANDOPT AHEAD → Claim 1 qualified as model-dependent on GSM8K)
On GSM8K the outcome depends on the model. With Qwen2.5 (1.5B, 3B) the selected models are barely better than base
and SC votes better; with OLMo-2-1B under RandOpt's prompt the selected models are individually +5 pp better (a shared
shift, as on GQA) and RandOpt beats SC by 8.7 pp.

## Open, and what this data cannot answer
OLMo-2-1B scores 35% here under RandOpt's prompt ("Let's think step by step and output the final answer after ####"),
which looks low for this model (its developers report a substantially higher GSM8K score under their own evaluation
setup; exact figure to be checked and cited). That suggests, as on GQA
(G4), that the shared shift may be one a prompt also gives. **Not tested:** RandOpt's code and `p2_sc.py` store only the
extracted final answers (the extractor falls back to the last number), so format failures cannot be measured from
this data. A G4-style control (base greedy and SC@50 with a different prompt, saving texts) would answer it.

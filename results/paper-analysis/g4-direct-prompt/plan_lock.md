# G4 plan lock: does prompting the base model to answer directly match RandOpt on GQA?

Committed before any G4 output. 1× H100; session cap **$10** (pod guard).

## Why
The exploratory shift analysis (`gqa-shift/SHIFT_RESULT.md`, 7912ad4) found that RandOpt's selected GQA models mostly
stop reasoning and answer directly (median 10 tokens vs 146), and that shorter answers are more accurate. If the base
model prompted to answer directly gives the same gain, RandOpt's GQA advantage over self-consistency (G2/G2R) is a
prompt effect that weight search finds expensively. G4 tests this.

## Design
Model, revision, test questions (G2's 1238), images, scorer, greedy and SC sampling settings (T = 0.7, n = 50, seed
20261007) and max_tokens 256 are identical to G3. Runner `scripts/g3_eval.py` with a new `--prompt` option (default
'cot' = RandOpt's prompt, unchanged). Prompts, fixed now (both keep `\boxed{}` so answer extraction is identical):
- **direct (primary):** "…Question: {q}\n\nAnswer directly without explanation, and put your final answer within \boxed{}."
- **short (secondary):** "…Question: {q}\n\nAnswer the question using a single word or phrase, and put your final answer
  within \boxed{}."

Arms: cot BASE (validity only); direct BASE, SC@50 and G2's top-50 MEMBERS; short BASE and SC@50.
References under the CoT prompt come from G3's stored 256-token outputs (G3 reproduced G2 exactly): RandOpt K = 50
(seed 42), SC@50, BASE, members; seed-43 RandOpt votes from G2R.

## Validity
G4's cot BASE accuracy equals G3's (53.39%) within 0.5 pp; otherwise G4-1 is reported as INVALID-ENV.

## Primary G4-1
D = acc(RandOpt K = 50 vote, CoT prompt, seed 42) − acc(SC@50 vote, **direct** prompt); paired item bootstrap (10,000,
seed 0). **RANDOPT AHEAD** if CI lower > 0 (weight search gives more than a direct-answer prompt);
**PROMPT-SC AHEAD** if CI upper < 0; else **NO DIFFERENCE DETECTED**, plus **EQUIVALENT** if the CI is within ±2 pp.

## Secondary (reported regardless)
- Same D for K = 10, for the seed-43 population, and with the short prompt.
- Direct/short BASE greedy vs CoT BASE greedy and vs the CoT members' mean (does a prompt alone give the member-level gain?).
- SC(direct) − SC(CoT).
- Selected members evaluated with the direct prompt: RandOpt K = 50 − SC@50, both direct (does search add on top of
  the prompt? The members were selected under the CoT prompt, so this is descriptive).
- Mean tokens, share of direct answers (≤ 32 tokens), boxed share, per arm.

## How it is used
RANDOPT AHEAD → the paper states that the GQA advantage is more than a prompt effect. EQUIVALENT or PROMPT-SC AHEAD →
the paper states that on GQA weight search recovers what a direct-answer prompt gives, and the GQA result is framed as
search finding a format shift, not as a capability gain. NO DIFFERENCE DETECTED without equivalence → reported as
inconclusive.

## Analysis
`scripts/g4_analysis.py`. Self-test with every prompt directory replaced by G3's CoT outputs reproduces G3
(D = +3.63 [+1.70, +5.57]; prompt differences 0; seed-43 RandOpt 63.65 = G2R).

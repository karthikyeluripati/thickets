# OS plan lock: can a prompt reach the OmniSpatial winner's transferable gain?

Committed before any OS output. 1× H100 (80 GB); session cap **$10** (pod guard).

## Why
The OmniSpatial winner (seed 9504111, σ = 0.002, all parameters, Qwen3-VL-8B) is the only selected perturbation in
this study whose gain transfers to fresh matched items (+2.67 [+0.17, +4.93], M1) and is not explained by its
answer-content tilt (+2.33 of +2.67 remain, C1-1). It is also the one setting where no prompt control was run. OS asks
whether the benchmark's own prompt variants, chosen on the winner's own selection data, reach that gain, and whether
the winner's gain survives with the prompt held fixed.

## Design
- Model, revision, engine, fingerprints and restore exactly as M1 (`scripts/m1_measure.py`): Qwen3-VL-8B @ 0c351dd,
  vLLM 0.11 in-process, eager, pinned RandOpt worker, PERTURB_VISUAL = 1; BASE fingerprint 5bc2499c…; winner fingerprint
  = its frozen expected_state_id; exact base restore checked at the end.
- **Prompts: the benchmark's own three evaluation configurations** (`third_party/omnispatial/system_prompts.py`):
  - **direct** = SYS `none` + FORMAT `direct`; answer = first generated token (max_tokens 1), exactly M1.
  - **zeroshot_cot** = SYS `zeroshot_cot` + FORMAT `re`; **manual_cot** = SYS `manual_cot` + FORMAT `re`; greedy, up to
    2048 tokens; answer = the last "Answer: X" (X ∈ A–D); no match = parse failure, scored wrong.
- Items: prompt selection on the original **SEARCH200** set (`examples/omnispatial-perspective-taking/search.jsonl`, the
  winner's own selection data); evaluation on the **M1 hold-out** (600 items, 484 images; `m1/pod/holdout.jsonl`).
  Images fetched from the pinned OmniSpatial revision and verified by sha256 against the item records (`os_prep.py`).
- Arms: BASE × 3 prompts on SEARCH200; BASE × 3 and WINNER × 3 on the hold-out. Runner `scripts/os_prompt_eval.py`.

## Rule
Chosen prompt = highest BASE accuracy on SEARCH200; ties: direct, zeroshot_cot, manual_cot.

## Validity
This run's direct-prompt BASE and WINNER hold-out correct counts equal M1's (208 and 224) within ±3 items; fingerprints
match. Otherwise OS-1 is INVALID-ENV.

## Primary OS-1
D = acc(WINNER, direct prompt, as in M1) − acc(BASE, chosen prompt) on the hold-out; image-cluster bootstrap exactly as
M1 (2000 resamples, seed 20261006). **WINNER AHEAD** if CI lower > 0; **PROMPT AHEAD** if CI upper < 0; else
**NO DIFFERENCE DETECTED**; plus **EQUIVALENT** if the CI lies within ±2 pp. (If direct is chosen, OS-1 equals M1-1.)

## Secondary (reported regardless)
- **OS-2, prompt held fixed:** acc(WINNER, chosen) − acc(BASE, chosen), same bootstrap.
- Winner gain under each prompt; all six hold-out accuracies; SEARCH200 accuracies; parse failures; answer agreement of
  this run's direct arms with M1's.

## How it is used (fixed now)
- WINNER AHEAD → the benchmark's prompt variants do not reach the winner's gain: in this direct-answer setting, weight
  search found a transferable improvement these prompts do not give. The paper states it as the one positive case.
- NO DIFFERENCE / PROMPT AHEAD → a prompt chosen on the same selection data reaches it; the story is complete across
  every setting studied.
- OS-2 says whether the winner's gain persists once the prompt is fixed.

## Analysis
`scripts/os_analysis.py`. Self-test with M1's own outputs standing in for all arms (direct chosen) reproduces M1-1
exactly: +2.67 [+0.17, +4.93]; environment counts 208/208 and 224/224.

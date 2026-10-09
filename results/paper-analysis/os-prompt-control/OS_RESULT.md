# OS result: the OmniSpatial winner's gain exists only under the direct prompt; a prompt chosen on its own selection data reaches it

Lock c14ff18 (before any OS output). Analysis `scripts/os_analysis.py` → `pod/os/os_results.json`. 1× H100, ≈ $1.6.
No engineering events.

## Validity
Direct-prompt BASE and WINNER on the M1 hold-out: 208/600 and 224/600, identical to M1, with **identical answers on all
600 items** in both; BASE fingerprint and the winner's expected_state_id matched; exact base restore (mismatch 0).
Direct-prompt BASE on SEARCH200: 72/200 (36.0%), the original search's recorded base accuracy. → **VALID**.

## Prompt selection (BASE accuracy on SEARCH200, the winner's own selection data)
direct 36.0 / zeroshot_cot 38.5 / **manual_cot 39.5** → **manual_cot chosen** (parse failures counted wrong: 0 / 15 / 10).

## Primary OS-1
D = WINNER (direct prompt, as M1) − BASE (manual_cot) on the hold-out = **−1.50 pp [−5.35, +2.23]** (image-cluster
bootstrap as M1) → **NO DIFFERENCE DETECTED** (not equivalent). The point estimate favours the prompt.

## Secondary
| Hold-out (600 items, 484 images) | direct | zeroshot_cot | manual_cot |
|---|---|---|---|
| BASE | 34.67 | **39.83** | 38.83 |
| WINNER (seed 9504111) | 37.33 | 36.00 | 38.17 |
| **Winner gain** [image-cluster 95% CI] | **+2.67 [+0.17, +4.93]** | **−3.83 [−6.81, −0.84]** | −0.67 [−3.95, +2.51] |
| Parse failures (BASE / WINNER) | 0 / 0 | 39 / 31 | 32 / 21 |

- **OS-2, prompt held fixed (manual_cot): winner gain −0.67 [−3.95, +2.51]**, no difference.
- Under zeroshot_cot the winner is worse than the unperturbed model (−3.83, CI excludes 0).
- Parse failures under the CoT prompts are scored wrong, which can only lower the prompted BASE (the comparison is
  conservative toward the winner's direct-prompt score).

## Reading
The only selected perturbation in this study whose gain transferred to fresh data and was not explained by its
answer-content tilt has that gain only under the prompt it was selected with (+2.67 under direct). Under the benchmark's
step-by-step prompts its advantage disappears (manual_cot) or reverses (zeroshot_cot), and a prompt chosen on its own
selection data gives the unperturbed model a hold-out accuracy at least as high as the winner's (38.83 vs 37.33; no
difference detected, CI [−5.35, +2.23]). This is the third setting (with GQA and OLMo) where the selected perturbation's
advantage is specific to the prompt it was selected under. The comparison is underpowered for equivalence (600 items,
484 image clusters).

# O2 result: RandOpt's GSM8K win on OLMo-2-1B is a prompt effect; asked plainly, one base generation beats RandOpt

Lock 6ecafcf (before any O2 output). Analysis `scripts/o2_analysis.py` → `pod/o2/o2_results.json`. 1× H100, ≈ $5.2
(O1 + O2 ≈ $51.8 of the user's $60).

## Engineering note (no effect on results)
The runner's stored per-generation `c` field is wrong (it passed the ground truth as "#### x"; RandOpt's scorer expects
"x"), found in the smoke run. The locked analysis never reads it: it rescores every arm from the saved answers with the
correct ground truth (as its self-test did). Texts are saved for every arm, including SC.

## Validity
- Environment: RandOpt-prompt base greedy 33.66% = O1's greedy 33.66% (same scorer); answers identical on all 1319
  items → **VALID**.
- **Member fidelity gate: FAILED** (as locked, the members secondary is reported as not valid). Rebuilt members agree
  with O1's dumped answers on 70.2% of items on average (min 65.2%; gate ≥ 90%), while their accuracy matches
  (mean |Δ| 0.63 pp; gate ≤ 2). For scale: two different O1 members agree on 41%, a member and base on 37%. So the
  seeds are right and the models nearly the same; the likely cause is randopt.py's in-place subtract-to-restore in bf16
  (weights drift over ~625 cycles per engine), which O2's exact restore does not reproduce.

## Primary O2-1
D = RandOpt K = 50 (O1, RandOpt's prompt) − SC@50 (plain prompt) = **−22.21 pp [−24.87, −19.56] → PROMPT-SC AHEAD.**
47 RandOpt-only vs 340 SC-only correct.

## Secondary
| GSM8K / OLMo-2-1B, 1319 items | RandOpt's prompt | plain (question only) | boxed |
|---|---|---|---|
| BASE greedy (1 generation) | 33.66 | **65.28** | 67.85 |
| SC@10 / SC@50 | – / 43.75 (O1) | 73.16 / **74.68** | 74.68 / 76.12 |
| RandOpt K = 50 (O1) | 52.46 | – | – |
| mean tokens, BASE | 168 | 223 | 212 |
| BASE contains "####" / \boxed | 97.0% / 4.1% | 0% / 96.7% | 0% / 96.8% |
| BASE hits max_tokens | 2.4% | 3.9% | 3.9% |

- RandOpt − one plain-prompt BASE generation: **−12.81 [−15.69, −9.86]**; boxed: −15.39 [−18.20, −12.43].
- Boxed prompt: RandOpt − SC@50 −23.65 [−26.23, −21.08] (PROMPT-SC AHEAD).
- Search on top of the prompt (**not valid**, fidelity gate failed; descriptive only): rebuilt members with the plain
  prompt average 62.2% (below plain BASE 65.3) and vote 73.8 vs SC@50 74.7: −0.91 [−2.27, +0.53].

## Reading
Under RandOpt's GSM8K prompt ("…output the final answer after ####"), OLMo-2-1B follows the format (97% write "####")
but reasons worse and shorter, and scores half of what it scores when simply asked the question (its own habit is a
\boxed answer). RandOpt's selection finds perturbations that partly undo that damage (+5 pp per member, O1), which is
why it beat SC under the same prompt. With the plain question, one greedy generation (65.3%) beats RandOpt's
5000-perturbation search plus 50-model vote (52.5%) by 12.8 pp, and SC@50 beats it by 22.2 pp. Together with G4 (GQA),
both settings where RandOpt beat SC are cases where RandOpt's prompt costs the base model heavily and a prompt fix
gives more than weight search.

# Results index

Every experiment folder under `paper-analysis/`, what it asked, where its pre-registration lock and result are, and
what the paper does with it. Locks were committed before any output of the experiment; a lock commit can be checked
with `git show <commit> --stat`. Full numbers and every locked test with its outcome: `paper/RESULTS_MASTER.md`
(section in brackets; R9 is the confirmatory ledger). Paper role follows `paper/STORY.md`.

Paper role: **main** = main text; **appx** = appendix; **sup** = superseded (kept as a record, mentioned once).

## Claim 1: what RandOpt's weight search buys (same-run comparisons and controls)
| Folder | Question | Lock | Result | Verdict | Role |
|---|---|---|---|---|---|
| `c-sameRun` | RandOpt (N = 5000, K = 50) vs SC@50, GSM8K / Qwen2.5-1.5B | 1437d44 + 6118222 | `C_RESULT.md` | **SC ahead**, −2.65 [−4.32, −0.99] | main [R8b] |
| `c3b-sameRun` | same, GSM8K / Qwen2.5-3B | e78cfd0 | `C3B_RESULT.md` | **SC ahead**, −1.59 [−2.65, −0.53] | main [R8b] |
| `g1` | frozen GQA selection items and candidates (RandOpt setting, matched image-disjoint splits) | f57b8ae | `plan_lock.md` | items used by G2–G4 | appx |
| `g2-sameRun` | same-run RandOpt vs SC@50, GQA / Qwen2.5-VL-3B | d2b3d68 + 5be1720 | `G2_RESULT.md` | **RandOpt ahead**, +3.47 [+1.62, +5.41] | main [R8b] |
| `g2r-seed` | does G2 replicate with a disjoint population (seed 43)? | 9e84c4f + 0754376 | `G2R_RESULT.md` | **replicated**, +3.63 [+1.78, +5.49] | main [R8b] |
| `g3-termination` | is the GQA advantage a termination repair (256 vs 1024 tokens)? | dc19fd4 | `G3_RESULT.md` | **partial** (about a quarter) | main [R8b] |
| `robustness` | Claim 1 under strict scoring, McNemar + Holm, image-cluster CIs, K = 10; data alignment | none (exploratory) | `ROBUSTNESS.md` | holds (strengthens under strict scoring) | appx, one sentence in main [R8b.6] |
| `expert-specificity` | are selected perturbations prompt-specific; do they reproduce the good prompt's answers? | none (exploratory) | `expert_specificity.json` | prompt-specific (ρ 0.025 / 0.171, overlap 0); GQA yes, OLMo no | main [R8b.8] |
| `transfer` | selection-set gain vs test gain across all ten searches (same-engine bases) | none (exploratory) | `transfer_table.md` | transfers only where the prompt leaves ≥ 10 pp unclaimed | main, bridge [R8b.7] |
| `gqa-shift` | what selected GQA models do differently (CPU, on G3 texts) | none (exploratory) | `SHIFT_RESULT.md` | answer directly; shorter = more accurate | main, labelled exploratory |
| `g4-direct-prompt` | does a direct-answer prompt match RandOpt on GQA? | 2ad4e46 | `G4_RESULT.md` | **no difference**; 1 direct generation 64.7 vs RandOpt 63.5; search adds nothing on top | main [R8b] |
| `o1-olmo-sameRun` | same-run RandOpt vs SC@50 on a non-Qwen model, GSM8K / OLMo-2-1B | e2e2400 | `O1_RESULT.md` | **RandOpt ahead**, +8.72 [+6.75, +10.77]; members +5.1 over base | main [R8b] |
| `ps-prompt-selection` | prompt chosen on RandOpt's own selection set vs RandOpt, all four rows | a1caf88 | `PS_RESULT.md` | **no row RandOpt ahead** (2 SC ahead, 2 n.d., one equivalent) | main [R8b] |
| `q2-qwen-prompt` | same prompt control on the Qwen GSM8K rows | 2a55ccc | `Q2_RESULT.md` | plain prompt worse for Qwen (RandOpt beats plain SC +2.58, +7.05); damage prediction supported 4/4 | main [R8b] |
| `o2-olmo-prompt` | is O1's RandOpt win a prompt effect? (plain and boxed prompts) | 6ecafcf | `O2_RESULT.md` | **prompt effect**: plain-prompt SC@50 74.7 vs RandOpt 52.5, −22.21 [−24.87, −19.56]; member fidelity gate failed | main [R8b] |
| `gb-gsm8k-boxed-search` | RandOpt's search under the boxed prompt on GSM8K (prompt held fixed) | 77332dd | `GB_RESULT.md` | OLMo: **SC ahead** −2.12 [−3.49, −0.83]; Qwen-1.5B: INVALID-ENV (not run; run in RV-2b) | main [R8b] |
| `gd-gqa-direct-search` | RandOpt's search under the direct prompt on GQA (prompt held fixed) | 0bb89eb | `GD_RESULT.md` | **equivalent**: −0.32 [−1.29, +0.65]; members no better than base | main [R8b] |
| `rv-reviewer-round` | reviewer round: public-template prompt selection (RV-1), the prompt held fixed on the Qwen rows (RV-2a/2b), randopt.py's base print (RV-4), SC temperature (RV-5); a second OLMo search (RV-3) | c33fec2; amendments `amendment_1.md`–`amendment_3.md` | `RV_RESULT.md` | RV-1: 3/4 rows with public templates alone (Qwen-1.5B **RandOpt ahead** +4.17); RV-2a **SC ahead** −1.36; RV-2b **SC ahead** −4.32; RV-3 OLMo **replicated** +10.39; RV-4 engine-specific base print; RV-5 no change | main [R8b.5, R8b.9, R8b.10] |
| `p2` | SC@50 vs published RandOpt, GSM8K-0.5B/1.5B, GQA | 01efd9d | `P2_RESULT.md` | GSM8K-1.5B matches; others fail the base gate | main (gated rows), appx |
| `p3` | same, GSM8K-3B, MATH-500 | c60b600 | `P3_RESULT.md` | GSM8K-3B matches; MATH-500 not comparable | main (gated row), appx |
| `p1` | do perturbations behave like sampling (CoT regime)? | 464aed0 | `P1_RESULT.md` | same items flip; persistent models | main (A/B), appx (C) [R7] |
| `p0` | first-order law per question in the RandOpt GQA setting (direct answers) | d5152f7 | `P0_RESULT.md` | **GO** r = 0.930; CoT link absent | main [R4, R7] |

## Claim 2: the first-order tilt law and its limits
| Folder | Question | Lock | Result | Verdict | Role |
|---|---|---|---|---|---|
| `stage12` | is a perturbation's answer tilt predictable from its noise (Qwen3-VL-8B)? | 09c1f1f | `STAGE12_RESULT.md` | **pass**, r = 0.938 | main [R4] |
| `stage3` | localization (3B) and noise-only predictor vs search (3A) | ce60c5d | `STAGE3_RESULT.md` | 3B pass (middle layers, r = 0.978); 3A mixed | main [R5, R6] |
| `r1` | does the law replicate on a new task and content class? | 00155cc | `R1_RESULT.md` | **replicated**, r = 0.915 | main [R4] |
| `r2` | same, Qwen2.5-VL-7B | 47d2102 | `R2_RESULT.md` | inconclusive (reliability gate failed) | appx |
| `r2b` | R2 with a remedy fixed in advance | 1cb3252 | `R2B_RESULT.md` | **replicated**, r = 0.925 | main [R4] |
| `i5` | does the law hold per item, not only on average? | a1c2a7a | `I5_RESULT.md` | **item-level**, r = 0.771 | main [R4] |
| `s1` | law on a non-Qwen model (OLMo-2-1B, ARC); winner per item (S1-B) | 797fa96 | `S1_RESULT.md` | A-1 **GO** r = 0.790; B-1 NO-GO | main [R4] |
| `s1-7b` | law at 7B (OLMo-2-7B, ARC) | d64a1ad | `S1_7B_RESULT.md` | **GO**, r = 0.787 | main [R4] |
| `geometry-gpu-a` | whole-model first-order with vision weights perturbed | 198c352; A1 79252d4 | `GPU_A_RESULT_A1.md` | F1 **falsified** (r = 0.137): a stated limit | main (limit), appx [R4] |
| `theory` | flip probability, unselected vote, sampling-temperature analogy | f06f3c6 (criteria-first) | `theory_results.json` | T1 AUC 0.935; T2 96/96 | main [R8c] |

## Claim 3: the OmniSpatial case study
| Folder | Question | Lock | Result | Verdict | Role |
|---|---|---|---|---|---|
| `m1` | does the search winner transfer to fresh matched items? | 5eb864b | `M1_C1_RESULT.md` | transfers, +2.67 [0.17, 4.93] (not full) | main [R2] |
| `os-prompt-control` | can the benchmark's own prompts, chosen on SEARCH200, reach the winner's gain? | c14ff18 | `OS_RESULT.md` | **no difference** (−1.50 [−5.35, +2.23]); winner's gain prompt-specific (−0.67 held fixed) | main [R2b] |
| `c1` | does removing answer-content priors remove the winner's gain? | 5aa4c06 | `plan_lock.md`, `c1_results.json` | **not explained** (2.33 of 2.67 remain) | main [R3] |
| `tau-check` | does the noise-only predictor work through the front tilt? | 8c52ba1 (criteria first) | `TAU_CHECK_RESULT.md` | holds (exploratory data) | main [R6] |
| `answer-prior` | answer-content priors of candidates (post-hoc) | none | `answer_prior_*.json` | label-aligned front tilt | main, labelled post-hoc [R3] |

## Superseded (split mismatch; see RESULTS_MASTER R1)
Kept because later analyses read their data (Stage 3, answer priors, GPU-A), directly or through
`selection_vs_specificity.py`, which imports `selection_vs_shared.py`.

| Folder | Question | Lock | Role |
|---|---|---|---|
| `random-control-transfer` | random-control transfer of candidate 9504111 | 42c9718 | sup |
| `selection-vs-specificity` | selection vs candidate specificity | faf824a | sup |
| `selection-vs-shared-response` | selection vs shared response | 00aab64 | sup |
| `causal-diagnostic` | causal diagnostic for 9504111 | (session records) | sup |

## Other
- `perspective-taking-n5000-20261004/`: raw outputs of the original OmniSpatial N = 5000 search (Qwen3-VL-8B) that the
  case study and the tilt-law experiments build on.
- `paper-analysis/paper_master_*`: tables generated by `paper/figures/scripts/build_master.py`.
- Each GPU experiment's `pod/` subfolder holds the pod's job logs, return codes, cost record (`rate.txt`) and outputs.

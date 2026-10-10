| Study | Lock | Test | Statistic | Rule (fixed in the lock) | Result | Outcome |
|---|---|---|---|---|---|---|
| GPU-A run 1 | 198c352 | S2a HF vs vLLM fidelity | median per-item max \|Δlog p\|; argmax agreement | ≤ 0.10; ≥ 99% | 0.25; 98% | **NO-GO** (BF16 output resolution; led to amendment A1) |
| GPU-A A1 | 79252d4 | V0 score-definition gate; S2b candidate bytes | fp32 contrast checks; packed-tensor SHA | as locked | pass; 642/642 | pass |
| GPU-A A1 | 79252d4 | F1 whole-model first-order (vision perturbed) | pooled r, 13 candidates × 761 items | ≥ 0.6 support; < 0.3 falsify | 0.137 | **falsified** |
| GPU-A A1 | 79252d4 | F2 changed answers predicted to flip | share | < 0.5 falsify | 0.53 | not falsified (weak) |
| GPU-A A1 | 79252d4 | F3 RERANK gain prediction | r over 543 | ≥ 0.5 support; < 0.3 falsify | 0.376 | inconclusive |
| GPU-A A1 | 79252d4 | S2d group partials vs insertions | pooled r | < 0.5 falsify | 0.239 | **falsified** |
| Stage 1+2 | 09c1f1f | S1a reliability; S1b fidelity | Spearman–Brown; exactness | ≥ 0.7; all exact | 0.878; exact | pass |
| Stage 1+2 | 09c1f1f | **S2 tilt law** (Qwen3-VL-8B) | r(pred_NOV, T_NOV), 80 perturbations | ≥ 0.6 pass; < 0.3 falsify | 0.938 (slope 0.83) | **pass** |
| Stage 1+2 | 09c1f1f | S2 share carried by non-vision | r(T_NOV, T_FULL)² | ≥ 0.5 | 0.843 | pass |
| Stage 3 | ce60c5d | 3A-0 pipeline consistency | r | ≥ 0.9 | 0.938 | pass |
| Stage 3 | ce60c5d | 3A-1 τ vs SEARCH gain | r over 5000 | ≥ 0.25 pass; < 0.10 falsify | 0.240 | inconclusive |
| Stage 3 | ce60c5d | 3A-2 τ vs RERANK gain | r over 543 | same | 0.402 | pass |
| Stage 3 | ce60c5d | 3A-3 enrichment of advanced candidates | mean z difference | ≥ 0.3 SD | +0.09 [−0.01, 0.18] | not supported |
| Stage 3 | ce60c5d | 3B-1 block-level law | pooled r, 160 pairs | ≥ 0.6 pass; < 0.3 falsify | 0.978 (slope 0.95) | **pass** |
| Stage 3 | ce60c5d | 3B-2 additivity; 3B-3 dominance | r; variance share | ≥ 0.9; ≥ 50% for one block | 0.96; none | valid; distributed |
| R1 | 00155cc | reliability; position share; **law** | SB; share; r | ≥ 0.7; < 0.20; ≥ 0.6 | 0.866; 0.184; **0.915** | **replicated** |
| R1 | 00155cc | middle-layer share | share of predicted variance, layers 6–23 | ≥ 0.5 | 0.73 | supported |
| R2 | 47d2102 | reliability; **law** (Qwen2.5-VL-7B, 96 items) | SB; r | ≥ 0.7; ≥ 0.6 | **0.518**; 0.920 | **inconclusive** (reliability gate failed) |
| R2b | 1cb3252 | reliability; position share; **law** (363 items; remedy fixed in advance) | SB; share; r | ≥ 0.7; < 0.20; ≥ 0.6 | 0.949; 0.073; **0.925** | **replicated** |
| I5 | a1c2a7a | consistency gate; **per-item law** | r; pooled r, 7680 pairs | ≥ 0.99; ≥ 0.6 item-level, < 0.3 average-only | 0.99999; **0.771** | **item-level** |
| M1 | 5eb864b | M1-1 winner on fresh matched items | gain, image-cluster CI | transfers if CI > 0; full if also ≥ +6 | +2.67 [0.17, 4.93] | transfers (not full) |
| M1 | 5eb864b | M1-2 gold-front concentration | gain difference, CI | supports if CI > 0 | +1.95 [−2.57, 6.47] | not supported |
| M1 | 5eb864b | M1-3 τ vs fresh gain | r over 59 (Fisher CI) | ≥ 0.25 supports | 0.61 [0.42, 0.75] | supports |
| M1 | 5eb864b | M1-4 top-50 vs controls | mean difference, bootstrap CI | reported | +1.16 [0.31, 2.03] | reported |
| C1 | 5aa4c06 | **C1-1 prior calibration of the winner** | calibrated vs raw fresh gain | explains if ≤ 0.5×raw with CI; not explained if ≥ 0.8×raw | 2.33 vs 2.67 | **not explained** |
| C1 | 5aa4c06 | C1-2 surviving share (candidates with raw g > 0) | Σg_cal / Σg | ≤ 0.5 supports | ≈ 0.00 [−0.42, 0.32] | locked label "supports"; **flagged**: a cross-fitted exploratory check shows it is regression-to-mean, so it is not evidence |
| τ-check | 8c52ba1 | criteria fixed before computing (exploratory data) | within-top-50 r; partial r; control sign | ≥ 0.30; ≥ 0.20; ≥ 0 | 0.59; 0.31; 0.57 | holds (works through the front tilt) |
| P0 | d5152f7 | **per-question first-order** (LM-only GQA) | pooled r, σ ≤ 0.002 | ≥ 0.5 GO; < 0.3 NO-GO | **0.930** | **GO** (σ = 0.005: 0.31) |
| P1 | 464aed0 | P1-A flip propensities | Spearman over 400 items | ≥ 0.6 supports | 0.946 | supports |
| P1 | 464aed0 | P1-B persistence | r(selection, held-out) over 64 | persistent if ≥ 0.3 with CI > 0 | 0.74 [0.61, 0.84] | **persistent** (rejects pure re-sampling) |
| P1 | 464aed0 | P1-C top-8 vote vs SC at T\* | difference, item bootstrap CI | advantage if CI > 0 | +2.5 [−3.5, 8.5] | no advantage (underpowered) |
| P2 | 01efd9d | base gate; SC@50 vs published RandOpt, GSM8K-1.5B | greedy diff; SC − RandOpt | ±2.0; match if ≥ −1.0 | 0.5; +3.4 | gate pass; MATCHES (label) |
| P2 | 01efd9d | same, GSM8K-0.5B / GQA-VL-3B | greedy diff | ±2.0 / ±2.5 | +3.3 / −2.6 | gate **fail**: not comparable |
| P3 | c60b600 | base gate; SC@50 vs published RandOpt, GSM8K-3B | greedy diff; SC − RandOpt | ±2.0; ≥ −1.0 | 0.5; +1.1 | gate pass; MATCHES (label) |
| P3 | c60b600 | same, MATH-500-1.5B / 3B | greedy diff | ±2.0 | +6.4 / +5.8 | gate **fail**: not comparable |
| S1 | 797fa96 | **A-1 law on a non-Qwen model** (OLMo-2-1B, ARC-Challenge) | pooled r, σ ≤ 0.002, 1600 pairs | ≥ 0.5 GO; < 0.3 NO-GO | **0.790** [0.753, 0.824] | **GO** |
| S1 | 797fa96 | B-1 winner per-item first-order (all parameters) | r over 600 fresh items | ≥ 0.5 GO; < 0.3 NO-GO | 0.162 [0.067, 0.252] | **NO-GO** |
| S1 | 797fa96 | B-2 sign agreement on changed items | share, n = 52 | ≥ 0.70 supported; < 0.60 not | 0.654 | inconclusive |
| C | 1437d44 + 6118222 | validity; base gate | recomputed = printed; ±2.0 | exact; required | 1018 = 1018; +1.5 / +0.5 | valid; pass |
| C | 1437d44 + 6118222 | **C-1 same-run RandOpt (N = 5000, K = 50) − SC@50** | paired item bootstrap | CI > 0 RandOpt ahead; CI < 0 SC ahead | **−2.65 [−4.32, −0.99]** | **SC AHEAD** |
| C | 1437d44 + 6118222 | K = 10 (secondary) | same | same | +1.97 [0.00, 3.87] | no difference detected |
| C3B | e78cfd0 | validity; base gate | recomputed = printed; ±2.0 | required | 1143 = 1143; +0.9 / +0.5 | valid; pass |
| C3B | e78cfd0 | **C3B-1 same-run RandOpt − SC@50, GSM8K-3B** | paired item bootstrap | as C-1 | **−1.59 [−2.65, −0.53]** | **SC AHEAD** |
| C3B | e78cfd0 | K = 10 | same | same | −1.06 [−2.35, 0.15] | no difference detected |
| G2 | d2b3d68 + 5be1720 | environment gate | base vs P2 greedy, same items | ≤ 1.0 pp | 0.16 | valid |
| G2 | d2b3d68 + 5be1720 | **G2-1 same-run RandOpt − SC@50, GQA** | paired item bootstrap | as C-1 | **+3.47 [+1.62, +5.41]** | **RANDOPT AHEAD** |
| G2 | d2b3d68 + 5be1720 | K = 10 | same | same | +4.36 [+2.26, +6.54] | RandOpt ahead |
| G2R | 9e84c4f + 0754376 | **G2R-1 GQA advantage with a disjoint population (seed 43)** | paired item bootstrap | REPLICATED if CI > 0 | **+3.63 [+1.78, +5.49]**; D₂ − D₁ +0.16 [−0.97, +1.29] | **REPLICATED** |
| O1 | e2e2400 | **O1-1 same-run RandOpt − SC@50, GSM8K / OLMo-2-1B (non-Qwen)** | paired item bootstrap | as C-1 | **+8.72 [+6.75, +10.77]** | **RANDOPT AHEAD** (Claim 1 model-dependent on GSM8K) |
| O1 | e2e2400 | K = 10; O1b second seed | same; budget rule | – | +7.58 [+5.31, +9.93]; O1b not run ($87 > $58.5) | RandOpt ahead; skipped by rule |
| PS | a1caf88 | **PS: RandOpt K = 50 − SC@50 under the prompt chosen on RandOpt's selection set** (Qwen-1.5B / Qwen-3B / OLMo / GQA) | paired item bootstrap | as G4 | **−2.96 [−4.62, −1.29] / −0.38 [−1.67, 0.91] / −23.65 [−26.23, −21.08] / −1.21 [−2.83, 0.40]** | **SC ahead / equivalent / SC ahead / n.d.: no row RandOpt ahead** |
| OS | c14ff18 | **OS-1 OmniSpatial winner (direct) − BASE under the prompt chosen on SEARCH200 (manual_cot)** | image-cluster bootstrap (as M1) | as G4 | **−1.50 [−5.35, +2.23]** | **NO DIFFERENCE DETECTED** (valid; answers identical to M1) |
| OS | c14ff18 | OS-2 winner gain with the prompt held fixed (manual_cot) | same | reported | −0.67 [−3.95, +2.51] | no difference (under zeroshot_cot −3.83 [−6.81, −0.84]) |
| GD | 0bb89eb | **GD-1 RandOpt searched under the direct prompt − SC@50 direct, GQA (prompt held fixed)** | paired item bootstrap | as G4 | **−0.32 [−1.29, +0.65]** | **NO DIFFERENCE, EQUIVALENT (±2 pp)** (valid; finished on a second pod, cross-pod answers identical) |
| GB | 77332dd | **GB-1 RandOpt searched under the boxed prompt − SC@50 boxed, GSM8K OLMo-2-1B (prompt held fixed)** | paired item bootstrap | as G4 | **−2.12 [−3.49, −0.83]** | **SC AHEAD** (env and fidelity gates passed) |
| GB | 77332dd | GB-2 same, Qwen2.5-1.5B | environment gate | ±1.0 pp | base selection 68.00 vs 73.00 printed by randopt.py | **INVALID-ENV, not run** |
| Q2 | 2a55ccc | **Q2-1/2 RandOpt − SC@50 with the plain prompt, Qwen 1.5B / 3B** | paired item bootstrap | as G4 | **+2.58 [0.53, 4.62] / +7.05 [5.23, 8.95]** | **RANDOPT AHEAD** (plain prompt is worse for Qwen) |
| Q2 | 2a55ccc | damage prediction (all four rows) | plain − RandOpt-prompt base | supported if Qwen both < 5 pp | +4.70, −7.20 (GQA +11.3, OLMo +31.6) | **supported** (4/4 rows, locked measure); not decisive: vs the PS-chosen prompt Qwen-1.5B damage is +10.2 and RandOpt still lost |
| O2 | 6ecafcf | **O2-1 RandOpt (O1) − SC@50 with the plain question prompt, OLMo-2-1B** | paired item bootstrap | as G4 | **−22.21 [−24.87, −19.56]** | **PROMPT-SC AHEAD** (valid env) |
| O2 | 6ecafcf | member fidelity gate; search on top of prompt | answer agreement ≥ 0.90, acc ≤ 2 pp | as locked | 0.70 / 0.63 pp; −0.91 [−2.27, 0.53] | **gate failed** → members secondary not valid |
| G4 | 2ad4e46 | **G4-1 RandOpt (CoT) − SC@50 with a direct-answer prompt, GQA** | paired item bootstrap | as C-1, + equivalence ±2 | **−1.21 [−2.83, +0.40]** | **NO DIFFERENCE DETECTED** (not equivalent) |
| G4 | 2ad4e46 | direct-prompt BASE − CoT BASE; search on top of prompt | paired bootstrap | reported | +11.31 [8.48, 14.14]; −0.08 [−0.97, +0.81] | prompt effect; search adds nothing |
| G3 | dc19fd4 | **G3-1 does a 1024-token budget remove the GQA advantage?** | Δ = D256 − D1024; D1024 | supported if Δ CI > 0 and D1024 CI ∋ 0 | Δ +0.97 [0.32, 1.70]; D1024 +2.67 [0.73, 4.52] | **PARTIAL** |
| G3 | dc19fd4 | G3-2 base vs members non-termination at 256 | difference, CI | supported if CI > 0 | +1.90 [0.31, 3.48] | supported (small) |
| S1-7B | d64a1ad | **7B-1 law at 7B** (OLMo-2-7B, ARC) | pooled r, σ ≤ 0.002 | ≥ 0.5 GO; < 0.3 NO-GO | **0.787** [0.758, 0.820] | **GO** |
| RV | c33fec2 | **RV-1 RandOpt − SC@50 under the public template chosen on the selection set** (Qwen-1.5B / Qwen-3B / OLMo / GQA) | paired item bootstrap | as PS | **+4.17 [+1.97, +6.44] / −0.45 [−1.67, +0.76] / −5.23 [−7.66, −2.81] / −2.67 [−4.60, −0.73]** | **RandOpt ahead / equivalent / SC ahead / SC ahead** (pooled six-candidate rule: SC ahead / equiv. / SC ahead / SC ahead) |
| RV | c33fec2 | **RV-2a RandOpt searched under boxed − SC@50 boxed, Qwen2.5-3B (prompt held fixed)** | paired item bootstrap | as GB | **−1.36 [−2.35, −0.38]** | **SC AHEAD** (gates passed; resumed across sessions, amendment 1) |
| RV | c33fec2 | **RV-2b same, Qwen2.5-1.5B** (amendment 2; GB-2 re-run with the lock's gates) | paired item bootstrap | as GB | **−4.32 [−5.91, −2.81]** | **SC AHEAD** (gates passed) |
| RV | c33fec2 | RV-3 second OLMo-2-1B search (seed 43) | – | – | – | **not run** (GPU budget; amendments 1–2) |
| RV | c33fec2 | RV-4 randopt.py base print, Qwen2.5-1.5B, fresh pod | reproduction | reading fixed in the lock | randopt.py 73.00 (4 and 1 engines) vs our runner 68.00 | **engine-specific base print**: fidelity limitation; exploratory R8b.1/R8b.7 recomputed |
| RV | c33fec2 | RV-5 SC@50 at T = 0.5 / 1.0 (Qwen-3B boxed; GQA direct) | paired item bootstrap | reported; T = 0.7 stays primary | −0.23 / −1.14; −1.05 / +0.24 (all CIs ∋ 0) | no outcome changes |

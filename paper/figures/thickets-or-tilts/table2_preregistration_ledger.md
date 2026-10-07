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

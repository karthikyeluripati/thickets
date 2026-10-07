| Test | Lock | Statistic | Rule | Result | Outcome |
|---|---|---|---|---|---|
| GPU-A H*, F1 (whole model incl. vision) | `79252d4` | pooled r(pred, measured contrast) | ≥0.6 support / <0.3 falsify | 0.137 | falsified |
| GPU-A H*, S2d (group partials) | `79252d4` | pooled r | <0.5 falsify | 0.239 | falsified |
| Stage 2: tilt law | `09c1f1f` | r(pred, measured), 80 pert. | ≥0.6 pass | 0.938 | pass |
| Stage 3A-1: τ vs SEARCH gain | `ce60c5d` | r over 5000 | ≥0.25 pass / <0.10 falsify | 0.240 | inconclusive |
| Stage 3A-2: τ vs RERANK gain | `ce60c5d` | r over 543 | ≥0.25 pass | 0.402 | pass |
| Stage 3B: block localization | `ce60c5d` | pooled r, 160 pairs | ≥0.6 pass | 0.978 | pass |
| R1: new task / content word | `00155cc` | r(pred, measured) | ≥0.6 pass | 0.915 | replicated |
| R2: second model (96 items) | `47d2102` | r; reliability gate | reliability ≥0.7 | 0.920; rel. 0.52 | inconclusive |
| R2b: second model (363 items) | `1cb3252` | r; reliability gate | same | 0.925; rel. 0.95 | replicated |
| I5: per-item law | `a1c2a7a` | pooled r, 7680 pairs | ≥0.6 pass | 0.771 | pass |
| M1-1: winner on fresh matched items | `5eb864b` | gain, cluster CI | CI > 0 | +2.67 [0.17, 4.93] | transfers (not full) |
| M1-2: front concentration | `5eb864b` | front − non-front gain | CI > 0 | +1.95 [−2.57, 6.47] | not supported |
| M1-3: τ vs fresh gain | `5eb864b` | r over 59 | ≥0.25 | 0.61 | pass |
| M1-4: top-50 vs controls | `5eb864b` | mean gain difference | reported | +1.16 [0.31, 2.03] | – |
| C1-1: prior calibration (winner) | `5aa4c06` | calibrated vs raw gain | ≤0.5× explains | 2.33 vs 2.67 | not explained |
| P0: per-question first-order (LM-only GQA) | `d5152f7` | pooled r, σ≤0.002 | ≥0.5 GO | 0.930 | GO |
| P1-A: flips match sampling | `464aed0` | Spearman over 400 items | ≥0.6 | 0.946 | supports |
| P1-B: persistence | `464aed0` | r(sel, held-out) over 64 | persistent if ≥0.3 | 0.74 | persistent |
| P1-C: top-8 vote vs SC | `464aed0` | diff vs SC@T*, CI | advantage if CI > 0 | +2.5 [−3.5, 8.5] | no advantage |
| P2: GSM8K-1.5B SC vs RandOpt | `01efd9d` | SC@50 − RandOpt | match ≥ −1.0 | +3.4 | matches |
| P3: GSM8K-3B SC vs RandOpt | `c60b600` | SC@50 − RandOpt | match ≥ −1.0 | +1.1 | matches |
| P2/P3: other 4 rows | `01efd9d / c60b600` | base-reproduction gate | ±2.0 pp (GQA ±2.5) | failed | not comparable |

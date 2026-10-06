# C1 plan lock (item 3): does removing each candidate's answer-content prior shift remove its "expert" gain?

Locked before any M1 result is read. CPU only: it uses the M1 outputs (first-token A–D log-probs for BASE and the
measured candidates on the fresh compass-format hold-out H). No GPU.

## Calibration (per candidate, cross-fitted, no use of gold labels in the fit)
- Split H into two halves by image: **half A** holds items whose SHA256('c1-half-v1:'+image_sha256) has an even first
  hex digit; the rest are **half B**. Every item of an image lands in the same half.
- **Features** x_k for option k: indicators for the option text containing front/forward, back, left, right (the
  `answer_prior_analysis.feats` rule), centered within the item.
- **Fit on one half:** b_c = least-squares fit of the centered score change (z_cand − z_base, each centered within the
  item) on the centered x. This is a 4-parameter global content prior shift. Gold answers are not used.
- **Apply to the other half:** calibrated scores z_cal = z_cand − X b_c.
- Then swap the halves, so every item is calibrated by a fit from the other half.
- **Scoring:** answer = argmax of the A–D scores, for BASE, raw candidate and calibrated candidate alike, so raw and
  calibrated use the same rule. Agreement of argmax with the parsed first token is reported.
- **Gains** (pp vs BASE, argmax-scored): raw g and calibrated g_cal.

## Tests
| Test | Statistic | Reading |
|---|---|---|
| **C1-1 (primary): 9504111** | raw g and g_cal on H with a paired cluster bootstrap over images (2000, seed 20261006) of (g − g_cal) | **PRIOR-SHIFT EXPLAINS** if raw g > 0 and g_cal ≤ 0.5·g with the CI of (g − g_cal) excluding 0; **NOT EXPLAINED** if g_cal ≥ 0.8·g; else PARTIAL. If raw g ≤ 0 (no gain to explain), report **NOT APPLICABLE** |
| C1-2: population | over measured candidates with raw g > 0: Σ g_cal / Σ g (share of positive gain surviving calibration), with a bootstrap over candidates | reported. ≤ 0.5 supports "most expert gain is prior shift" |
| C1-3 | r(raw g, g_cal) across candidates; mean g_cal for SEARCH-top-50 vs controls | reported |
| C1-4 sanity | calibration on BASE vs BASE (b = 0 by construction) and the share of variance of the score changes explained by the 4-parameter shift | reported |

No re-tuning of the features, split, scoring or thresholds.

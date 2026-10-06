# M1 plan lock: does the selected advantage transfer to fresh items of the SAME format?

Locked before any M1 measurement. Motivation: `paper/CORRECTION_SPLIT_MISMATCH.md`. TEST is a different question format,
so the original transfer question was never tested. M1 is the matched-split test that question needed.
GPU: proposed cap **$6.00**, one pod session (needs the user's go and a pod).

## Frozen inputs
- **Hold-out items** (`holdout_candidates.json`): OmniSpatial-train Perspective_Taking, official data.json (sha256 verified
  against the pinned manifest). Items qualify if they are:
  - not in SEARCH or RERANK;
  - on an image file not used by SEARCH, RERANK or TEST;
  - 4 options, compass format (the `answer_prior_analysis.fmt` rule).

  That gives 1911 eligible items. M1 uses the 600 with the smallest SHA256('m1-holdout-v1:'+uid): 484 images,
  303 gold-front.
- **Leakage rule (applied on the pod, before any model call):** drop any item whose image is byte-identical (sha256) to a
  SEARCH/RERANK/TEST image. The remaining items form H (`gpu/holdout.jsonl`).
- **Candidates** (`frozen_candidates.json`, 61), measured in this order:
  1. 9504111;
  2. the 12 frozen random controls;
  3. the 48 other SEARCH-top-50 members.

## Protocol
- Identical to the original study: Qwen3-VL-8B @ 0c351dd, vLLM 0.11 in-process, pinned RandOpt worker, official direct
  prompt, greedy.
- **Answer** = the first generated token (max_tokens = 1), stripped and upper-cased; correct iff it equals the gold letter.
- A–D first-token log-probs are recorded for later analyses. They are not used here.
- Fidelity:
  - base fingerprint 5bc2499c…;
  - each candidate's fingerprint equals its stored candidate_state_id;
  - exact base restore after each candidate.

  Any failure stops the run.

## Statistics (H accuracy gain g = 100·(acc_cand − acc_base))
Uncertainty for a single candidate uses a paired bootstrap over **images** (cluster bootstrap: 2000 resamples,
seed 20261006).

| Test | Statistic | Reading |
|---|---|---|
| **M1-1 (primary)** | 9504111's g with a 95% cluster-bootstrap CI | **TRANSFERS** if the CI lower bound > 0; **FULL TRANSFER** if additionally g ≥ +6 pp (its RERANK compass gain was +8.5); **DOES NOT TRANSFER** if the CI includes 0. The answer-prior account predicts positive, about +2 to +4. |
| M1-2 | 9504111's g on gold-front items minus g on non-front items, cluster-bootstrap CI | > 0 with the CI excluding 0 supports the prior-shift account |
| M1-3 | across measured candidates: Pearson r(RERANK gain, g) and r(τ, g), with Fisher 95% CI. τ = Stage-3 predicted non-vision tilt (τ_S + τ_R) | r(τ, g) ≥ 0.25 supports "the tilt is the transferable component"; r(RERANK gain, g) is reported |
| M1-4 | mean g of SEARCH-top-50 members (incl. 9504111) minus mean g of the 12 controls; bootstrap over candidates | does selection buy anything on fresh same-format items? |
| reported | base accuracy on H; 9504111's rank among the measured candidates; parse failures | – |

- **Minimum:** M1-3/M1-4 need ≥ 40 measured candidates, else INCONCLUSIVE. M1-1/M1-2 need only 9504111.
- No re-tuning of items, candidates, scoring or thresholds.

## Budget
- Estimate ≈ $5.4: setup about $0.5, images about 1 min, 61 candidates × about 80 s.
- The in-script guard stops before the cap, and partial results are kept.

# EXPLORATORY robustness checks of Claim 1 (not pre-registered)

Script `scripts/robustness_checks.py` → `robustness_results.json`. CPU only, on committed outputs.

## 1. Data alignment (GSM8K)
Ground truth from `openai/gsm8k` equals RandOpt's parquet on 1319/1319 items; question text matches on 1319/1319
(checked against the parquet's chat-message content; the script's first text check stringified the message list
and is superseded by this direct check). Every run indexes items in this order; the O2 environment gate also showed
identical BASE answers on all 1319 items across pods.

## 2. Scorer leniency
RandOpt's GSM8K scorer takes the answer after "####" and otherwise falls back to the last number in the text; the
GQA scorer accepts unboxed short answers. We re-scored with strict rules: GSM8K answers count only inside \boxed{}
(boxed/plain prompts) or after "####" (RandOpt's prompt); GQA answers count only on exact normalized match.

| GSM8K BASE greedy, lenient → strict | RandOpt's prompt | plain | boxed |
|---|---|---|---|
| Qwen2.5-1.5B | 60.05 → 7.96 | 64.75 → 53.75 | 70.20 → **74.45** |
| Qwen2.5-3B | 80.14 → 56.63 | 72.93 → 51.25 | 82.34 → **86.50** |
| OLMo-2-1B | 33.66 → 0.00 | 65.28 → 68.54 | 67.85 → **69.90** |

- Under RandOpt's own prompt, its accuracy depends heavily on the lenient fallback (the models rarely emit a
  strict "#### number"). RandOpt's published numbers use the same lenient scorer, as do all our arms.
- For the **boxed** prompt (chosen by PS for every GSM8K model) the lenient scorer is *conservative*: it misreads some
  correct boxed answers (a later number in the text), so strict scoring is higher.
- **PS with strict SC (boxed answers only) vs RandOpt on its own lenient scorer** (RandOpt's member texts were not
  saved; keeping its lenient score can only favour RandOpt): Qwen-1.5B −7.81 [−9.63, −6.06], Qwen-3B −4.78
  [−6.07, −3.49], OLMo −26.00 [−28.58, −23.43]; strict SC@50 84.99 / 91.43 / 78.47 (lenient 80.14 / 87.04 / 76.12).
  The PS conclusion strengthens; the Qwen-3B row moves from "no difference" to SC ahead.
- **GQA strict exact match:** RandOpt 60.42 vs direct SC 61.47, D −1.05 [−2.67, +0.57] (lenient −1.21 [−2.83,
  +0.40]); unchanged conclusion.

## 3. Statistical method
- Exact McNemar tests agree with the paired bootstrap on every PS row: Qwen-1.5B p = 0.00083, Qwen-3B p = 0.65, OLMo
  p ≈ 1e−67, GQA p = 0.17. **Holm-corrected across the four rows:** 0.0025, 0.65, ≈ 4e−67, 0.34; the two "ahead" rows
  survive correction.
- GQA image-clustered bootstrap (questions share images): D −1.21 [−2.79, +0.41], same as the item bootstrap.

## 4. Choice of K
At K = 10 (both RandOpt and SC use 10): Qwen-1.5B −0.38 [−2.20, +1.52], Qwen-3B −0.53 [−1.97, +0.91], OLMo −24.56
[−27.37, −21.83], GQA −0.57 [−2.18, +1.05]. RandOpt is ahead in no row at K = 10 either, but Qwen-1.5B's "SC ahead"
holds at K = 50 only. State: "matches or beats" holds at both K; the Qwen-1.5B advantage is K-dependent.

## 5. Known bugs and their (non-)effect
- o2_eval's stored `c` field (wrong ground-truth format; O2/Q2/PS): never read by the analyses, which rescore saved
  answers (checked in the code and by self-tests).
- p2_sc's own `c` rule differs slightly from RandOpt's scorer (O1 greedy 33.36 vs 33.66): all comparisons use
  RandOpt's scorer for every arm.
- O1 top-50 reconstruction: matches all 10 rows randopt.py printed; rebuilt members' accuracy matches O1's within
  0.63 pp (answer agreement 70%, so the O2 members secondary was declared not valid, as locked).
- A Windows text-encoding bug made one verification pass vacuously (0 rows compared); caught and re-run (10/10).
- Worker OOMs (G2, G2R): re-run alone with identical settings; G2R replicates G2, and G3 at 256 tokens reproduces G2
  exactly.

## 6. Thresholds and fixed choices
SC temperature 0.7 was fixed in P2's lock before any SC output (T = 0.3 was lower on GSM8K-1.5B: 73.3); K = 50 is the
paper's setting (K = 10 above); the ±2 pp equivalence margin only adds a label; base-reproduction gates passed with
margin (largest gap +1.47 of ±2.0, C). Prompt selection chose with margins of ≥ 4.5 pp on the 200 selection questions
(Qwen-3B 90.0 vs 85.0 is the smallest), so the choice is not a near-tie.

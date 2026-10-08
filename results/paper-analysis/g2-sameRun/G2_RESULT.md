# G2 result: same-run RandOpt (N = 5000) vs self-consistency on GQA / Qwen2.5-VL-3B-Instruct

Lock d2b3d68 + amendment 1 (5be1720), both before any G2 output. Analysis `scripts/g2_analysis.py` → `g2_results.json`.
6× H100; configuration chosen by the smoke run: **fast** (CUDA graphs, 12 workers); projection $23.36 (GO).

## Engineering event (reported, no effect on data)
Test phase: worker 6 (sharing GPU 0 with worker 0, which also evaluated BASE) hit CUDA OOM; ranks 6/18/30/42 were missing.
Worker 6 was re-run alone on GPU 1 with identical settings (resumable runner; existing files skipped;
`pod/g2/test_6_rerun.log`). All 50 ranks + BASE present before analysis.

## Validity
- Test set: P2's 2000-question sample minus questions whose image is in the selection set → **1238** questions
  (762 removed; 389 images). Selection: 200 G1 questions.
- **Environment gate:** G2 base greedy 53.39% vs P2 greedy on the same 1238 questions 53.55% (|Δ| = 0.16 ≤ 1.0) → **VALID**.
- Base selection reward 0.535; top-50 selection reward 0.580–0.635; σ of the top 50: 46 × 0.002, 3 × 0.001, 1 × 0.0005.

## Primary G2-1 (K = 50, paired item bootstrap, 10,000 resamples)
RandOpt 63.49% vs SC@50 (T = 0.7, P2 run) 60.02%: **D = +3.47 pp [+1.62, +5.41] → RANDOPT AHEAD.** Image-cluster CI
[+1.41, +5.53]. Discordant: 95 RandOpt-only vs 52 SC-only correct.

## Secondary
- K = 10: RandOpt 63.33 vs SC@10 58.97, D = +4.36 [+2.26, +6.54] → RANDOPT AHEAD.
- Members: the 50 selected models average 58.4% individually (52.5–63.1) vs base 53.4% (**+5.0 pp per model**, unlike
  GSM8K where members were +4.0 (1.5B) and +0.2 (3B)). Vote 63.5.
- RandOpt gain over base here +10.1 pp (paper: +12.4 on all of testdev with train-split selection; descriptive).

## EXPLORATORY: where the GQA gain comes from (not pre-registered)
- Yes/no questions (443): base 72.2%, members 72.3%; vote RandOpt 76.5 vs SC 75.6. Yes-share unchanged (44.9 → 43.1).
- Open questions (795): base 42.9%, members 50.6%; vote RandOpt 56.2 vs SC 51.3. **The difference is on open questions.**
- Base often ends without a usable final answer (extractions like "step step", "determine", "information provided";
  consistent with reasoning running past max_tokens = 256). Keyword heuristic for such non-answers: base 10.7% of
  questions, SC samples 10.8%, selected members 5.5%. On the 133 base-non-answer questions, members are correct 43.4%
  of the time (base 0%); on the other questions members gain only +1.3 pp (58.9 → 60.2).
- Reading: on GQA, selection finds perturbations that **repair answer termination/format** (shorter answers: 1.31 vs
  1.58 words on open questions; exact string match 35.8 vs 28.3%), a shared, transferable shift that sampling the
  base model cannot supply. This matches the original paper's own "format thickets" analysis and the sequence-length
  observation in "When does RandOpt work?". Heuristic; a direct check needs the generated texts / token counts
  (not saved in the test files).

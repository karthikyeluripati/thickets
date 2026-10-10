# RV result: the reviewer round

Lock c33fec2 (before any RV output); amendments `amendment_1.md` (session record, RV-2a resumption, RV-2b/RV-3 not run),
`amendment_2.md` (RV-2b run after all) and `amendment_3.md` (RV-3 run after all). Analyses: `scripts/rv_analysis.py` → `rv_results.json` (RV-1, RV-4, RV-5;
run once), `scripts/gb_analysis.py` → `rv2a_results.json`, `rv2b_results.json`. Pod logs per session in `pod/session1-5/`,
outputs in `pod/rv/`. GPU ≈ $86.8 over five 4× H100 sessions ($20.5, $20.6, $11.8, $18.0, $16.0).

| Item | Test | Result | Outcome |
|---|---|---|---|
| RV-1 | RandOpt − SC@50 under the public template chosen on the 200 selection questions | Qwen-1.5B +4.17 [+1.97, +6.44]; Qwen-3B −0.45 [−1.67, +0.76]; OLMo −5.23 [−7.66, −2.81]; GQA −2.67 [−4.60, −0.73] | RandOpt ahead / equivalent / SC ahead / SC ahead; pooled six-candidate rule: matches or beats in all four |
| RV-2a | RandOpt searched under boxed − SC@50 boxed, Qwen2.5-3B | −1.36 [−2.35, −0.38]; members −0.47 [−1.45, +0.53] vs base | **SC ahead** (gates passed; resumed across sessions) |
| RV-2b | same, Qwen2.5-1.5B | −4.32 [−5.91, −2.81]; members −0.51 [−1.71, +0.73] vs base | **SC ahead** (gates passed) |
| RV-3 | second OLMo search (seed 43), RandOpt − SC@50 under RandOpt's prompt | +10.39 [+8.26, +12.51]; members +7.92 vs base; seed 43 − seed 42 +1.67 [+0.15, +3.18] | **RandOpt ahead: O1 replicated** |
| RV-4 | randopt.py's base print, Qwen2.5-1.5B, fresh pod | 73.00 (4 and 1 engines) vs our runner 68.00; test bases agree | engine-specific print: fidelity limitation; exploratory R8b.1 / R8b.7 recomputed |
| RV-5 | SC@50 at T = 0.5 / 1.0 (Qwen-3B boxed; GQA direct) | −0.23 / −1.14; −1.05 / +0.24 (all CIs contain 0) | no outcome changes |

## Reading
- With the prompt held fixed, weight search adds nothing over sampling in all four same-run rows (with GD and GB).
- The cheap route needs a candidate that fixes the answer format: public harness templates alone lose on Qwen-1.5B.
- The Qwen rows' "population mean below base" (R8b.1) and their selection gains (R8b.7) are corrected with the
  same-engine base; no confirmatory comparison used randopt.py's printed base.

Full numbers: `paper/RESULTS_MASTER.md` R8b.5, R8b.7, R8b.9, R8b.10 and the R9 ledger.

## Folder layout
- `plan_lock.md`, `amendment_1.md`, `amendment_2.md`: the lock and its amendments (each committed before the outputs it covers).
- `jobs/` (sessions 1–2: the locked queue 01–07), `jobs_resume/` (session 3: RV-2a resumption), `jobs_2b/` (session 4: RV-2b),
  `jobs_rv3/` (session 5: RV-3).
- `selftest/rv_selftest.json`: the analysis self-test recorded in the lock.
- `pod/rv/`: all outputs (`rv1/`, `rv4/`, `rv5/`, `q3/` = RV-2a, `q15/` = RV-2b, `olmo2/` = RV-3); `pod/session1-5/`: per-session pod logs
  (`session1/rv_q3`, `rv_q15`: the partial RV-2a and RV-4 fidelity pass of session 1).
- `rv_results.json`, `rv2a_results.json`, `rv2b_results.json`: analysis outputs.

# GD status: interrupted before the test phase finished (no result yet)

The pod stopped at ~05:08 UTC on 2026-10-09, before the $25 guard (~05:19), most likely because the account credit
ran out (spent ≈ $20.5). Pulled and committed: all 5000 selection rewards (`pod/gd/out/select_*.jsonl`), the
direct-prompt BASE test (`test_base.json`) and 36 of the 50 top-K test files. Missing: ranks 9–11, 20–23, 32–35, 44–47
(workers 8–11 failed vLLM's startup memory check in the second wave and were being re-run). No analysis has been run;
the locked rule needs all 50 ranks. To finish: re-run the test phase for workers 8–11 (identical settings, resumable
runner) on any GPU; ~15 min on 1× H100.

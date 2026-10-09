# RV amendment 1: session record, resumption of RV-2a, and scope after the stops

Written after the events below and before any RV-2a test output; nothing in any test, gate or analysis changes.

## Sessions
- **Session 1** (4× H100, cap $115): prep, RV-4, RV-1 and RV-5 completed and were pulled (results committed in
  3d4f88a). The pod was reset at ~22:23 UTC during RV-2a's selection (container wiped; no volume). ≈ $20.5.
- **Session 2** (same host, fresh container, cap $94 = $115 − session 1): RV-2a's gates passed again identically
  (`pod/rv/q3/gates.txt`); 3209 of 5000 selection rewards were pulled before the pod stopped at ~23:42 UTC with its job
  still running and its own guard not triggered. ≈ $20.6.
- Engineering in session 2: Qwen2.5-3B workers use ≈ 27.5 GB each whatever the vLLM share, so the third worker per GPU
  cannot start; the launcher re-ran failed workers afterwards (no effect on results).

## Resumption of RV-2a (session 3; the user's final GPU budget)
Same population (seed 42), prompt (boxed), scorer, ranking and test design. Only the split of the remaining
perturbations changes: the pulled `select_<w>.jsonl` (W = 12) are uploaded; residue w mod 12 is split into
sub-workers w, w+12, w+24 (W = 36) whose done-files are seeded with that residue's finished k
(`scripts/rv_resume_q3.py`), run 2 per GPU (gpu-mem 0.38). Selection is complete only if all 5000 k are scored. Test
phase as GB with 8 workers. Analysis exactly as locked (`gb_analysis.py`).

## Scope after the stops (fixed now)
The user's GPU budget ends with session 3. **RV-3 (second OLMo seed) and RV-2b (Qwen-1.5B boxed search) are not run**
and are reported as such in the ledger and the limitations: the OLMo row rests on one search under RandOpt's prompt
(plus GB's search under the boxed prompt), and the Qwen-1.5B row has no prompt-held-fixed search.

# Visual line tracing v2.1 result — 2026-10-03

**STOP: the frozen benchmark capability gate failed.** Medium+Hard held-out
accuracy is **27.03% (90/333)**. No nonzero RandOpt candidates were generated.

1. **What is base Medium+Hard accuracy?** Selection: **26.00% (26/100)**.
   Held-out: **27.03% (90/333)**. The complete baseline is:

   | Endpoint | Selection correct / total | Selection accuracy | Held-out correct / total | Held-out accuracy |
   |---|---:|---:|---:|---:|
   | Easy — diagnostic | 25 / 50 | 50.00% | 75 / 167 | 44.91% |
   | Medium | 13 / 50 | 26.00% | 49 / 167 | 29.34% |
   | Hard | 13 / 50 | 26.00% | 41 / 166 | 24.70% |
   | **Medium+Hard — primary** | **26 / 100** | **26.00%** | **90 / 333** | **27.03%** |
   | Overall — diagnostic | 51 / 150 | 34.00% | 165 / 500 | 33.00% |

   All answers were valid single-digit responses; none hit the token cap.
   The 243 primary held-out errors were wrong endpoints. Base primary-selection
   and easy-selection calls were separate; held-out generation included all
   500 images in their frozen order.

2. **Did the benchmark pass the capability gate?** **No.** The combined result
   falls below the required inclusive 30–70% range, and neither Medium nor Hard
   individually reaches 30%. At least 100/333 primary correct answers were
   required; the base obtained 90. Easy and overall accuracy did not enter this
   decision. The committed [baseline lock](../results/visual-line-tracing-v21-20261003/locks/baseline.json)
   records `STOP_benchmark_capability_gate`.

3. **Which sigma won calibration?** None. Calibration was not run after the
   failed capability gate; no sigma lock exists.

4. **What did the 300-candidate distribution look like?** Unmeasured. No final
   search candidates, histogram, expert-density estimate or ranking were produced.

5. **How much did the frozen best expert improve on held-out Medium+Hard?**
   Unmeasured. No expert was selected or evaluated.

6. **How much did top-5/top-10 improve?** Unmeasured. No committees were selected,
   and neither individual means nor majority votes were computed.

7. **Did gains appear on Medium and Hard separately?** There are no candidate
   gains to assess. The base scores are 29.34% and 24.70%, respectively.

8. **Did the frozen GO gate pass?** It was not reached. Baseline controls passed:
   zero perturbation reproduced base outputs on all 650 images, snapshot restore
   was bitwise exact, repeated base outputs were identical, and the native state
   matched the pinned reference. Nonzero-candidate and cross-phase controls were
   not applicable because execution stopped after the baseline.

9. **Final decision: GO or NO-GO?** **NO-GO to proceeding on this benchmark**, with
   execution status `STOP_benchmark_capability_gate`. v2 still does not place the
   base VLM in the required capability regime. This is **not a scientific negative
   for vanilla RandOpt**: no nonzero perturbation search was run, so the question
   of a better weight-space expert in a usable regime remains unanswered. Stop
   here; no v3, prompt/label/model change or additional sweep was attempted.

The [v2.1 protocol](VISUAL_LINE_TRACING_V21_PROTOCOL.md) was frozen in
`7ca95bb633bceab6bff6f605225bffb36b5b91b9` at 20:46 EDT, before inference at
20:48–20:51 EDT on October 3 (00:48–00:51 UTC October 4). The baseline decision
was committed as `2b9012c31a7d45d978df3d8c4f0ee11f7ff7711c`. The model, tokenizer
and processor revision remained
`Qwen/Qwen2.5-VL-3B-Instruct@66285546d2b821cf421d4f5eb2576359d3770cd3`, using
the same BF16/TP1/greedy vLLM 0.10.2 and Ray 2.49.2 executor on one H100.

[Evidence](../results/visual-line-tracing-v21-20261003/) includes raw base/zero/
repeat generations, native state and parameter hashes, exact commands, package
versions, GPU details, measured source, the lock, and the independently
[reproduced decision](../results/visual-line-tracing-v21-20261003/analysis/gate.json).
All **105 H100 tests passed**, with one CPU-only guard skipped. All 654 frozen v2
dataset files match commit `6d30f60`; images and labels were not changed.
The [integrity record](../results/visual-line-tracing-v21-20261003/integrity.json)
verifies prior artifacts and execution-source bytes. Existing v1/v2 artifacts
and `main` are unchanged. The GPU was idle after cleanup.

# P1 pilot result: perturbations flip the same items as sampling, are persistent models, and their vote ≈ self-consistency

Lock: `plan_lock.md` (464aed0, committed before output). Qwen2.5-VL-3B, GQA, RandOpt CoT prompt and scorer, language-model
only. S = 200 selection items, H = 200 held-out items; 64 perturbations (σ 0.001/0.002); base sampling at T ∈ {0.3, 0.5,
0.7, 1.0}, n = 32. Pod ≈ $2.5 of the $10 cap.
- Base greedy accuracy: S 53.5, H 56.0 (RandOpt's reported GQA base is 56.6).
- After the run, the analysis's sample-file name parsing was fixed (a ".json.gz" stem bug); no logic changed.

| Test | Result | Locked verdict |
|---|---|---|
| Flip rates | perturbations 49.1% of item answers differ from greedy base; T = 0.3 46.1%, 0.5 52.4%, 0.7 58.3%, 1.0 69.7% → T\* = 0.3 | – |
| **P1-A** Spearman(f_P, f_T\*) over 400 items | **0.946** (0.95 / 0.93 / 0.89 at T = 0.5 / 0.7 / 1.0) | **SUPPORTS** (the same items flip) |
| **P1-B** r(selection gain, held-out gain), 64 perturbed models | **0.74** [0.61, 0.84]; pseudo-models from T\* samples: 0.10 | **PERSISTENT** |
| **P1-C** RandOpt-style top-K vote vs self-consistency (H) | K = 8: RandOpt 62.0 vs SC@T\* 59.5 vs SC@0.7 63.0 vs random-8 perturbation vote 60.1 (base 56.0); diff vs SC@T\* +2.5 [−3.5, +8.5]. K = 16: 62.0 / 61.0 / 63.0; K = 32: 61.5 / 60.5 / 63.5 | **NO ADVANTAGE** (underpowered: n = 200 items) |
| **Overall** (pre-stated rule) | B is PERSISTENT | **NOT SUPPORTED** as pure re-sampling |

**Post-hoc check on B** (does persistence reflect expertise or damage?):
- **Much of it is damage.** At σ = 0.002, r = 0.84 vs 0.39 at σ = 0.001. The boxed-answer rate falls from 90.5% (base)
  to as low as 50.7%, and r(boxed rate, held-out gain) = 0.44. The bottom-8 held-out gains are −18 to +1.5 pp.
- **There is modest real persistence among non-damaged models:** r = 0.40 for selection gain ≥ 0 (n = 11).
- **The top-8 by selection** have held-out gains of 0 to +7, mean +2.9 pp.

## Reading
- Weight perturbations explore **the same uncertain items** that temperature sampling explores (ρ = 0.95). But each
  perturbed model is a **consistent** model: its quality differences persist, mostly as format damage, with a modest
  positive tail.
- The **vote** of RandOpt-selected perturbations is **not better than base-model self-consistency** at K ≤ 32 on GQA. At
  T = 0.7, SC was numerically highest.

## Literature check (arXiv 2603.12228v1, full text)
- **GQA:** the paper reports only Base 56.6 vs RandOpt 69.0. **No base majority-vote baseline for GQA.**
- **Text tasks (Table 4):** RandOpt beats "TT-MV" substantially (e.g. Qwen2.5-1.5B GSM8K 76.4 vs 69.1; MATH-500 59.7 vs
  50.0). The footnote defines TT-MV as "majority vote over test samples from a single trained model with **different
  seeds**". The temperature is not stated. RandOpt's own code generates at temperature 0.0. Several TT-MV gains are
  tiny (0.5B GSM8K 39.9 → 41.0), which is consistent with near-greedy sampling. That is an **open question, not a
  finding.**
- **Already in the paper:** "format thickets" (GSM8K: 19.0% format vs 12.3% reasoning).
- **Elsewhere:** selection bias (arXiv 2608.10867).

# P2 result: self-consistency vs the Neural Thickets paper's reported RandOpt. MATCH on GSM8K-1.5B; RandOpt far ahead on GQA

Lock: `plan_lock.md` (01efd9d, committed before output). Pod ≈ $5 of the $10 cap; pod stopped after a verified pull.
- The first two job attempts failed on a package conflict: an unpinned `datasets` install upgraded huggingface-hub to
  2.x. No outputs were produced. The environment was pinned, verified, and the jobs re-run.
- After the run, the analysis serializer (numpy bool) was fixed; no logic changed.

| Row | our greedy | paper Base | gate (±2.0; GQA ±2.5) | **SC@50 T = 0.7** [95% CI] | SC@50 T = 0.3 | paper TT-MV | **paper RandOpt** | locked verdict |
|---|---|---|---|---|---|---|---|---|
| **GSM8K / Qwen2.5-1.5B-Inst** (1319) | 59.3 | 58.8 | pass | **79.8** [77.6, 82.0] | 73.3 | 69.1 | 76.4 | **MATCHES** (+3.4) |
| GSM8K / Qwen2.5-0.5B-Inst (1319) | 43.2 | 39.9 | **fail** (+3.3) | 56.0 [53.1, 58.6] | 54.9 | 41.0 | 54.1 | NOT COMPARABLE |
| GQA / Qwen2.5-VL-3B (2000 of testdev-balanced) | 54.0 | 56.6 | **fail** (−2.6) | 59.2 [57.1, 61.4] | 58.1 | – | 69.0 | NOT COMPARABLE |

SC@K curves (T = 0.7):
| | K = 1 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|
| GSM8K-1.5B | 57.5 | 69.9 | 75.2 | 78.7 | 79.8 |
| GSM8K-0.5B | 36.6 | 47.2 | 52.5 | 54.8 | 56.0 |
| GQA | 47.7 | 54.6 | 57.7 | 57.8 | 59.2 |

## Reading (no claims beyond these rows)
- **GSM8K-1.5B (comparable).** Properly sampled self-consistency (T = 0.7, 50 samples), with **no weight search**,
  reaches 79.8%.
  - That is above the paper's RandOpt (76.4%) and **10.7 pp above the paper's TT-MV (69.1%)**.
  - Our T = 0.3 SC (73.3%) is much closer to their TT-MV, consistent with a low-temperature TT-MV, but **that is
    unconfirmed**.
- **GSM8K-0.5B (not comparable by the gate).** SC gains +12.7 pp over our base vs RandOpt's reported +14.2 pp over
  theirs. Similar magnitude; descriptive only.
- **GQA (not comparable by the gate).** SC gains only +5.2 pp over our base vs RandOpt's reported +12.4 pp. Even
  allowing for the 2.6 pp base gap, **RandOpt's reported GQA gain is far larger than self-consistency's**.
  - In P1 (N = 64 perturbations), RandOpt-style top-8 ≈ SC on 200 items.
  - So either selection at N = 5000 adds a lot on GQA, or the 69.0 does not reproduce in our hands. **Unknown**:
    resolving it requires running RandOpt at scale.
- **Caveat:** these are cross-paper comparisons (the paper reports 3-run means, on different hardware). Only the
  GSM8K-1.5B row passed the pre-registered base-reproduction gate.

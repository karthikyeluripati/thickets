# Complementarity selection: offline complete, GPU validation pending

There is **no GO/NO-GO result yet**. The previous RunPod endpoint returned
`container not found`; a replacement H100 endpoint is required for the fresh
300-prompt test. The existing adaptive matrix was not regenerated. All prior
study artifacts and `main` remain unchanged.

The method, gate, tie seeds and fresh inputs were frozen in commit `8ebae53`.
All committees and the exact 232-expert union were locked at `af4823c` before any
new held-out generation. The later voting-details adjustment normalizes tuples
to JSON lists for exact reload comparison; it changes no vote, committee or gate.
See the [protocol](COMPLEMENTARITY_PROTOCOL.md) and
[selection artifacts](../results/complementarity-selection-20261003/offline/).

## Test 1: selection prompts only

These accuracies use the same 200 prompts used to select each committee. They
are training objectives, not evidence of held-out improvement. Primary greedy
tie seed is 0; two additional frozen tie seeds and three exact-quality random
controls are preserved. The optional lambda heuristic is not used.

| Population | K | Standard top-K | Greedy committee | Gain |
|---|---:|---:|---:|---:|
| Development | 3 | 54.0% | 55.5% | +1.5 pp |
| Development | 5 | 56.0% | 62.0% | +6.0 pp |
| Development | 10 | 60.5% | 67.5% | +7.0 pp |
| Development | 20 | 60.5% | 69.0% | +8.5 pp |
| Development | 50 | 61.5% | 70.0% | +8.5 pp |
| Validation A | 3 | 53.5% | 55.5% | +2.0 pp |
| Validation A | 5 | 59.5% | 62.0% | +2.5 pp |
| Validation A | 10 | 61.0% | 66.5% | +5.5 pp |
| Validation A | 20 | 63.5% | 70.5% | +7.0 pp |
| Validation A | 50 | 61.5% | 70.0% | +8.5 pp |
| Validation B | 3 | 54.5% | 56.5% | +2.0 pp |
| Validation B | 5 | 58.0% | 62.0% | +4.0 pp |
| Validation B | 10 | 58.5% | 65.0% | +6.5 pp |
| Validation B | 20 | 60.5% | 68.0% | +7.5 pp |
| Validation B | 50 | 60.5% | 70.0% | +9.5 pp |

The union has 74 development, 80 validation-A and 78 validation-B experts. It
includes standard top-50, greedy K<=20 for all three tie seeds, and random K<=20
for all three control seeds. Each random committee prefix has exactly the same
individual-score multiset as primary greedy, with distinct expert IDs and no
error optimization. Greedy K=50 is selection-only, as specified in the GPU plan.

## Prepared Test 2

Fresh GSM8K main/test indices **40-339** are frozen at dataset revision
`740312add88f781978c0658806c59bc2815b9866`, with no exact prompt overlap against
earlier example inputs. JSONL SHA256:
`acc0c407b6377bced91c72604b709df22afa363bceb79c42a8c485cecdbfd239`.
The workload is 232 x 300 = 69,600 selected-expert request sequences, plus six
full-benchmark correctness-control generations. No fresh held-out output exists
in this branch yet.

The collector uses the original pinned RandOpt worker and Qwen2.5-0.5B revision,
BF16 TP=1, greedy eager Ray/vLLM, snapshot anchoring and the same 512-token cap.
Every expert's expected state fingerprint is copied from the original matrix and
checked **before inference**. It rejects changed protocol/lock/input bytes and
refuses existing output directories. Correctness audits are outside timing.

The prepared analysis rescans raw generations with the pinned reward/extractor,
reproduces selection votes, reports all fresh committee votes and margins,
pairwise error correlations, disagreement, individual quality, exact token and
expert-pass costs, selection-to-test gains, matched controls and the fixed gate.
The gate requires the same qualifying comparison on both held-out populations.
It separately reports whether ordinary small top-K committees already satisfy an
apparent efficiency win, to avoid attributing that to complementarity selection.

Tests: the full implementation suite passed 73 tests with one CPU CUDA skip;
after the JSON-list normalization, all six targeted committee/validation tests
passed. Incremental vote calculations are checked against independent Counter
votes, including empty answers and tie changes. A fingerprint mismatch test
verifies that inference is never called and restoration still occurs.

On a replacement H100 with the recorded PyTorch 2.8.0+cu128 / vLLM 0.10.2 /
Ray 2.49.2 / Transformers 4.56.2 environment and the pinned upstream checkout:

```bash
export PYTHONPATH="$PWD/src"
bash scripts/run_committee_validation.sh /workspace/RandOpt-feasibility \
  results/complementarity-selection-20261003/offline \
  results/complementarity-selection-20261003/gpu-01
python scripts/analyze_complementarity.py \
  --upstream-root /workspace/RandOpt-feasibility \
  --selection results/complementarity-selection-20261003/offline \
  --run results/complementarity-selection-20261003/gpu-01 \
  --out results/complementarity-selection-20261003/analysis
```

The question about real ensemble improvement remains unanswered until this fresh
test completes. Missing compute access must not be reported as a scientific NO.

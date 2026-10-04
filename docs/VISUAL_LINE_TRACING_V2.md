# Visual Line Tracing v2: easier geometry

Status: **dataset created and validated; no model evaluation run**. This is a
separate, user-requested difficulty revision after the completed v1 study.
The v1 dataset, generator, protocol, report and result artifacts remain intact.

| Difficulty | v1 swaps | v2 swaps | v1 minimum target crossings | v2 minimum target crossings |
|---|---:|---:|---:|---:|
| Easy | 4 | 1 | 1 | 1 |
| Medium | 7 | 2–3 | 3 | 2 |
| Hard | 10 | 4–5 | 5 | 3 |

The reduced minimum target-crossing counts make the shorter paths feasible.
Spacing remains 92/78/66 pixels and wiggle remains 3/7/10 for easy/medium/hard.
Everything else in the drawing stays the same: four black paths, red start dot,
white underpasses, endpoint digits, 448×448 resolution, line widths, font and
3× supersampling with Pillow 12.3.0. The v2 renderer reproduces the saved v1
pixels when given v1 geometry.

The prompt, answer parser, voting, model/tokenizer/processor revision, runtime,
decoding, perturbation recipe, candidate budgets and decision thresholds are
unchanged. The [v2 configuration](../experiments/visual_line_tracing_v2_protocol.json)
differs from v1 only in dataset path and version/parent metadata. Its presence
does not authorize or imply a new GPU experiment.

## Frozen dataset

[examples/visual-line-tracing-v2](../examples/visual-line-tracing-v2/) contains all
650 PNGs, prompts, labels, graph metadata, seeds and hashes. It preserves the
150 selection / 500 held-out split sizes, example IDs, seeds and difficulty
assignments from v1. Geometry and labels are regenerated for feasible pairs;
these are not label-identical paired examples. Selection and held-out seeds
remain disjoint, and all 650 v2 image hashes are unique.

| Split | Easy | Medium | Hard |
|---|---|---|---|
| Selection | 50 × 1 swap | 25 × 2; 25 × 3 | 25 × 4; 25 × 5 |
| Held-out | 167 × 1 swap | 84 × 2; 83 × 3 | 83 × 4; 83 × 5 |

With one adjacent swap and a target that must cross, only six start/end pairs
are possible. Two swaps with two target crossings allow eight pairs; the other
v2 settings allow all sixteen. The generator enumerates feasible pairs, shuffles
their order deterministically, and cycles them independently within each split,
difficulty and swap count. Every feasible pair appears; counts differ by at most
one within each group. Requiring v1's sixteen pairs at one swap would be impossible.

This necessarily changes endpoint marginals. The easy bucket has positional
shortcuts, so its accuracy should not be interpreted against a 25% reference
alone. The table below evaluates a **model-free topology prior**: predict the
most common reachable endpoint given the start lane and swap count, breaking
ties by the lowest endpoint. The rule is fixed from enumerated geometry, without
fitting either split's answers. It does not follow the rendered path.

| Difficulty | Selection shortcut accuracy | Held-out shortcut accuracy |
|---|---:|---:|
| Easy | 68.00% | 66.47% |
| Medium | 36.00% | 37.13% |
| Hard | 22.00% | 24.70% |

The [manifest](../examples/visual-line-tracing-v2/manifest.json) also records label
counts, feasible cells, swap counts, straight-endpoint accuracy and the prior
without start position. These are structural dataset checks, not VLM results.
The existing requirement for held-out gains in multiple difficulty levels,
including medium or hard, remains unchanged.

## Validation and reproduction

**88 tests passed, one CUDA test skipped** on the local CPU environment. Checks
replay all 650 graph labels independently, verify split/image disjointness and
swap distributions, reconstruct every topology, reproduce saved images for all
five swap counts, establish renderer equivalence on v1 images, reject impossible
pairs and overwrites, and verify that the v2 protocol changes no inference or
scoring settings. Samples with 1/2/3/4/5 swaps were also visually inspected.
The [validation artifacts](../results/visual-line-tracing-v2-20261003/) record
the test output, source and dataset hashes, commands and preservation checks.

Generate into a new directory using the pinned Pillow version:

```bash
PYTHONPATH=src python scripts/freeze_line_tracing.py \
  --version v2 --out runs/visual-line-tracing-v2-regenerated
python -m pytest -q
```

The generator refuses existing output directories. Omitting `--version` keeps
the original v1 behavior. v2 is a prospective benchmark revision motivated by
the v1 result; it does not revise that result or establish a new accuracy,
calibration decision, expert density or GO/NO-GO conclusion.

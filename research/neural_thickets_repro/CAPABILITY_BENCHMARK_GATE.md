# Capability Benchmark Gate — measurement-instrument validation for 8 visual capabilities

Living design-decision log, same convention as `REPRO_SPEC.md`/`SCOPED_PERTURBATION_DESIGN.md`:
one row per decision, tagged Confirmed / Resolved-by-assumption / UNRESOLVED, appended to as
work progresses rather than written once at the end.

**Scope of this milestone**: a base-model, ZERO-perturbation evaluation pipeline for 8
capabilities, validating that each is a trustworthy measurement instrument before any future
multi-capability perturbation sweep. No RandOpt, no perturbation, in this entire milestone —
`scopes.py`/`scoped_perturbation.py`/`thicket_metrics.py`/`ledger.py`/`candidate_sampling.py`/
`topk_voting.py`/`run_scoped_randopt.py`/`external/RandOpt/*` are all untouched.

## Step 0 audit — existing reusable infrastructure

| Component | Reused as | Status |
|---|---|---|
| `vlm_adapter.resolve_model_snapshot` | Unchanged, reused directly | Confirmed |
| `vlm_adapter.format_chat_prompt` | Unchanged, reused directly | Confirmed |
| `vlm_adapter.build_image_aware_requests` / `generate_with_images` | Unchanged, untouched — GQA-specific, only used by the two GQA-derived adapters via `_gqa_filtered_base.py`'s reuse of `GQAHandler` | Confirmed |
| `vlm_adapter.build_multimodal_requests` (NEW) | Generalizes the above for arbitrary (messages, optional image) pairs, supports `image=None` for the text-only sanity condition | New, additive |
| `eval_base_image_aware.py`'s single-local-`vllm.LLM`, no-Ray, no-perturbation shape | Template for `run_capability_benchmark_gate.py` | Confirmed as the right template (not `run_scoped_randopt.py`'s multi-actor/perturbation machinery) |
| `env_check.py` (`assert_feasible`, `check_cuda`, `check_module`, `check_disk`) | Reused directly, unchanged | Confirmed |
| `config.py`'s `ModelConfig`/`ReproducibilityConfig`/`HardwareConfig` | Reused as-is (dataset-independent) | Confirmed |
| `config.py`'s `EvaluationConfig`, `DatasetConfig` | NOT reused — RandOpt-ensemble-specific (voting/tie_break) or single-dataset-shaped; new `GenerationConfig`/`CapabilityDatasetConfig`/`BenchmarkGatesConfig`/`CapabilityBenchmarkConfig` added instead, additive | New, additive |
| `GQAHandler` (external, unmodified) 4-method contract (`load_data`/`compute_reward`/`extract_answer_for_voting`/`is_voted_answer_correct`) | Reused unchanged by the two GQA-derived adapters via `_gqa_filtered_base.py` | Confirmed — no second, incompatible GQA scoring path introduced |
| `prepare_gqa_data.py`'s prefix-slice subset convention (`dataset.select(range(n))`) | The two GQA-derived adapters override `subset_selection_rule()` back to `"prefix"` to match it; every other adapter defaults to `"shuffled_prefix"` since their pools risk class/category-ordered rows (see "Subset selection rule" row below) | Confirmed as the right per-dataset choice, not a uniform rule |

New abstractions (all in `src/neural_thickets_repro/benchmarks/`): `base.py` (`CapabilityBenchmark` ABC, `Example`/`ParsedPrediction`/`ExampleScore`), `subset_selection.py`, `integrity.py`, `normalization.py`, `box_iou.py`, `vqa_soft_accuracy.py`, `prompting.py`, `runner.py`, `image_sanity.py`, `card.py`, `summary.py`, and `adapters/` (8 capability adapters + `gqa_raw_schema.py` + `_gqa_filtered_base.py`).

## Dataset source decisions

| Capability | Source | Confidence |
|---|---|---|
| `visual_grounding` | `lmms-lab-encoder/RefCOCO` | **Confirmed live** this session (HF viewer: val=8.81k/test=5k/testA=1.98k/testB=1.81k, schema `image`/`question`/`bbox`[xywh]/`question_id`, ungated) |
| `ocr_text_recognition` | `lmms-lab-encoder/textvqa` | **Confirmed live** this session (train=34.6k/val=5k/test=5.73k, `answers`: list of 10, ungated) |
| `counting` | `HuggingFaceM4/the_cauldron`, config `tallyqa` | **Confirmed live** this session (schema `images:[image]`, `texts:[{"user","assistant","source"}]`, multi-turn per image, ~98.7k rows, single split) |
| `attribute_recognition` | `ranjaykrishna/visual_genome`, config `attributes_v1.2.0` | Resolved-by-assumption — canonical authors' repo confirmed to exist with an `attributes` config family (region/object/attribute annotations, WordNet-canonicalized); **exact version-suffix string and split name UNCONFIRMED**, `config_name` is a constructor param specifically so this is correctable without touching adapter logic |
| `object_recognition` | `ILSVRC/imagenet-1k`, split `validation` | **Confirmed live** this session — dataset exists, splits/schema/`int2str` class-name mapping confirmed; **GATED**, requires an HF token with accepted ImageNet license (user's explicit decision: build for gated access, hard-fail clearly, never substitute) |
| `fine_grained_recognition` | `bentrevett/caltech-ucsd-birds-200-2011` | Resolved-by-assumption — one of several community mirrors (`randall-lab/cub200`, `galilai-group/cub200`, others also exist), none singularly canonical; adapter reads canonical species names from the mirror's own `features["label"].names` and hard-fails (`CUBSchemaError`) rather than guessing if that's absent |
| `spatial_reasoning` / `relational_reasoning` | `external/RandOpt/data/gqa/testdev.parquet` via `GQAHandler`, filtered by `gqa_raw_schema.py` | Filter logic Confirmed-by-test against the ASSUMED public GQA schema; **the assumed field names (`types.semantic`/`types.structural`, the "semantic" reasoning-program shape) are UNCONFIRMED against the actual raw dataset** — see next section, the required first pod-side step |

## GQA spatial/relational filter — raw schema investigation (REQUIRED FIRST POD-SIDE STEP)

`adapters/gqa_raw_schema.py`'s field-name constants (`SEMANTIC_TYPE_FIELD = "types.semantic"`,
`STRUCTURAL_TYPE_FIELD = "types.structural"`, and the `_extract_relation_name` reasoning-
program shape) are written against GQA's **publicly documented** official schema (Hudson &
Manning, CVPR 2019) — **not yet verified** against whichever specific raw HF dataset/parquet
form is actually loaded here (distinct from `GQAHandler`'s own already-prompt-formatted
output, which does not expose this metadata at all).

**Required first step on the pod**, before trusting the spatial/relational filter:

```python
from neural_thickets_repro.benchmarks.adapters.gqa_raw_schema import inspect_raw_schema
# load the raw GQA annotation rows (NOT via GQAHandler.load_data -- that returns prompt-
# formatted records, not the raw type/semantic-program metadata) and run:
report = inspect_raw_schema(raw_rows)
print(report)  # report["assumed_schema_confirmed"] must be True before proceeding
```

If `assumed_schema_confirmed` is `False`, update `SEMANTIC_TYPE_FIELD`/`STRUCTURAL_TYPE_FIELD`/
`_extract_relation_name` in `gqa_raw_schema.py` to match the real field names, and update this
row, before generating the spatial/relational subset IDs for real.

**Scientific note on spatial vs. relational (explicit, not hidden)**: GQA's own "relation"
semantic-type category **naturally contains** both spatial relations (left/right/above/etc.)
and non-spatial relations (holding/wearing/riding/etc.) — spatial is a sub-filter of it, not
a category GQA hands us separately. `relational_reasoning` is defined as
`natural-relational MINUS spatial` — an **explicit experimental choice** made so the two
capability benchmarks measure distinct skills, not a claim that GQA itself provides two
naturally disjoint categories. `build_spatial_relational_filters()` reports both the natural
containment (spatial ⊆ natural-relational, `natural_intersection_over_spatial` = 1.0 by
construction) and the final experimental exclusion (`n_experimental_intersection` = 0, by
this explicit choice) — never one without the other.

Spatial-relation keyword list (`SPATIAL_RELATION_KEYWORDS`, closed and explicit): left, right,
above, below, behind, front, near, next to, on, under, underneath, inside, outside, between,
beside, atop, over, top of, bottom of.

## Subset selection rule

`prepare_gqa_data.py`'s existing "first 200 rows" convention is a **prefix slice**, not seeded
sampling. This is appropriate for GQA's `train_balanced` (order doesn't encode answer/class
structure) but would be **wrong** for ImageNet-1K's validation split and most CUB-200 mirrors,
which are ordered by class — a raw prefix slice would sample only 1–4 classes total, not a
meaningful evaluation. Resolution: `CapabilityBenchmark.subset_selection_rule()` defaults to
`"shuffled_prefix"` (a seeded shuffle of a copy, then prefix — still fully deterministic,
IDs persisted, no resampling ever) for every new adapter; the two GQA-derived adapters
explicitly override back to `"prefix"` to match the existing, validated GQA-pilot convention.
Both rules are implemented and tested identically in `subset_selection.py` — this is a
per-dataset choice, not a universal rule change.

## Status-decision thresholds (`BenchmarkGatesConfig`, `card.py::decide_status`)

| Threshold | Value (all 8 configs) | Rationale |
|---|---|---|
| `max_parser_failure_rate_pass` | 0.02 | Below this, parsing noise is negligible |
| `max_parser_failure_rate_needs_review` | 0.10 | Above this, forced FAIL — too much signal lost to parsing to trust the score at all |
| `image_sanity_min_gap_pass` | 0.05 | At n≈40, a proportion metric's one-sample SE is ≈0.08; a gap below this isn't yet distinguishable from noise, but any non-positive gap is never ambiguous (forced FAIL) |
| `image_sanity_subset_size` | 40 | Small enough to be cheap, large enough that a real effect (image genuinely reaching the model) should be visible above the noise floor above |
| `floor_ceiling_low` / `floor_ceiling_high` | 0.05 / 0.95 | A benchmark scoring at the floor or ceiling at N=200 may lack the dynamic range needed for density estimation — flagged NEEDS_REVIEW as a scientifically informative finding, not a defect |

Full decision order is documented in `card.py`'s module docstring; every card exposes the raw
measurements regardless of the assigned Status.

## Open items (UNRESOLVED, tracked here, not silently resolved)

- GQA raw schema field names (see above) — first pod-side step.
- `lmms-lab-encoder/RefCOCO+`'s exact repo id — assumed analogous to the confirmed RefCOCO
  repo, not independently verified.
- Visual Genome `attributes` config's exact version suffix and split name.
- CUB-200-2011 mirror choice — `bentrevett/caltech-ucsd-birds-200-2011` is a documented,
  revisable pick among several community mirrors.
- ImageNet-1K gated access — requires an HF token with accepted license configured on
  whichever machine actually runs `load_examples()`; the adapter hard-fails with a clear
  message (`ImageNetGatedAccessError`) if this isn't set up, never substitutes anything else.

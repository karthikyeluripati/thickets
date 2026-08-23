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
| `visual_grounding` | `lmms-lab-encoder/RefCOCO` | **Schema corrected this repair pass**: `question` is a fixed region-captioning instruction, NOT the referring expression (confirmed via real N=5 Qwen output + live re-inspection this session) — the real referring expression(s) live in `answer` (a list); this adapter uses `answer[0]`. Otherwise as before: val=8.81k/test=5k/testA=1.98k/testB=1.81k, `bbox`[xywh]/`question_id`, ungated. |
| `ocr_text_recognition` | `lmms-lab-encoder/textvqa` | **Confirmed live** this session (train=34.6k/val=5k/test=5.73k, `answers`: list of 10, ungated) |
| `ocr_text_recognition_grounded` (NEW, this repair pass) | Same source, filtered by `prepare_textvqa_ocr_filter.py` | EXPERIMENTAL, not an official TextVQA category — see "N=5 repair pass" section below, item 4 |
| `counting` | `HuggingFaceM4/the_cauldron`, config `tallyqa` | **Confirmed live** this session (schema `images:[image]`, `texts:[{"user","assistant","source"}]`, multi-turn per image, ~98.7k rows, single split) |
| `attribute_recognition` | `AnnaZ1103/visual_genome_revised` (config `attributes`, split `train`) — annotations + per-row image URLs, both from the same source | **Source changed again this repair pass** — both the original `ranjaykrishna/visual_genome` (config `attributes_v1.2.0`) AND a prior `mikewang/vaw` replacement FAIL on `datasets==5.0.1` ("Dataset scripts are no longer supported", each confirmed on a real RunPod in turn). `AnnaZ1103/visual_genome_revised` was directly tested on a real RunPod with `datasets==5.0.1` and confirmed working as script-free Parquet — see "Visual Genome" section below. A community repackaging of VG's own annotations, not the canonical upstream distribution — documented, not claimed otherwise. |
| `object_recognition` | `ILSVRC/imagenet-1k`, split `validation` | **Confirmed live** this session — dataset exists, splits/schema/`int2str` class-name mapping confirmed; **GATED**, requires an HF token with accepted ImageNet license (user's explicit decision: build for gated access, hard-fail clearly, never substitute) |
| `fine_grained_recognition` | `bentrevett/caltech-ucsd-birds-200-2011` | Resolved-by-assumption — one of several community mirrors (`randall-lab/cub200`, `galilai-group/cub200`, others also exist), none singularly canonical; adapter reads canonical species names from the mirror's own `features["label"].names` and hard-fails (`CUBSchemaError`) rather than guessing if that's absent |
| `spatial_reasoning` / `relational_reasoning` | `external/RandOpt/data/gqa/testdev.parquet` via `GQAHandler`, filtered by `gqa_raw_schema.py` + `prepare_gqa_capability_filters.py` | Field NAMES **confirmed live** this session (see next section); the exact "relate" argument-encoding format is still pod-side-unconfirmed |

## GQA capability-filter wiring bug (fixed this repair pass)

**Root cause of the reported `RuntimeError: GQASpatialReasoningBenchmark needs either
question_ids or filter_ids_path`**: `GQAFilteredBenchmark.__init__` defaulted BOTH
`question_ids` and `filter_ids_path` to `None` with no fallback, and
`run_capability_benchmark_gate.py`'s `load_adapter()` always instantiates adapters with zero
constructor args (`adapter_cls()`) — there was simply no code path, anywhere, that could ever
supply either value in a real CLI run. This was a genuine wiring gap, not a data-availability
problem (confirmed separately: `external/RandOpt/data/gqa/` was also empty on the same fresh
pod, but that alone would have surfaced as a *different*, later error inside
`GQAHandler.load_data()`, not this one).

**Fix**: each GQA adapter subclass now declares a `DEFAULT_FILTER_IDS_FILENAME` class
attribute (`gqa_spatial_ids.json` / `gqa_relational_ids.json`); `GQAFilteredBenchmark.__init__`
resolves `filter_ids_path` to `artifacts/benchmark_subsets/<DEFAULT_FILTER_IDS_FILENAME>`
whenever it isn't explicitly passed, so the no-arg CLI instantiation now always has a real
default path. `_resolve_question_ids()` also now gives an actionable error (naming the exact
`prepare_gqa_capability_filters.py` command to run) if that default artifact doesn't exist
yet, instead of a bare "needs either..." message.

**New CLI**: `prepare_gqa_capability_filters.py` — loads GQA's raw annotations independently
of `GQAHandler`, runs `inspect_raw_schema()`, builds and persists both ID files plus a stats
file, and prints the spatial/relational/intersection/neither counts. See the bootstrap
sequence below.

## GQA spatial/relational filter — raw schema investigation

`adapters/gqa_raw_schema.py`'s field-name constants — **CONFIRMED live** this session via
direct HF dataset-viewer inspection of `lmms-lab-encoder/GQA`'s `testdev_balanced_instructions`
config: the raw rows DO carry `types` (`structural`/`semantic`/`detailed`), `semantic` (a
reasoning-program operation-step list), `semanticStr`, `groups`, `isBalanced`, `entailed`,
`equivalent` — `prepare_gqa_data.py`'s own parquet (`id`/`imageId`/`question`/`answer`/
`fullAnswer` only) is a deliberately NARROWED projection of these same rows for GQAHandler's
needs, not evidence the richer fields don't exist upstream. The semantic-category value for
relational questions is **`"rel"`**, not `"relation"` as initially guessed from public
documentation alone — corrected in `gqa_raw_schema.RELATION_SEMANTIC_VALUE` after this live
check.

**Still pod-side-unconfirmed**: the exact argument-encoding shape of `semantic`'s "relate"
operation steps (`_extract_relation_name`) — only the field NAMES and the semantic-category
value set were independently confirmed, not individual "relate" argument strings at the
row-content level. Run `prepare_gqa_capability_filters.py` (bootstrap step 7 below) and read
its printed `inspect_raw_schema()` report plus the spatial/relational counts before trusting
the resulting ID files as final.

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

## Visual Genome: source migration (VAW replaced this repair pass, second migration)

**Confirmed failure on a real RunPod, twice, for the same underlying reason**:
- `ranjaykrishna/visual_genome` (config `attributes_v1.2.0`) raises `RuntimeError: Dataset
  scripts are no longer supported, but found visual_genome.py` on `datasets==5.0.1`.
- The first replacement, `mikewang/vaw` (VAW), was ALSO confirmed failing on a real RunPod
  with `datasets==5.0.1` — it still ships its own `vaw.py` loading script and hits the exact
  same "Dataset scripts are no longer supported" error. **Do not use `mikewang/vaw` — it does
  not work with the pinned `datasets` version.**

Both repos require `trust_remote_code` execution of a legacy Python loading script, which
recent `datasets` versions refuse to run automatically. No HF auto-converted-Parquet mirror
exists for either (the `datasets-server` parquet API 404s for both), so no configuration
change fixes this — a genuinely different, script-free source is required.

**Resolution, confirmed on a real RunPod this repair pass**: `AnnaZ1103/visual_genome_revised`,
config `"attributes"`, split `"train"` — loads successfully as script-free Parquet under
`datasets==5.0.1`. This is a **community repackaging** of Visual Genome's own attribute
annotations, not the canonical upstream `ranjaykrishna/visual_genome` distribution — documented
here and in the adapter's `known_caveats()`, never claimed to be canonical. Real observed
schema (105414 rows):

```
{
    "image_id": int64, "url": string, "width": int64, "height": int64,
    "coco_id": int64, "flickr_id": int64,
    "attributes": [
        {"attributes": [string], "h": int64, "names": [string], "object_id": int64,
         "synsets": [string], "w": int64, "x": int64, "y": int64}
    ],
}
```
One row = one image; `attributes` is a list of annotated objects, each with its own bbox
(`x, y, w, h`), a synonym-name list, and its positive visual-attribute list.

**Design change: no more full image-archive download.** Unlike the VAW-era design (which
needed to fetch VG's entire `VG_100K(_2).zip` image archives separately, since VAW carries no
image bytes), this source's own rows already carry each image's canonical `url` — so
`prepare_visual_genome_data.py` now fetches images **one at a time, by URL**, only for images
actually referenced by the flattened examples it selects. The old
`DEFAULT_IMAGES_ZIP_URL_PART1`/`PART2` zip-archive URLs and `extract_needed_images_from_zip`
logic are removed entirely — there is no longer an "UNRESOLVED archive URL" caveat, because
there is no archive.

**Flattening and filtering** (`flatten_attribute_examples`, deterministic, documented, never
model-performance-driven): one flattened example per (image, object) pair; excludes objects
with an empty `attributes` list (no target), an empty `names` list (no queryable object name),
or a bbox failing `validate_bbox()` (`x>=0, y>=0, w>0, h>0`, plus `x+w<=width` / `y+h<=height`
with a documented ±1px tolerance for VG's known annotation-rounding quirk). `object_name` is
the object's FIRST `names` entry (a documented simplification, separate from the multi-
attribute target, which is always preserved in full — never collapsed to one answer).

**Not done in this pass — flagged, not silently resolved**: no appearance-vs-state semantic
filtering exists on the raw attribute vocabulary (e.g. `"walking"` is a real observed VG
attribute value that arguably describes an action/state, not a visual appearance). There is no
existing, defensible criterion in this codebase for that distinction; inventing one now would
be silent ontology engineering. This is a scientific-review item for the upcoming N=5 manual
inspection pass.

`prepare_visual_genome_data.py` writes `external/RandOpt/data/visual_genome/{vg_attributes.parquet,images/,vg_prepare_stats.json}`
(prefix-sliced to `--max-candidates` image ROWS, default 500 — bounds dataset-load size and
the number of images fetched; a documented deviation from "load everything," same "first N"
discipline `prepare_gqa_data.py` already uses; NOT the final N=200 benchmark subset, which is
drawn later from this larger candidate pool). `vg_prepare_stats.json` persists candidate/
flattened/skip counts for reproducibility. `verify_visual_genome_data.py` re-derives (never
trusts prepare's own exit status) row count, unique images, duplicate `example_id`, empty
`positive_attributes`/`object_name`, invalid bboxes, the accepted-attribute cardinality
distribution, and every referenced image present/not-corrupt, before the artifact is trusted.
The adapter (`attribute_recognition_visualgenome.py`) reads ONLY this local prepared artifact
— it never calls `datasets.load_dataset(...)` at all.

## Visual Genome: benchmark-example identity (fixed this repair pass)

**Real RunPod finding**: `verify_visual_genome_data.py` reported many `duplicate_instance_ids`
on a real prepared artifact. Root cause: the source VG `object_id` is NOT a safe benchmark
identity in this repackaged dataset — live inspection confirmed image_id=2 contains TWO
distinct "building" object records both with `object_id=22`, but different bboxes (`x=363,
y=0, w=146, h=265` vs. `x=108, y=0, w=166, h=205`). Even `(image_id, object_id)` is therefore
not guaranteed unique.

**Fix**: `build_example_id()` in `prepare_visual_genome_data.py` derives the benchmark
`example_id` from `(image_id, object_id, x, y, w, h)`, with a deterministic per-base-key
occurrence counter appended only when that full tuple genuinely repeats (a still-possible
malformed-duplicate-record case) — provably collision-free rather than "collision-free in the
cases observed so far." The source `object_id` is preserved as its own separate column
(prepare/verify) and `Example.metadata["object_id"]` (adapter) — never overwritten to
manufacture uniqueness; `example_id` and `object_id` are different concepts throughout this
package. `verify_visual_genome_data.py`'s duplicate check now keys on `example_id`, not
`object_id`.

## N=5 repair pass: real Qwen2.5-VL findings (this session)

The first real N=5 smoke test (7 of 8 capabilities, commit `0ffa06e`) surfaced several
protocol defects invisible to `parser_failure_rate` (reported as 0 throughout). Fixed here,
each narrowly scoped, none touching RandOpt/perturbation/CUB/TallyQA/ImageNet:

**1. Visual grounding — coordinate contract.** Qwen reliably emitted PIXEL-space boxes
despite the prompt asking for `[0,1]`-normalized ones (e.g. predicting `[112,189,444,362]` for
a 640x425 image against a target of `~[0.165,0.461,0.686,0.860]` — a real ~0.91 IoU match,
scored as ~0 by comparing the raw numbers directly). Fixed at the evaluator layer, never by
special-casing Qwen or changing the prompt: `box_iou.detect_coordinate_mode()`/
`canonicalize_prediction_box()` classify a prediction as `normalized_xyxy_0_1` /
`pixel_xyxy` / `qwen_normalized_0_1000` / `unrecognized` using ONLY deterministic value-range +
this example's real image dimensions (never accuracy), and `score_example()` converts both
prediction and target into one canonical pixel-space representation before IoU. Per-example
`raw_prediction_box`/`canonical_prediction_box`/`coordinate_mode` are now recorded in
`ExampleScore.detail`. An `unrecognized` box counts toward `parser_failure_rate`.

**1b. Visual grounding — RefCOCO field mixup.** `row["question"]` was, in every N=5 example,
the fixed instruction "Please carefully observe the area circled in the image and come up
with a caption for the area." — confirmed via live schema re-inspection this session that
`lmms-lab-encoder/RefCOCO` repackages RefCOCO as an instruction-tuned region-captioning
dataset, NOT a faithful referring-expression representation. The real referring expression(s)
live in `row["answer"]` (a list of independent human-written region descriptions, e.g.
`["bowl behind the others can only see part", "Dish in top right corner"]`). Fixed via Option
A (recover from available fields): `load_examples()` now uses `answer[0]` deterministically;
all annotations are kept in `Example.metadata["all_referring_expressions"]` for audit. Hard-
fails (`RefCOCOSchemaError`) on an empty `answer` list rather than falling back to the
misleading `question` field.

**2. GQA parser + prompt.** GQAHandler's own `extract_answer_for_voting` (external, frozen)
mis-extracted `\boxed{\text{the person in the blue shirt}}` as `\text{the person` (a
non-balanced-brace bug), and fabricated `"step step"` for a generation truncated by the token
ceiling before any `\boxed{}` appeared — `parser_failure_rate` stayed 0 throughout. Fixed by a
NEW capability-benchmark-only module, `adapters/gqa_boxed_answer.py`
(`extract_boxed_answer()`): balanced-brace matching (correctly handles nesting), unwraps
simple LaTeX wrappers (`\text{}`/`\mathrm{}`/etc.), returns `None` — a real parser failure,
never a fabricated fallback — for a missing/truncated `\boxed{}`. `_gqa_filtered_base.py`'s
`parse_prediction()` now uses this instead of `extract_answer_for_voting()`; actual reward
scoring is UNCHANGED (`compute_reward()` still runs on the raw generation). Separately,
`build_prompt()` now appends a capability-benchmark-ONLY "keep reasoning brief, still use
`\boxed{}`" instruction to GQAHandler's own last message turn (never mutating it, never
editing GQAHandler's own historical text) — addressing the observed truncation cause without
touching GQAHandler or Gate 1's own scripts, which never see this addition.

**3. GQA capability taxonomy audit.** A manual N=5 sample raised the concern that "around"/"on"
might be spatial relations misclassified as non-spatial. Two things were done, per explicit
instruction NOT to fabricate a correction until real data is inspected: (a) a real,
schema-independent bug was fixed regardless of what the real values turn out to be —
`is_spatial_relation` used bare substring matching (`"on" in lowered`), which could
false-positive inside an unrelated word (e.g. "person", "along", "onion"); it now matches each
keyword with `\b`-bounded regex. (b) a new CPU-side audit utility,
`gqa_raw_schema.describe_question_classification()` (exposed via
`prepare_gqa_capability_filters.py --audit-question-ids`), prints the full real raw record —
question, `types.semantic`/`types.structural`, the raw `semantic` program, `semanticStr`, the
extracted relation name, and the classification decision + matched keyword — for a given
question ID. `SPATIAL_RELATION_KEYWORDS` and the persisted `gqa_spatial_ids.json`/
`gqa_relational_ids.json` filter files are UNCHANGED in this commit; regenerate only after
running the audit command below on real data.

**4. OCR/text-recognition — capability leakage.** A real N=5 sample showed TextVQA questions
("how many wheels does this van have?" → "4", not OCR-supported; "is this book material?", not
an OCR reading question) mixed in with genuine OCR questions ("what type of laptop is this?" →
"macbook air", supported by OCR tokens "MacBook Air"). `ocr_text_recognition`
(`TextVQAOCRBenchmark`) is UNCHANGED — it remains full TextVQA. A NEW, separate, EXPERIMENTAL
capability `ocr_text_recognition_grounded` (`TextVQAOCRGroundedBenchmark`) narrows to examples
where at least one reference answer's word sequence is recoverable from the row's own OCR
token sequence (`benchmarks/ocr_grounding.py`, supporting multi-token answers like "macbook
air"/"chicken noodle"), using target answers + provided OCR tokens ONLY — never model
predictions. `prepare_textvqa_ocr_filter.py` persists the ID filter (same prepare-then-filter
pattern as GQA) and reports total/retained/rejected/percent. Every TextVQA `Example` (both
variants) now also carries `metadata["ocr_grounded"]` for audit.

**5. Visual Genome attribute prompt.** A real N=5 sample showed the model sometimes answering
with the attribute's CATEGORY ("Material", "Color: Brown") instead of its VALUE ("wooden",
"brown"). `INSTRUCTION` now explicitly asks for the VALUE and explicitly rules out a bare
category word, without ever revealing the ground-truth attribute. The same sample showed a
target with stray whitespace (`"wooden "`); `load_examples()` now strips target attribute
values (the value actually scored against) while `Example.metadata["raw_positive_attributes"]`
keeps the exact, unmodified values. VG's raw vocabulary mixing colors/materials with
action/state words (`"hanging"`, `"walking"`) is now surfaced via
`Example.metadata["flagged_state_action_attributes"]` against a small, explicitly
non-exhaustive watchlist (`STATE_ACTION_ATTRIBUTE_WATCHLIST`) — for later manual ontology
review only; nothing is filtered, dropped, or reweighted.

**6/7. Clean tasks untouched; new audit tooling.** Counting (TallyQA) and fine-grained (CUB)
adapters are unmodified. A new CPU-side utility, `inspect_capability_predictions.py`, reads a
`predictions.jsonl` and prints a compact per-example report (query/target/raw_generation/
parsed_prediction/score, plus capability-specific detail/metadata fields — grounding's
IoU/boxes/coordinate_mode, GQA's `extracted`, OCR's `ocr_grounded`, attributes' bbox/object_id/
raw attributes) — every real bug in this section was originally found by exactly this kind of
manual side-by-side read, not by any aggregate metric.

## Fresh RunPod bootstrap

The exact command sequence to go from an empty pod to all 8 capabilities passing `--dry-run`
integrity checks. Confirmed against a real fresh RunPod this session up through the
environment/runtime pins (521 passed / 0 failed once `transformers`/`tokenizers`/`vllm`/
`datasets` were pinned as below); the GQA/VG-specific steps below are new as of this repair
pass and have not yet been re-run end-to-end on a fresh pod.

```bash
# 1. Clone repo
git clone <this-repo-url> thickets
cd thickets/research/neural_thickets_repro

# 2. Setup external RandOpt pinned dependency (gitignored, never vendored)
python external/setup_external_repo.py

# 3. Install the known-good runtime (see requirements/requirements-gpu.txt's own comment for
#    WHY these three are pinned exactly -- an unpinned vllm==0.11.0 install pulled in
#    transformers 5.15.1 by default on a real pod, breaking vllm+Qwen2.5-VL with an
#    mrope/rope_type configuration conflict)
pip install -r requirements/requirements-gpu.txt

# 4. HF login (needed for GQA/RefCOCO/TextVQA/TallyQA -- all ungated -- and REQUIRED for the
#    gated ImageNet-1K; VG no longer touches HF's gated ranjaykrishna/visual_genome repo, or
#    mikewang/vaw, at all -- see the Visual Genome section above)
huggingface-cli login   # for ImageNet-1K: this account must have separately accepted the
                         # license at https://huggingface.co/datasets/ILSVRC/imagenet-1k

# 5. Prepare GQA evaluation data (GQAHandler's own parquet + images -- historical semantics
#    unchanged; this is source A, see "Fresh-pod GQA data preparation" above)
python -m neural_thickets_repro.prepare_gqa_data --config configs/gqa_repro.yaml

# 6. Verify GQA evaluation data
python -m neural_thickets_repro.verify_gqa_data --config configs/gqa_repro.yaml

# 7. Prepare GQA spatial/relational capability filters (source B -- raw annotation metadata,
#    loaded independently of step 5/6; writes artifacts/benchmark_subsets/gqa_spatial_ids.json
#    + gqa_relational_ids.json + gqa_spatial_relational_stats.json). Inspect the printed
#    schema-confirmation report and the spatial/relational/intersection/neither counts before
#    trusting the result -- do not skip reading this output.
python -m neural_thickets_repro.prepare_gqa_capability_filters --config configs/gqa_repro.yaml

# 8. Prepare Visual Genome attribute-recognition data (AnnaZ1103/visual_genome_revised, see
#    "Visual Genome" section above -- fetches only the individual images actually needed, by
#    URL, no full archive download)
python -m neural_thickets_repro.prepare_visual_genome_data

# 9. Verify Visual Genome data
python -m neural_thickets_repro.verify_visual_genome_data

# 10. (optional, this repair pass) Audit real GQA relation-argument values for specific
#     question IDs before trusting/regenerating the spatial/relational filters -- read-only,
#     never persists anything. See "N=5 repair pass" section above, item 3.
python -m neural_thickets_repro.prepare_gqa_capability_filters --config configs/gqa_repro.yaml --audit-question-ids <comma-separated-question-ids>

# 11. (this repair pass) Prepare the EXPERIMENTAL OCR-grounded TextVQA subset -- read the
#     printed total/retained/rejected/percent_retained before trusting it. See item 4 above.
python -m neural_thickets_repro.prepare_textvqa_ocr_filter

# 12. Dry-run every capability (data loading + integrity only, no GPU/model call)
for cfg in object_recognition visual_grounding counting spatial_reasoning relational_reasoning ocr_text_recognition ocr_text_recognition_grounded attribute_recognition fine_grained_recognition; do
    python -m neural_thickets_repro.run_capability_benchmark_gate --config configs/benchmarks/$cfg.yaml --dry-run
done

# 13. GPU model smoke (confirms the pinned runtime actually loads Qwen2.5-VL-3B-Instruct
#     under vLLM before spending time on any real benchmark run)
python -m neural_thickets_repro.eval_base_image_aware --config configs/gqa_repro.yaml --max-examples 1

# 14. (this repair pass) Manually audit N=5 predictions before trusting aggregate metrics --
#     see "N=5 repair pass" section above, item 6/7. `--capability` selects the extra
#     grounding/GQA/OCR/attribute fields to surface.
python -m neural_thickets_repro.inspect_capability_predictions --predictions <results-dir>/visual_grounding/predictions.jsonl --capability visual_grounding
```

## Open items (UNRESOLVED, tracked here, not silently resolved)

- GQA raw "relate" operation argument-encoding shape (field NAMES are now confirmed; the
  individual argument STRING format at the row-content level is not) — run
  `prepare_gqa_capability_filters.py` and read its printed report.
- `lmms-lab-encoder/RefCOCO+`'s exact repo id — assumed analogous to the confirmed RefCOCO
  repo, not independently verified.
- `AnnaZ1103/visual_genome_revised`'s own long-term stability/maintenance as a community
  repackaging (as opposed to a maintained canonical distribution) is not something this
  project controls — confirmed working on a real RunPod at the time of this repair pass, but
  not guaranteed to remain so indefinitely; the earlier `mikewang/vaw` failure is a concrete
  precedent for a script-free-Parquet source silently reverting to script-based loading.
- Visual Genome attribute vocabulary is not filtered for appearance-vs-state terms (e.g.
  `"walking"`, `"hanging"`) — a real N=5 manual inspection confirmed both occur in the wild;
  `Example.metadata["flagged_state_action_attributes"]` now surfaces matches against a small,
  explicitly non-exhaustive watchlist for later manual ontology review, but nothing is
  filtered/dropped/reweighted (see "N=5 repair pass" section below).
- GQA raw relation-argument values for the 10 audited question IDs (see "N=5 repair pass"
  section below) — the word-boundary substring-matching bug is fixed regardless of these
  values, but the persisted `gqa_spatial_ids.json`/`gqa_relational_ids.json` filters
  themselves are UNCHANGED pending that real-data audit; do not regenerate them until it runs.
- TextVQA OCR-groundedness real retained/rejected counts and percentage — the filter logic is
  tested against synthetic data here; the real numbers come only from actually running
  `prepare_textvqa_ocr_filter.py` on the pod (see "N=5 repair pass" section below).
- CUB-200-2011 mirror choice — `bentrevett/caltech-ucsd-birds-200-2011` is a documented,
  revisable pick among several community mirrors.
- ImageNet-1K gated access — requires an HF token with accepted license configured on
  whichever machine actually runs `load_examples()`; the adapter hard-fails with a clear
  message (`ImageNetGatedAccessError`) if this isn't set up, never substitutes anything else
  (confirmed as a real, expected user-access blocker on the RunPod this session, not a bug).

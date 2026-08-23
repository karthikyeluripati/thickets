"""Visual Genome attributes (attribute_recognition) adapter.

Dataset source: a LOCALLY PREPARED artifact (see prepare_visual_genome_data.py), not a live
`datasets.load_dataset()` call. Source history (see CAPABILITY_BENCHMARK_GATE.md for the full
record): `ranjaykrishna/visual_genome` (original config `attributes`) and then `mikewang/vaw`
(VAW) were BOTH tried and BOTH fail on modern `datasets` with "Dataset scripts are no longer
supported" -- confirmed on a real RunPod, datasets==5.0.1, for each in turn. The operational
source is now `AnnaZ1103/visual_genome_revised` (config "attributes", split "train") --
CONFIRMED script-free Parquet on a real RunPod with datasets==5.0.1 -- a community
repackaging of Visual Genome's own attribute annotations (not the canonical upstream VG
distribution; documented, not claimed otherwise). Each row carries its own image `url`
(Stanford-hosted), so `prepare_visual_genome_data.py` fetches only the images actually needed
by the flattened examples -- never the full ~100k-image VG archive -- and writes
`<source>/vg_attributes.parquet` + `<source>/images/`. This adapter reads ONLY that prepared
local artifact, mirroring GQAHandler's own "prepare offline, adapter reads locally" split of
labor.

To keep scoring automatic and unambiguous without reducing this to unrestricted captioning:
one Example = one (image, object) pair. The `positive_attributes` list is preserved in FULL
as Example.target (never collapsed to one answer) -- a prediction is scored correct if it
matches ANY of them (see score_example), the same "preserve the valid target set" discipline
TextVQA's 10-answer list already uses in this package.

LOCALIZED-CROP PROTOCOL (this repair pass, replacing the earlier full-image + red-marker
protocol): a real N=50 image-sanity run on the marker-overlay protocol found ZERO visual
dependence -- correct=0.15, shuffled=0.10, text-only=0.15 (text-only EXACTLY matched
correct-image). Root cause: the full scene image (plus the object's own NAME, always spoken
in the prompt) gave the model enough of a prior to guess a plausible attribute for that
object category without ever needing to look at the marked region -- "wooden"/"brown" are
generic, high-prior guesses for "chair" regardless of which chair is actually pictured, and a
drawn rectangle competing with a whole busy scene is a weak localization signal. Fixed by
cropping the image down to the annotated ground-truth bbox (see benchmarks/image_crop.py) --
prepare_image() below now returns ONLY the object's own (padded) region, not the full scene,
so there is no remaining scene content left for the model to answer from without attending to
the crop itself. The bbox is used ONLY for localization (WHERE to crop) -- it is never
inferred, altered, or treated as attribute/target information, and the dataset/subset
sampling/scoring are all otherwise unchanged.

PROMPT FIX (this repair pass): a real N=5 manual inspection found the model sometimes
answering with the attribute's CATEGORY name ("Material", "Color: Brown") instead of its
VALUE ("wooden", "brown") -- the previous instruction asked for "ONE visual attribute"
without ever clarifying that a category name is not itself an answer. INSTRUCTION below now
explicitly asks for the attribute VALUE and explicitly rules out answering with a bare
category word. The prompt never reveals the ground-truth category or value.

TARGET WHITESPACE / RAW-ATTRIBUTE PRESERVATION (this repair pass): the same N=5 inspection
found a raw target value with stray trailing whitespace ("wooden "). load_examples() now
strips whitespace from each attribute value going into Example.target (the value actually
used for scoring), while Example.metadata["raw_positive_attributes"] keeps the exact,
unmodified values as read from the parquet, so nothing is silently discarded.

STATE/ACTION ATTRIBUTE FLAGGING (this repair pass, reporting only, no filtering): VG's raw
attribute vocabulary mixes colors/materials with action/state words (e.g. "hanging",
"walking") observed in the same N=5 sample. Example.metadata["flagged_state_action_attributes"]
surfaces these against a small, explicitly non-exhaustive watchlist (STATE_ACTION_ATTRIBUTE_
WATCHLIST below) for later manual ontology review -- examples are NEVER filtered, dropped, or
reweighted based on this flag, since no principled inclusion/exclusion rule exists yet.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from ..base import CapabilityBenchmark, Example, ExampleScore, ParsedPrediction
from ..image_crop import CROP_CONTEXT_PADDING_FRACTION, CropError, compute_padded_crop_box, crop_to_bbox
from ..normalization import normalize_answer
from ..prompting import build_image_text_messages

# {object_name} filled from the row's own object_name -- NOT a ground-truth attribute value,
# just the (already-known, pre-generation) object category, matching the LOCALIZED-CROP
# PROTOCOL note above. Never mentions bbox coordinates or the accepted attribute values.
INSTRUCTION_TEMPLATE = (
    "The image shows a close-up crop centered on a single {object_name}. State ONE visible "
    "attribute VALUE of this {object_name} -- for example a specific color such as 'brown', "
    "a material such as 'wooden', a texture, or a visible state. Answer with the attribute "
    "VALUE itself, never the attribute's category name (do not answer with a bare category "
    "word like 'Color', 'Material', or 'Texture'). Answer with a single word or short phrase "
    "giving the actual attribute, and nothing else."
)
PREPARE_COMMAND = "python -m neural_thickets_repro.prepare_visual_genome_data"

# Small, EXPLICITLY non-exhaustive watchlist of VG attribute values that describe an
# action/state rather than a lasting visual-appearance property -- for later manual ontology
# review ONLY (see Example.metadata["flagged_state_action_attributes"]). Never used to filter,
# drop, or reweight examples: no principled inclusion/exclusion rule exists yet, and inventing
# one here would be silent ontology engineering.
STATE_ACTION_ATTRIBUTE_WATCHLIST = frozenset({
    "hanging", "walking", "running", "standing", "sitting", "sleeping", "flying",
    "swimming", "jumping", "riding",
})


def _flagged_state_action_attributes(attributes: List[str]) -> List[str]:
    return [a for a in attributes if a.strip().lower() in STATE_ACTION_ATTRIBUTE_WATCHLIST]


class VisualGenomeSchemaError(RuntimeError):
    """The prepared local Visual Genome artifact is missing or doesn't match the expected
    schema (example_id/image_id/object_id/object_name/positive_attributes/bbox_x/y/w/h) --
    refuses to guess a different one.
    """


class VisualGenomeAttributeBenchmark(CapabilityBenchmark):
    capability = "attribute_recognition"
    name = "visual_genome_attributes"

    def dataset_source(self) -> str:
        return "AnnaZ1103/visual_genome_revised (attributes config, train split) + per-row image URLs, prepared locally by prepare_visual_genome_data.py"

    def known_caveats(self) -> List[str]:
        return [
            "Annotation source is AnnaZ1103/visual_genome_revised (config 'attributes'), a "
            "community repackaging of Visual Genome's own attribute annotations -- NOT the "
            "canonical ranjaykrishna/visual_genome upstream distribution. Migrated because "
            "BOTH ranjaykrishna/visual_genome AND a prior mikewang/vaw replacement fail on "
            "modern `datasets` with 'Dataset scripts are no longer supported' (each confirmed "
            "failing on a real RunPod, datasets==5.0.1); this source was confirmed working "
            "(script-free Parquet) on a real RunPod before being adopted.",
            "Images are fetched individually via each row's own `url` field (Stanford-hosted "
            "originals), only for images actually referenced by the flattened examples -- "
            "never the full ~100k-image Visual Genome archive.",
            "`object_name` is the FIRST entry of the object's `names` list -- VG objects can "
            "carry multiple synonym names; only the first is used as the single-noun prompt "
            "target. This is separate from the attribute TARGET set, which is always kept in "
            "full (see below).",
            "No appearance-vs-state semantic filtering is applied to VG's raw attribute "
            "vocabulary -- e.g. 'walking' and 'hanging' are real observed VG attribute values "
            "(confirmed via a real N=5 manual inspection) that arguably describe an "
            "action/state rather than a visual appearance attribute. There is no existing, "
            "defensible criterion in this codebase for that distinction, and one is "
            "deliberately NOT invented here. Example.metadata['flagged_state_action_"
            "attributes'] surfaces matches against STATE_ACTION_ATTRIBUTE_WATCHLIST (a small, "
            "explicitly non-exhaustive list) for later manual ontology review only -- examples "
            "are never filtered, dropped, or reweighted based on this flag.",
            "The prompt explicitly asks for the attribute VALUE, not its category name (e.g. "
            "not 'Material') -- a real N=5 manual inspection found the model sometimes "
            "answering with a bare category label instead of a value.",
            "Target attribute values are whitespace-stripped (e.g. 'wooden ' -> 'wooden'); "
            "the exact, unmodified raw values are preserved in "
            "Example.metadata['raw_positive_attributes'].",
            "The model is shown a CROP of the annotated bbox (with a fixed "
            f"{CROP_CONTEXT_PADDING_FRACTION:.0%} context padding, see benchmarks/image_crop.py), "
            "not the full scene image -- replaces an earlier full-image + red-bbox-marker "
            "protocol that a real N=50 image-sanity run showed had NO measurable visual "
            "dependence (correct=0.15, shuffled=0.10, text-only=0.15 -- text-only exactly "
            "matched correct-image). The bbox is used ONLY to localize the crop -- it is "
            "never inferred, altered, or treated as attribute/target information. A row whose "
            "bbox cannot produce a valid (non-degenerate) crop is excluded from the candidate "
            "pool at load_examples() time, not silently cropped to garbage.",
            "The shuffled-image sanity condition swaps in a DIFFERENT example's own "
            "(image, bbox) pair together -- never this example's own bbox applied to a "
            "different photo, which would produce a misaligned/meaningless crop rather than a "
            "genuine 'different but valid' visual distractor.",
            "The full positive_attributes list is preserved as the target set; a prediction "
            "is scored correct if it matches ANY of them.",
            "Example.example_id is derived from (image_id, object_id, bbox) via "
            "build_example_id() in prepare_visual_genome_data.py, NOT the raw source "
            "object_id alone -- confirmed on a real RunPod that this repackaged dataset can "
            "contain multiple distinct object records sharing the same object_id within one "
            "image. The source object_id is preserved separately in Example.metadata.",
        ]

    def load_examples(self, cfg: Any) -> List[Example]:
        import pandas as pd
        from PIL import Image

        data_dir = Path(cfg.dataset.source)
        parquet_path = data_dir / "vg_attributes.parquet"
        images_dir = data_dir / "images"
        if not parquet_path.exists():
            raise VisualGenomeSchemaError(
                f"No prepared Visual Genome attributes parquet found at {parquet_path}. "
                f"Generate it first with:\n    {PREPARE_COMMAND}"
            )

        df = pd.read_parquet(parquet_path)
        missing_columns = {"example_id", "image_id", "object_id", "object_name", "positive_attributes", "bbox_x", "bbox_y", "bbox_w", "bbox_h"} - set(df.columns)
        if missing_columns:
            raise VisualGenomeSchemaError(f"{parquet_path} is missing expected column(s) {sorted(missing_columns)} -- refusing to guess a different schema.")

        has_image_dims = {"image_width", "image_height"} <= set(df.columns)

        examples: List[Example] = []
        for _, row in df.iterrows():
            image_path = images_dir / f"{row['image_id']}.jpg"
            image = Image.open(image_path).convert("RGB") if image_path.exists() else None
            bbox_xywh = [row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"]]

            # LOCALIZED-CROP PROTOCOL: a row whose bbox cannot produce a valid, non-degenerate
            # crop against its OWN real image dimensions is excluded from the candidate pool
            # here, deterministically -- never silently cropped to a garbage/empty region.
            # image.size is used (not a possibly-stale recorded image_width/height column) so
            # this always reflects the actual downloaded image file.
            crop_box_xyxy = None
            if image is not None:
                try:
                    crop_box_xyxy = list(compute_padded_crop_box(bbox_xywh, image.size[0], image.size[1]))
                except CropError:
                    continue

            raw_attributes = list(json.loads(row["positive_attributes"]))
            # target: whitespace-normalized (scoring already tolerates stray whitespace via
            # normalize_answer's own .strip(), but the stored target itself should be clean --
            # see this module's TARGET WHITESPACE docstring note). metadata keeps the exact,
            # unmodified raw values separately.
            normalized_attributes = [a.strip() for a in raw_attributes if a.strip()]
            metadata: Dict[str, Any] = {
                "bbox_xywh": bbox_xywh,
                "crop_box_xyxy": crop_box_xyxy,  # the ACTUAL padded+clipped crop bounds -- input localization metadata only, never a target
                "image_id": str(row["image_id"]),
                # source VG object_id, preserved distinctly from example_id -- NOT unique on
                # its own in this repackaged dataset (see prepare_visual_genome_data.py's
                # "BENCHMARK EXAMPLE IDENTITY" docstring section), so never used as the identity.
                "object_id": str(row["object_id"]),
                "raw_positive_attributes": raw_attributes,
                "flagged_state_action_attributes": _flagged_state_action_attributes(raw_attributes),
            }
            if has_image_dims:
                metadata["image_width"] = row["image_width"]
                metadata["image_height"] = row["image_height"]
            examples.append(Example(
                example_id=str(row["example_id"]),
                image=image,
                image_ref=str(image_path),
                prompt_input={"object_name": row["object_name"]},
                target=normalized_attributes,
                metadata=metadata,
            ))
        return examples

    def prepare_image(self, example: Example):
        """Returns the LOCALIZED CROP (never the full scene) -- see this module's
        LOCALIZED-CROP PROTOCOL docstring note. `crop_to_bbox` never mutates
        `example.image` (PIL's own `.crop()` always returns a new Image) and derives the
        crop bounds from the image's own real `.size`, so this is correct for the
        shuffled-image sanity condition too (a different image, with a different real size,
        paired with its OWN bbox via make_shuffled_image_variant() below).
        """
        if example.image is None:
            return None
        cropped, _ = crop_to_bbox(example.image, example.metadata["bbox_xywh"], CROP_CONTEXT_PADDING_FRACTION)
        return cropped

    def make_shuffled_image_variant(self, example: Example, source_example: Example) -> Example:
        """Overrides CapabilityBenchmark's default (which would swap in `source_example`'s
        image while keeping `example`'s own bbox metadata -- silently applying one example's
        localization box to a DIFFERENT photo, producing a misaligned/meaningless crop, not a
        genuine "different but valid" visual distractor). The shuffled condition here is
        `source_example`'s own (image, bbox) pair -- a real, validly-localized crop of a
        different object -- paired with `example`'s own prompt/target, exactly as the
        LOCALIZED ATTRIBUTE RECOGNITION protocol requires. `source_example` is guaranteed to
        already have a valid crop (or it would have been excluded at load_examples() time).
        """
        new_metadata = dict(example.metadata)
        new_metadata["sanity_shuffle_source_id"] = source_example.example_id
        new_metadata["bbox_xywh"] = source_example.metadata["bbox_xywh"]
        new_metadata["crop_box_xyxy"] = source_example.metadata.get("crop_box_xyxy")
        return Example(
            example_id=example.example_id, image=source_example.image, image_ref=f"shuffled_from:{source_example.image_ref}",
            prompt_input=example.prompt_input, target=example.target, metadata=new_metadata,
        )

    def build_prompt(self, example: Example) -> List[dict]:
        instruction = INSTRUCTION_TEMPLATE.format(object_name=example.prompt_input["object_name"])
        return build_image_text_messages(instruction)

    def parse_prediction(self, raw_generation: str, example: Example) -> ParsedPrediction:
        stripped = raw_generation.strip()
        if not stripped:
            return ParsedPrediction(parsed="", parse_ok=False, parse_error="empty generation")
        return ParsedPrediction(parsed=stripped, parse_ok=True)

    def score_example(self, parsed: ParsedPrediction, example: Example) -> ExampleScore:
        if not parsed.parse_ok:
            return ExampleScore(score=0.0, correct=False, detail={"reason": "parse_failure"})

        predicted_norm = normalize_answer(parsed.parsed)
        padded_prediction = f" {predicted_norm} "
        valid_targets_norm = [normalize_answer(a) for a in example.target]
        matched = next((a for a in valid_targets_norm if a and f" {a} " in padded_prediction), None)

        correct = matched is not None
        return ExampleScore(score=1.0 if correct else 0.0, correct=correct, detail={"matched_attribute": matched, "valid_targets": example.target})

    def aggregate_metrics(self, scores: List[ExampleScore]) -> Dict[str, float]:
        n = len(scores)
        if n == 0:
            return {"accuracy": 0.0, "primary_metric": 0.0, "parser_failure_rate": 0.0}
        parser_failures = sum(1 for s in scores if s.detail.get("reason") == "parse_failure")
        accuracy = sum(s.score for s in scores) / n
        return {"accuracy": accuracy, "primary_metric": accuracy, "parser_failure_rate": parser_failures / n}

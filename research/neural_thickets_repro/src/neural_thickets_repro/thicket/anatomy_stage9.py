"""Stage 9: hierarchical (L2) child-region partition for the two drilled-down L1 parents
(vision, language) -- multimodal_connector_or_merger is deliberately NOT drilled down (Stage 8
found no stable capability-selective dominance there, and it is architecturally far smaller
than vision/language; it remains a single coarse L1 reference region).

Builds on thicket.anatomy.build_anatomy_atlas's ALREADY-EXISTING L2 depth-band regions
(vision_early/vision_middle/vision_late, language_early/language_middle/language_late -- a
deterministic near-equal-thirds split of the parent's own discovered block/layer indices) --
never reimplements that discovery/partitioning logic. What this module adds, additively:

1. Renames "vision_middle"/"language_middle" to "vision_mid"/"language_mid" (Stage 9's own
   naming convention throughout the spec), a pure relabeling of the SAME underlying tensor set.

2. Resolves the two LANGUAGE PARENT tensors thicket.anatomy.validate_atlas already reports as
   `uncovered_by_parent["language"]` (confirmed, live, both here and in Stage 7A's own prior
   inventory: the input token-embedding tensor and the final pre-head normalization tensor,
   whose exact string names depend on which LM namespace convention -- "runtime_wrapped"
   (language_model.model.*) vs "flat_checkpoint" (model.*) -- this model happens to use, so
   NEVER hardcoded as literal strings) via a generic, deterministic architectural-position rule
   (see `_classify_uncovered_tensor`): a name containing "embed" is input-side -> language_early;
   a name containing "norm" (and not already inside a numbered layer, which by construction it
   cannot be, since "uncovered" means the layer-index regex did NOT match it) is output-side ->
   language_late. Any OTHER kind of uncovered tensor (should not exist per Stage 7A's own
   confirmed inventory, but never assumed) hard-fails rather than being silently dropped or
   guessed -- this is the SAME generic rule applied to whichever L1 parent needs it, so it is
   equally ready for vision if a future model revision ever introduces an uncovered vision
   tensor (see point 3).

3. Applies the IDENTICAL uncovered-tensor audit to vision. Live-code-audited (this repair pass,
   via a fixture mirroring this project's own confirmed real safetensors tensor-name structure --
   REPRO_SPEC.md: 100% of vision-tower tensors are `visual.patch_embed.*` / `visual.blocks.*` /
   `visual.merger.*`, nothing else): vision_early/vision_middle/vision_late's own EXISTING
   membership (vision_early already additionally owns patch_embed + any trainable
   rotary_pos_emb, per anatomy.py's own established rule) already exactly partitions the vision
   L1 parent with ZERO uncovered tensor -- confirmed by `validate_atlas`, never merely assumed.
   `build_stage9_hierarchical_partition` still runs the SAME audit+resolve step for vision as
   for language (never a special case), so if a real live inventory ever surfaces an uncovered
   vision tensor, it is deterministically resolved (or hard-fails as unclassifiable) rather than
   silently ignored.

Stage 9's SIX child regions are exactly:
    vision_early, vision_mid, vision_late, language_early, language_mid, language_late
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

from .anatomy import AnatomyAtlas, AnatomyRegion, build_anatomy_atlas, compute_mask_hash, validate_atlas

STAGE9_DRILLDOWN_PARENTS: Tuple[str, ...] = ("vision", "language")
STAGE9_CHILD_REGIONS: Tuple[str, ...] = (
    "vision_early", "vision_mid", "vision_late",
    "language_early", "language_mid", "language_late",
)

_BAND_RENAME: Dict[str, str] = {"early": "early", "middle": "mid", "late": "late"}


class Stage9PartitionError(RuntimeError):
    """The Stage-9 hierarchical partition failed a hard-verification requirement -- an
    uncovered parent tensor could not be classified by deterministic architectural position, a
    child overlaps a sibling, or the children's union does not exactly equal the parent. Never
    silently proceeds with an incomplete/overlapping partition.
    """


def _classify_uncovered_tensor(name: str) -> str:
    """Deterministic architectural-position rule (never a per-model special case): a name
    containing "embed" is an input-side tensor (word/token/patch embedding) -> "early"; a name
    containing "norm" is an output-side final-normalization tensor -> "late" (by construction,
    an UNCOVERED name never matched the numbered-layer/block regex, so a "norm" match here can
    only be a top-level, not per-layer, normalization tensor). Anything else hard-fails rather
    than guessing a third case.
    """
    lowered = name.lower()
    if "embed" in lowered:
        return "early"
    if "norm" in lowered:
        return "late"
    raise Stage9PartitionError(
        f"Uncovered parent tensor {name!r} could not be classified as input-side ('embed') or "
        f"output-side ('norm') by deterministic architectural position -- refusing to guess. "
        f"Document and resolve this explicitly before any Stage-9 GPU execution."
    )


@dataclass(frozen=True)
class Stage9PartitionAudit:
    parent: str
    child_band_names: Dict[str, str]  # e.g. {"vision_early": "early", ...} -> pre-rename band label
    uncovered_tensors: Tuple[str, ...]
    uncovered_tensor_assignment: Dict[str, str]  # tensor name -> band it was assigned to ("early"/"late")
    union_equals_parent: bool
    children_pairwise_disjoint: bool


def build_stage9_hierarchical_partition(
    param_names: Sequence[str], model_family: str = "qwen2_5_vl",
) -> Tuple[Dict[str, AnatomyRegion], Dict[str, Stage9PartitionAudit]]:
    """Returns (child_regions, audits). `child_regions` has exactly the 6 STAGE9_CHILD_REGIONS
    keys, each an AnatomyRegion (level=2, parent="vision"/"language", already-sorted
    param_names, its own stable mask_hash). `audits` has one Stage9PartitionAudit per drilled
    -down parent ("vision", "language"), recording the uncovered-tensor resolution and the hard
    -verification results -- never silently trusted, always returned for persistence.
    """
    atlas = build_anatomy_atlas(param_names, model_family=model_family)
    validation = validate_atlas(atlas, allow_empty=())

    child_regions: Dict[str, AnatomyRegion] = {}
    audits: Dict[str, Stage9PartitionAudit] = {}

    for parent_name in STAGE9_DRILLDOWN_PARENTS:
        parent = atlas.region(parent_name)
        band_names: Dict[str, Tuple[str, ...]] = {}
        for old_band, new_band in _BAND_RENAME.items():
            source_region = atlas.region(f"{parent_name}_{old_band}")
            band_names[new_band] = tuple(source_region.param_names)

        uncovered = validation.uncovered_by_parent.get(parent_name, ())
        uncovered_assignment: Dict[str, str] = {}
        for tensor_name in uncovered:
            target_band = _classify_uncovered_tensor(tensor_name)
            band_names[target_band] = tuple(sorted(band_names[target_band]) + [tensor_name])
            uncovered_assignment[tensor_name] = target_band

        for new_band, names in band_names.items():
            region_name = f"{parent_name}_{new_band}"
            sorted_names = tuple(sorted(names))
            child_regions[region_name] = AnatomyRegion(
                name=region_name, level=2, parent=parent_name,
                param_names=sorted_names, mask_hash=compute_mask_hash(sorted_names),
            )

        children_of_parent = [child_regions[f"{parent_name}_{b}"] for b in _BAND_RENAME.values()]
        union = set()
        pairwise_disjoint = True
        for i, a in enumerate(children_of_parent):
            union |= set(a.param_names)
            for b in children_of_parent[i + 1:]:
                if set(a.param_names) & set(b.param_names):
                    pairwise_disjoint = False
        union_equals_parent = union == set(parent.param_names)

        audits[parent_name] = Stage9PartitionAudit(
            parent=parent_name,
            child_band_names={f"{parent_name}_{b}": b for b in _BAND_RENAME.values()},
            uncovered_tensors=tuple(sorted(uncovered)),
            uncovered_tensor_assignment=uncovered_assignment,
            union_equals_parent=union_equals_parent,
            children_pairwise_disjoint=pairwise_disjoint,
        )

    return child_regions, audits


def ensure_stage9_partition_valid(child_regions: Dict[str, AnatomyRegion], audits: Dict[str, Stage9PartitionAudit]) -> None:
    """Hard verification (Section 4 of the spec): exactly the 6 frozen child regions; for each
    drilled-down parent, children form an exact partition (union==parent, pairwise disjoint);
    globally, no child includes any multimodal_connector_or_merger parameter.
    """
    if set(child_regions.keys()) != set(STAGE9_CHILD_REGIONS):
        raise Stage9PartitionError(f"Expected exactly {STAGE9_CHILD_REGIONS}, got {sorted(child_regions.keys())}")
    for parent_name, audit in audits.items():
        if not audit.union_equals_parent:
            raise Stage9PartitionError(f"Stage-9 children of {parent_name!r} do not union to exactly the parent's own parameter set.")
        if not audit.children_pairwise_disjoint:
            raise Stage9PartitionError(f"Stage-9 children of {parent_name!r} are not pairwise disjoint.")
    for region_name, region in child_regions.items():
        connector_leak = [n for n in region.param_names if "merger" in n.lower()]
        if connector_leak:
            raise Stage9PartitionError(f"Stage-9 child region {region_name!r} unexpectedly includes connector/merger parameter(s): {connector_leak[:5]}")

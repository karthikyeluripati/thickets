"""Stage 7A: live L1/L2 anatomy inventory -- numeric report over an ALREADY-BUILT AnatomyAtlas
(thicket.anatomy.build_anatomy_atlas, frozen and unmodified) plus the model's real tensors.

anatomy.py itself is purely name-based (never loads weights). This module is the thin numeric
layer on top: given the atlas AND the actual (name -> tensor) mapping (real GPU tensors on the
pod, or synthetic CPU fixtures in tests), it computes per-region tensor/parameter counts,
||theta_a||_2, RMS magnitude, first/last 10 parameter names (for audit -- so a human can see
which real tensors close out or open each region), and layer indices where applicable. Never
guesses these numbers from names alone, and never invents a region beyond what the atlas
already discovered.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence, Tuple

from .anatomy import AnatomyAtlas, AnatomyRegion, validate_atlas


def _region_layer_indices(atlas: AnatomyAtlas, region_name: str) -> Tuple[int, ...]:
    """Layer/block indices represented in this region, where the region is one of the L2
    depth-band regions (vision_early/middle/late, language_early/middle/late) -- derived from
    the SAME regex patterns anatomy.py itself used to discover them, never re-guessed. Returns
    an empty tuple for any region without a well-defined single layer-index axis (full_model,
    vision, multimodal_connector_or_merger, language).
    """
    import re

    if region_name.startswith("vision_"):
        pattern = re.compile(r"^visual\.blocks\.(\d+)\.")
    elif region_name.startswith("language_"):
        from .anatomy import LM_NAMESPACE_CONVENTIONS

        if atlas.lm_namespace_convention is None:
            return ()
        pattern = LM_NAMESPACE_CONVENTIONS[atlas.lm_namespace_convention]
    else:
        return ()

    region = atlas.region(region_name)
    indices = sorted({int(m.group(1)) for name in region.param_names if (m := pattern.match(name))})
    return tuple(indices)


def compute_tensor_norm_stats(region: AnatomyRegion, named_parameters: Mapping[str, Any]) -> Dict[str, float]:
    """||theta_a||_2 and RMS magnitude (||theta_a||_2 / sqrt(d_a)) over exactly this region's
    own parameter set. `named_parameters` maps name -> a tensor-like object supporting
    `.numel()` and either `.detach().float().pow(2).sum().item()` (a real torch.Tensor) or a
    plain float-castable `.pow(2).sum()` result -- in practice always a torch.Tensor, real or
    synthetic-fixture. Missing names hard-fail (never silently skipped -- a region whose
    parameters aren't all present in the handed-in named_parameters means the caller built the
    atlas from a DIFFERENT parameter-name set than it is now computing norms against).
    """
    missing = [n for n in region.param_names if n not in named_parameters]
    if missing:
        raise KeyError(
            f"Region {region.name!r} references {len(missing)} parameter name(s) not present "
            f"in the handed-in named_parameters (first missing: {missing[:5]}). The atlas and "
            f"the tensor mapping must be built from the exact same live model."
        )

    total_elements = 0
    sum_sq = 0.0
    for name in region.param_names:
        p = named_parameters[name]
        total_elements += p.numel()
        sum_sq += p.detach().float().pow(2).sum().item()

    l2_norm = math.sqrt(sum_sq)
    rms = l2_norm / math.sqrt(total_elements) if total_elements > 0 else 0.0
    return {"total_element_count": total_elements, "l2_norm": l2_norm, "rms_magnitude": rms}


def build_region_report(
    atlas: AnatomyAtlas, region_name: str, named_parameters: Mapping[str, Any], *, total_model_param_count: int,
) -> Dict[str, Any]:
    """One region's full Section-1 report row: tensor count, parameter count, percent of total
    model parameters, ||theta_a||_2, RMS magnitude, first/last 10 parameter names, layer
    indices (empty for regions without a single depth axis), and the region's own stable
    mask_hash (already computed by anatomy.py -- reused, not recomputed).
    """
    region = atlas.region(region_name)
    norm_stats = compute_tensor_norm_stats(region, named_parameters)
    names_sorted = list(region.param_names)  # AnatomyRegion.param_names is already sorted (anatomy._make_region)
    param_count = sum(named_parameters[n].numel() for n in names_sorted)

    return {
        "region": region_name,
        "level": region.level,
        "parent": region.parent,
        "tensor_count": region.param_count,
        "parameter_count": param_count,
        "percent_of_total_model_parameters": (100.0 * param_count / total_model_param_count) if total_model_param_count > 0 else 0.0,
        "l2_norm": norm_stats["l2_norm"],
        "rms_magnitude": norm_stats["rms_magnitude"],
        "first_10_parameter_names": names_sorted[:10],
        "last_10_parameter_names": names_sorted[-10:],
        "layer_indices": list(_region_layer_indices(atlas, region_name)),
        "mask_hash": region.mask_hash,
    }


def build_full_anatomy_inventory(
    atlas: AnatomyAtlas, named_parameters: Mapping[str, Any], *, model_family: str, model_revision: str,
) -> Dict[str, Any]:
    """The complete Section-1 deliverable: every atlas region's numeric report, plus the
    validate_atlas() disjointness/coverage evidence (raises AnatomyValidationError -- "hard
    fail unexpected overlap" -- before this function can return a report claiming success).
    """
    report = validate_atlas(atlas)  # raises on empty region / sibling overlap -- never swallowed

    total_model_param_count = sum(p.numel() for p in named_parameters.values())

    regions = {
        name: build_region_report(atlas, name, named_parameters, total_model_param_count=total_model_param_count)
        for name in atlas.regions
    }

    return {
        "model_family": model_family,
        "model_revision": model_revision,
        "total_model_parameter_count": total_model_param_count,
        "lm_namespace_convention": atlas.lm_namespace_convention,
        "lm_layer_indices": list(atlas.lm_layer_indices),
        "vision_block_indices": list(atlas.vision_block_indices),
        "regions": regions,
        "validation": {
            "ok": report.ok,
            "empty_regions": list(report.empty_regions),
            "sibling_overlaps": {f"{a}|{b}": list(names) for (a, b), names in report.sibling_overlaps.items()},
            "uncovered_by_parent": {parent: list(names) for parent, names in report.uncovered_by_parent.items()},
            "uncovered_by_parent_counts": {parent: len(names) for parent, names in report.uncovered_by_parent.items()},
        },
    }

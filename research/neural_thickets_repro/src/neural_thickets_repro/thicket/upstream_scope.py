"""Stage 7A Section 2: what Stage 6's RELEASED upstream perturbation rule actually included,
measured against the LIVE model's real parameter names -- never assumed from the rule's name.

Pinned RandOpt's `WorkerExtension._should_perturb` (external/RandOpt/utils/worker_extn.py) and
this project's own `perturb_cpu.should_perturb` are the SAME rule: `not name.startswith(
("visual.", "model.visual."))`. `multimodal_connector_or_merger` (`visual.merger.*`) is a
subset of that `visual.` prefix, so it is EXCLUDED by the upstream rule exactly like the rest
of the vision encoder -- but this module computes and reports that fact against the real
atlas/parameter set rather than asserting it a priori, per the explicit instruction not to
assume Stage 6 was purely "language" without checking.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Mapping

from ..perturb_cpu import should_perturb
from .anatomy import AnatomyAtlas, AnatomyRegion

L1_REGION_NAMES = ("vision", "multimodal_connector_or_merger", "language")


def _region_upstream_split(region: AnatomyRegion) -> Dict[str, Any]:
    perturbed = [n for n in region.param_names if should_perturb(n)]
    excluded = [n for n in region.param_names if not should_perturb(n)]
    return {"perturbed": perturbed, "excluded": excluded}


def compute_upstream_scope_inventory(atlas: AnatomyAtlas, named_parameters: Mapping[str, Any]) -> Dict[str, Any]:
    """Section 2's per-L1-region table (tensors/params perturbed vs excluded, percent
    perturbed) PLUS the aggregate upstream-perturbed-scope-wide stats (tensor/param count,
    ||theta||_2, RMS) needed by Section 3's sigma-to-relative-L2 mapping, PLUS an explicit,
    measured (not assumed) comparison against the `language` L1 region's own membership.
    """
    full_model = atlas.region("full_model")
    per_region: Dict[str, Any] = {}

    for region_name in ("full_model",) + L1_REGION_NAMES:
        region = atlas.region(region_name)
        split = _region_upstream_split(region)
        perturbed_param_count = sum(named_parameters[n].numel() for n in split["perturbed"])
        excluded_param_count = sum(named_parameters[n].numel() for n in split["excluded"])
        total_param_count = perturbed_param_count + excluded_param_count
        per_region[region_name] = {
            "tensors_perturbed": len(split["perturbed"]),
            "tensors_excluded": len(split["excluded"]),
            "parameters_perturbed": perturbed_param_count,
            "parameters_excluded": excluded_param_count,
            "percent_perturbed": (100.0 * perturbed_param_count / total_param_count) if total_param_count > 0 else 0.0,
        }

    upstream_perturbed_names = [n for n in full_model.param_names if should_perturb(n)]
    sum_sq = sum(named_parameters[n].detach().float().pow(2).sum().item() for n in upstream_perturbed_names)
    upstream_param_count = sum(named_parameters[n].numel() for n in upstream_perturbed_names)
    upstream_l2_norm = math.sqrt(sum_sq)
    upstream_rms = upstream_l2_norm / math.sqrt(upstream_param_count) if upstream_param_count > 0 else 0.0

    language_names = set(atlas.region("language").param_names)
    upstream_names_set = set(upstream_perturbed_names)
    equals_language_region = upstream_names_set == language_names

    return {
        "per_region": per_region,
        "upstream_perturbed_scope": {
            "tensor_count": len(upstream_perturbed_names),
            "parameter_count": upstream_param_count,
            "l2_norm": upstream_l2_norm,
            "rms_magnitude": upstream_rms,
        },
        "upstream_scope_vs_language_region": {
            "equals_language_region": equals_language_region,
            "in_upstream_not_in_language_count": len(upstream_names_set - language_names),
            "in_language_not_in_upstream_count": len(language_names - upstream_names_set),
            "in_upstream_not_in_language_sample": sorted(upstream_names_set - language_names)[:10],
            "in_language_not_in_upstream_sample": sorted(language_names - upstream_names_set)[:10],
        },
    }

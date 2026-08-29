"""TP-aware analog of scaling_common.report_scaling_anatomy_audit -- Stage-11 32B LIVE readiness
(task spec Section 11: "obtain exact live parameter counts: total, vision, connector, language.
Require: vision + connector + language = total, zero overlap, zero unassigned parameters.").

WHY report_scaling_anatomy_audit ITSELF IS NOT REUSED DIRECTLY AT TP>1: that function sums
`t.numel()` over `model.named_parameters()` -- correct only at TP=1 (3B/7B, unchanged, still
calling it directly via their own single-worker RPC path). Under TP>1, a parameter is EITHER
sharded (each rank holds a disjoint LOCAL slice -- `model.named_parameters()` on any one rank
already reflects only that slice, per vllm_shard_mapping.py's own sourcing notes) OR replicated
(every rank holds the FULL tensor). Naively summing one rank's local numel() would undercount
sharded parameters (by a factor of world_size) and get replicated ones right only by accident;
naively summing across ALL ranks would overcount replicated parameters by world_size. Both are
wrong for a "give me the TRUE total parameter count" audit.

THE FIX, and why it needs NO cross-rank all-reduce: vllm_shard_mapping.build_shard_spec_from_
attributes already computes, PER PARAMETER, `ShardSpec.global_shape` -- the parameter's REAL,
UNSHARDED shape, reconstructed from the local shard's shape + the owning module's tp_size/
tp_rank/output_dim/input_dim. This is a property of the parameter's ARCHITECTURE alone, not of
which rank asks or how the tensor happens to be sharded on THIS rank -- so `global_shape.numel()`
summed over a region's parameters gives the correct total on EVERY rank identically, without any
collective communication. Every rank computing the identical number is then usable as a rank-
consensus check (a real mapping bug would make ranks disagree), not merely a convenience.

Coverage/overlap checks (`union_equals_full_model`, `pairwise_disjoint`, `uncovered_by_full_model`)
are already correct at ANY TP size unchanged -- they operate on the atlas's PARAMETER NAME sets
(via `validate_atlas`), never on tensor values or shapes, so this module reuses
`report_scaling_anatomy_audit`'s own atlas-construction/validation path unmodified rather than
reimplementing it, and only replaces the element-count arithmetic.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch


def report_global_anatomy_audit_rpc(worker_self, region_labels: Sequence[str], model_family: str) -> Dict[str, Any]:
    """Worker-side RPC (dispatched identically to every TP rank via collective_rpc). Returns the
    SAME shape as scaling_common.report_scaling_anatomy_audit's return dict (`regions`,
    `total_model_elements`, `union_equals_full_model`, `pairwise_disjoint`,
    `uncovered_by_full_model`), except every `n_elements`/`total_model_elements` value is the
    TRUE global count (see module docstring) rather than this rank's local shard count.
    `l2_norm` is intentionally omitted (would require a real cross-rank all-reduce over actual
    tensor values, not needed by task spec Section 11's element-count requirement) --
    `l2_norm_note` explains this explicitly rather than silently dropping the field.
    """
    from ..scaling_common import MODEL_FAMILY, _atlas_key_for_label
    from .anatomy import AnatomyValidationError, build_anatomy_atlas, validate_atlas
    from .vllm_shard_mapping import build_shard_specs_for_region

    model_family = model_family or MODEL_FAMILY
    model = worker_self.model_runner.model
    named = list(model.named_parameters())
    names = [n for n, _ in named]

    atlas = build_anatomy_atlas(names, model_family=model_family)

    union_equals_full_model = True
    pairwise_disjoint = True
    uncovered_full_model: tuple = ()
    try:
        report = validate_atlas(atlas)
        uncovered_full_model = report.uncovered_by_parent.get("full_model", ())
        union_equals_full_model = len(uncovered_full_model) == 0
    except AnatomyValidationError as exc:
        pairwise_disjoint = "overlap" not in str(exc).lower()
        union_equals_full_model = False

    full_model_atlas_key = _atlas_key_for_label("whole_model")
    full_model_region = atlas.region(full_model_atlas_key)
    full_model_specs = build_shard_specs_for_region(model, full_model_region.param_names)
    total_elements = sum(int(torch.Size(s.global_shape).numel()) for s in full_model_specs.values())

    region_reports: Dict[str, Any] = {}
    for label in region_labels:
        atlas_key = _atlas_key_for_label(label)
        atlas_region = atlas.region(atlas_key)
        specs = build_shard_specs_for_region(model, atlas_region.param_names)
        n_elements = sum(int(torch.Size(s.global_shape).numel()) for s in specs.values())
        region_reports[label] = {
            "region": label, "atlas_key": atlas_key, "n_tensors": len(atlas_region.param_names),
            "n_elements": n_elements, "mask_hash": atlas_region.mask_hash,
            "percentage_of_total_elements": (100.0 * n_elements / total_elements) if total_elements else 0.0,
            "l2_norm_note": "omitted -- global L2 requires a real cross-rank all-reduce over tensor values, not needed for the Section 11 element-count audit",
        }

    return {
        "rank": getattr(worker_self, "rank", None),
        "regions": region_reports, "total_model_elements": total_elements,
        "union_equals_full_model": union_equals_full_model, "pairwise_disjoint": pairwise_disjoint,
        "uncovered_by_full_model": list(uncovered_full_model),
    }


class AnatomyAuditRankConsensusError(RuntimeError):
    """Every TP rank must compute IDENTICAL global element counts/coverage facts (they are a pure
    function of parameter architecture, never of which rank asks) -- any disagreement is a real
    shard-mapping bug, not floating-point noise (these are integer counts and booleans), and is a
    hard fail per this project's established "hard-fail on rank disagreement, never average or
    pick one" posture (see thicket.distributed_v3_solver.verify_solver_rank_consensus).
    """


_CONSENSUS_FIELDS = ("total_model_elements", "union_equals_full_model", "pairwise_disjoint", "uncovered_by_full_model")


def verify_anatomy_audit_rank_consensus(per_rank_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not per_rank_results:
        raise ValueError("verify_anatomy_audit_rank_consensus requires at least one per-rank result.")
    first = per_rank_results[0]
    for field in _CONSENSUS_FIELDS:
        values = [r[field] for r in per_rank_results]
        if any(v != values[0] for v in values):
            raise AnatomyAuditRankConsensusError(f"Rank disagreement on {field!r}: {values}")
    for region_label, first_region in first["regions"].items():
        for r in per_rank_results:
            region = r["regions"].get(region_label)
            if region is None:
                raise AnatomyAuditRankConsensusError(f"Rank {r.get('rank')} is missing region {region_label!r}.")
            if region["n_elements"] != first_region["n_elements"] or region["mask_hash"] != first_region["mask_hash"]:
                raise AnatomyAuditRankConsensusError(
                    f"Rank disagreement on region {region_label!r}: n_elements/mask_hash differ across ranks."
                )
    return {"ok": True, "n_ranks": len(per_rank_results), "consensus_fields": _CONSENSUS_FIELDS}

"""Scale-generic infrastructure shared by every child of the unified Stage-11 scaling
experiment (parent identity `stage11_visual_thicket_scaling_v1`, see
run_stage11_visual_thicket_scaling.py). Nothing here is 7B-specific -- every function is a pure
function of an explicit `scale_label`/`model_family`/`region_labels` argument, so the identical
code drives 3B, 7B, and (once genuinely enabled) 32B/72B.

FROZEN SCIENTIFIC OBJECT (unchanged across every scale/track):

    P(Delta_t | capability t, anatomy a, perturbation radius r, model scale s)

with s in {3B, 7B, 32B, 72B}. Two coordinated tracks share this module:

  TRACK S1 ("whole_model"): whole-model Neural-Thickets-style scaling. region_labels =
    ("whole_model",) -- a single L1 region whose mask is the union of ALL trainable model
    parameters. See WHOLE_MODEL_REGION_LABEL below for why this needs no new anatomy-discovery
    code: thicket.anatomy.build_anatomy_atlas already builds a LEVEL-0 "full_model" region that
    is, by construction, the exact union of every parameter -- "whole_model" is simply Stage 11's
    scientific NAME for that already-built, already-validated region, never a second
    independently-computed mask that could silently drift from the real completeness invariant.

  TRACK S2 ("anatomy"): the existing Stage-8-style coarse anatomy experiment (vision /
    multimodal_connector_or_merger / language). region_labels = STAGE8_REGIONS.

Both tracks reuse, BY IMPORT, the SAME frozen radii/capabilities/D_map/direction-family-size/
batch-size/perturbation-semantics/cache-policy as Stage 8 -- only the model (and, for S1, the
region set) ever changes. See run_stage11_coarse_anatomical_atlas_7b.py's own docstring for the
full derivation of each imported constant (unchanged, reused here by identity).

WHY THE HISTORICAL 3B "GLOBAL" RUN CANNOT ANCHOR TRACK S1
=================================================================================================
run_global_visual_thicket_pilot.py's existing 3B run applies `global_gaussian_upstream`
perturbations that SKIP `visual.*` parameters (see thicket/perturbation.py's own C1 docstring:
"skipping visual.* parameters") -- it was discovered, after the fact, to be LANGUAGE-scoped, not
whole-model. It is explicitly disallowed as a Track-S1 anchor (WHOLE_MODEL_HISTORICAL_
DISQUALIFICATION_NOTE below) and a dedicated 3B whole-model backfill run
(stage11_3b_whole_model_v1) is required instead -- same model revision, same D_map examples, same
six tasks, same frozen radii, 64 directions/radius, v3 BF16 solver, cache-reset v2, prefix
caching false, fixed-base restoration -- using the anatomical_relative_l2 perturbation semantics
(never global_gaussian_upstream) so vision IS included in theta_0's L2 norm and epsilon's support.

MODEL-REVISION-PINNING DISCIPLINE (unchanged from the 7B-only version): resolve_immutable_model_
revision() is the ONLY place a revision is ever accepted from -- an already-pinned 40-hex SHA
passes through as-is, a mutable ref (e.g. "main") is resolved LIVE via huggingface_hub, and any
Hub failure or malformed/missing SHA is a hard stop. No commit hash is ever hand-typed in source.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

from .run_stage8_coarse_anatomical_atlas import (
    STAGE8_CAPABILITIES,
    STAGE8_D_MAP_N,
    STAGE8_GENERATE_BATCH_SIZE,
    STAGE8_N_DIRECTIONS_PER_CELL,
    STAGE8_RADII,
    STAGE8_REGIONS,
    STAGE8_SMOKE_D_MAP_N,
    STAGE8_SMOKE_N_DIRECTIONS,
)
from .thicket.anatomy import AnatomyValidationError, build_anatomy_atlas, validate_atlas
from .thicket.seeds import derive_seed

# =================================================================================================
# Section 4: unified scale configuration -- ScalingModelSpec + registry
# =================================================================================================

MODEL_FAMILY = "qwen2_5_vl"


@dataclass(frozen=True)
class ScalingModelSpec:
    """The complete, scale-specific identity a shared runner needs -- everything ELSE (radii,
    capabilities, direction-family size, D_map, perturbation semantics, cache policy) is a global
    invariant imported from Stage 8, never part of this spec.
    """
    scale_label: str
    model_name: str
    revision_ref: str
    model_family: str = MODEL_FAMILY
    expected_model_family: str = MODEL_FAMILY


SCALING_MODEL_REGISTRY: Dict[str, ScalingModelSpec] = {
    "3B": ScalingModelSpec(scale_label="3B", model_name="Qwen/Qwen2.5-VL-3B-Instruct", revision_ref="main"),
    "7B": ScalingModelSpec(scale_label="7B", model_name="Qwen/Qwen2.5-VL-7B-Instruct", revision_ref="main"),
    "32B": ScalingModelSpec(scale_label="32B", model_name="Qwen/Qwen2.5-VL-32B-Instruct", revision_ref="main"),
    "72B": ScalingModelSpec(scale_label="72B", model_name="Qwen/Qwen2.5-VL-72B-Instruct", revision_ref="main"),
}

# Explicit execution gate: 32B/72B are registered (so the scale-generic machinery below is fully
# exercised/tested against them) but are NEVER runnable yet -- "DO NOT START 32B OR 72B" is
# enforced here, in one place, rather than trusted to every call site individually.
RUNNABLE_SCALES: Tuple[str, ...] = ("3B", "7B")


class ScaleNotYetEnabledError(RuntimeError):
    """A caller asked to actually PLAN/EXECUTE a scale outside RUNNABLE_SCALES. The scale is
    registered (ScalingModelSpec exists, comparability-audit code can address it) but running any
    perturbation sweep against it is out of scope for this milestone.
    """


def ensure_scale_runnable(scale_label: str) -> None:
    if scale_label not in RUNNABLE_SCALES:
        raise ScaleNotYetEnabledError(
            f"Scale {scale_label!r} is registered in SCALING_MODEL_REGISTRY but is NOT in "
            f"RUNNABLE_SCALES {RUNNABLE_SCALES} -- 32B/72B execution is explicitly out of scope "
            f"for this milestone (infrastructure-only)."
        )


def get_scaling_model_spec(scale_label: str) -> ScalingModelSpec:
    if scale_label not in SCALING_MODEL_REGISTRY:
        raise KeyError(f"Unknown scale {scale_label!r}; known scales: {sorted(SCALING_MODEL_REGISTRY)}")
    return SCALING_MODEL_REGISTRY[scale_label]


# =================================================================================================
# Section 5: model family / revision audit -- factual metadata only, no scientific claims
# =================================================================================================

# A STANDING caveat applied uniformly to every scale: this project has no verified live evidence
# that every Qwen2.5-VL checkpoint in the 3B/7B/32B/72B series shares an identical post-training
# recipe merely because the names form a size series. This is deliberately NOT a specific claim
# about any one scale (which would require live model-card verification this module cannot
# perform) -- it is a standing instruction to VERIFY before treating any two scales as a
# perfectly controlled comparison, recorded once here so every comparability report carries it.
COMPARABILITY_CAVEAT_TEMPLATE = (
    "Verify via the live HuggingFace model card whether {scale} shares the same post-training / "
    "instruction-tuning recipe and release date as the other scales compared against it in this "
    "study. Do not assume an identical training pipeline merely because the checkpoint belongs to "
    "the same named model family/size series."
)


def build_model_family_comparability_report(
    specs: Sequence[ScalingModelSpec], hf_model_info_fn=None,
) -> Dict[str, Any]:
    """Factual metadata only (Section 5): model_name/requested_revision_ref/resolved_commit_sha/
    config architecture/parameter count/family metadata, where available. Never makes a
    scientific claim -- `caveats` records what to VERIFY, not an asserted difference. Requires NO
    GPU; `hf_model_info_fn`, if given, is a live-network callable `(model_name) -> dict` (e.g.
    wrapping huggingface_hub.HfApi().model_info) so this stays independently unit-testable with a
    fake; when omitted, every per-scale entry honestly reports `fetched=False` rather than
    fabricating architecture/parameter-count numbers.
    """
    entries: Dict[str, Any] = {}
    for spec in specs:
        entry: Dict[str, Any] = {
            "scale_label": spec.scale_label, "model_name": spec.model_name, "requested_revision_ref": spec.revision_ref,
            "expected_model_family": spec.expected_model_family, "fetched": False,
            "resolved_commit_sha": None, "config_architecture": None, "parameter_count": None,
            "caveats": [COMPARABILITY_CAVEAT_TEMPLATE.format(scale=spec.scale_label)],
        }
        if hf_model_info_fn is not None:
            try:
                info = hf_model_info_fn(spec.model_name)
                entry.update({
                    "fetched": True, "resolved_commit_sha": info.get("sha"),
                    "config_architecture": info.get("config_architecture"), "parameter_count": info.get("parameter_count"),
                })
            except Exception as exc:  # noqa: BLE001 -- a fetch failure degrades to fetched=False, never a fabricated value
                entry["fetch_error"] = str(exc)
        entries[spec.scale_label] = entry
    return {"scales": entries, "note": "Factual metadata only -- see per-scale `caveats`. No scientific claim is made from this audit."}


# =================================================================================================
# Section 1: model resolution -- resolve the exact immutable HF revision, never invent one
# (moved here, unchanged, from run_stage11_coarse_anatomical_atlas_7b.py so both tracks/scales
# share ONE implementation; that module re-exports these names for backward compatibility)
# =================================================================================================


class ModelRevisionResolutionError(RuntimeError):
    """Could not establish a genuine, immutable (full 40-hex-character commit SHA) HuggingFace
    revision for the requested model -- hard stop. Never proceeds with a mutable ref (e.g.
    "main") passed straight through to snapshot_download, and never fabricates a SHA.
    """


_FULL_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _looks_like_full_git_sha(value: str) -> bool:
    return bool(_FULL_GIT_SHA_RE.match(value))


def resolve_immutable_model_revision(model_name: str, revision_ref: str) -> Dict[str, Any]:
    if _looks_like_full_git_sha(revision_ref):
        return {"model_name": model_name, "requested_ref": revision_ref, "resolved_revision": revision_ref, "resolution_method": "already_pinned"}

    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo_id=model_name, revision=revision_ref)
        resolved_sha = getattr(info, "sha", None)
    except Exception as exc:  # noqa: BLE001 -- any Hub/network failure is a hard stop, never silently retried with a guess
        raise ModelRevisionResolutionError(
            f"Could not resolve an immutable revision for {model_name!r} at ref {revision_ref!r}: {exc}"
        ) from exc

    if not resolved_sha or not _looks_like_full_git_sha(resolved_sha):
        raise ModelRevisionResolutionError(
            f"HuggingFace Hub did not return a genuine full commit SHA for {model_name!r} at ref "
            f"{revision_ref!r} (got {resolved_sha!r}) -- refusing to proceed with an unpinned or "
            f"malformed revision."
        )
    return {"model_name": model_name, "requested_ref": revision_ref, "resolved_revision": resolved_sha, "resolution_method": "resolved_via_hf_api"}


# =================================================================================================
# Section 1 (Track S1): "whole_model" is Stage 11's name for Level-0's already-built "full_model"
# =================================================================================================

WHOLE_MODEL_REGION_LABEL = "whole_model"
_WHOLE_MODEL_ATLAS_KEY = "full_model"

WHOLE_MODEL_HISTORICAL_DISQUALIFICATION_NOTE = (
    "The historical 3B run_global_visual_thicket_pilot.py run applies global_gaussian_upstream "
    "perturbations, which explicitly SKIP visual.* parameters (thicket/perturbation.py's own C1 "
    "docstring) -- it is LANGUAGE-scoped, not whole-model, and is NEVER used as a Track-S1 "
    "whole_model anchor at any scale. A dedicated stage11_{scale}_whole_model_v1 backfill run "
    "using anatomical_relative_l2 semantics (vision included) is required instead."
)


def _atlas_key_for_label(label: str) -> str:
    return _WHOLE_MODEL_ATLAS_KEY if label == WHOLE_MODEL_REGION_LABEL else label


# =================================================================================================
# Section 1/2: scale- and track-generic live anatomy audit (tensor/element counts, norms,
# percentages, hashes) + region-param-name reporting -- both are pure functions of the REAL live
# model's named_parameters(), never assumed to transfer numerically across scales.
# =================================================================================================


def report_scaling_anatomy_audit(worker_self, region_labels: Sequence[str], model_family: str = MODEL_FAMILY) -> Dict[str, Any]:
    """Generalizes run_stage11_coarse_anatomical_atlas_7b.report_stage11_anatomy_audit to an
    arbitrary set of region LABELS (e.g. ("whole_model",) or STAGE8_REGIONS) via
    _atlas_key_for_label -- "whole_model" is translated to the atlas's own "full_model" Level-0
    region, never a second independently-computed mask.
    """
    model = worker_self.model_runner.model
    named = list(model.named_parameters())
    names = [n for n, _ in named]
    tensor_by_name = dict(named)

    atlas = build_anatomy_atlas(names, model_family=model_family)

    union_equals_full_model = True
    pairwise_disjoint = True
    uncovered_full_model: Tuple[str, ...] = ()
    try:
        report = validate_atlas(atlas)
        uncovered_full_model = report.uncovered_by_parent.get("full_model", ())
        union_equals_full_model = len(uncovered_full_model) == 0
    except AnatomyValidationError as exc:
        pairwise_disjoint = "overlap" not in str(exc).lower()
        union_equals_full_model = False

    total_elements = int(sum(t.numel() for t in tensor_by_name.values()))

    region_reports: Dict[str, Any] = {}
    for label in region_labels:
        atlas_key = _atlas_key_for_label(label)
        atlas_region = atlas.region(atlas_key)
        region_tensors = [tensor_by_name[n] for n in atlas_region.param_names]
        n_elements = int(sum(t.numel() for t in region_tensors))
        l2_norm_sq = 0.0
        for t in region_tensors:
            l2_norm_sq += float(t.detach().float().pow(2).sum().item())
        region_reports[label] = {
            "region": label, "atlas_key": atlas_key, "n_tensors": len(atlas_region.param_names), "n_elements": n_elements,
            "l2_norm": l2_norm_sq ** 0.5, "mask_hash": atlas_region.mask_hash,
            "percentage_of_total_elements": (100.0 * n_elements / total_elements) if total_elements else 0.0,
        }

    return {
        "regions": region_reports, "total_model_elements": total_elements,
        "union_equals_full_model": union_equals_full_model, "pairwise_disjoint": pairwise_disjoint,
        "uncovered_by_full_model": list(uncovered_full_model),
    }


def compute_anatomy_audit_hash(audit: Dict[str, Any]) -> str:
    canonical = json.dumps({
        region: {"mask_hash": r["mask_hash"], "n_tensors": r["n_tensors"], "n_elements": r["n_elements"]}
        for region, r in sorted(audit["regions"].items())
    }, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def ensure_scaling_anatomy_audit_passes(audit: Dict[str, Any], region_labels: Sequence[str]) -> None:
    if not audit["union_equals_full_model"]:
        raise RuntimeError(f"Scaling live anatomy audit FAILED: uncovered full-model parameters: {audit['uncovered_by_full_model']}")
    if not audit["pairwise_disjoint"]:
        raise RuntimeError("Scaling live anatomy audit FAILED: L1 regions are not pairwise disjoint.")
    missing = set(region_labels) - set(audit["regions"])
    if missing:
        raise RuntimeError(f"Scaling live anatomy audit is missing region(s): {sorted(missing)}")
    empty = [r for r, info in audit["regions"].items() if info["n_tensors"] == 0]
    if empty:
        raise RuntimeError(f"Scaling live anatomy audit found EMPTY region(s): {empty}")


def ensure_whole_model_covers_100_percent(audit: Dict[str, Any]) -> None:
    """Track-S1-specific audit gate (Section 1): whole_model element count == total model
    parameter element count, percentage == 100%, no uncovered parameters. Distinct from (and
    additional to) ensure_scaling_anatomy_audit_passes's generic checks.
    """
    info = audit["regions"].get(WHOLE_MODEL_REGION_LABEL)
    if info is None:
        raise RuntimeError(f"Whole-model audit is missing the {WHOLE_MODEL_REGION_LABEL!r} region entirely.")
    if info["n_elements"] != audit["total_model_elements"]:
        raise RuntimeError(
            f"whole_model element count ({info['n_elements']}) != total model parameter element "
            f"count ({audit['total_model_elements']}) -- whole_model does not actually cover 100% "
            f"of the model."
        )
    if abs(info["percentage_of_total_elements"] - 100.0) > 1e-9:
        raise RuntimeError(f"whole_model percentage_of_total_elements is {info['percentage_of_total_elements']}, expected exactly 100.0.")


def report_region_param_names_for_scaling(worker_self, region_labels: Sequence[str], model_family: str = MODEL_FAMILY) -> Dict[str, Dict[str, Any]]:
    """Scale/track-generic sibling of run_stage7b_anatomical_calibration.report_region_param_names
    -- translates "whole_model" to the atlas's own "full_model" key via _atlas_key_for_label, else
    behaves identically (a read-only report over the real live model's named_parameters()).
    """
    model = worker_self.model_runner.model
    names = [n for n, _ in model.named_parameters()]
    atlas = build_anatomy_atlas(names, model_family=model_family)
    out: Dict[str, Dict[str, Any]] = {}
    for label in region_labels:
        atlas_key = _atlas_key_for_label(label)
        region = atlas.region(atlas_key)
        out[label] = {"param_names": list(region.param_names), "mask_hash": region.mask_hash}
    return out


# =================================================================================================
# Section 7: direction-family design -- independent deterministic namespace per (model_scale,
# region, direction_index). Directions are NEVER geometrically paired across model scales (
# different parameter dimensionality); seed-index correspondence is bookkeeping only.
# =================================================================================================


def build_scaling_direction_seed_bank(base_seed: int, scale_label: str, region_labels: Sequence[str], n_directions: int) -> Dict[str, Tuple[int, ...]]:
    """Namespace ("stage11_direction_family", scale_label, region, i) -- scale_label is an
    explicit namespace component (unlike run_stage11_coarse_anatomical_atlas_7b.py's own
    3-argument ("stage11_direction_family", region, i) namespace, which predates the scaling
    generalization and remains valid because it is ONLY ever used for scale="7B"/track="anatomy").
    Including scale_label here means even a hypothetical future re-use of the SAME base seed and
    region label at a DIFFERENT scale is guaranteed independent by construction, never merely by
    the two numeric values differing coincidentally.
    """
    return {
        region: tuple(derive_seed(base_seed, "stage11_direction_family", scale_label, region, str(i)) for i in range(n_directions))
        for region in region_labels
    }


def compute_direction_seed_bank_hash(seed_bank: Dict[str, Tuple[int, ...]]) -> str:
    canonical = json.dumps({region: list(seeds) for region, seeds in sorted(seed_bank.items())}, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# Re-exported so callers need only import from this one module for every frozen Stage-8-derived
# invariant a scaling child needs, alongside the scale/track-generic machinery above.
__all__ = [
    "MODEL_FAMILY", "ScalingModelSpec", "SCALING_MODEL_REGISTRY", "RUNNABLE_SCALES", "ScaleNotYetEnabledError",
    "ensure_scale_runnable", "get_scaling_model_spec", "COMPARABILITY_CAVEAT_TEMPLATE",
    "build_model_family_comparability_report", "ModelRevisionResolutionError", "resolve_immutable_model_revision",
    "WHOLE_MODEL_REGION_LABEL", "WHOLE_MODEL_HISTORICAL_DISQUALIFICATION_NOTE", "report_scaling_anatomy_audit",
    "compute_anatomy_audit_hash", "ensure_scaling_anatomy_audit_passes", "ensure_whole_model_covers_100_percent",
    "report_region_param_names_for_scaling", "build_scaling_direction_seed_bank", "compute_direction_seed_bank_hash",
    "STAGE8_CAPABILITIES", "STAGE8_D_MAP_N", "STAGE8_GENERATE_BATCH_SIZE", "STAGE8_N_DIRECTIONS_PER_CELL",
    "STAGE8_RADII", "STAGE8_REGIONS", "STAGE8_SMOKE_D_MAP_N", "STAGE8_SMOKE_N_DIRECTIONS",
]

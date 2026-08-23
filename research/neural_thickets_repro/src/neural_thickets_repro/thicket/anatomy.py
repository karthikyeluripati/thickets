"""Deterministic hierarchical model-anatomy registry (spec section B).

Generalizes ..scopes.py's fixed, 3B-specific scope registry (which hard-requires exactly 32
vision blocks and an LM layer count exactly divisible by 3) into an architecture-scale
-independent hierarchy: layer/block COUNTS are always discovered from the actual parameter
names handed in, never hardcoded, so this same code path works unmodified for a 7B/72B ladder
member whose depths differ from the 3B model. Reuses ..scopes.py's already-validated LM
-namespace-convention discovery (`detect_lm_namespace_convention`, `discover_lm_layer_indices`)
rather than reimplementing it, and reuses `..perturb_cpu.DEFAULT_VISUAL_PREFIXES` /
`..scopes.VISUAL_MERGER_PREFIXES` for the vision/connector/language split -- it does NOT import
or depend on scopes.py's fixed PERTURBATION_SCOPES registry or its hardcoded block/layer counts.

LEVEL 0: full_model (every parameter).
LEVEL 1: vision (vision-encoder params, i.e. visual.* minus visual.merger.*),
         multimodal_connector_or_merger (visual.merger.*),
         language (everything NOT visual.*).
         These three partition full_model exactly (validated by validate_atlas).
LEVEL 2: vision_early / vision_middle / vision_late -- a deterministic near-equal-thirds split
         of the vision encoder's own discovered block indices (see `partition_into_thirds`).
         language_early / language_middle / language_late -- the same partition rule applied
         to the LM's own discovered layer indices.
LEVEL 3: attention / mlp -- STRUCTURAL classification only (`classify_attention_or_mlp`), not
         built into the default atlas and not exhaustively swept in Stage 5 (spec section B1).

DEPTH-BAND PARTITION RULE (documented per spec B2): for n contiguous indices 0..n-1, let
base = n // 3 and remainder = n % 3. The first `remainder` bands (in early, middle, late order)
get `base + 1` indices; the rest get `base`. This is the same rule already established (as a
hardcoded special case) by scopes.py's vision_early/middle/late 11/11/10 split for n=32
(32 // 3 = 10, remainder 2 -> early=11, middle=11, late=10) and its lm_early/middle/late
12/12/12-style exact-thirds split for any n divisible by 3 -- generalized here to work for ANY
n >= 3, which a future model's different depth may require. n < 3 hard-fails (cannot form three
non-empty contiguous bands) rather than silently returning empty bands.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..perturb_cpu import DEFAULT_VISUAL_PREFIXES
from ..scopes import LM_NAMESPACE_CONVENTIONS, VISUAL_MERGER_PREFIXES, discover_lm_layer_indices

# Qwen2.5-VL's vision-tower block naming -- stable across the 3B/7B/72B family (all are the
# same architecture family, just deeper/wider); a genuinely different vision-tower naming
# convention would need its own registry entry here, mirroring scopes.LM_NAMESPACE_CONVENTIONS'
# own multi-convention discovery discipline -- not silently assumed to always match.
_VISION_BLOCK_PATTERN = re.compile(r"^visual\.blocks\.(\d+)\.")
VISUAL_PATCH_EMBED_PREFIXES: Tuple[str, ...] = ("visual.patch_embed.",)
VISUAL_ROTARY_POS_EMB_PREFIXES: Tuple[str, ...] = ("visual.rotary_pos_emb.",)

_ATTENTION_NAME_MARKERS: Tuple[str, ...] = (".self_attn.", ".attn.", ".attention.")
_MLP_NAME_MARKERS: Tuple[str, ...] = (".mlp.", ".feed_forward.", ".ffn.")


class AnatomyDiscoveryError(RuntimeError):
    """Layer/block indices could not be unambiguously discovered from the given parameter
    names (none matched, or the matched set is not a complete contiguous range from 0) --
    never guessed or silently partial.
    """


class AnatomyValidationError(RuntimeError):
    """A built AnatomyAtlas violates one of its required invariants (an unexpectedly empty
    region, or sibling regions that overlap) -- never silently accepted.
    """


def _is_visual(name: str) -> bool:
    return name.startswith(DEFAULT_VISUAL_PREFIXES)


def _is_visual_merger(name: str) -> bool:
    return name.startswith(VISUAL_MERGER_PREFIXES)


def _is_visual_encoder(name: str) -> bool:
    return _is_visual(name) and not _is_visual_merger(name)


def compute_mask_hash(param_names: Iterable[str]) -> str:
    """Stable hash of a region's exact parameter-name set -- order-independent (sorted before
    hashing), so two independently-built regions with the same membership always hash
    identically, and any membership change is detected.
    """
    canonical = json.dumps(sorted(param_names))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def discover_contiguous_block_indices(param_names: Sequence[str], pattern: re.Pattern) -> List[int]:
    """Sorted unique indices captured by `pattern`'s single capture group. Hard-fails if
    nothing matches, or if the matched indices are not the complete {0, ..., n-1} range --
    never partitions a partial/gapped block set, and never hardcodes an expected n.
    """
    indices = sorted({int(m.group(1)) for name in param_names if (m := pattern.match(name))})
    if not indices:
        sample = list(param_names)[:10]
        raise AnatomyDiscoveryError(
            f"No parameter names matched pattern {pattern.pattern!r} among "
            f"{len(param_names)} names. First names seen: {sample}."
        )
    if indices != list(range(len(indices))):
        raise AnatomyDiscoveryError(
            f"Indices matched by {pattern.pattern!r} are not a complete contiguous range "
            f"starting at 0: found {indices}."
        )
    return indices


def partition_into_thirds(indices: Sequence[int]) -> Dict[str, List[int]]:
    """Deterministic near-equal contiguous thirds of `indices` (see module docstring for the
    exact rule) -- works for any n = len(indices) >= 3, unlike scopes.partition_layers_into
    _thirds (n must be divisible by 3) and scopes.partition_vision_blocks (n must be exactly
    32). Hard-fails only for n < 3, since three non-empty contiguous bands are impossible.
    """
    ordered = sorted(indices)
    n = len(ordered)
    if n < 3:
        raise AnatomyDiscoveryError(f"Cannot partition {n} indices into three non-empty contiguous bands.")
    base, remainder = divmod(n, 3)
    sizes = [base + 1 if i < remainder else base for i in range(3)]
    early_end = sizes[0]
    middle_end = sizes[0] + sizes[1]
    return {
        "early": ordered[:early_end],
        "middle": ordered[early_end:middle_end],
        "late": ordered[middle_end:],
    }


def classify_attention_or_mlp(name: str) -> Optional[str]:
    """Best-effort STRUCTURAL classification only (spec B1 Level 3: "support structurally, but
    do NOT execute exhaustive experiments yet") -- returns "attention", "mlp", or None (neither
    marker present, e.g. a layernorm/embedding/bias-free projection). Not used to build the
    default AnatomyAtlas; callers doing Level-3 localization call this directly per-name.
    """
    if any(marker in name for marker in _ATTENTION_NAME_MARKERS):
        return "attention"
    if any(marker in name for marker in _MLP_NAME_MARKERS):
        return "mlp"
    return None


@dataclass(frozen=True)
class AnatomyRegion:
    name: str
    level: int
    parent: Optional[str]
    param_names: Tuple[str, ...]
    mask_hash: str

    @property
    def param_count(self) -> int:
        return len(self.param_names)


@dataclass
class AnatomyAtlas:
    model_family: str
    regions: Dict[str, AnatomyRegion]
    lm_namespace_convention: Optional[str]
    lm_layer_indices: Tuple[int, ...]
    vision_block_indices: Tuple[int, ...]

    def region(self, name: str) -> AnatomyRegion:
        return self.regions[name]

    def children(self, parent_name: str) -> List[AnatomyRegion]:
        return [r for r in self.regions.values() if r.parent == parent_name]


def _make_region(name: str, level: int, parent: Optional[str], param_names: Sequence[str]) -> AnatomyRegion:
    names = tuple(sorted(param_names))
    return AnatomyRegion(name=name, level=level, parent=parent, param_names=names, mask_hash=compute_mask_hash(names))


def build_anatomy_atlas(param_names: Sequence[str], model_family: str = "qwen2_5_vl") -> AnatomyAtlas:
    """Builds the full LEVEL 0/1/2 hierarchy from real (or synthetic-fixture) parameter names.
    Never loads weights or requires GPU -- purely a function of the name strings themselves.
    """
    all_names = list(param_names)
    regions: Dict[str, AnatomyRegion] = {}

    regions["full_model"] = _make_region("full_model", level=0, parent=None, param_names=all_names)

    vision_names = [n for n in all_names if _is_visual_encoder(n)]
    connector_names = [n for n in all_names if _is_visual_merger(n)]
    language_names = [n for n in all_names if not _is_visual(n)]
    regions["vision"] = _make_region("vision", level=1, parent="full_model", param_names=vision_names)
    regions["multimodal_connector_or_merger"] = _make_region(
        "multimodal_connector_or_merger", level=1, parent="full_model", param_names=connector_names
    )
    regions["language"] = _make_region("language", level=1, parent="full_model", param_names=language_names)

    vision_block_indices = discover_contiguous_block_indices(vision_names, _VISION_BLOCK_PATTERN)
    vision_thirds = partition_into_thirds(vision_block_indices)
    for band, indices in vision_thirds.items():
        index_set = set(indices)
        band_names = [n for n in vision_names if (m := _VISION_BLOCK_PATTERN.match(n)) and int(m.group(1)) in index_set]
        if band == "early":
            # Mirrors scopes.py's established convention: vision_early additionally owns
            # patch_embed and (if any trainable rotary_pos_emb parameters exist at runtime --
            # typically a non-trainable buffer, so usually a no-op) rotary_pos_emb.
            band_names += [n for n in vision_names if n.startswith(VISUAL_PATCH_EMBED_PREFIXES + VISUAL_ROTARY_POS_EMB_PREFIXES)]
        regions[f"vision_{band}"] = _make_region(f"vision_{band}", level=2, parent="vision", param_names=band_names)

    lm_convention, lm_layer_indices = discover_lm_layer_indices(language_names)
    lm_regex = LM_NAMESPACE_CONVENTIONS[lm_convention]
    lm_thirds = partition_into_thirds(lm_layer_indices)
    for band, indices in lm_thirds.items():
        index_set = set(indices)
        band_names = [n for n in language_names if (m := lm_regex.match(n)) and int(m.group(1)) in index_set]
        regions[f"language_{band}"] = _make_region(f"language_{band}", level=2, parent="language", param_names=band_names)

    return AnatomyAtlas(
        model_family=model_family,
        regions=regions,
        lm_namespace_convention=lm_convention,
        lm_layer_indices=tuple(lm_layer_indices),
        vision_block_indices=tuple(vision_block_indices),
    )


@dataclass
class AnatomyValidationReport:
    empty_regions: Tuple[str, ...]
    sibling_overlaps: Dict[Tuple[str, str], Tuple[str, ...]]
    uncovered_by_parent: Dict[str, Tuple[str, ...]]

    @property
    def ok(self) -> bool:
        return not self.empty_regions and not self.sibling_overlaps


def validate_atlas(atlas: AnatomyAtlas, *, allow_empty: Sequence[str] = ()) -> AnatomyValidationReport:
    """Deterministic (pure function of `atlas`, no RNG) mask validation (spec B4):
      - fails (raises) if any region other than one named in `allow_empty` is empty;
      - reports pairwise overlap between SIBLING regions (same parent) -- expected to be
        empty for a correctly-partitioned atlas; a non-empty overlap is a real bug, so this
        also raises, unlike uncovered-parameter reporting below;
      - reports (does not raise on) parameters present in a parent region but not covered by
        the union of its own children -- expected for e.g. "language" (embeddings/final norm/
        lm_head sit outside any numbered layer), not necessarily a bug.
    """
    empty_regions = tuple(sorted(name for name, region in atlas.regions.items() if region.param_count == 0 and name not in allow_empty))
    if empty_regions:
        raise AnatomyValidationError(f"Unexpectedly empty anatomy region(s): {empty_regions}")

    by_parent: Dict[Optional[str], List[AnatomyRegion]] = {}
    for region in atlas.regions.values():
        by_parent.setdefault(region.parent, []).append(region)

    sibling_overlaps: Dict[Tuple[str, str], Tuple[str, ...]] = {}
    for siblings in by_parent.values():
        for i in range(len(siblings)):
            for j in range(i + 1, len(siblings)):
                a, b = siblings[i], siblings[j]
                overlap = set(a.param_names) & set(b.param_names)
                if overlap:
                    sibling_overlaps[(a.name, b.name)] = tuple(sorted(overlap))
    if sibling_overlaps:
        raise AnatomyValidationError(f"Sibling anatomy regions overlap (should be disjoint): {sibling_overlaps}")

    uncovered_by_parent: Dict[str, Tuple[str, ...]] = {}
    for parent_name, siblings in by_parent.items():
        if parent_name is None:
            continue
        parent = atlas.regions[parent_name]
        children_union = set()
        for child in siblings:
            children_union |= set(child.param_names)
        uncovered = set(parent.param_names) - children_union
        if uncovered:
            uncovered_by_parent[parent_name] = tuple(sorted(uncovered))

    return AnatomyValidationReport(empty_regions=empty_regions, sibling_overlaps=sibling_overlaps, uncovered_by_parent=uncovered_by_parent)

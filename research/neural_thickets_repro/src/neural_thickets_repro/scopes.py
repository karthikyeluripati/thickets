"""Perturbation-scope registry for scoped RandOpt (WACV component-localization milestone).

Pure Python/torch, no GPU/ray/vllm dependency -- CPU-testable against any
named_parameters()-like iterable, real or synthetic. Nothing here is copied from
external/RandOpt (which has no scope concept at all -- see SCOPED_PERTURBATION_DESIGN.md's
WorkerExtension investigation: apply_perturbation/reset_to_base_weights take no
scope/component argument, and _should_perturb only distinguishes "LM" from "everything"
via a single env-var-gated visual/non-visual split). This module is the local, package-side
extension that fills that gap, per the reproduction-integrity rule of never editing
external/RandOpt.

CRITICAL correction baked into this design (see SCOPED_PERTURBATION_DESIGN.md): the HF
checkpoint's flat safetensors keys (model.layers.*) are NOT necessarily what
worker_self.model_runner.model.named_parameters() yields once vLLM loads and wraps the
checkpoint -- prior GPU logs show the loaded runtime module actually nests the LM under
language_model.model.*. Every function here takes actual parameter NAMES as input (real
runtime names on GPU, synthetic fixture names in CPU tests) and *discovers* which naming
convention is present rather than assuming either one.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .perturb_cpu import DEFAULT_VISUAL_PREFIXES, should_perturb

PERTURBATION_SCOPES: Tuple[str, ...] = (
    "full_lm", "vision_encoder", "vision_merger", "lm_early", "lm_middle", "lm_late", "full_vlm",
)

VISUAL_MERGER_PREFIXES: Tuple[str, ...] = ("visual.merger.",)

# Explicit, closed registry of recognized LM-decoder-layer naming conventions. Neither is
# assumed correct a priori -- detect_lm_namespace_convention() below picks whichever one
# actually appears among the real parameter names handed to it.
LM_NAMESPACE_CONVENTIONS: Dict[str, re.Pattern] = {
    "runtime_wrapped": re.compile(r"^language_model\.model\.layers\.(\d+)\."),  # observed on the pod
    "flat_checkpoint": re.compile(r"^model\.layers\.(\d+)\."),  # HF safetensors index form
}


class ScopeSelectionError(RuntimeError):
    """A scope selected zero parameters, or the LM layer namespace could not be resolved
    unambiguously. Never silently falls back to full-model perturbation.
    """


def _is_visual(name: str) -> bool:
    return name.startswith(DEFAULT_VISUAL_PREFIXES)


def _is_visual_merger(name: str) -> bool:
    return name.startswith(VISUAL_MERGER_PREFIXES)


def _is_visual_encoder(name: str) -> bool:
    return _is_visual(name) and not _is_visual_merger(name)


def detect_lm_namespace_convention(param_names: Sequence[str]) -> str:
    """Counts how many of the given names match each recognized LM-layer naming convention
    and returns the one that actually matched. Hard-fails (does not guess) if zero
    conventions match (namespace not recognized at all) or if more than one matches
    (ambiguous -- e.g. a stray parameter name coincidentally matching the "wrong" pattern).
    """
    match_counts = {
        convention: sum(1 for name in param_names if pattern.match(name))
        for convention, pattern in LM_NAMESPACE_CONVENTIONS.items()
    }
    matched = [conv for conv, count in match_counts.items() if count > 0]

    if not matched:
        sample = list(param_names)[:10]
        raise ScopeSelectionError(
            f"No recognized LM-layer namespace found among {len(param_names)} parameter "
            f"names (tried: {list(LM_NAMESPACE_CONVENTIONS)}). First names seen: {sample}. "
            f"Refusing to guess -- add the actual convention to LM_NAMESPACE_CONVENTIONS "
            f"once confirmed, rather than assuming one."
        )
    if len(matched) > 1:
        raise ScopeSelectionError(
            f"Ambiguous LM-layer namespace: more than one recognized convention matched "
            f"({dict((c, match_counts[c]) for c in matched)}). Refusing to guess which is "
            f"authoritative."
        )
    return matched[0]


def discover_lm_layer_indices(param_names: Sequence[str]) -> Tuple[str, List[int]]:
    """Returns (detected_convention, sorted unique layer indices). Hard-fails if the
    discovered indices are not a complete contiguous range starting at 0 (gaps are treated
    as a real problem, not "however many layers we happened to find").
    """
    convention = detect_lm_namespace_convention(param_names)
    pattern = LM_NAMESPACE_CONVENTIONS[convention]
    indices = sorted({int(m.group(1)) for name in param_names if (m := pattern.match(name))})

    expected = list(range(len(indices)))
    if indices != expected:
        raise ScopeSelectionError(
            f"LM layer indices under convention {convention!r} are not a complete "
            f"contiguous range starting at 0: found {indices}. Refusing to partition an "
            f"incomplete/gapped layer set into thirds."
        )
    return convention, indices


def partition_layers_into_thirds(indices: Sequence[int]) -> Dict[str, List[int]]:
    """Contiguous equal thirds by count, in sorted order. Hard-fails if the count isn't
    evenly divisible by 3 -- a deliberate simplification (true for the real model's 36
    layers under either recognized convention), not a silent uneven-split guess.
    """
    n = len(indices)
    if n == 0 or n % 3 != 0:
        raise ScopeSelectionError(
            f"Cannot partition {n} LM layers into three equal contiguous thirds -- only "
            f"counts evenly divisible by 3 are supported (got indices={list(indices)})."
        )
    third = n // 3
    ordered = sorted(indices)
    return {
        "early": ordered[:third],
        "middle": ordered[third:2 * third],
        "late": ordered[2 * third:],
    }


# --- selector factories: (all_param_names) -> Callable[[str], bool] ---


def _full_lm_selector_factory(all_names: Sequence[str]) -> Callable[[str], bool]:
    # Literally "not visual", mirroring upstream _should_perturb's own semantics exactly --
    # never reconstructed from an enumerated union of embed/layers/norm names, which would
    # need to know the LM naming convention even for scopes that don't care about it.
    return lambda name: should_perturb(name)


def _full_vlm_selector_factory(all_names: Sequence[str]) -> Callable[[str], bool]:
    return lambda name: True


def _vision_encoder_selector_factory(all_names: Sequence[str]) -> Callable[[str], bool]:
    return _is_visual_encoder


def _vision_merger_selector_factory(all_names: Sequence[str]) -> Callable[[str], bool]:
    return _is_visual_merger


def _make_lm_third_selector_factory(third: str) -> Callable[[Sequence[str]], Callable[[str], bool]]:
    def _factory(all_names: Sequence[str]) -> Callable[[str], bool]:
        convention, indices = discover_lm_layer_indices(all_names)
        thirds = partition_layers_into_thirds(indices)
        selected_indices = set(thirds[third])
        pattern = LM_NAMESPACE_CONVENTIONS[convention]

        def _selector(name: str) -> bool:
            m = pattern.match(name)
            return m is not None and int(m.group(1)) in selected_indices

        return _selector
    return _factory


_SCOPE_SELECTOR_FACTORIES: Dict[str, Callable[[Sequence[str]], Callable[[str], bool]]] = {
    "full_lm": _full_lm_selector_factory,
    "vision_encoder": _vision_encoder_selector_factory,
    "vision_merger": _vision_merger_selector_factory,
    "lm_early": _make_lm_third_selector_factory("early"),
    "lm_middle": _make_lm_third_selector_factory("middle"),
    "lm_late": _make_lm_third_selector_factory("late"),
    "full_vlm": _full_vlm_selector_factory,
}

# Scope-specific hard exclusion assertions, run after selection -- catches a selector-factory
# bug (or a future namespace convention that doesn't cleanly separate components the way we
# expect) rather than trusting the factory silently.
_SCOPE_EXCLUSION_CHECKS: Dict[str, Callable[[str], bool]] = {
    "vision_encoder": lambda name: not _is_visual_merger(name) and _is_visual(name),
    "vision_merger": lambda name: _is_visual_merger(name),
    "full_lm": lambda name: not _is_visual(name),
    "lm_early": lambda name: not _is_visual(name),
    "lm_middle": lambda name: not _is_visual(name),
    "lm_late": lambda name: not _is_visual(name),
}


@dataclass
class ScopeManifest:
    scope: str
    selected_param_names: List[str]
    selected_param_count: int
    total_element_count: int
    base_l2_norm: float
    representative_names: List[str]
    detected_lm_convention: Optional[str]
    aliases: List[str] = field(default_factory=list)


def build_scope_manifest(scope_name: str, named_parameters, alias_named_parameters=None) -> ScopeManifest:
    """named_parameters: an iterable of (name, tensor) pairs -- the model's real
    named_parameters() on GPU, or a synthetic fixture's on CPU. This is the manifest that
    directly drives noise application and the relative-L2 formula, so it is deduplicated by
    STORAGE (tensor.data_ptr()), not merely by name, guaranteed by this function regardless
    of what the caller passes in. In ordinary use this is close to a no-op: PyTorch's own
    named_parameters() already deduplicates shared/tied parameters by default
    (remove_duplicate=True) -- this function's own dedup is a defensive guarantee for
    whatever it's actually handed, not a reliance on that default.

    alias_named_parameters: OPTIONAL, informational only. If the caller separately obtained
    an untied view (e.g. named_parameters(remove_duplicate=False), where supported), pass it
    here so any name sharing storage with an already-selected tensor, under a *different*
    name, is recorded in the returned manifest's `aliases` for visibility. Never used to
    build the perturbed set itself.
    """
    if scope_name not in _SCOPE_SELECTOR_FACTORIES:
        raise ValueError(f"Unknown perturbation scope {scope_name!r}, expected one of {PERTURBATION_SCOPES}")

    named_parameters = list(named_parameters)
    all_names = [name for name, _ in named_parameters]

    seen_storage = {}  # data_ptr -> first name seen for it
    deduped: List[Tuple[str, "object"]] = []
    for name, p in named_parameters:
        ptr = p.data_ptr()
        if ptr in seen_storage:
            continue
        seen_storage[ptr] = name
        deduped.append((name, p))

    deduped_names = [name for name, _ in deduped]
    detected_convention = None
    if scope_name in ("lm_early", "lm_middle", "lm_late"):
        detected_convention, _ = discover_lm_layer_indices(deduped_names)

    selector = _SCOPE_SELECTOR_FACTORIES[scope_name](deduped_names)
    selected = [(name, p) for name, p in deduped if selector(name)]

    if not selected:
        raise ScopeSelectionError(
            f"Scope {scope_name!r} selected zero parameters out of {len(deduped)} "
            f"(deduplicated) candidates -- refusing to silently fall back to full-model "
            f"perturbation. First names available: {deduped_names[:10]}"
        )

    exclusion_check = _SCOPE_EXCLUSION_CHECKS.get(scope_name)
    if exclusion_check is not None:
        violations = [name for name, _ in selected if not exclusion_check(name)]
        if violations:
            raise ScopeSelectionError(
                f"Scope {scope_name!r} selected parameter(s) that violate its own exclusion "
                f"rule: {violations[:10]}. Refusing to proceed with an incorrectly-scoped "
                f"selection."
            )

    selected_names_sorted = sorted(name for name, _ in selected)
    total_elements = sum(p.numel() for _, p in selected)
    base_l2_norm = sum(p.detach().float().pow(2).sum().item() for _, p in selected) ** 0.5

    aliases: List[str] = []
    if alias_named_parameters is not None:
        selected_ptrs = {p.data_ptr() for _, p in selected}
        selected_name_set = set(selected_names_sorted)
        for name, p in alias_named_parameters:
            if p.data_ptr() in selected_ptrs and name not in selected_name_set:
                aliases.append(name)

    return ScopeManifest(
        scope=scope_name,
        selected_param_names=selected_names_sorted,
        selected_param_count=len(selected),
        total_element_count=total_elements,
        base_l2_norm=base_l2_norm,
        representative_names=selected_names_sorted[:5],
        detected_lm_convention=detected_convention,
        aliases=sorted(set(aliases)),
    )


def compute_relative_l2_sigma(base_l2_norm: float, param_count: int, r: float) -> float:
    """sigma_m = r * ||theta_m||_2 / sqrt(d_m), so E[||Delta_m||_2^2] = d_m * sigma_m^2 =
    r^2 * ||theta_m||_2^2 -- i.e. E[||Delta_m||_2] ~= r * ||theta_m||_2 in expectation. This
    holds regardless of the per-tensor-reseed noise convention used to actually sample
    Delta_m (see scoped_perturbation.py) -- every selected coordinate still has variance
    sigma_m^2 individually; this derivation makes no claim that the resulting joint
    perturbation is strictly isotropic across the concatenated scope.
    """
    if param_count <= 0:
        raise ValueError(f"param_count must be positive, got {param_count}")
    return r * base_l2_norm / (param_count ** 0.5)

"""Two scientifically distinct perturbation modes (spec section C) plus perturbation identity
(spec section D) and shared-population generation across capabilities (spec section D1).

--------------------------------------------------------------------------------------------
C1. global_gaussian_upstream -- the historical, upstream-compatible mode. This is a THIN,
non-modifying wrapper around ..perturb_cpu.perturb/restore (theta' = theta + sigma * epsilon,
epsilon ~ N(0, I) per-tensor-reseed by `seed`, skipping visual.* parameters) -- the exact
existing Gate-1/2 reproduction path. Nothing here reimplements or rewrites that math; it is
reused unchanged so this mode stays comparable to Neural Thickets / RandOpt.

C2. anatomical_relative_l2 -- a NEW anatomy-controlled mode. For anatomical region a:

    r = ||epsilon_a||_2 / ||theta_a||_2

Unlike ..scoped_perturbation.py's existing "relative_l2" scale mode (which derives a single
scalar sigma from E[||epsilon||_2] = r * ||theta||_2 in EXPECTATION, via
scopes.compute_relative_l2_sigma, and applies it without checking the realized sampled norm),
this mode samples the noise FIRST, measures its actual realized L2 norm, then rescales by an
exact scalar factor so the applied perturbation's L2 norm EXACTLY equals r * ||theta_a||_2 (up
to floating-point rounding) -- not merely in expectation. This is a deliberate, documented
scientific distinction (not a silent behavior change to the existing scoped-perturbation code
path, which is untouched): spec section C3 requires numerically verifying the realized ratio
matches the requested r to tolerance, which an expectation-only scalar sigma cannot guarantee
for any single finite-dimensional sample. Every parameter outside region a is never even read.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import torch

from ..perturb_cpu import DEFAULT_VISUAL_PREFIXES, _generate_noise, perturb, restore, should_perturb

PERTURBATION_MODES: Tuple[str, ...] = ("global_gaussian_upstream", "anatomical_relative_l2")


class UnknownPerturbationModeError(ValueError):
    pass


# --- C1: global_gaussian_upstream -- re-exported, not reimplemented -----------------------

global_gaussian_upstream_perturb = perturb
global_gaussian_upstream_restore = restore
global_gaussian_upstream_should_perturb = should_perturb


@dataclass(frozen=True)
class GlobalGaussianRecord:
    mode: str = field(default="global_gaussian_upstream", init=False)
    seed: int = 0
    sigma: float = 0.0
    visual_prefixes: Tuple[str, ...] = DEFAULT_VISUAL_PREFIXES


def apply_global_gaussian_upstream(model: torch.nn.Module, seed: int, sigma: float, visual_prefixes: Sequence[str] = DEFAULT_VISUAL_PREFIXES) -> GlobalGaussianRecord:
    perturb(model, seed=seed, sigma=sigma, visual_prefixes=tuple(visual_prefixes))
    return GlobalGaussianRecord(seed=seed, sigma=sigma, visual_prefixes=tuple(visual_prefixes))


def undo_global_gaussian_upstream(model: torch.nn.Module, record: GlobalGaussianRecord) -> None:
    restore(model, seed=record.seed, sigma=record.sigma, visual_prefixes=record.visual_prefixes)


# --- C2: anatomical_relative_l2 -------------------------------------------------------------


@dataclass(frozen=True)
class AnatomicalRelativeL2Record:
    mode: str = field(default="anatomical_relative_l2", init=False)
    region: str = ""
    seed: int = 0
    requested_r: float = 0.0
    theta_l2_norm: float = 0.0
    raw_noise_l2_norm: float = 0.0
    scale: float = 0.0
    realized_epsilon_l2_norm: float = 0.0
    region_param_names: Tuple[str, ...] = ()

    @property
    def param_count(self) -> int:
        return len(self.region_param_names)


class DegenerateRegionError(RuntimeError):
    """The requested region is empty, or its sampled noise had zero norm (impossible to
    rescale to a nonzero target ratio) -- refuses to silently proceed.
    """


@torch.no_grad()
def apply_anatomical_relative_l2(
    model: torch.nn.Module, region: str, region_param_names: Sequence[str], seed: int, r: float
) -> AnatomicalRelativeL2Record:
    """Sampling procedure (spec C2):
      1. sample independent Gaussian noise over only `region_param_names` (per-tensor reseed,
         the same convention as perturb_cpu._generate_noise / scoped_perturbation.py -- never
         an independent continuous stream across the concatenated region);
      2. compute the combined sampled noise's L2 norm across the whole region;
      3. rescale by a single scalar so the applied perturbation's L2 norm EXACTLY equals
         r * ||theta_a||_2;
      4. every parameter outside the region is never touched (not even read).
    """
    region_param_names = tuple(sorted(set(region_param_names)))
    if not region_param_names:
        raise DegenerateRegionError(f"Region {region!r} has zero parameters -- refusing to perturb an empty region.")

    named = dict(model.named_parameters())
    missing = [n for n in region_param_names if n not in named]
    if missing:
        raise DegenerateRegionError(f"Region {region!r} references parameter name(s) not found on the model: {missing[:10]}")

    theta_sq_sum = 0.0
    noises: Dict[str, torch.Tensor] = {}
    noise_sq_sum = 0.0
    for name in region_param_names:
        p = named[name]
        theta_sq_sum += p.detach().float().pow(2).sum().item()
        noise = _generate_noise(p, seed)
        noises[name] = noise
        noise_sq_sum += noise.detach().float().pow(2).sum().item()

    theta_l2_norm = theta_sq_sum ** 0.5
    raw_noise_l2_norm = noise_sq_sum ** 0.5
    if raw_noise_l2_norm == 0.0:
        raise DegenerateRegionError(f"Sampled noise for region {region!r} has zero norm -- cannot rescale to a nonzero target ratio.")

    scale = (r * theta_l2_norm) / raw_noise_l2_norm

    realized_sq_sum = 0.0
    for name in region_param_names:
        p = named[name]
        delta = scale * noises[name]
        p.add_(delta.to(dtype=p.dtype))
        realized_sq_sum += delta.detach().float().pow(2).sum().item()
    realized_epsilon_l2_norm = realized_sq_sum ** 0.5

    return AnatomicalRelativeL2Record(
        region=region, seed=seed, requested_r=r, theta_l2_norm=theta_l2_norm, raw_noise_l2_norm=raw_noise_l2_norm,
        scale=scale, realized_epsilon_l2_norm=realized_epsilon_l2_norm, region_param_names=region_param_names,
    )


@torch.no_grad()
def undo_anatomical_relative_l2(model: torch.nn.Module, record: AnatomicalRelativeL2Record) -> None:
    """Regenerates the identical seeded noise (never a stored weight copy) and subtracts the
    SAME scale recorded at apply-time -- must be called with the exact record returned by the
    matching apply_anatomical_relative_l2() call.
    """
    named = dict(model.named_parameters())
    for name in record.region_param_names:
        p = named[name]
        noise = _generate_noise(p, record.seed)
        p.sub_((record.scale * noise).to(dtype=p.dtype))


# --- D: perturbation identity ----------------------------------------------------------------


def compute_perturbation_id(
    seed: int, perturbation_mode: str, anatomy_region: Optional[str], radius: Optional[float], sigma: Optional[float],
    model_family: str, model_scale: str, model_revision: str, parameter_mask_hash: str,
) -> str:
    """Deterministic id: sha256 of the canonical (sorted-key) JSON of every manifest field,
    INCLUDING seed -- the same manifest fields with a different seed always yields a different
    id; identical fields (including seed) always yield the identical id. Truncated to 24 hex
    chars (96 bits) -- ample collision resistance for this project's population sizes, kept
    short for readability in filenames/logs.
    """
    payload = {
        "seed": seed, "perturbation_mode": perturbation_mode, "anatomy_region": anatomy_region,
        "radius": radius, "sigma": sigma, "model_family": model_family, "model_scale": model_scale,
        "model_revision": model_revision, "parameter_mask_hash": parameter_mask_hash,
    }
    canonical = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True)
class PerturbationManifest:
    seed: int
    perturbation_mode: str
    model_family: str
    model_scale: str
    model_revision: str
    parameter_mask_hash: str
    anatomy_region: Optional[str] = None
    radius: Optional[float] = None
    sigma: Optional[float] = None
    perturbation_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.perturbation_mode not in PERTURBATION_MODES:
            raise UnknownPerturbationModeError(f"Unknown perturbation_mode {self.perturbation_mode!r}, expected one of {PERTURBATION_MODES}")
        pid = compute_perturbation_id(
            self.seed, self.perturbation_mode, self.anatomy_region, self.radius, self.sigma,
            self.model_family, self.model_scale, self.model_revision, self.parameter_mask_hash,
        )
        object.__setattr__(self, "perturbation_id", pid)

    def to_dict(self) -> Dict:
        return {
            "perturbation_id": self.perturbation_id, "seed": self.seed, "perturbation_mode": self.perturbation_mode,
            "anatomy_region": self.anatomy_region, "radius": self.radius, "sigma": self.sigma,
            "model_family": self.model_family, "model_scale": self.model_scale, "model_revision": self.model_revision,
            "parameter_mask_hash": self.parameter_mask_hash,
        }


# --- D1: shared population across capabilities ------------------------------------------------


def generate_perturbation_population(
    *, mode: str, n: int, base_seed: int, model_family: str, model_scale: str, model_revision: str,
    parameter_mask_hash: str, anatomy_region: Optional[str] = None, radius: Optional[float] = None, sigma: Optional[float] = None,
) -> Tuple[PerturbationManifest, ...]:
    """Builds ONE population of `n` PerturbationManifests for a single (mode, region,
    radius-or-sigma) cell, with seeds derived purely as a function of this cell's own
    identifying fields (via ..thicket.seeds.derive_seed) -- calling this twice with identical
    arguments always returns the IDENTICAL population (same seeds, same perturbation_ids).

    This is what makes perturbation i align across capabilities (spec D1): a capability-
    evaluation driver that calls this function once per (mode, region, radius) cell and then
    evaluates every capability against each manifest in the returned tuple, in order, gets the
    SAME perturbation identity at index i regardless of which capability is being scored --
    never an independently-resampled population per task.
    """
    from .seeds import derive_seed

    manifests = []
    for i in range(n):
        seed = derive_seed(base_seed, "perturbation_population", mode, str(anatomy_region), str(radius), str(sigma), str(i))
        manifests.append(
            PerturbationManifest(
                seed=seed, perturbation_mode=mode, anatomy_region=anatomy_region, radius=radius, sigma=sigma,
                model_family=model_family, model_scale=model_scale, model_revision=model_revision,
                parameter_mask_hash=parameter_mask_hash,
            )
        )
    return tuple(manifests)

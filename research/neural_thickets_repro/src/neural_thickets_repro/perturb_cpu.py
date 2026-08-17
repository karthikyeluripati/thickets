"""Perturbation math, reimplemented from our own understanding of REPRO_SPEC.md's
"Perturbation formula" / "Perturbation scope (exact)" rows -- not copied from the upstream
RandOpt repo, which has no declared license (see external/EXTERNAL_COMMIT.txt) and is only
read, described, and invoked externally, never transcribed.

SCAFFOLD / UNIT-TEST ONLY: exercised in tests/test_perturb_cpu.py against a synthetic dummy
nn.Module (tests/conftest.py), never against the real Qwen2.5-VL-3B-Instruct checkpoint.
Passing tests here show this code is internally consistent with our reconstructed
specification; they are not proof of equivalence to the upstream VLM implementation, which
remains entirely unexercised until Gate 1 runs on real GPU hardware.
"""
from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn

# Mirrors REPRO_SPEC.md "Perturbation scope (exact)". Do not reinterpret or generalize this
# prefix rule -- it is the exact scope rule described there, cross-checked against the real
# checkpoint's actual tensor name prefixes.
DEFAULT_VISUAL_PREFIXES: Tuple[str, ...] = ("visual.", "model.visual.")


def should_perturb(name: str, visual_prefixes: Iterable[str] = DEFAULT_VISUAL_PREFIXES) -> bool:
    """True unless `name` starts with one of the vision-encoder prefixes."""
    return not name.startswith(tuple(visual_prefixes))


def _generate_noise(param: torch.Tensor, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=param.device)
    gen.manual_seed(int(seed))
    return torch.randn(param.shape, dtype=param.dtype, device=param.device, generator=gen)


@torch.no_grad()
def perturb(
    model: nn.Module,
    seed: int,
    sigma: float,
    visual_prefixes: Iterable[str] = DEFAULT_VISUAL_PREFIXES,
) -> None:
    """In-place theta' = theta + sigma * epsilon, epsilon ~ N(0, I) seeded by `seed`,
    skipping parameters matched by should_perturb().
    """
    for name, p in model.named_parameters():
        if not should_perturb(name, visual_prefixes):
            continue
        noise = _generate_noise(p, seed)
        p.add_(sigma * noise)


@torch.no_grad()
def restore(
    model: nn.Module,
    seed: int,
    sigma: float,
    visual_prefixes: Iterable[str] = DEFAULT_VISUAL_PREFIXES,
) -> None:
    """Undo perturb() by regenerating the identical seeded noise and subtracting it -- not
    by restoring from a stored weight copy. Must be called with the same (seed, sigma) used
    in the matching perturb() call, or the result will not be the original weights.
    """
    for name, p in model.named_parameters():
        if not should_perturb(name, visual_prefixes):
            continue
        noise = _generate_noise(p, seed)
        p.sub_(sigma * noise)

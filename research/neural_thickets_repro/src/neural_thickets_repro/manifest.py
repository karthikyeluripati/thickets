"""Parameter manifest generation: introspects a model's named_parameters() and records,
per tensor, whether the scope rule in REPRO_SPEC.md would perturb it.

Works against any nn.Module. tests/test_manifest.py exercises it on a synthetic dummy
module (SCAFFOLD validation). Running it against the real Qwen2.5-VL-3B-Instruct checkpoint
is a Gate 1 setup step -- see REPRO_SPEC.md's "Perturbation scope (exact)" row, which already
cross-checked should_perturb()'s prefix rule against the real checkpoint's actual tensor
names fetched from the HF hub (metadata only, no weights downloaded).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Dict

import torch.nn as nn

from .perturb_cpu import DEFAULT_VISUAL_PREFIXES, should_perturb


def build_parameter_manifest(
    model: nn.Module, visual_prefixes: Iterable[str] = DEFAULT_VISUAL_PREFIXES
) -> List[Dict]:
    manifest = []
    for name, p in model.named_parameters():
        perturbed = should_perturb(name, visual_prefixes)
        module = name.rsplit(".", 1)[0] if "." in name else name
        manifest.append(
            {
                "name": name,
                "module": module,
                "shape": list(p.shape),
                "dtype": str(p.dtype),
                "num_params": p.numel(),
                "perturbed": perturbed,
            }
        )
    return manifest


def write_manifest(
    model: nn.Module,
    path: "str | Path",
    visual_prefixes: Iterable[str] = DEFAULT_VISUAL_PREFIXES,
) -> List[Dict]:
    manifest = build_parameter_manifest(model, visual_prefixes)
    Path(path).write_text(json.dumps(manifest, indent=2))
    return manifest


def summarize(manifest: List[Dict]) -> Dict:
    perturbed = [m for m in manifest if m["perturbed"]]
    frozen = [m for m in manifest if not m["perturbed"]]
    return {
        "total_tensors": len(manifest),
        "perturbed_tensors": len(perturbed),
        "frozen_tensors": len(frozen),
        "perturbed_params": sum(m["num_params"] for m in perturbed),
        "frozen_params": sum(m["num_params"] for m in frozen),
    }

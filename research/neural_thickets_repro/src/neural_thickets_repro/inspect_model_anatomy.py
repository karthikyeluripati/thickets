"""Model-anatomy parameter-inventory utility (spec section B3). Reports, per anatomical region
(see .thicket.anatomy): region name, module/name patterns (represented by the region's own
discovered param-name set), layer indices, parameter (tensor) count, element (numel) count,
percentage of total elements, dtype(s) present, and -- when real weights are available (any
live nn.Module, not merely a name list) -- ||theta_a||_2. Output is machine-readable JSON.

Two usage modes:
  1. inspect_model_anatomy(named_parameters, ...) -- the pure, CPU-testable function, works
     against ANY (name, tensor) iterable, real or synthetic. Unit tests exercise this against
     a synthetic dummy nn.Module (tests/conftest.py's existing fixtures) and never download a
     real model, matching this project's established convention (perturb_cpu.py, manifest.py).
  2. main() -- a thin CLI that lazily imports torch/transformers ONLY inside the function body
     (never at module scope, so `import neural_thickets_repro.inspect_model_anatomy` stays
     GPU/heavy-dependency-free) to load a real checkpoint's named_parameters() on the pod and
     write the JSON report to disk. Actually loading Qwen2.5-VL-7B/72B requires real GPU/disk
     resources this milestone does not execute -- see VISUAL_THICKET_EXPERIMENT_SPEC.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from .thicket.anatomy import build_anatomy_atlas, validate_atlas


def inspect_model_anatomy(
    named_parameters: Iterable[Tuple[str, Any]], model_name: str, model_revision: Optional[str] = None, model_scale: Optional[str] = None,
) -> Dict[str, Any]:
    """`named_parameters`: an iterable of (name, tensor-like) pairs, where each tensor-like
    object exposes `.numel()`, `.dtype`, and (optionally) `.detach().float().pow(2).sum()
    .item()` for the L2-norm contribution -- real torch.nn.Parameter objects satisfy all of
    these; a plain namedtuple/object stub used by a future non-torch caller need only satisfy
    the ones actually exercised.
    """
    named = list(named_parameters)
    param_names = [name for name, _ in named]
    tensor_by_name = dict(named)

    atlas = build_anatomy_atlas(param_names, model_family=model_name)
    validation = validate_atlas(atlas)

    total_elements = sum(int(t.numel()) for t in tensor_by_name.values())

    regions_report = {}
    for name, region in atlas.regions.items():
        tensors = [tensor_by_name[n] for n in region.param_names]
        element_count = sum(int(t.numel()) for t in tensors)
        dtypes = sorted({str(t.dtype) for t in tensors})
        l2_norm_sq = sum(t.detach().float().pow(2).sum().item() for t in tensors)
        regions_report[name] = {
            "level": region.level,
            "parent": region.parent,
            "tensor_count": region.param_count,
            "element_count": element_count,
            "percentage_of_total_elements": (100.0 * element_count / total_elements) if total_elements else 0.0,
            "dtypes": dtypes,
            "l2_norm": l2_norm_sq ** 0.5,
            "mask_hash": region.mask_hash,
            "representative_names": list(region.param_names[:5]),
        }

    return {
        "model_name": model_name,
        "model_revision": model_revision,
        "model_scale": model_scale,
        "lm_namespace_convention": atlas.lm_namespace_convention,
        "lm_layer_indices": list(atlas.lm_layer_indices),
        "vision_block_indices": list(atlas.vision_block_indices),
        "total_tensor_count": len(param_names),
        "total_element_count": total_elements,
        "regions": regions_report,
        "validation": {
            "ok": validation.ok,
            "sibling_overlaps": {f"{a}|{b}": list(names) for (a, b), names in validation.sibling_overlaps.items()},
            "uncovered_by_parent": {k: list(v) for k, v in validation.uncovered_by_parent.items()},
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True, help="HF repo id, e.g. Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--model-scale", default=None, help="e.g. '3B', '7B', '72B' -- free-form label, not parsed")
    parser.add_argument("--output", default=None, help="path to write the JSON report; defaults to stdout")
    args = parser.parse_args(argv)

    import torch  # lazy: keeps `import neural_thickets_repro.inspect_model_anatomy` GPU-free
    from transformers import AutoModel

    model = AutoModel.from_pretrained(args.model_name, revision=args.revision, torch_dtype=torch.bfloat16)
    report = inspect_model_anatomy(model.named_parameters(), model_name=args.model_name, model_revision=args.revision, model_scale=args.model_scale)

    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(text)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

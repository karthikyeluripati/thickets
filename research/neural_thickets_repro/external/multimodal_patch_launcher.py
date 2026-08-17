#!/usr/bin/env python3
"""OPTIONAL launcher that monkeypatches external/RandOpt's launch_engines() to pass
multimodal=True for the GQA run, WITHOUT ever editing the external clone's files on disk.

Background: external/RandOpt/randopt.py's main() calls
    launch_engines(args.num_engines, base_model_path, precision=args.precision, tensor_parallel_size=args.tp)
without a `multimodal` argument, even though core/engine.py:launch_engines() accepts one
(multimodal=True sets vLLM's limit_mm_per_prompt={"image": 1}). Whether this omission is a
real bug for the GQA/VLM path is UNRESOLVED from reading alone (see REPRO_SPEC.md) -- it can
only be settled by testing on GPU hardware.

Try eval_base.py / run_randopt.py WITHOUT this launcher first (they invoke randopt.py
unmodified). Only reach for this launcher if that run fails, hangs, or clearly mishandles
image inputs. Whichever way it goes, record what you observed in REPRO_SPEC.md and
src/neural_thickets_repro/vlm_adapter.py's MULTIMODAL_FIX_NOTES.

Usage: same CLI args as external/RandOpt/randopt.py, e.g.
    python external/multimodal_patch_launcher.py --dataset gqa --model_name Qwen/Qwen2.5-VL-3B-Instruct [...]
"""
import sys
from pathlib import Path

EXTERNAL_RANDOPT_DIR = Path(__file__).resolve().parent / "RandOpt"
if not EXTERNAL_RANDOPT_DIR.exists():
    print(f"{EXTERNAL_RANDOPT_DIR} not found -- run external/setup_external_repo.py first.", file=sys.stderr)
    raise SystemExit(1)

sys.path.insert(0, str(EXTERNAL_RANDOPT_DIR))

import core.engine as engine_module  # type: ignore  # noqa: E402

_original_launch_engines = engine_module.launch_engines


def _patched_launch_engines(
    num_engines, model_name, precision="bfloat16", batch_size=25,
    tensor_parallel_size=1, enable_prefix_caching=False,
    gpu_memory_utilization=0.75, multimodal=False,
):
    # Forces multimodal=True regardless of what the caller passed -- this launcher exists
    # solely to test that one change; see module docstring.
    return _original_launch_engines(
        num_engines, model_name, precision=precision, batch_size=batch_size,
        tensor_parallel_size=tensor_parallel_size, enable_prefix_caching=enable_prefix_caching,
        gpu_memory_utilization=gpu_memory_utilization, multimodal=True,
    )


engine_module.launch_engines = _patched_launch_engines

# core/__init__.py does `from .engine import launch_engines`, so randopt.py's
# `from core import launch_engines` binds a reference to the ORIGINAL function unless the
# patched name is also installed on the `core` package itself.
import core  # type: ignore  # noqa: E402

core.launch_engines = _patched_launch_engines

import randopt  # type: ignore  # noqa: E402

if __name__ == "__main__":
    parsed_args = randopt.parse_args()
    randopt.main(parsed_args)

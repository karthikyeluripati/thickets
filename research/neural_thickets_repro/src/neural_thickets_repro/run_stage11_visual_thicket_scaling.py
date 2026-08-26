"""SCALING LAWS OF VISUAL NEURAL THICKETS -- unified parent entry point.

Parent experiment identity: `stage11_visual_thicket_scaling_v1`. Frozen scientific object:

    P(Delta_t | capability t, anatomy a, perturbation radius r, model scale s),  s in {3B,7B,32B,72B}

with two coordinated child tracks:
  S1 "whole_model" -- run_stage11_whole_model_scaling.py (NEW this milestone)
  S2 "anatomy"      -- run_stage11_coarse_anatomical_atlas_7b.py for scale=7B (the EXISTING,
                       already-tested 47-test implementation, untouched -- "the existing 7B
                       implementation may be refactored into the 7b_anatomy child" is satisfied
                       here by making it REACHABLE as a child of this dispatcher, rather than by
                       rewriting its internals and risking its already-validated test suite).
                       scale=3B REUSES the authoritative stage8_coarse_anatomical_atlas_3b_v2_
                       batched10 run (never rerun -- see Section 16 of the task spec).

Child run identities:
    stage11_3b_whole_model_v1   (this dispatcher -> run_stage11_whole_model_scaling, --scale 3B)
    stage11_3b_anatomy          (NOT rerun -- reuses stage8_coarse_anatomical_atlas_3b_v2_batched10)
    stage11_7b_whole_model_v1   (this dispatcher -> run_stage11_whole_model_scaling, --scale 7B)
    stage11_7b_anatomy_v1       (this dispatcher -> run_stage11_coarse_anatomical_atlas_7b, unchanged)
    stage11_32b_whole_model_v1  (registered, NOT runnable yet -- ScaleNotYetEnabledError)
    stage11_32b_anatomy_v1      (registered, NOT runnable yet -- ScaleNotYetEnabledError)
    stage11_72b_whole_model_v1  (registered, NOT runnable yet -- ScaleNotYetEnabledError)
    stage11_72b_anatomy_v1      (registered, NOT runnable yet -- ScaleNotYetEnabledError)

DO NOT DO YET (unchanged from the single-scale Stage-11 spec, plus the two new prohibitions this
milestone adds): 7B depth Stage 9, 72B/32B EXECUTION, heads, attention-vs-MLP, parameter-space
Stage 10B, post-training, routing, distillation, D_confirm, atlas-guided search. 32B/72B are
registered (ScalingModelSpec exists, comparability-audit code can address them) but
ensure_scale_runnable() hard-blocks any attempt to actually plan/execute them.

This module makes NO GPU call, NO Hub call, and starts NO run itself beyond forwarding argv to
the appropriate already-gated child entry point -- it is pure dispatch + the (also GPU-free)
--report-model-family-comparability path (Section 5).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from .scaling_common import (
    SCALING_MODEL_REGISTRY,
    ScaleNotYetEnabledError,
    build_model_family_comparability_report,
    ensure_scale_runnable,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PARENT_RUN_SIGNATURE = "stage11_visual_thicket_scaling_v1"
TRACKS = ("whole_model", "anatomy")

STAGE8_3B_ANATOMY_ANCHOR = "stage8_coarse_anatomical_atlas_3b_v2_batched10"
REUSE_NOT_RERUN_NOTE = (
    f"Track S2 (anatomy) at scale=3B REUSES the authoritative {STAGE8_3B_ANATOMY_ANCHOR} run as "
    f"its 3B anchor -- it is NEVER rerun. Nothing to execute here; point cross-scale analysis at "
    f"the existing results directory instead."
)


def _child_run_signature(scale_label: str, track: str) -> str:
    if scale_label == "3B" and track == "anatomy":
        # No "_v1" -- this is a REUSE POINTER to the pre-existing stage8_coarse_anatomical_atlas_
        # 3b_v2_batched10 run, never a new run identity of its own (Section 16).
        return "stage11_3b_anatomy"
    return f"stage11_{scale_label.lower()}_{track}_v1"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scale", choices=sorted(SCALING_MODEL_REGISTRY), help="Which scale to dispatch to (required unless --report-model-family-comparability is given).")
    parser.add_argument("--track", choices=TRACKS, help="Which track to dispatch to (required unless --report-model-family-comparability is given).")
    parser.add_argument("--report-model-family-comparability", action="store_true", help="Write model_family_comparability.json for ALL registered scales (Section 5) -- factual metadata only, no GPU/Hub call, no scientific claim.")
    parser.add_argument("--output-root", default=str(REPO_ROOT / "results" / "stage11_visual_thicket_scaling"))
    args, remaining = parser.parse_known_args(argv)

    if args.report_model_family_comparability:
        report = build_model_family_comparability_report(list(SCALING_MODEL_REGISTRY.values()))
        output_dir = Path(args.output_root)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "model_family_comparability.json"
        out_path.write_text(json.dumps(report, indent=2))
        print(f"Wrote {out_path} (fetched=False for every scale -- no live HF Hub call was made; see the module docstring's `hf_model_info_fn` hook to enable one on the pod).")
        return 0

    if not args.scale or not args.track:
        parser.error("--scale and --track are both required unless --report-model-family-comparability is given.")

    print(f"[{PARENT_RUN_SIGNATURE}] dispatching scale={args.scale!r} track={args.track!r} child_run_signature={_child_run_signature(args.scale, args.track)!r}")

    try:
        ensure_scale_runnable(args.scale)
    except ScaleNotYetEnabledError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.track == "anatomy" and args.scale == "3B":
        print(REUSE_NOT_RERUN_NOTE)
        return 0

    if args.track == "anatomy":
        # scale == "7B" (the only other runnable anatomy scale) -- delegate to the EXISTING,
        # already-tested 47-test module, completely unmodified.
        from . import run_stage11_coarse_anatomical_atlas_7b as anatomy_7b

        return anatomy_7b.main(remaining)

    # args.track == "whole_model", scale in {"3B", "7B"}
    from . import run_stage11_whole_model_scaling as whole_model

    return whole_model.main(["--scale", args.scale, *remaining])


if __name__ == "__main__":
    raise SystemExit(main())

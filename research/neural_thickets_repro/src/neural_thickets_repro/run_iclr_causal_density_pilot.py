"""Isolated 7B causal-density pilot -- CLI entry point. Branch `iclr-causal-density-pilot`
ONLY; never merged into the original branch by this script or any other. This module is
STRICTLY 7B-only: it does not import, reference, construct, or dispatch anything from
run_stage11_coarse_anatomical_atlas_32b.py, stage11_32b_s2_live_evidence.py, diagnostics/
stage11_32b_s2_live_v3_solver_probe.py, or run_stage11_visual_thicket_scaling.py's 32B path --
see test_run_iclr_causal_density_pilot.py::test_module_never_imports_or_references_32b_or_72b
for the structural proof this holds, checked by source inspection so a future edit that
accidentally adds a 32B/72B import or command string is caught immediately.

Phases 0-4 (audit, preregistration, subset/shuffle-manifest construction, candidate-population
construction) run here, fully CPU-only. Phases 5-6 (base-control gate, decisive pilot) REQUIRE
a live TP-capable GPU engine (vllm/ray) and are NOT executed by this script in this
environment -- `--dry-run` prints the exact plan and immediately returns without touching
vllm/ray/torch at all, matching this project's established --dry-run convention throughout the
Stage 6-11 lineage.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]

_FORBIDDEN_32B_72B_SUBSTRINGS = (
    "stage11_coarse_anatomical_atlas_32b", "stage11_32b_s2_live_evidence", "stage11_32b_s2_live_v3_solver_probe",
    "--scale 32B", "--scale 72B", "32B", "72B",
)


def _ensure_no_32b_72b_in_argv(argv: Optional[Sequence[str]]) -> None:
    """Runtime guard, additional to the static/source-inspection tests: refuses to even parse
    argv if any 32B/72B marker leaked in (e.g. a copy-pasted command from the 32B milestone) --
    this pilot is strictly 7B-only, at every layer, not merely by omission.
    """
    if not argv:
        return
    joined = " ".join(argv)
    for token in _FORBIDDEN_32B_72B_SUBSTRINGS:
        if token in joined:
            raise ValueError(f"run_iclr_causal_density_pilot.py refuses argv containing {token!r} -- this pilot is strictly 7B-only.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    _ensure_no_32b_72b_in_argv(argv)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", default=str(REPO_ROOT / "reports" / "iclr_causal_density"))
    parser.add_argument("--dry-run", action="store_true", help="print the frozen design plan and exit -- no GPU/Hub call, no vllm/ray/torch import")
    args = parser.parse_args(argv)

    from .iclr_causal_density.design import FROZEN_DESIGN, expected_base_control_row_count, expected_row_count

    if args.dry_run:
        print(f"ICLR causal-density pilot plan (model={FROZEN_DESIGN.model_name}, scale={FROZEN_DESIGN.model_scale}):")
        print(f"capabilities={FROZEN_DESIGN.capabilities}")
        print(f"scopes={FROZEN_DESIGN.scopes}")
        print(f"radii={FROZEN_DESIGN.radii}")
        print(f"n_seeds_per_cell={FROZEN_DESIGN.n_seeds_per_cell} n_unique_perturbations={FROZEN_DESIGN.n_unique_perturbations}")
        print(f"selection_set_size={FROZEN_DESIGN.selection_set_size} audit_set_size={FROZEN_DESIGN.audit_set_size}")
        print(f"visual_conditions={FROZEN_DESIGN.visual_conditions}")
        print(f"expected_candidate_result_rows={expected_row_count()}")
        print(f"expected_base_control_result_rows={expected_base_control_row_count()}")
        print("(dry-run: no GPU/Hub call made, no vllm/ray/torch import)")
        return 0

    print(
        "GPU execution (Phase 5 base-control gate / Phase 6 decisive pilot) is not performed by "
        "this script in this environment -- see reports/iclr_causal_density/decision.md for the "
        "registered INCONCLUSIVE -- EXECUTION BLOCKED verdict and the exact resumable command.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

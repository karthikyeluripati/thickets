"""python -m neural_thickets_repro.report --config configs/gqa_repro.yaml

Refuses to run until Gate 3 (results/randopt_N{N}_K{K}_*/results.json) exists -- there is
nothing to report before then. Producing REPRODUCTION_REPORT.md itself is a later-phase
deliverable, not implemented in this Gate-0 pass.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

from .config import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "gqa_repro.yaml"))
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    pattern = str(REPO_ROOT / "results" / f"randopt_N{cfg.randopt.N}_K{cfg.randopt.K}_*" / "results.json")
    matches = glob.glob(pattern)
    if not matches:
        print(
            "No Gate 3 (full N/K) results found under results/. report.py has nothing to "
            "report until run_randopt.py has actually completed a full run on GPU hardware -- "
            "REPRODUCTION_REPORT.md is a later-phase deliverable, not produced by this "
            "Gate-0 scaffold.",
            file=sys.stderr,
        )
        return 1

    print(f"Found {len(matches)} Gate 3 result file(s): {matches}")
    print("Report generation is not implemented in this Gate-0 phase.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""python -m neural_thickets_repro.validate_env [--config configs/gqa_repro.yaml]

Prints a FEASIBLE/BLOCKED table for each pipeline gate (see plan / REPRO_SPEC.md for the
gate definitions). Purely a readiness check -- never attempts to run anything, safe to
run repeatedly.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from .config import load_config
from .env_check import CheckResult, check_cuda, check_disk, check_gate_artifact, check_module

REPO_ROOT = Path(__file__).resolve().parents[2]  # research/neural_thickets_repro/
RESULTS_DIR = REPO_ROOT / "results"


def _print_row(stage: str, checks: List[CheckResult]) -> bool:
    failed = [c for c in checks if not c.ok]
    status = "FEASIBLE" if not failed else "BLOCKED"
    reason = "-" if not failed else "; ".join(f"{c.name}: {c.detail}" for c in failed)
    print(f"{stage:<34}{status:<10}{reason}")
    return not failed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "gqa_repro.yaml"))
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    print(f"{'STAGE':<34}{'STATUS':<10}REASON")

    _print_row("Gate 0: config/spec/scaffold", [])

    cuda = check_cuda()
    vllm = check_module("vllm")
    ray = check_module("ray")
    disk = check_disk(REPO_ROOT, cfg.hardware.min_free_disk_gb)
    hw_checks = [cuda, vllm, ray, disk]

    _print_row("Gate 1: baseline eval", hw_checks)

    gate1_artifact = check_gate_artifact(RESULTS_DIR / "base" / "metrics.json")
    _print_row("Gate 2: small-scale RandOpt", hw_checks + [gate1_artifact])

    gate2_artifact = check_gate_artifact(RESULTS_DIR / "randopt_smoke" / "results.json")
    _print_row("Gate 3: full N=5000, K=50", hw_checks + [gate1_artifact, gate2_artifact])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

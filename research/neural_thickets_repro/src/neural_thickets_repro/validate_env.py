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
from .env_check import (
    CheckResult,
    check_cuda,
    check_disk,
    check_filesystem_consistency,
    check_gate_artifact,
    check_module,
    resolve_hf_home,
)

REPO_ROOT = Path(__file__).resolve().parents[2]  # research/neural_thickets_repro/
RESULTS_DIR = REPO_ROOT / "results"
GQA_DATA_DIR = REPO_ROOT / "external" / "RandOpt" / "data" / "gqa"


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

    hf_home = resolve_hf_home()

    cuda = check_cuda()
    vllm = check_module("vllm")
    ray = check_module("ray")
    # Checked separately and by name: REPO_ROOT (code/data-prep defaults) and HF_HOME
    # (model cache, often multi-GB) can be on different filesystems -- e.g. HF_HOME
    # defaults to ~/.cache/huggingface, which on RunPod is the ephemeral container root,
    # not the persistent /workspace volume, unless HF_HOME is explicitly set there.
    disk_repo = check_disk(REPO_ROOT, cfg.hardware.min_free_disk_gb)
    disk_repo.name = "disk(repo_root)"
    disk_hf = check_disk(hf_home, cfg.hardware.min_free_disk_gb)
    disk_hf.name = "disk(hf_home)"
    # Advisory, not a hard feasibility blocker: both locations can independently have
    # enough free space even if they're on different filesystems, so this is reported
    # separately rather than folded into the FEASIBLE/BLOCKED gate check.
    fs_consistency = check_filesystem_consistency({
        "repo_root": REPO_ROOT,
        "hf_home": hf_home,
        "gqa_data": GQA_DATA_DIR,
        "results": RESULTS_DIR,
    })
    hw_checks = [cuda, vllm, ray, disk_repo, disk_hf]

    print(f"  HF_HOME resolves to: {hf_home}")
    print(f"  filesystem consistency (advisory): {'OK' if fs_consistency.ok else 'MISMATCH'} -- {fs_consistency.detail}")
    _print_row("Gate 1: baseline eval", hw_checks)

    gate1_artifact = check_gate_artifact(RESULTS_DIR / "base" / "metrics.json")
    _print_row("Gate 2: small-scale RandOpt", hw_checks + [gate1_artifact])

    gate2_artifact = check_gate_artifact(RESULTS_DIR / "randopt_smoke" / "results.json")
    _print_row("Gate 3: full N=5000, K=50", hw_checks + [gate1_artifact, gate2_artifact])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

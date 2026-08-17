"""Hardware/software feasibility gate. Gate 1-3 entrypoints must call assert_feasible()
as their first action and exit immediately with a specific reason -- never attempt a
partial run that fails mid-download or deep inside the external RandOpt engine.
"""
from __future__ import annotations

import importlib.util
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def check_cuda() -> CheckResult:
    try:
        import torch

        ok = torch.cuda.is_available()
        detail = (
            f"{torch.cuda.device_count()} CUDA device(s)"
            if ok
            else "torch.cuda.is_available() is False"
        )
    except ImportError:
        ok, detail = False, "torch is not installed"
    return CheckResult("cuda", ok, detail)


def check_module(module_name: str) -> CheckResult:
    ok = importlib.util.find_spec(module_name) is not None
    return CheckResult(module_name, ok, "installed" if ok else "not installed")


def check_disk(path: "str | Path", min_gb: float) -> CheckResult:
    p = Path(path).resolve()
    usage = shutil.disk_usage(p.anchor)
    free_gb = usage.free / (1024**3)
    ok = free_gb >= min_gb
    return CheckResult("disk", ok, f"{free_gb:.1f} GB free (need >= {min_gb} GB) at {p.anchor}")


def check_gate_artifact(path: "str | Path") -> CheckResult:
    p = Path(path)
    ok = p.exists()
    return CheckResult(f"gate_artifact:{p.name}", ok, f"{'present' if ok else 'missing'}: {p}")


class GateBlockedError(RuntimeError):
    pass


def assert_feasible(stage: str, checks: List[CheckResult]) -> None:
    failed = [c for c in checks if not c.ok]
    if failed:
        reasons = "; ".join(f"{c.name}: {c.detail}" for c in failed)
        raise GateBlockedError(f"{stage} is BLOCKED -- {reasons}")

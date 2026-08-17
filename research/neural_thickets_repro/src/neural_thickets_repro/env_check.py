"""Hardware/software feasibility gate. Gate 1-3 entrypoints must call assert_feasible()
as their first action and exit immediately with a specific reason -- never attempt a
partial run that fails mid-download or deep inside the external RandOpt engine.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


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


def _nearest_existing(path: Path) -> Path:
    """Walk up to the nearest ancestor that actually exists (e.g. a not-yet-created cache
    dir), so disk_usage always has something real to stat.
    """
    p = path
    while not p.exists() and p != p.parent:
        p = p.parent
    return p


def check_disk(path: "str | Path", min_gb: float) -> CheckResult:
    """Checks free space on the filesystem that actually CONTAINS `path`.

    IMPORTANT: must call shutil.disk_usage(p), not shutil.disk_usage(p.anchor). `.anchor`
    collapses any POSIX path to "/" regardless of mount points further down the tree, so it
    silently reports the container's root filesystem instead of e.g. a mounted persistent
    volume (RunPod's /workspace) that `path` actually lives under. shutil.disk_usage
    correctly resolves through mount points (POSIX) / drives (Windows) when given the real
    path, which is why we pass p itself here.
    """
    p = _nearest_existing(Path(path).resolve())
    usage = shutil.disk_usage(p)
    free_gb = usage.free / (1024**3)
    ok = free_gb >= min_gb
    return CheckResult("disk", ok, f"{free_gb:.1f} GB free (need >= {min_gb} GB) at {p}")


def resolve_hf_home() -> Path:
    """Where the HF hub cache (model weights etc.) will actually be written -- honors
    HF_HOME if set (as RUNPOD_SETUP.md instructs, pointed at persistent storage), else
    transformers/huggingface_hub's own default (~/.cache/huggingface), which on RunPod is
    typically the ephemeral container filesystem, NOT the persistent volume.
    """
    env_value = os.environ.get("HF_HOME")
    if env_value:
        return Path(env_value)
    return Path.home() / ".cache" / "huggingface"


def check_filesystem_consistency(paths: Dict[str, "str | Path"]) -> CheckResult:
    """Confirms the given named paths (HF_HOME, GQA data dir, results dir, ...) all resolve
    onto the SAME filesystem/device. Catches the case where e.g. HF_HOME defaults to the
    ephemeral container root while the repo/data live on a mounted persistent volume --
    disk-space checks on one don't protect the other, and a multi-GB model download could
    silently fail or vanish on container restart if it landed on the wrong one.
    """
    resolved = {name: _nearest_existing(Path(p).resolve()) for name, p in paths.items()}
    devices = {name: os.stat(p).st_dev for name, p in resolved.items()}
    unique_devices = set(devices.values())
    ok = len(unique_devices) <= 1
    detail = ", ".join(f"{name}={resolved[name]} (dev={dev})" for name, dev in devices.items())
    return CheckResult("filesystem_consistency", ok, detail)


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

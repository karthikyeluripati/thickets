import os
import shutil
from pathlib import Path

import pytest

from neural_thickets_repro.env_check import (
    CheckResult,
    GateBlockedError,
    assert_feasible,
    check_disk,
    check_filesystem_consistency,
    check_gate_artifact,
    check_module,
    resolve_hf_home,
)


def test_check_module_missing_reports_not_installed():
    result = check_module("definitely_not_a_real_module_xyz123")
    assert result.ok is False
    assert "not installed" in result.detail


def test_check_module_present_reports_installed():
    result = check_module("json")  # stdlib, always present
    assert result.ok is True


def test_check_disk_blocked_when_threshold_too_high(tmp_path):
    result = check_disk(tmp_path, min_gb=10**9)  # absurdly high, guaranteed to fail
    assert result.ok is False
    assert "need >=" in result.detail


def test_check_disk_ok_when_threshold_trivial(tmp_path):
    result = check_disk(tmp_path, min_gb=0)
    assert result.ok is True


def test_check_disk_checks_the_real_path_not_its_drive_root(tmp_path, monkeypatch):
    """Regression test: check_disk must call shutil.disk_usage on the actual (nearest
    existing) path, NOT on path.anchor. `.anchor` collapses any POSIX path to "/",
    which would silently report the container root filesystem instead of e.g. a mounted
    persistent volume the path actually lives under (the RunPod /workspace bug).
    """
    nested = tmp_path / "workspace" / "thickets" / "research"
    nested.mkdir(parents=True)

    calls = []
    real_disk_usage = shutil.disk_usage

    def spy_disk_usage(path):
        calls.append(Path(path))
        return real_disk_usage(path)

    monkeypatch.setattr(shutil, "disk_usage", spy_disk_usage)
    check_disk(nested, min_gb=0)

    assert calls == [nested.resolve()]
    assert calls[0] != Path(nested.resolve().anchor)


def test_check_disk_walks_up_to_nearest_existing_ancestor(tmp_path):
    not_yet_created = tmp_path / "hf_cache" / "hub"
    result = check_disk(not_yet_created, min_gb=0)
    assert result.ok is True  # should not raise even though the exact path doesn't exist yet


def test_resolve_hf_home_honors_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "custom_hf_home"))
    assert resolve_hf_home() == tmp_path / "custom_hf_home"


def test_resolve_hf_home_defaults_when_unset(monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    assert resolve_hf_home() == Path.home() / ".cache" / "huggingface"


def test_check_filesystem_consistency_true_for_same_tree(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    result = check_filesystem_consistency({"a": a, "b": b})
    assert result.ok is True


def test_check_gate_artifact_missing(tmp_path):
    result = check_gate_artifact(tmp_path / "nope.json")
    assert result.ok is False
    assert "missing" in result.detail


def test_check_gate_artifact_present(tmp_path):
    f = tmp_path / "metrics.json"
    f.write_text("{}")
    result = check_gate_artifact(f)
    assert result.ok is True
    assert "present" in result.detail


def test_assert_feasible_passes_when_all_ok():
    checks = [CheckResult("a", True, "fine"), CheckResult("b", True, "fine")]
    assert_feasible("stage", checks)  # should not raise


def test_assert_feasible_reports_all_failed_reasons():
    checks = [
        CheckResult("cuda", False, "no GPU"),
        CheckResult("vllm", False, "not installed"),
        CheckResult("disk", True, "plenty"),
    ]
    with pytest.raises(GateBlockedError) as exc_info:
        assert_feasible("Gate 1", checks)
    message = str(exc_info.value)
    assert "Gate 1 is BLOCKED" in message
    assert "cuda: no GPU" in message
    assert "vllm: not installed" in message
    assert "disk" not in message.split("--")[1]  # passing check should not be listed as a reason

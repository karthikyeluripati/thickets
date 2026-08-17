import pytest

from neural_thickets_repro.env_check import (
    CheckResult,
    GateBlockedError,
    assert_feasible,
    check_disk,
    check_gate_artifact,
    check_module,
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

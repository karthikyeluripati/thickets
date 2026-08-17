"""Regression test: eval_base.py / run_randopt.py MUST invoke the external RandOpt repo's
randopt.py with cwd set to that repo's own root. GQAHandler resolves data/gqa/train.parquet
etc. relative to the process's cwd, not relative to randopt.py's location -- running with
our reproduction repo as cwd (the old default) fails to find data prepared under
external/RandOpt/data/gqa/ with FileNotFoundError: data/gqa/train.parquet.

Uses the real configs/gqa_repro.yaml (already has resolved model/dataset fields) and
monkeypatches: the hardware feasibility gate (this machine has no GPU), the external repo
paths (point at a throwaway fake randopt.py so .exists() is True without a real clone), and
subprocess.run (recorded instead of executed) -- so this runs anywhere, no GPU/external
clone/network needed.
"""
from pathlib import Path

import neural_thickets_repro.eval_base as eval_base_module
import neural_thickets_repro.run_randopt as run_randopt_module

REAL_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "gqa_repro.yaml"


def _make_fake_external_repo(tmp_path):
    external_root = tmp_path / "external_randopt_root"
    external_root.mkdir()
    (external_root / "randopt.py").write_text("# fake randopt.py for testing\n")
    return external_root


def _fake_run_recorder(recorded):
    def fake_run(cmd, check=False, cwd=None):
        recorded["cmd"] = cmd
        recorded["cwd"] = cwd
        recorded["check"] = check
    return fake_run


def test_eval_base_invokes_subprocess_with_external_repo_root_as_cwd(tmp_path, monkeypatch):
    external_root = _make_fake_external_repo(tmp_path)
    monkeypatch.setattr(eval_base_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(eval_base_module, "EXTERNAL_RANDOPT", external_root / "randopt.py")
    monkeypatch.setattr(eval_base_module, "EXTERNAL_RANDOPT_ROOT", external_root)
    monkeypatch.setattr(eval_base_module, "assert_feasible", lambda stage, checks: None)

    recorded = {}
    monkeypatch.setattr(eval_base_module.subprocess, "run", _fake_run_recorder(recorded))

    rc = eval_base_module.main(["--config", str(REAL_CONFIG)])

    assert rc == 0
    assert recorded["cwd"] == external_root
    assert recorded["cwd"] != tmp_path  # must NOT be our reproduction repo (the old bug)
    assert recorded["check"] is True
    assert str(external_root / "randopt.py") in recorded["cmd"]


def test_run_randopt_invokes_subprocess_with_external_repo_root_as_cwd(tmp_path, monkeypatch):
    external_root = _make_fake_external_repo(tmp_path)
    monkeypatch.setattr(run_randopt_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(run_randopt_module, "EXTERNAL_RANDOPT", external_root / "randopt.py")
    monkeypatch.setattr(run_randopt_module, "EXTERNAL_RANDOPT_ROOT", external_root)
    monkeypatch.setattr(run_randopt_module, "GATE1_ARTIFACT", tmp_path / "unused_gate1.json")
    monkeypatch.setattr(run_randopt_module, "GATE2_ARTIFACT", tmp_path / "unused_gate2.json")
    monkeypatch.setattr(run_randopt_module, "assert_feasible", lambda stage, checks: None)

    recorded = {}
    monkeypatch.setattr(run_randopt_module.subprocess, "run", _fake_run_recorder(recorded))

    rc = run_randopt_module.main([
        "--config", str(REAL_CONFIG),
        "--N", "5", "--K", "2",
        "--sigma-candidate", "sigma_default",
    ])

    assert rc == 0
    assert recorded["cwd"] == external_root
    assert recorded["cwd"] != tmp_path  # must NOT be our reproduction repo (the old bug)
    assert recorded["check"] is True
    assert str(external_root / "randopt.py") in recorded["cmd"]

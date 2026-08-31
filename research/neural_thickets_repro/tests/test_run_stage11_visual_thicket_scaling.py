"""Tests for run_stage11_visual_thicket_scaling.py -- the unified scale/track dispatcher. All
paths tested here are GPU/Hub-free (--dry-run for the two runnable-execution paths, the pure
reuse-not-rerun message for 3B anatomy, the pure block message for 32B/72B, and the network-free
--report-model-family-comparability path).
"""
import json

import pytest

import neural_thickets_repro.run_stage11_visual_thicket_scaling as dispatcher
from neural_thickets_repro.scaling_common import SCALING_MODEL_REGISTRY


def test_report_model_family_comparability_writes_json_for_all_scales(tmp_path, capsys):
    exit_code = dispatcher.main(["--report-model-family-comparability", "--output-root", str(tmp_path)])
    assert exit_code == 0
    out_path = tmp_path / "model_family_comparability.json"
    assert out_path.exists()
    report = json.loads(out_path.read_text())
    assert set(report["scales"].keys()) == set(SCALING_MODEL_REGISTRY.keys())


def test_requires_scale_and_track_unless_reporting_comparability():
    with pytest.raises(SystemExit):
        dispatcher.main([])


def test_dispatch_3b_anatomy_is_reuse_not_rerun_and_never_executes(capsys):
    exit_code = dispatcher.main(["--scale", "3B", "--track", "anatomy"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "REUSES" in captured.out
    assert "stage8_coarse_anatomical_atlas_3b_v2_batched10" in captured.out
    assert "never rerun" in captured.out.lower() or "NEVER rerun" in captured.out


@pytest.mark.parametrize("track", ["whole_model", "anatomy"])
def test_dispatch_blocks_72b_unconditionally(track, capsys):
    """72B remains hard-disabled exactly as before the 32B-readiness milestone -- still routed
    through the unmodified ensure_scale_runnable, still the same message.
    """
    exit_code = dispatcher.main(["--scale", "72B", "--track", track])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "not yet enabled" in (captured.out + captured.err).lower() or "NOT in RUNNABLE_SCALES" in (captured.out + captured.err)


def test_dispatch_32b_anatomy_routes_to_the_dedicated_s2_runner(monkeypatch):
    """32B anatomy (S2) is no longer hard-blocked at the dispatcher -- it routes to the dedicated,
    narrow, TP=4-aware run_stage11_coarse_anatomical_atlas_32b runner, which enforces its OWN
    live-evidence readiness gate (see test_run_stage11_coarse_anatomical_atlas_32b.py) rather than
    being refused here purely for lacking live evidence at dispatch time.
    """
    import neural_thickets_repro.run_stage11_coarse_anatomical_atlas_32b as anatomy_32b

    reached = {}

    def _marker(argv=None):
        reached["argv"] = argv
        return 0

    monkeypatch.setattr(anatomy_32b, "main", _marker)
    exit_code = dispatcher.main(["--scale", "32B", "--track", "anatomy"])
    assert exit_code == 0
    assert reached["argv"] == []


def test_dispatch_no_longer_blocks_32b_whole_model_without_smoke_at_the_dispatcher_level(monkeypatch):
    """The former dispatcher-level '--smoke required, unconditionally' guard is REMOVED -- 32B
    whole_model (--smoke or full) now reaches the SAME live-evidence gate either way, evaluated
    inside run_stage11_whole_model_scaling.main() (the only place real evidence can be gathered/
    validated), never rejected here purely for lacking --smoke. Proven by patching that main()
    with a marker so this test never touches real env/GPU/Hub checks -- it does not assert what
    run_stage11_whole_model_scaling.main() itself ultimately decides (see test_stage11_32b_smoke_
    wiring.py for that gate's own full/smoke-agnostic behavior).
    """
    import neural_thickets_repro.run_stage11_whole_model_scaling as whole_model_module

    reached = {}

    def _marker(argv=None):
        reached["argv"] = argv
        return 0

    monkeypatch.setattr(whole_model_module, "main", _marker)

    exit_code = dispatcher.main(["--scale", "32B", "--track", "whole_model"])
    assert exit_code == 0
    assert reached["argv"] == ["--scale", "32B"]  # no --smoke passed through -- a FULL invocation reached the runner unmodified


def test_dispatch_7b_whole_model_dry_run_produces_correct_counts(capsys):
    exit_code = dispatcher.main(["--scale", "7B", "--track", "whole_model", "--dry-run"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "total_unique_perturbations=192" in out
    assert "total_perturbation_x_capability_evaluations=1152" in out
    assert "total_perturbed_model_example_evaluations=57600" in out


def test_dispatch_3b_whole_model_dry_run_smoke_produces_correct_counts(capsys):
    exit_code = dispatcher.main(["--scale", "3B", "--track", "whole_model", "--smoke", "--dry-run"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "total_unique_perturbations=3" in out
    assert "total_perturbation_x_capability_evaluations=18" in out
    assert "total_perturbed_model_example_evaluations=90" in out


def test_dispatch_7b_anatomy_dry_run_delegates_to_the_existing_untouched_module(capsys):
    """Delegates to the EXISTING, already-tested run_stage11_coarse_anatomical_atlas_7b module
    unmodified -- its own internal run_signature convention
    ("stage11_coarse_anatomical_atlas_7b_v1") is preserved exactly as that module's own 47-test
    suite already validates, rather than renamed by this dispatcher.
    """
    exit_code = dispatcher.main(["--scale", "7B", "--track", "anatomy", "--dry-run"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "total_unique_perturbations=576" in out
    assert "total_perturbation_x_capability_evaluations=3456" in out
    assert "total_perturbed_model_example_evaluations=172800" in out
    assert "stage11_coarse_anatomical_atlas_7b_v1" in out


def test_dispatch_prints_child_run_signature_before_delegating(capsys):
    dispatcher.main(["--scale", "7B", "--track", "whole_model", "--dry-run"])
    out = capsys.readouterr().out
    assert "stage11_7b_whole_model_v1" in out
    assert dispatcher.PARENT_RUN_SIGNATURE in out

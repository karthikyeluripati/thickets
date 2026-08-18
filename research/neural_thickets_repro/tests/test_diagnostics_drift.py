import torch

from neural_thickets_repro.diagnostics.perturb_restore_drift import (
    check_vision_frozen,
    measure_drift,
    run_drift_audit,
)


def test_measure_drift_zero_when_unchanged(dummy_vlm_factory):
    model = dummy_vlm_factory()
    original = {k: v.clone() for k, v in model.state_dict().items()}
    drift = measure_drift(model, original)
    assert drift["max_abs_drift"] == 0.0
    assert drift["mean_abs_drift"] == 0.0
    assert drift["relative_norm_drift"] == 0.0
    assert drift["fraction_elements_differing"] == 0.0


def test_measure_drift_fraction_elements_differing_counts_exact_mismatches(dummy_vlm_factory):
    model = dummy_vlm_factory()
    original = {k: v.clone() for k, v in model.state_dict().items()}
    _, p = next(iter(model.named_parameters()))
    with torch.no_grad():
        p.view(-1)[0] += 1.0  # perturb exactly one element of one parameter tensor
    total_param_elems = sum(pp.numel() for _, pp in model.named_parameters())

    drift = measure_drift(model, original)

    assert drift["fraction_elements_differing"] == 1 / total_param_elems
    assert drift["max_abs_drift"] == 1.0


def test_run_drift_audit_reports_requested_checkpoints(dummy_vlm_factory):
    model = dummy_vlm_factory().to(torch.bfloat16)
    results = run_drift_audit(model, sigma=0.01, cycle_checkpoints=[1, 5, 20])
    assert [r["cycle"] for r in results] == [1, 5, 20]
    for r in results:
        assert r["max_abs_drift"] >= 0.0
        assert r["mean_abs_drift"] >= 0.0
        assert r["relative_norm_drift"] >= 0.0
        assert "elapsed_seconds" in r


def test_vision_encoder_never_drifts_across_many_cycles(dummy_vlm_factory):
    model = dummy_vlm_factory().to(torch.bfloat16)
    original = {k: v.clone() for k, v in model.state_dict().items()}
    run_drift_audit(model, sigma=0.05, cycle_checkpoints=[50])
    assert check_vision_frozen(model, original) is True


def test_drift_is_actually_detected_not_silently_zero(dummy_vlm_factory):
    """At bf16 with a nonzero sigma, 100 perturb->restore cycles should leave measurable
    drift -- regression guard against a broken restore() that always fully cancels (which
    would silently defeat the whole point of this diagnostic).
    """
    model = dummy_vlm_factory().to(torch.bfloat16)
    results = run_drift_audit(model, sigma=0.05, cycle_checkpoints=[100])
    assert results[0]["relative_norm_drift"] > 0.0
    assert results[0]["cycle"] == 100

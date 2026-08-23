import pytest

from neural_thickets_repro.thicket.experiment_size import ExperimentSizeInputs, estimate_experiment_size


def test_spec_worked_example_matches_item_10():
    """The exact instance from the Stage-5 task spec: models=1, capabilities=3, anatomy=3,
    radii=3, perturbations_per_condition=64, examples_per_capability=50.
    """
    inputs = ExperimentSizeInputs(
        n_models=1, n_capabilities=3, n_anatomy_regions=3, n_radii=3,
        n_perturbations_per_condition=64, n_examples_per_capability=50,
    )
    report = estimate_experiment_size(inputs)

    assert report.unique_candidate_models == 1 * 3 * 3 * 64  # 576
    assert report.baseline_evaluations == 1 * 3 * 50  # 150
    assert report.total_model_example_evaluations == 576 * 3 * 50  # 86400
    assert report.evaluations_per_capability == 86400 // 3
    assert report.evaluations_per_anatomy == 86400 // 3
    assert report.evaluations_per_radius == 86400 // 3
    assert report.multiplier_vs_one_baseline == pytest.approx(576.0)


def test_multiplier_equals_candidates_per_model_when_repeats_and_ensemble_are_one():
    """Both unique_candidate_models and baseline_evaluations scale with n_models (each model
    gets its own candidate sweep AND its own baseline sweep), so the multiplier reduces to
    candidates-per-model, not the raw candidate count across all models.
    """
    inputs = ExperimentSizeInputs(
        n_models=2, n_capabilities=4, n_anatomy_regions=5, n_radii=2,
        n_perturbations_per_condition=10, n_examples_per_capability=100,
    )
    report = estimate_experiment_size(inputs)
    candidates_per_model = report.unique_candidate_models / inputs.n_models
    assert report.multiplier_vs_one_baseline == pytest.approx(candidates_per_model)


def test_multiplier_equals_unique_candidate_models_for_a_single_model():
    inputs = ExperimentSizeInputs(
        n_models=1, n_capabilities=4, n_anatomy_regions=5, n_radii=2,
        n_perturbations_per_condition=10, n_examples_per_capability=100,
    )
    report = estimate_experiment_size(inputs)
    assert report.multiplier_vs_one_baseline == pytest.approx(float(report.unique_candidate_models))


def test_repeats_and_ensemble_multiply_total_evaluations():
    base = ExperimentSizeInputs(n_models=1, n_capabilities=2, n_anatomy_regions=1, n_radii=1, n_perturbations_per_condition=4, n_examples_per_capability=10)
    doubled_repeat = ExperimentSizeInputs(n_models=1, n_capabilities=2, n_anatomy_regions=1, n_radii=1, n_perturbations_per_condition=4, n_examples_per_capability=10, n_repeats=2)
    tripled_ensemble = ExperimentSizeInputs(n_models=1, n_capabilities=2, n_anatomy_regions=1, n_radii=1, n_perturbations_per_condition=4, n_examples_per_capability=10, ensemble_k=3)

    r0 = estimate_experiment_size(base)
    r1 = estimate_experiment_size(doubled_repeat)
    r2 = estimate_experiment_size(tripled_ensemble)

    assert r1.total_model_example_evaluations == 2 * r0.total_model_example_evaluations
    assert r2.total_model_example_evaluations == 3 * r0.total_model_example_evaluations


def test_sanity_evaluations_are_reported_separately_from_perturbed_evaluations():
    inputs = ExperimentSizeInputs(n_models=1, n_capabilities=2, n_anatomy_regions=1, n_radii=1, n_perturbations_per_condition=4, n_examples_per_capability=10, n_sanity_runs=3)
    report = estimate_experiment_size(inputs)
    assert report.sanity_evaluations == 3 * 1 * 2 * 10  # 60
    inputs_no_sanity = ExperimentSizeInputs(n_models=1, n_capabilities=2, n_anatomy_regions=1, n_radii=1, n_perturbations_per_condition=4, n_examples_per_capability=10)
    assert estimate_experiment_size(inputs_no_sanity).sanity_evaluations == 0


def test_non_positive_fields_are_rejected():
    with pytest.raises(ValueError):
        ExperimentSizeInputs(n_models=0, n_capabilities=1, n_anatomy_regions=1, n_radii=1, n_perturbations_per_condition=1, n_examples_per_capability=1)


def test_negative_sanity_runs_rejected():
    with pytest.raises(ValueError):
        ExperimentSizeInputs(n_models=1, n_capabilities=1, n_anatomy_regions=1, n_radii=1, n_perturbations_per_condition=1, n_examples_per_capability=1, n_sanity_runs=-1)


def test_report_has_no_dollar_cost_field():
    inputs = ExperimentSizeInputs(n_models=1, n_capabilities=1, n_anatomy_regions=1, n_radii=1, n_perturbations_per_condition=1, n_examples_per_capability=1)
    report = estimate_experiment_size(inputs)
    for field_name in report.__dataclass_fields__:
        assert "cost" not in field_name.lower()
        assert "dollar" not in field_name.lower()

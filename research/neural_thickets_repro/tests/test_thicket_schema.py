import json

import pytest

from neural_thickets_repro.thicket.schema import ExperimentResultRecord


def _record(**overrides):
    kwargs = dict(
        experiment_id="exp1", perturbation_id="pert1", model_family="qwen2_5_vl", model_scale="3B", model_revision="rev1",
        perturbation_mode="anatomical_relative_l2", anatomy_region="vision_early", radius=0.05, sigma=None, seed=1,
        parameter_mask_hash="hash1", capability="counting", dataset_role="map", subset_hash="subhash1",
        base_score=0.5, perturbed_score=0.6, delta=0.1, parser_failure_rate=0.0,
        per_example_result_path="results/foo.jsonl", per_example_result_hash="filehash1", runtime_metadata={},
    )
    kwargs.update(overrides)
    return ExperimentResultRecord(**kwargs)


def test_record_round_trips_through_dict():
    record = _record()
    d = record.to_dict()
    restored = ExperimentResultRecord.from_dict(d)
    assert restored == record


def test_record_is_json_serializable():
    record = _record()
    json.dumps(record.to_dict())  # must not raise


def test_record_rejects_inconsistent_delta():
    with pytest.raises(ValueError):
        _record(base_score=0.5, perturbed_score=0.6, delta=999.0)

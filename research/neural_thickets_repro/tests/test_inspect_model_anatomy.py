"""Tests inspect_model_anatomy() against a synthetic dummy nn.Module (the shared 32-vision
-block / 12-LM-layer fixture) -- no real model download, matching this repo's established
no-GPU-needed convention. `main()`'s actual argparse/AutoModel-loading path is intentionally
untested here (it needs a real pod-side checkpoint), mirroring run_capability_benchmark_gate.py
's own import-avoidance testing convention.
"""
from neural_thickets_repro.inspect_model_anatomy import inspect_model_anatomy


def test_inspect_model_anatomy_reports_every_region(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    report = inspect_model_anatomy(model.named_parameters(), model_name="qwen2_5_vl_synthetic", model_revision="rev1", model_scale="synthetic")

    assert report["model_name"] == "qwen2_5_vl_synthetic"
    assert report["lm_namespace_convention"] == "runtime_wrapped"
    assert report["vision_block_indices"] == list(range(32))
    assert report["lm_layer_indices"] == list(range(12))
    assert set(report["regions"]) >= {"full_model", "vision", "multimodal_connector_or_merger", "language", "vision_early", "vision_middle", "vision_late", "language_early", "language_middle", "language_late"}


def test_inspect_model_anatomy_percentages_sum_to_100_across_level1_regions(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    report = inspect_model_anatomy(model.named_parameters(), model_name="x")
    level1 = ["vision", "multimodal_connector_or_merger", "language"]
    total_pct = sum(report["regions"][name]["percentage_of_total_elements"] for name in level1)
    assert abs(total_pct - 100.0) < 1e-6


def test_inspect_model_anatomy_reports_l2_norm_and_dtype(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    report = inspect_model_anatomy(model.named_parameters(), model_name="x")
    vision_region = report["regions"]["vision"]
    assert vision_region["l2_norm"] > 0.0
    assert vision_region["dtypes"] == ["torch.float32"]
    assert vision_region["tensor_count"] > 0
    assert vision_region["element_count"] > 0


def test_inspect_model_anatomy_reports_validation_block(runtime_wrapped_vlm_32vision_factory):
    model = runtime_wrapped_vlm_32vision_factory()
    report = inspect_model_anatomy(model.named_parameters(), model_name="x")
    assert report["validation"]["ok"] is True
    assert "language" in report["validation"]["uncovered_by_parent"]


def test_inspect_model_anatomy_is_json_serializable(runtime_wrapped_vlm_32vision_factory):
    import json

    model = runtime_wrapped_vlm_32vision_factory()
    report = inspect_model_anatomy(model.named_parameters(), model_name="x")
    json.dumps(report)  # must not raise

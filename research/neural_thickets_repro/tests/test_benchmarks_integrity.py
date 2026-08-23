"""Tests for benchmarks/integrity.py -- pure Python, no GPU/ray/vllm needed."""
from neural_thickets_repro.benchmarks.base import Example
from neural_thickets_repro.benchmarks.integrity import merge_parser_failure_stats, validate_examples


def test_clean_pool_passes(tiny_image_factory):
    img = tiny_image_factory()
    examples = [Example(example_id=str(i), image=img, target="t") for i in range(5)]
    report = validate_examples(examples, n_requested=5)
    assert report.passed is True
    assert report.n_loaded == 5
    assert report.n_duplicate_ids == 0
    assert report.n_missing_targets == 0
    assert report.n_invalid_images == 0


def test_duplicate_id_detected(tiny_image_factory):
    img = tiny_image_factory()
    examples = [
        Example(example_id="1", image=img, target="t"),
        Example(example_id="1", image=img, target="t"),
        Example(example_id="2", image=img, target="t"),
    ]
    report = validate_examples(examples, n_requested=3)
    assert report.passed is False
    assert report.n_duplicate_ids == 1
    assert report.duplicate_ids == ["1"]


def test_missing_target_detected(tiny_image_factory):
    img = tiny_image_factory()
    examples = [Example(example_id="1", image=img, target=None)]
    report = validate_examples(examples, n_requested=1)
    assert report.passed is False
    assert report.n_missing_targets == 1
    assert report.missing_target_ids == ["1"]


def test_invalid_image_detected():
    examples = [Example(example_id="1", image=None, target="t")]
    report = validate_examples(examples, n_requested=1, require_images=True)
    assert report.passed is False
    assert report.n_invalid_images == 1
    assert report.invalid_image_ids == ["1"]


def test_zero_size_image_detected():
    from PIL import Image
    zero_size_image = Image.new("RGB", (0, 0))
    examples = [Example(example_id="1", image=zero_size_image, target="t")]
    report = validate_examples(examples, n_requested=1)
    assert report.n_invalid_images == 1


def test_require_images_false_skips_image_check():
    examples = [Example(example_id="1", image=None, target="t")]
    report = validate_examples(examples, n_requested=1, require_images=False)
    assert report.n_invalid_images == 0
    assert report.passed is True


def test_merge_parser_failure_stats_computes_rate(tiny_image_factory):
    img = tiny_image_factory()
    examples = [Example(example_id=str(i), image=img, target="t") for i in range(10)]
    report = validate_examples(examples, n_requested=10)
    merge_parser_failure_stats(report, parser_failures=2, n=10)
    assert report.parser_failures == 2
    assert report.parser_failure_rate == 0.2


def test_merge_parser_failure_stats_zero_n_no_divide_by_zero():
    report = validate_examples([], n_requested=0)
    merge_parser_failure_stats(report, parser_failures=0, n=0)
    assert report.parser_failure_rate == 0.0


def test_to_dict_includes_passed_field(tiny_image_factory):
    img = tiny_image_factory()
    examples = [Example(example_id="1", image=img, target="t")]
    report = validate_examples(examples, n_requested=1)
    d = report.to_dict()
    assert d["passed"] is True
    assert "n_requested" in d and "parser_failure_rate" in d

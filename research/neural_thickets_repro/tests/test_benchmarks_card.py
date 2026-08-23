"""Tests for benchmarks/card.py -- pure Python, no GPU/ray/vllm needed. Table-driven over
every branch of decide_status().
"""
from types import SimpleNamespace

import pytest

from neural_thickets_repro.benchmarks.card import (
    REQUIRED_CARD_FIELDS,
    BenchmarkCardData,
    decide_status,
    render_json,
    render_markdown,
    write_card,
)
from neural_thickets_repro.benchmarks.image_sanity import ImageSanityResult
from neural_thickets_repro.benchmarks.integrity import validate_examples


def _gates(**overrides):
    defaults = dict(
        max_parser_failure_rate_pass=0.02,
        max_parser_failure_rate_needs_review=0.10,
        image_sanity_min_gap_pass=0.05,
        image_sanity_subset_size=40,
        floor_ceiling_low=0.05,
        floor_ceiling_high=0.95,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _clean_sanity(gap_shuffled=0.2, text_only_supported=True, gap_text_only=0.2):
    return ImageSanityResult(
        n=40, correct_image_primary_metric=0.6,
        shuffled_image_primary_metric=0.6 - gap_shuffled,
        text_only_primary_metric=(0.6 - gap_text_only) if text_only_supported else None,
        text_only_supported=text_only_supported,
        text_only_unsupported_reason=None if text_only_supported else "not supported",
    )


def _card(**overrides):
    defaults = dict(
        dataset="fake_dataset", capability="fake_capability", dataset_revision="abc",
        split="validation", subset_size=200, subset_seed=42, subset_ids_path="artifacts/x.json",
        integrity=validate_examples([], n_requested=0),
        prompt_template="template", generation_config={"max_tokens": 64},
        prediction_parser="fake parser", metric_description="fake metric",
        base_metrics={"primary_metric": 0.6, "parser_failure_rate": 0.0},
        repeat_metrics={"primary_metric": 0.6, "parser_failure_rate": 0.0},
        repeatability_status="PASS", generation_hash_match=True, parsed_prediction_hash_match=True,
        image_sanity=_clean_sanity(), known_caveats=[],
    )
    defaults.update(overrides)
    return BenchmarkCardData(**defaults)


def test_clean_card_passes():
    status, reasons = decide_status(_card(), _gates())
    assert status == "PASS"
    assert reasons == []


def test_repeatability_fail_forces_fail_regardless_of_everything_else():
    card = _card(repeatability_status="FAIL")
    status, reasons = decide_status(card, _gates())
    assert status == "FAIL"
    assert "repeatability" in reasons[0]


def test_parser_failure_rate_above_needs_review_ceiling_forces_fail():
    card = _card(base_metrics={"primary_metric": 0.6, "parser_failure_rate": 0.15})
    status, _ = decide_status(card, _gates())
    assert status == "FAIL"


def test_parser_failure_rate_between_pass_and_needs_review_is_needs_review():
    card = _card(base_metrics={"primary_metric": 0.6, "parser_failure_rate": 0.05})
    status, reasons = decide_status(card, _gates())
    assert status == "NEEDS_REVIEW"
    assert any("parser failure rate" in r for r in reasons)


def test_image_sanity_gap_shuffled_non_positive_forces_fail():
    card = _card(image_sanity=_clean_sanity(gap_shuffled=0.0))
    status, reasons = decide_status(card, _gates())
    assert status == "FAIL"
    assert "correct - shuffled" in reasons[0]

    card_negative = _card(image_sanity=_clean_sanity(gap_shuffled=-0.1))
    status, _ = decide_status(card_negative, _gates())
    assert status == "FAIL"


def test_image_sanity_gap_text_only_non_positive_forces_fail():
    card = _card(image_sanity=_clean_sanity(gap_shuffled=0.2, gap_text_only=0.0))
    status, reasons = decide_status(card, _gates())
    assert status == "FAIL"
    assert "correct - text_only" in reasons[0]


def test_image_sanity_gap_below_pass_threshold_is_needs_review():
    card = _card(image_sanity=_clean_sanity(gap_shuffled=0.01))
    status, reasons = decide_status(card, _gates())
    assert status == "NEEDS_REVIEW"
    assert any("shuffled" in r for r in reasons)


def test_no_repeat_run_caps_below_pass():
    card = _card(repeatability_status="NOT_RUN", repeat_metrics=None)
    status, reasons = decide_status(card, _gates())
    assert status == "NEEDS_REVIEW"
    assert any("no repeat run" in r for r in reasons)


def test_no_image_sanity_check_caps_below_pass():
    card = _card(image_sanity=None)
    status, reasons = decide_status(card, _gates())
    assert status == "NEEDS_REVIEW"
    assert any("no image-sanity check" in r for r in reasons)


def test_text_only_not_supported_is_handled_honestly_not_scored_as_pass():
    card = _card(image_sanity=_clean_sanity(text_only_supported=False))
    status, reasons = decide_status(card, _gates())
    assert status == "PASS"  # text-only axis simply excluded, not penalized, not silently passed as if tested
    md = render_markdown(card, status, reasons)
    assert "NOT_SUPPORTED" in md


def test_floor_ceiling_low_triggers_needs_review():
    card = _card(base_metrics={"primary_metric": 0.02, "parser_failure_rate": 0.0})
    status, reasons = decide_status(card, _gates())
    assert status == "NEEDS_REVIEW"
    assert any("floor/ceiling" in r for r in reasons)


def test_floor_ceiling_high_triggers_needs_review():
    card = _card(base_metrics={"primary_metric": 0.99, "parser_failure_rate": 0.0})
    status, reasons = decide_status(card, _gates())
    assert status == "NEEDS_REVIEW"
    assert any("floor/ceiling" in r for r in reasons)


def test_rendered_markdown_and_json_contain_every_required_field():
    card = _card()
    status, reasons = decide_status(card, _gates())
    md = render_markdown(card, status, reasons)
    for field_label in REQUIRED_CARD_FIELDS:
        assert field_label in md, f"missing required card field: {field_label}"

    payload = render_json(card, status, reasons)
    assert payload["status"] == status
    assert payload["base_metrics"]["primary_metric"] == 0.6


def test_write_card_creates_markdown_and_json_files(tmp_path):
    card = _card()
    status, reasons = write_card(card, _gates(), tmp_path)
    assert status == "PASS"
    assert (tmp_path / "card.md").exists()
    assert (tmp_path / "card.json").exists()

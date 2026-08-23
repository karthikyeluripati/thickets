"""Tests for benchmarks/summary.py -- pure Python + filesystem, no GPU/ray/vllm needed."""
import json

from neural_thickets_repro.benchmarks.summary import build_summary_json, build_summary_table, write_summary


def _card(capability, dataset, status, primary_metric=0.6):
    return {
        "capability": capability, "dataset": dataset, "subset_size": 200,
        "base_metrics": {"primary_metric": primary_metric, "parser_failure_rate": 0.01},
        "repeat_metrics": {"primary_metric": primary_metric, "parser_failure_rate": 0.01},
        "integrity": {"n_valid_images": 200, "n_loaded": 200},
        "image_sanity": {"correct_minus_shuffled": 0.2},
        "status": status,
    }


def test_build_summary_table_includes_all_cards():
    cards = [_card("counting", "tallyqa", "PASS"), _card("visual_grounding", "refcoco", "NEEDS_REVIEW")]
    table = build_summary_table(cards)
    assert "counting" in table
    assert "visual_grounding" in table
    assert "PASS" in table
    assert "NEEDS_REVIEW" in table


def test_build_summary_table_handles_missing_repeat_and_sanity():
    card = _card("counting", "tallyqa", "NEEDS_REVIEW")
    card["repeat_metrics"] = None
    card["image_sanity"] = None
    table = build_summary_table([card])
    assert "N/A" in table


def test_build_summary_json_counts_statuses():
    cards = [
        _card("a", "d1", "PASS"), _card("b", "d2", "PASS"),
        _card("c", "d3", "FAIL"), _card("d", "d4", "NEEDS_REVIEW"),
    ]
    summary = build_summary_json(cards)
    assert summary["n_capabilities"] == 4
    assert summary["status_counts"] == {"PASS": 2, "FAIL": 1, "NEEDS_REVIEW": 1}
    assert summary["all_pass"] is False


def test_all_pass_true_only_when_every_card_passes():
    cards = [_card("a", "d1", "PASS"), _card("b", "d2", "PASS")]
    summary = build_summary_json(cards)
    assert summary["all_pass"] is True


def test_all_pass_false_when_no_cards():
    summary = build_summary_json([])
    assert summary["all_pass"] is False
    assert summary["n_capabilities"] == 0


def test_write_summary_reads_cards_from_disk_and_writes_both_files(tmp_path):
    for name, status in [("counting", "PASS"), ("visual_grounding", "NEEDS_REVIEW")]:
        card_dir = tmp_path / name
        card_dir.mkdir()
        (card_dir / "card.json").write_text(json.dumps(_card(name, name, status)))

    summary = write_summary(tmp_path)

    assert (tmp_path / "summary.md").exists()
    assert (tmp_path / "summary.json").exists()
    assert summary["n_capabilities"] == 2
    on_disk = json.loads((tmp_path / "summary.json").read_text())
    assert on_disk["n_capabilities"] == 2


# ---------------------------------------------------------------------------------------
# expected_capabilities / missing-capability robustness (this repair pass) -- a real
# multi-capability run was observed with a capability's row silently absent from summary.md.
# ---------------------------------------------------------------------------------------

def test_build_summary_table_shows_an_explicit_missing_row_when_a_card_is_absent():
    cards = [_card("counting", "tallyqa", "PASS")]
    table = build_summary_table(cards, expected_capabilities=["counting", "visual_grounding"])
    assert "counting" in table
    assert "visual_grounding" in table
    assert "MISSING" in table
    # visual_grounding's row must exist even though no card was provided for it.
    rows = [line for line in table.splitlines() if line.startswith("| visual_grounding")]
    assert len(rows) == 1
    assert "MISSING" in rows[0]


def test_build_summary_table_all_seven_capabilities_appear_when_all_have_cards():
    capabilities = ["visual_grounding", "counting", "spatial_reasoning", "ocr_text_recognition_grounded", "attribute_recognition", "relational_reasoning", "fine_grained_recognition"]
    cards = [_card(c, c, "PASS") for c in capabilities]
    table = build_summary_table(cards, expected_capabilities=capabilities)
    for capability in capabilities:
        assert capability in table
    assert "MISSING" not in table


def test_build_summary_table_without_expected_capabilities_behaves_as_before():
    """Backward compatibility: omitting expected_capabilities falls back to exactly the
    previous behavior (one row per discovered card, in whatever order they're given).
    """
    cards = [_card("counting", "tallyqa", "PASS"), _card("visual_grounding", "refcoco", "NEEDS_REVIEW")]
    table = build_summary_table(cards)
    assert "counting" in table
    assert "visual_grounding" in table
    assert "MISSING" not in table


def test_build_summary_json_reports_missing_capabilities():
    cards = [_card("counting", "tallyqa", "PASS")]
    summary = build_summary_json(cards, expected_capabilities=["counting", "visual_grounding", "attribute_recognition"])
    assert summary["missing_capabilities"] == ["visual_grounding", "attribute_recognition"]
    assert summary["n_expected_capabilities"] == 3
    assert summary["n_capabilities"] == 1
    assert summary["all_pass"] is False  # missing capabilities can never count as all_pass


def test_build_summary_json_no_missing_capabilities_when_all_present():
    cards = [_card("counting", "tallyqa", "PASS"), _card("visual_grounding", "refcoco", "PASS")]
    summary = build_summary_json(cards, expected_capabilities=["counting", "visual_grounding"])
    assert summary["missing_capabilities"] == []
    assert summary["all_pass"] is True


def test_write_summary_passes_through_expected_capabilities(tmp_path):
    card_dir = tmp_path / "counting"
    card_dir.mkdir()
    (card_dir / "card.json").write_text(json.dumps(_card("counting", "tallyqa", "PASS")))

    summary = write_summary(tmp_path, expected_capabilities=["counting", "visual_grounding"])

    assert summary["missing_capabilities"] == ["visual_grounding"]
    table_on_disk = (tmp_path / "summary.md").read_text()
    assert "visual_grounding" in table_on_disk
    assert "MISSING" in table_on_disk

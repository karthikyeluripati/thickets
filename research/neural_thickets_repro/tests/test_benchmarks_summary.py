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

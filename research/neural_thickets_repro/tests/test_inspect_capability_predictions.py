"""Tests for inspect_capability_predictions.py -- pure Python, reads a synthetic JSONL file,
no GPU/ray/vllm/network needed.
"""
import json

import neural_thickets_repro.inspect_capability_predictions as m


def _write_predictions(tmp_path, records):
    path = tmp_path / "predictions.jsonl"
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


def test_load_predictions_reads_all_lines(tmp_path):
    records = [
        {"example_id": "1", "raw_generation": "a"},
        {"example_id": "2", "raw_generation": "b"},
    ]
    path = _write_predictions(tmp_path, records)
    loaded = m.load_predictions(path)
    assert loaded == records


def test_load_predictions_skips_blank_lines(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text('{"example_id": "1"}\n\n{"example_id": "2"}\n')
    loaded = m.load_predictions(path)
    assert len(loaded) == 2


def test_build_report_rows_includes_generic_fields():
    predictions = [{
        "example_id": "1", "query": {"question": "what color?"}, "target": "red",
        "raw_generation": "It is red.", "parsed_prediction": "red",
        "per_example_score": 1.0, "correct": True, "detail": {}, "metadata": {},
    }]
    rows = m.build_report_rows(predictions)
    assert rows == [{
        "example_id": "1", "query": {"question": "what color?"}, "target": "red",
        "raw_generation": "It is red.", "parsed_prediction": "red", "score": 1.0, "correct": True,
    }]


def test_build_report_rows_respects_limit():
    predictions = [{"example_id": str(i)} for i in range(5)]
    rows = m.build_report_rows(predictions, limit=2)
    assert len(rows) == 2


def test_build_report_rows_surfaces_grounding_specific_fields():
    predictions = [{
        "example_id": "1", "detail": {"iou": 0.91, "raw_prediction_box": [112, 189, 444, 362], "coordinate_mode": "pixel_xyxy"},
        "metadata": {"image_width": 640, "image_height": 425},
    }]
    rows = m.build_report_rows(predictions, capability="visual_grounding")
    assert rows[0]["iou"] == 0.91
    assert rows[0]["coordinate_mode"] == "pixel_xyxy"
    assert rows[0]["image_width"] == 640


def test_build_report_rows_surfaces_ocr_grounded_flag():
    predictions = [{"example_id": "1", "detail": {}, "metadata": {"ocr_tokens": ["STOP"], "ocr_grounded": True}}]
    rows = m.build_report_rows(predictions, capability="ocr_text_recognition")
    assert rows[0]["ocr_grounded"] is True


def test_build_report_rows_surfaces_attribute_recognition_fields():
    predictions = [{
        "example_id": "1", "detail": {"matched_attribute": "wooden", "valid_targets": ["wooden", "brown"]},
        "metadata": {"bbox_xywh": [1, 2, 3, 4], "object_id": "22"},
    }]
    rows = m.build_report_rows(predictions, capability="attribute_recognition")
    assert rows[0]["matched_attribute"] == "wooden"
    assert rows[0]["bbox_xywh"] == [1, 2, 3, 4]
    assert rows[0]["object_id"] == "22"


def test_build_report_rows_missing_capability_specific_key_is_silently_skipped():
    predictions = [{"example_id": "1", "detail": {}, "metadata": {}}]  # no "iou" etc.
    rows = m.build_report_rows(predictions, capability="visual_grounding")
    assert "iou" not in rows[0]


def test_build_report_rows_unknown_capability_shows_only_generic_fields():
    predictions = [{"example_id": "1", "detail": {"iou": 0.9}, "metadata": {"foo": "bar"}}]
    rows = m.build_report_rows(predictions, capability="not_a_real_capability")
    assert "iou" not in rows[0]
    assert "foo" not in rows[0]


def test_render_json_is_valid_json():
    rows = [{"example_id": "1", "score": 1.0}]
    parsed = json.loads(m.render_json(rows))
    assert parsed == rows


def test_render_table_includes_example_id_and_fields():
    rows = [{"example_id": "1", "score": 1.0, "correct": True, "raw_generation": "red"}]
    table = m.render_table(rows)
    assert "1" in table
    assert "raw_generation: red" in table


def test_render_table_empty_rows():
    assert m.render_table([]) == "(no predictions)"


def test_main_end_to_end_json_format(tmp_path, capsys):
    records = [{"example_id": "1", "raw_generation": "red", "per_example_score": 1.0, "correct": True, "detail": {}, "metadata": {}}]
    path = _write_predictions(tmp_path, records)

    rc = m.main(["--predictions", str(path), "--format", "json"])

    assert rc == 0
    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert parsed[0]["example_id"] == "1"


def test_main_end_to_end_table_format(tmp_path, capsys):
    records = [{"example_id": "1", "raw_generation": "red", "per_example_score": 1.0, "correct": True, "detail": {}, "metadata": {}}]
    path = _write_predictions(tmp_path, records)

    rc = m.main(["--predictions", str(path)])

    assert rc == 0
    captured = capsys.readouterr()
    assert "raw_generation: red" in captured.out

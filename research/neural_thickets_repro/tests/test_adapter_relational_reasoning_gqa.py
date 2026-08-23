"""Tests for adapters/relational_reasoning_gqa.py -- shares _gqa_filtered_base.py's logic
with spatial_reasoning_gqa.py (thoroughly tested in test_adapter_spatial_reasoning_gqa.py);
this file covers what's specific to the relational variant.
"""
from types import SimpleNamespace

from neural_thickets_repro.benchmarks.adapters.relational_reasoning_gqa import GQARelationalReasoningBenchmark


def test_capability_and_name():
    bench = GQARelationalReasoningBenchmark(question_ids={"1"})
    assert bench.capability == "relational_reasoning"
    assert bench.name == "gqa_testdev_relational"


def test_subset_selection_rule_matches_existing_gqa_pilot_convention():
    bench = GQARelationalReasoningBenchmark(question_ids={"1"})
    assert bench.subset_selection_rule() == "prefix"


def test_known_caveats_documents_explicit_exclusion_choice_not_natural_disjointness():
    caveats = " ".join(GQARelationalReasoningBenchmark(question_ids={"1"}).known_caveats())
    assert "explicit experimental exclusion" in caveats
    assert "not a claim GQA itself provides two naturally disjoint categories" in caveats


def test_load_examples_filters_to_relational_ids_only(fake_gqa_handler_factory, tiny_image_factory, tmp_path):
    image_path = tmp_path / "img.png"
    tiny_image_factory().save(image_path)
    records = [
        {"question_id": "r1", "image_path": str(image_path), "messages": [{"role": "user", "content": []}], "ground_truth": {"answer": "holding"}},
        {"question_id": "s1", "image_path": str(image_path), "messages": [{"role": "user", "content": []}], "ground_truth": {"answer": "left"}},
    ]
    handler = fake_gqa_handler_factory(records)
    bench = GQARelationalReasoningBenchmark(gqa_handler=handler, question_ids={"r1"})
    cfg = SimpleNamespace(dataset=SimpleNamespace(source="testdev.parquet", split="test"))

    examples = bench.load_examples(cfg)

    assert len(examples) == 1
    assert examples[0].example_id == "r1"

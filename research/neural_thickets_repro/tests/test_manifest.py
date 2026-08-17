from neural_thickets_repro.manifest import build_parameter_manifest, summarize


def test_visual_params_marked_not_perturbed(dummy_vlm):
    manifest = build_parameter_manifest(dummy_vlm)
    visual_entries = [m for m in manifest if m["name"].startswith("visual.")]
    assert visual_entries, "fixture should contain visual.* params"
    assert all(not m["perturbed"] for m in visual_entries)


def test_language_model_params_marked_perturbed(dummy_vlm):
    manifest = build_parameter_manifest(dummy_vlm)
    lm_entries = [m for m in manifest if m["name"].startswith("model.")]
    assert lm_entries, "fixture should contain model.* params"
    assert all(m["perturbed"] for m in lm_entries)


def test_manifest_entries_have_expected_fields(dummy_vlm):
    manifest = build_parameter_manifest(dummy_vlm)
    for entry in manifest:
        assert set(entry) == {"name", "module", "shape", "dtype", "num_params", "perturbed"}
        assert entry["num_params"] > 0
        assert isinstance(entry["shape"], list)


def test_summarize_counts_match_manifest(dummy_vlm):
    manifest = build_parameter_manifest(dummy_vlm)
    summary = summarize(manifest)
    assert summary["total_tensors"] == len(manifest)
    assert summary["perturbed_tensors"] + summary["frozen_tensors"] == len(manifest)
    assert summary["frozen_tensors"] == sum(1 for m in manifest if m["name"].startswith("visual."))

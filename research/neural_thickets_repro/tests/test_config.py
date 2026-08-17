from pathlib import Path

import pytest

from neural_thickets_repro.config import UnresolvedFieldError, load_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "gqa_repro.yaml"


def test_load_real_config_parses_expected_values():
    cfg = load_config(CONFIG_PATH)
    assert cfg.model.name == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert cfg.randopt.N == 5000
    assert cfg.randopt.K == 50
    assert cfg.dataset.selection_set_size == 200
    assert cfg.randopt.visual_param_prefixes == ["visual.", "model.visual."]


def test_sigma_is_unresolved_by_default():
    cfg = load_config(CONFIG_PATH)
    assert cfg.randopt.sigmas is None
    assert set(cfg.randopt.sigma_candidates) == {
        "sigma_default", "sigma_example_scripts", "sigma_appendix_e3",
    }


def test_require_resolved_passes_for_resolved_field():
    cfg = load_config(CONFIG_PATH)
    cfg.require_resolved("model.name", "randopt.N")  # should not raise


def test_require_resolved_raises_for_unresolved_field():
    cfg = load_config(CONFIG_PATH)
    with pytest.raises(UnresolvedFieldError):
        cfg.require_resolved("randopt.sigmas")


def test_dataset_provenance_now_resolved():
    """As of the Gate 1 prep pass, dataset.selection_split/test_split/revision were
    resolved by documented reproduction assumption (see REPRO_SPEC.md) -- no longer None.
    """
    cfg = load_config(CONFIG_PATH)
    cfg.require_resolved("dataset.revision", "dataset.selection_split", "dataset.test_split")

from neural_thickets_repro.thicket.seeds import derive_seed


def test_same_inputs_give_same_seed():
    assert derive_seed(42, "a", "b") == derive_seed(42, "a", "b")


def test_different_namespace_parts_give_different_seeds():
    assert derive_seed(42, "a", "b") != derive_seed(42, "a", "c")


def test_different_base_seed_gives_different_seed():
    assert derive_seed(1, "a") != derive_seed(2, "a")


def test_namespace_part_order_matters():
    assert derive_seed(1, "a", "b") != derive_seed(1, "b", "a")


def test_seed_is_non_negative_int():
    seed = derive_seed(123, "x", "y", "z")
    assert isinstance(seed, int)
    assert seed >= 0

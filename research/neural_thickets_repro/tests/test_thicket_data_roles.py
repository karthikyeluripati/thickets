import pytest

from neural_thickets_repro.thicket.data_roles import (
    DataRoleDriftError,
    DataRoleError,
    DataRoleOverlapError,
    load_data_role_manifest,
    partition_data_roles,
    validate_against_pool,
    validate_disjoint,
    write_data_role_manifest,
)


def _ids(n):
    return [f"ex{i}" for i in range(n)]


def test_partition_is_deterministic_given_same_seed():
    ids = _ids(20)
    p1 = partition_data_roles(ids, {"map": 8, "confirm": 4, "select": 4, "test": 4}, seed=1)
    p2 = partition_data_roles(ids, {"map": 8, "confirm": 4, "select": 4, "test": 4}, seed=1)
    assert p1.roles == p2.roles
    assert p1.manifest_hash == p2.manifest_hash


def test_partition_differs_with_different_seed():
    ids = _ids(20)
    p1 = partition_data_roles(ids, {"map": 8, "confirm": 4, "select": 4, "test": 4}, seed=1)
    p2 = partition_data_roles(ids, {"map": 8, "confirm": 4, "select": 4, "test": 4}, seed=2)
    assert p1.roles != p2.roles


def test_roles_are_disjoint_by_construction():
    ids = _ids(20)
    partition = partition_data_roles(ids, {"map": 8, "confirm": 4, "select": 4, "test": 4}, seed=1)
    validate_disjoint(partition)  # must not raise
    seen = set()
    for role_ids in partition.roles.values():
        assert not (seen & set(role_ids))
        seen |= set(role_ids)


def test_sizes_are_respected():
    ids = _ids(20)
    partition = partition_data_roles(ids, {"map": 8, "confirm": 4, "select": 4, "test": 4}, seed=1)
    assert partition.sizes == {"map": 8, "confirm": 4, "select": 4, "test": 4}


def test_missing_role_defaults_to_zero():
    ids = _ids(10)
    partition = partition_data_roles(ids, {"map": 5, "test": 5}, seed=1)
    assert partition.sizes == {"map": 5, "confirm": 0, "select": 0, "test": 5}


def test_overlap_requested_sizes_exceeding_pool_raises():
    ids = _ids(5)
    with pytest.raises(DataRoleError):
        partition_data_roles(ids, {"map": 3, "confirm": 3}, seed=1)


def test_duplicate_ids_in_pool_raises():
    ids = ["a", "a", "b"]
    with pytest.raises(DataRoleError):
        partition_data_roles(ids, {"map": 1}, seed=1)


def test_unknown_role_name_raises():
    ids = _ids(5)
    with pytest.raises(DataRoleError):
        partition_data_roles(ids, {"bogus_role": 1}, seed=1)


def test_validate_disjoint_raises_on_constructed_overlap():
    from neural_thickets_repro.thicket.data_roles import DataRolePartition

    bad = DataRolePartition(roles={"map": ("a", "b"), "confirm": ("b", "c"), "select": (), "test": ()}, sizes={"map": 2, "confirm": 2, "select": 0, "test": 0}, seed=0, manifest_hash="x")
    with pytest.raises(DataRoleOverlapError):
        validate_disjoint(bad)


def test_write_and_load_manifest_round_trips(tmp_path):
    ids = _ids(20)
    partition = partition_data_roles(ids, {"map": 8, "confirm": 4, "select": 4, "test": 4}, seed=1)
    path = tmp_path / "data_roles.json"
    write_data_role_manifest(partition, path)
    loaded = load_data_role_manifest(path)
    assert loaded.roles == partition.roles
    assert loaded.manifest_hash == partition.manifest_hash


def test_validate_against_pool_passes_when_all_ids_present():
    ids = _ids(20)
    partition = partition_data_roles(ids, {"map": 8, "confirm": 4, "select": 4, "test": 4}, seed=1)
    validate_against_pool(partition, ids)  # must not raise


def test_validate_against_pool_raises_on_drift():
    ids = _ids(20)
    partition = partition_data_roles(ids, {"map": 8, "confirm": 4, "select": 4, "test": 4}, seed=1)
    shrunk_pool = ids[:15]
    with pytest.raises(DataRoleDriftError):
        validate_against_pool(partition, shrunk_pool)


def test_no_model_outputs_needed_to_construct_partition():
    """Sanity check on the design constraint itself: partitioning only ever consumes plain
    string IDs, sizes, and a seed -- never a score or a prediction.
    """
    import inspect

    from neural_thickets_repro.thicket import data_roles

    sig = inspect.signature(data_roles.partition_data_roles)
    assert set(sig.parameters) == {"example_ids", "sizes", "seed"}

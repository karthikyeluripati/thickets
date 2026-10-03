import pytest
import torch
from thicket_runtime import CandidateSpec, WeightState
from thicket_runtime.execution import state_hash
from thicket_runtime.upstream import load_worker


def model(dtype=torch.float32):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(12)
        return torch.nn.Sequential(torch.nn.Linear(16, 16, bias=False),
                                   torch.nn.Linear(16, 16, bias=False)).to(dtype).eval()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("sign", [1, -1])
def test_matches_literal_native_dtype_arithmetic(dtype, sign):
    state = WeightState(model(dtype), "snapshot-copy")
    c = CandidateSpec(state.base_id, 4, 0.03, sign=sign)
    expected = {n: p + sign * c.sigma * state.noise(n, p, c) for n, p in state.base.items()}
    state.apply(c)
    assert all(torch.equal(p, expected[n]) for n, p in state.params.items())
    state.reset()
    assert state.drift()["exact_base"]


def test_equal_shape_legacy_noise_and_named_noise():
    state = WeightState(model(), "snapshot-copy")
    items = list(state.params.items())
    legacy = CandidateSpec(state.base_id, 55, 0.1)
    assert torch.equal(state.noise(*items[0], legacy), state.noise(*items[1], legacy))
    named = CandidateSpec(state.base_id, 55, 0.1, rng="named-tensor-v1")
    assert not torch.equal(state.noise(*items[0], named), state.noise(*items[1], named))


def test_base_mismatch_is_rejected_without_mutation():
    state = WeightState(model(), "snapshot-copy")
    with pytest.raises(ValueError, match="different base"):
        state.apply(CandidateSpec("wrong", 42, 0.1))
    assert state.drift()["exact_base"]


def test_legacy_undo_is_not_an_exact_inverse():
    state = WeightState(model(torch.bfloat16), "legacy-add-subtract")
    state.apply(CandidateSpec(state.base_id, 42, 0.03))
    state.reset()
    assert state.drift()["changed_values"] > 0
    state.rebase()
    assert state_hash(state.model) == state.base_id


def test_exception_restores_exact_base_even_in_legacy_mode():
    state = WeightState(model(), "legacy-add-subtract")
    with pytest.raises(RuntimeError, match="rollout failed"):
        with state.using(CandidateSpec(state.base_id, 8, 0.01)):
            raise RuntimeError("rollout failed")
    assert state.drift()["exact_base"] and state.active is None


def test_nested_candidate_rejected():
    state = WeightState(model(), "snapshot-copy")
    c = CandidateSpec(state.base_id, 8, 0.01)
    with state.using(c):
        with pytest.raises(RuntimeError, match="active candidate"):
            state.apply(c)
    with pytest.raises(RuntimeError, match="no active"):
        state.reset()


def test_snapshot_is_order_independent():
    state = WeightState(model(), "snapshot-copy")
    c1, c2 = [CandidateSpec(state.base_id, s, 0.01) for s in (1, 2)]
    with state.using(c1):
        first = state_hash(state.model)
    with state.using(c2):
        pass
    with state.using(c1):
        assert state_hash(state.model) == first


def test_eval_only():
    with pytest.raises(ValueError, match="eval"):
        WeightState(torch.nn.Linear(2, 2), "snapshot-copy")


def test_tied_parameter_is_changed_once():
    m = model()
    m[1].weight = m[0].weight
    state = WeightState(m, "snapshot-copy")
    assert len(state.params) == 1
    with state.using(CandidateSpec(state.base_id, 2, 0.01)):
        assert m[1].weight is m[0].weight
    assert state.drift()["exact_base"]


def test_unrecognized_upstream_source_never_executed(tmp_path):
    (tmp_path / "utils").mkdir()
    (tmp_path / "utils/worker_extn.py").write_text("raise AssertionError('must not execute')")
    with pytest.raises(ValueError, match="mismatch"):
        load_worker(str(tmp_path))


def test_nonpersistent_buffers_are_hashed_and_audited():
    m = model()
    m.register_buffer("counter", torch.tensor([0.0]), persistent=False)
    state = WeightState(m, "snapshot-copy")
    m.counter.add_(1)
    assert state_hash(m) != state.base_id
    assert not state.drift()["exact_base"]
    assert state.drift()["changed_buffer_values"] == 1
    state.rebase()
    assert state_hash(m) == state.base_id


def test_named_rng_does_not_modify_global_rng():
    state = WeightState(model(), "snapshot-copy")
    before = torch.get_rng_state().clone()
    with state.using(CandidateSpec(state.base_id, 5, 0.01, rng="named-tensor-v1")):
        pass
    assert torch.equal(before, torch.get_rng_state())

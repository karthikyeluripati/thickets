import math
import pytest
from thicket_runtime import CandidateSpec

@pytest.mark.parametrize("changes", [
    {"seed": -1}, {"seed": 2**31}, {"seed": True}, {"seed": 3.2},
    {"sigma": -1}, {"sigma": math.nan}, {"sigma": math.inf}, {"sigma": True},
    {"base_id": ""}, {"rng": "unknown"}, {"sign": 0}, {"sign": True},
])
def test_invalid(changes):
    kwargs = dict(base_id="base-a", seed=42, sigma=0.001)
    kwargs.update(changes)
    with pytest.raises(ValueError):
        CandidateSpec(**kwargs)


def test_identity_covers_semantics():
    base = CandidateSpec("base-a", 42, 1)
    assert base.candidate_id == CandidateSpec("base-a", 42, 1.0).candidate_id
    alternatives = [CandidateSpec("base-b", 42, 1), CandidateSpec("base-a", 43, 1),
                    CandidateSpec("base-a", 42, 2), CandidateSpec("base-a", 42, 1, sign=-1),
                    CandidateSpec("base-a", 42, 1, rng="named-tensor-v1")]
    assert all(c.candidate_id != base.candidate_id for c in alternatives)


def test_versioned_tensor_seed_contract():
    legacy = CandidateSpec("base", 42, 0.1)
    assert legacy.tensor_seed("a") == legacy.tensor_seed("b") == 42
    named = CandidateSpec("base", 42, 0.1, rng="named-tensor-v1")
    assert named.tensor_seed("a") != named.tensor_seed("b")
    assert named.tensor_seed("a") == named.tensor_seed("a")

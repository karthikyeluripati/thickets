from dataclasses import asdict

import pytest
import torch

from thicket_runtime.candidate import CandidateSpec
from thicket_runtime.remote_trace import bind_recipes, run_remote_trace, audit_remote_trace
from thicket_runtime.vllm_audit import tensor_difference


def recipes():
    result = []
    for seed in (42, 43):
        c = CandidateSpec("hf-base", seed, 0.001)
        result.append(asdict(c) | {"candidate_id": c.candidate_id})
    return result


class FakeRPC:
    def __init__(self, fail=None):
        self.state = 0
        self.active = None
        self.fail = fail
        self.calls = []

    def rebase(self):
        self.state = 0
        self.active = None
        self.calls.append("rebase")

    def apply(self, candidate, strategy):
        if strategy == "snapshot-copy":
            self.state = 0
        self.active = candidate
        if self.fail == "apply":
            raise RuntimeError("apply failed")

    def restore(self, candidate, strategy):
        self.calls.append("restore")
        self.state = self.state + 1 if strategy == "legacy-add-subtract" else 0
        self.active = None

    def infer(self):
        if self.fail == "infer":
            raise RuntimeError("infer failed")
        return [{"token_ids": [self.active["seed"], self.state], "text": str(self.state),
                 "finish_reason": "stop"}]

    def score(self, output):
        if self.fail == "score":
            raise RuntimeError("score failed")
        return float(output[0]["text"])

    def memory(self, reset=False):
        self.calls.append("memory")
        return {"peak_allocated_bytes": 100}

    def drift(self):
        self.calls.append("drift")
        return {"exact_base": self.state == 0}

    def fingerprint(self):
        return (self.active["seed"], self.state)


def test_native_ids_never_masquerade_as_hf_ids():
    native = bind_recipes(recipes(), "packed-native-base")
    assert native[0]["source_candidate_id"] == recipes()[0]["candidate_id"]
    assert native[0]["candidate_id"] != native[0]["source_candidate_id"]
    assert native[0]["base_id"] == "packed-native-base"
    bad = recipes()
    bad[0]["seed"] += 1
    with pytest.raises(ValueError, match="identity"):
        bind_recipes(bad, "native")


def test_remote_memory_precedes_audit_and_warmups_are_excluded():
    api = FakeRPC()
    result = run_remote_trace(api, bind_recipes(recipes(), "native"), "snapshot-copy",
                              repeats=2, warmup=1)
    assert len(result["rows"]) == 4
    assert all(row["generated_tokens"] == 2 for row in result["rows"])
    for index, call in enumerate(api.calls):
        if call == "drift":
            assert api.calls[index - 1] == "memory"
    assert api.state == 0 and api.active is None
    for row in result["rows"]:
        assert row["total_wall_ms"] >= sum(row[p]["wall_ms"] for p in
                                            ("apply", "inference", "restore", "score"))


@pytest.mark.parametrize("phase", ["apply", "infer", "score"])
def test_remote_exception_rebases(phase):
    api = FakeRPC(fail=phase)
    with pytest.raises(RuntimeError, match="failed"):
        run_remote_trace(api, bind_recipes(recipes(), "native"), "legacy-add-subtract",
                         repeats=1, warmup=0)
    assert api.state == 0 and api.active is None
    if phase == "infer":
        assert "restore" in api.calls


def test_remote_audit_detects_sequential_candidate_and_output_changes():
    api = FakeRPC()
    candidates = bind_recipes(recipes(), "native")
    audit = audit_remote_trace(api, candidates, "legacy-add-subtract")
    assert not audit["all_resets_exact"]
    assert not audit["all_candidate_states_exact"]
    assert not audit["all_outputs_exact"] and not audit["all_rewards_exact"]
    assert audit["rows"][0]["output_exact"]
    assert not audit["rows"][1]["output_exact"]
    assert audit["all_reference_resets_exact"]
    assert audit_remote_trace(api, candidates, "snapshot-copy")["exact_semantics_gate"]
    assert api.state == 0 and api.active is None


def test_restoration_audit_distinguishes_signed_zero_bits():
    diff = tensor_difference(torch.tensor([0.0, -0.0]), torch.tensor([0.0, 0.0]))
    assert diff["changed_values"] == 1
    assert diff["max_abs"] == 0 and diff["finite"]
    assert not tensor_difference(torch.tensor([float("nan")]), torch.tensor([0.0]))["finite"]

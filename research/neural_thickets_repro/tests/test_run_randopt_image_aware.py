"""Tests for the pure-logic pieces of run_randopt_image_aware.py that don't need
GPU/vllm/ray/the external clone: sample_candidates(), the restoration-mode dispatch helpers
(_perturb_call/_restore_call), the spawn-fix regression check (same pattern as
test_eval_base_image_aware_spawn_fix.py), and -- via a fake Ray/engine -- that
restoration_mode changes ONLY which WorkerExtension method gets dispatched, never candidate
selection, scores, or voting.
"""
import importlib
import os
import sys
from types import SimpleNamespace

import pytest

from neural_thickets_repro.ledger import CandidateLedger
import neural_thickets_repro.run_randopt_image_aware as m


def test_sample_candidates_deterministic_given_same_seed():
    a = m.sample_candidates(20, [0.001, 0.002], seed=42)
    b = m.sample_candidates(20, [0.001, 0.002], seed=42)
    assert a == b


def test_sample_candidates_different_seeds_differ():
    a = m.sample_candidates(20, [0.001, 0.002], seed=1)
    b = m.sample_candidates(20, [0.001, 0.002], seed=2)
    assert a != b


def test_sample_candidates_returns_n_candidates():
    candidates = m.sample_candidates(20, [0.001], seed=42)
    assert len(candidates) == 20


def test_sample_candidates_seeds_are_unique():
    candidates = m.sample_candidates(50, [0.001, 0.002, 0.005], seed=42)
    seeds = [c[0] for c in candidates]
    assert len(seeds) == len(set(seeds))


def test_sample_candidates_sigmas_drawn_only_from_given_values():
    sigma_values = [0.001, 0.002, 0.005]
    candidates = m.sample_candidates(50, sigma_values, seed=42)
    sigmas = {c[1] for c in candidates}
    assert sigmas <= set(sigma_values)


def test_sample_candidates_types():
    candidates = m.sample_candidates(5, [0.001], seed=42)
    for seed, sigma in candidates:
        assert isinstance(seed, int)
        assert isinstance(sigma, float)


# --- spawn-fix regression checks (same pattern as eval_base_image_aware.py's) ---


def test_importing_module_forces_spawn_when_unset(monkeypatch):
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    importlib.reload(m)
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"


def test_importing_module_forces_spawn_even_if_already_set_to_fork(monkeypatch):
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "fork")
    importlib.reload(m)
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"


def test_assert_spawn_configured_passes_after_module_import():
    importlib.reload(m)
    m._assert_spawn_configured()  # should not raise


def test_assert_spawn_configured_raises_if_tampered_with_after_import(monkeypatch):
    importlib.reload(m)
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "fork")
    with pytest.raises(RuntimeError, match="spawn"):
        m._assert_spawn_configured()


# --- _perturb_call / _restore_call: per-mode WorkerExtension call shapes ---


def test_perturb_call_released_compat():
    assert m._perturb_call("released_compat", 111, 0.01) == ("perturb_self_weights", (111, 0.01, False))


def test_restore_call_released_compat():
    assert m._restore_call("released_compat", 111, 0.01) == ("restore_self_weights", (111, 0.01, False))


def test_perturb_call_fixed_base():
    assert m._perturb_call("fixed_base", 111, 0.01) == ("apply_perturbation", (111, 0.01))


def test_restore_call_fixed_base_takes_no_args():
    method, args = m._restore_call("fixed_base", 111, 0.01)
    assert method == "reset_to_base_weights"
    assert args == ()


def test_perturb_call_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unknown restoration_mode"):
        m._perturb_call("not_a_real_mode", 111, 0.01)


def test_restore_call_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unknown restoration_mode"):
        m._restore_call("not_a_real_mode", 111, 0.01)


# --- restoration_mode changes ONLY perturb/restore dispatch, never candidate selection,
# scores, or voting -- exercised via a fake Ray/engine so no real ray/vllm/GPU is needed ---


class _FakeCollectiveRpc:
    def __init__(self, calls):
        self._calls = calls

    def remote(self, method, args=()):
        self._calls.append((method, tuple(args)))
        return "ack"


class _FakeGenerate:
    def __init__(self, outputs):
        self._outputs = outputs

    def remote(self, requests, sampling_params, use_tqdm=False):
        return self._outputs


def _fake_engine(calls, texts):
    outputs = [SimpleNamespace(outputs=[SimpleNamespace(text=t)]) for t in texts]
    return SimpleNamespace(collective_rpc=_FakeCollectiveRpc(calls), generate=_FakeGenerate(outputs))


class _FakeHandler:
    """Deterministic stand-in for GQAHandler -- score/vote purely from response text, so any
    difference in results between restoration modes could only come from a difference in
    what generate() returned, never from the handler itself.
    """

    def compute_reward(self, response, ground_truth):
        return 1.0 if response == ground_truth["answer"] else 0.0

    def extract_answer_for_voting(self, text):
        return text

    def is_voted_answer_correct(self, answer, ground_truth):
        return answer == ground_truth["answer"]


@pytest.fixture(autouse=True)
def _fake_ray(monkeypatch):
    """run_sampling_phase/run_ensemble_phase do `import ray` internally; injecting a
    minimal stand-in into sys.modules lets these tests run without real ray installed
    (consistent with this project's local dev environment -- see test_vlm_adapter.py) while
    still exercising the actual dispatch code path (ray.get(engine.collective_rpc.remote(...))).
    """
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(get=lambda x: x))


def test_restoration_mode_only_changes_dispatch_not_sampling_scores(tmp_path):
    candidates = [(111, 0.001), (222, 0.002)]
    selection_datas = [
        {"question_id": "q0", "ground_truth": {"answer": "yes"}},
        {"question_id": "q1", "ground_truth": {"answer": "no"}},
    ]
    selection_requests = ["req0", "req1"]
    handler = _FakeHandler()

    results = {}
    calls_by_mode = {}
    for mode in m.RESTORATION_MODES:
        calls = []
        engine = _fake_engine(calls, texts=["yes", "no"])
        ledger = CandidateLedger(tmp_path / f"ledger_{mode}.jsonl")

        scores = m.run_sampling_phase(
            engine, handler, selection_requests, selection_datas, None, candidates, ledger, mode,
        )
        results[mode] = scores
        calls_by_mode[mode] = calls

    # Scores identical between modes -- same candidates, same (mocked) generation outputs.
    assert results["released_compat"] == results["fixed_base"]

    # The (seed, sigma) pairs actually dispatched are identical between modes -- restoration
    # mode never touches which candidates are sampled or in what order.
    def _perturb_seeds_sigmas(calls):
        return [args[:2] for method, args in calls if method in ("perturb_self_weights", "apply_perturbation")]

    assert _perturb_seeds_sigmas(calls_by_mode["released_compat"]) == candidates
    assert _perturb_seeds_sigmas(calls_by_mode["fixed_base"]) == candidates

    # Only the DISPATCHED METHOD NAMES differ between modes.
    assert {c[0] for c in calls_by_mode["released_compat"]} == {"perturb_self_weights", "restore_self_weights"}
    assert {c[0] for c in calls_by_mode["fixed_base"]} == {"apply_perturbation", "reset_to_base_weights"}

    # fixed_base's restore call takes no args at all, for every candidate.
    assert all(args == () for method, args in calls_by_mode["fixed_base"] if method == "reset_to_base_weights")

    # Each ledger record carries the mode that actually produced its score.
    for mode in m.RESTORATION_MODES:
        ledger = CandidateLedger(tmp_path / f"ledger_{mode}.jsonl")
        records = ledger.load_all()
        assert len(records) == len(candidates)
        assert all(rec.restoration_mode == mode for rec in records.values())


def test_restoration_mode_only_changes_dispatch_not_voting(tmp_path):
    top_k_candidates = [(111, 0.001), (222, 0.002)]
    test_datas = [
        {"question_id": "q0", "ground_truth": {"answer": "yes"}},
        {"question_id": "q1", "ground_truth": {"answer": "no"}},
    ]
    test_requests = ["req0", "req1"]
    handler = _FakeHandler()

    results = {}
    calls_by_mode = {}
    for mode in m.RESTORATION_MODES:
        calls = []
        engine = _fake_engine(calls, texts=["yes", "no"])

        ensemble_results = m.run_ensemble_phase(
            engine, handler, test_requests, test_datas, None, top_k_candidates, mode,
        )
        results[mode] = ensemble_results
        calls_by_mode[mode] = calls

    assert results["released_compat"]["accuracy"] == results["fixed_base"]["accuracy"]
    assert results["released_compat"]["predictions"] == results["fixed_base"]["predictions"]

    assert {c[0] for c in calls_by_mode["released_compat"]} == {"perturb_self_weights", "restore_self_weights"}
    assert {c[0] for c in calls_by_mode["fixed_base"]} == {"apply_perturbation", "reset_to_base_weights"}

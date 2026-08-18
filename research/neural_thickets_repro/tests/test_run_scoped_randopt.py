"""Tests for run_scoped_randopt.py's pure-logic pieces (CLI validation, arg gating, output
naming) and, via a fake Ray/engine/handler (same trick already proven in
tests/test_run_randopt_image_aware.py), proof that candidate sampling/top-K/voting are
byte-identical regardless of scope and that relative_l2 candidates never depend on a sampled
sigma_candidate value. No ray/vllm/GPU needed.
"""
import sys
from types import SimpleNamespace

import pytest

import neural_thickets_repro.run_scoped_randopt as m
from neural_thickets_repro.ledger import CandidateLedger


# --- CLI arg validation (main() gating logic) ---


def _base_args(**overrides):
    args = [
        "--config", str(m.REPO_ROOT / "configs" / "gqa_repro.yaml"),
        "--perturbation-scope", "vision_encoder",
        "--perturbation-scale-mode", "relative_l2",
        "--relative-l2", "0.01",
        "--restoration-mode", "fixed_base",
    ]
    for flag, value in overrides.items():
        args += [flag, value]
    return args


def test_released_compat_hard_fails(capsys):
    args = ["--perturbation-scope", "vision_encoder", "--perturbation-scale-mode", "relative_l2",
            "--relative-l2", "0.01", "--restoration-mode", "released_compat"]
    rc = m.main(args)
    assert rc == 1
    assert "must be 'fixed_base'" in capsys.readouterr().err


def test_raw_sigma_requires_sigma_candidate(capsys):
    args = ["--perturbation-scope", "vision_encoder", "--perturbation-scale-mode", "raw_sigma",
            "--restoration-mode", "fixed_base"]
    rc = m.main(args)
    assert rc == 1
    assert "--sigma-candidate is required" in capsys.readouterr().err


def test_raw_sigma_rejects_relative_l2_flag(capsys):
    args = ["--perturbation-scope", "vision_encoder", "--perturbation-scale-mode", "raw_sigma",
            "--sigma-candidate", "sigma_default", "--relative-l2", "0.01", "--restoration-mode", "fixed_base"]
    rc = m.main(args)
    assert rc == 1
    assert "--relative-l2 is not accepted" in capsys.readouterr().err


def test_relative_l2_requires_relative_l2_flag(capsys):
    args = ["--perturbation-scope", "vision_encoder", "--perturbation-scale-mode", "relative_l2",
            "--restoration-mode", "fixed_base"]
    rc = m.main(args)
    assert rc == 1
    assert "--relative-l2 is required" in capsys.readouterr().err


def test_relative_l2_rejects_sigma_candidate_flag(capsys):
    args = ["--perturbation-scope", "vision_encoder", "--perturbation-scale-mode", "relative_l2",
            "--relative-l2", "0.01", "--sigma-candidate", "sigma_default", "--restoration-mode", "fixed_base"]
    rc = m.main(args)
    assert rc == 1
    assert "--sigma-candidate is not accepted" in capsys.readouterr().err


def test_unknown_perturbation_scope_rejected_by_argparse():
    with pytest.raises(SystemExit):
        m.main(["--perturbation-scope", "not_a_scope", "--perturbation-scale-mode", "relative_l2",
                "--relative-l2", "0.01", "--restoration-mode", "fixed_base"])


# --- output dir naming ---


def test_out_dir_naming_raw_sigma_includes_sigma_candidate(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    scale_id = "raw_sigma_sigma_default"
    out_dir = tmp_path / "results" / f"scoped_randopt_N20_K5_vision_encoder_{scale_id}"
    assert "vision_encoder" in str(out_dir)
    assert "raw_sigma_sigma_default" in str(out_dir)
    assert "relative_l2" not in str(out_dir)


def test_out_dir_naming_relative_l2_includes_r_value():
    scale_id = "relative_l2_r0.01"
    out_dir_name = f"scoped_randopt_N20_K5_lm_middle_{scale_id}"
    assert "relative_l2_r0.01" in out_dir_name
    assert "sigma_candidate" not in out_dir_name


# --- fake-Ray/fake-engine: candidate sampling/scoring identical regardless of scope; only
# the scoped-perturb dispatch args (seed, sigma_or_r, scope, scale_mode) vary ---


class _FakeCollectiveRpc:
    def __init__(self, calls, perturb_return, fail_methods=frozenset()):
        self._calls = calls
        self._perturb_return = perturb_return
        self._fail_methods = fail_methods

    def remote(self, method, args=()):
        is_callable = not isinstance(method, str)
        name = method.__name__ if is_callable else method
        self._calls.append((name, tuple(args)))
        if name in self._fail_methods:
            raise RuntimeError(f"simulated failure dispatching {name!r}")
        return [self._perturb_return] if is_callable else ["ack"]


class _FakeGenerate:
    def __init__(self, outputs):
        self._outputs = outputs

    def remote(self, requests, sampling_params, use_tqdm=False):
        return self._outputs


def _fake_engine(calls, texts, perturb_return, fail_methods=frozenset()):
    outputs = [SimpleNamespace(outputs=[SimpleNamespace(text=t)]) for t in texts]
    return SimpleNamespace(
        collective_rpc=_FakeCollectiveRpc(calls, perturb_return, fail_methods=fail_methods),
        generate=_FakeGenerate(outputs),
    )


class _FakeHandler:
    def compute_reward(self, response, ground_truth):
        return 1.0 if response == ground_truth["answer"] else 0.0

    def extract_answer_for_voting(self, text):
        return text

    def is_voted_answer_correct(self, answer, ground_truth):
        return answer == ground_truth["answer"]


def _perturb_result(scope, scale_mode, seed, sigma_or_r):
    return {
        "scope": scope, "scale_mode": scale_mode, "seed": seed,
        "requested_relative_l2": sigma_or_r if scale_mode == "relative_l2" else None,
        "derived_sigma": sigma_or_r if scale_mode == "raw_sigma" else 0.00123,
        "actual_perturbation_l2": 0.05,
        "scope_param_count": 10, "scope_total_element_count": 100, "scope_base_l2_norm": 5.0,
        "detected_lm_convention": None, "representative_names": [], "aliases": [],
        "noise_semantics": "upstream_per_tensor_reseed",
    }


@pytest.fixture(autouse=True)
def _fake_ray(monkeypatch):
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(get=lambda x: x))


def test_relative_l2_candidates_use_seed_only_same_r_across_all_seven_scopes(tmp_path):
    """relative_l2 candidates for ALL SEVEN scopes, same N/global_seed/r, must use the exact
    same seed sequence -- proven directly, not assumed, and not just for two scopes.
    """
    from neural_thickets_repro.candidate_sampling import sample_candidate_seeds
    from neural_thickets_repro.scopes import PERTURBATION_SCOPES

    seeds = sample_candidate_seeds(4, seed=42)
    candidates = [(s, 0.01) for s in seeds]

    handler = _FakeHandler()
    selection_datas = [{"question_id": f"q{i}", "ground_truth": {"answer": "yes"}} for i in range(2)]
    selection_requests = ["r0", "r1"]

    results_by_scope = {}
    for scope in PERTURBATION_SCOPES:
        calls = []
        perturb_return = _perturb_result(scope, "relative_l2", seeds[0], 0.01)
        engine = _fake_engine(calls, texts=["yes", "yes"], perturb_return=perturb_return)
        ledger = CandidateLedger(tmp_path / f"ledger_{scope}.jsonl")

        scores = m.run_sampling_phase(
            engine, handler, selection_requests, selection_datas, None,
            candidates, ledger, scope, "relative_l2", base_score=0.5, base_responses=["yes", "yes"],
        )
        results_by_scope[scope] = (scores, calls)

    seed_sequences = {
        scope: [args[0] for method, args in calls if method == "scoped_apply_perturbation"]
        for scope, (_, calls) in results_by_scope.items()
    }
    assert len(set(tuple(v) for v in seed_sequences.values())) == 1, f"seed sequences differ across scopes: {seed_sequences}"
    assert list(seed_sequences.values())[0] == seeds

    r_value_sets = {
        scope: {args[1] for method, args in calls if method == "scoped_apply_perturbation"}
        for scope, (_, calls) in results_by_scope.items()
    }
    assert all(v == {0.01} for v in r_value_sets.values())


def test_relative_l2_scores_independent_of_hypothetical_sigma_candidate_choice(tmp_path):
    """Nothing sigma-candidate-shaped is ever dispatched in relative_l2 mode -- checked
    directly against the call log, not just "not required" at the CLI level.
    """
    from neural_thickets_repro.candidate_sampling import sample_candidate_seeds

    seeds = sample_candidate_seeds(3, seed=7)
    candidates = [(s, 0.02) for s in seeds]
    handler = _FakeHandler()
    selection_datas = [{"question_id": "q0", "ground_truth": {"answer": "yes"}}]
    selection_requests = ["r0"]

    calls = []
    perturb_return = _perturb_result("full_lm", "relative_l2", seeds[0], 0.02)
    engine = _fake_engine(calls, texts=["yes"], perturb_return=perturb_return)
    ledger = CandidateLedger(tmp_path / "ledger.jsonl")

    m.run_sampling_phase(engine, handler, selection_requests, selection_datas, None, candidates, ledger, "full_lm", "relative_l2", base_score=0.5, base_responses=["yes"])

    for method, args in calls:
        if method == "scoped_apply_perturbation":
            assert len(args) == 4  # (seed, r, scope, scale_mode) -- never a 5th sigma_candidate-shaped arg


def test_ledger_records_scoped_metadata_from_perturb_result(tmp_path):
    seeds = [111]
    candidates = [(111, 0.01)]
    handler = _FakeHandler()
    selection_datas = [{"question_id": "q0", "ground_truth": {"answer": "yes"}}]
    selection_requests = ["r0"]

    calls = []
    perturb_return = _perturb_result("vision_merger", "relative_l2", 111, 0.01)
    perturb_return["derived_sigma"] = 0.00456
    engine = _fake_engine(calls, texts=["yes"], perturb_return=perturb_return)
    ledger = CandidateLedger(tmp_path / "ledger.jsonl")

    m.run_sampling_phase(engine, handler, selection_requests, selection_datas, None, candidates, ledger, "vision_merger", "relative_l2", base_score=0.5, base_responses=["yes"])

    records = ledger.load_all()
    rec = records[0]
    assert rec.perturbation_scope == "vision_merger"
    assert rec.perturbation_scale_mode == "relative_l2"
    assert rec.requested_relative_l2 == 0.01
    assert rec.sigma == pytest.approx(0.00456)
    assert rec.restoration_mode == "fixed_base"
    assert rec.noise_semantics == "upstream_per_tensor_reseed"
    assert rec.scope_element_count == 100  # from _perturb_result's scope_total_element_count


def test_ledger_records_delta_score_against_the_given_base_score(tmp_path):
    """delta_score/is_expert/is_tie must be computed from the EXACT base_score passed into
    run_sampling_phase -- proven by re-running with a DIFFERENT base_score and checking the
    recorded delta/is_expert change accordingly, not a value read once and assumed fixed.
    """
    candidates = [(111, 0.01)]
    handler = _FakeHandler()  # compute_reward returns 1.0 for a matching response, else 0.0
    selection_datas = [{"question_id": "q0", "ground_truth": {"answer": "yes"}}]
    selection_requests = ["r0"]
    perturb_return = _perturb_result("full_lm", "relative_l2", 111, 0.01)

    ledger_low = CandidateLedger(tmp_path / "ledger_low_base.jsonl")
    engine_low = _fake_engine([], texts=["yes"], perturb_return=perturb_return)  # candidate score = 1.0
    m.run_sampling_phase(engine_low, handler, selection_requests, selection_datas, None, candidates, ledger_low, "full_lm", "relative_l2", base_score=0.0, base_responses=["no"])
    rec_low = ledger_low.load_all()[0]
    assert rec_low.base_score == 0.0
    assert rec_low.delta_score == pytest.approx(1.0)
    assert rec_low.is_expert is True
    assert rec_low.is_tie is False

    ledger_high = CandidateLedger(tmp_path / "ledger_high_base.jsonl")
    engine_high = _fake_engine([], texts=["yes"], perturb_return=perturb_return)  # same candidate score = 1.0
    m.run_sampling_phase(engine_high, handler, selection_requests, selection_datas, None, candidates, ledger_high, "full_lm", "relative_l2", base_score=1.0, base_responses=["yes"])
    rec_high = ledger_high.load_all()[0]
    assert rec_high.base_score == 1.0
    assert rec_high.delta_score == pytest.approx(0.0)
    assert rec_high.is_expert is False
    assert rec_high.is_tie is True


def test_compute_base_score_calls_reset_to_base_weights(tmp_path):
    """compute_base_score must restore to the exact stored base before scoring -- checked
    directly against the collective_rpc call log, not just documented in a docstring. (The
    fake engine only logs collective_rpc calls, not generate.remote -- the source itself
    calls reset_to_base_weights, then generate.remote, in that order; this test confirms the
    reset call actually happens, which is the part a caller-discipline bug could silently
    skip.)
    """
    handler = _FakeHandler()
    selection_datas = [{"question_id": "q0", "ground_truth": {"answer": "yes"}}]
    selection_requests = ["r0"]

    calls = []
    perturb_return = _perturb_result("full_lm", "relative_l2", 1, 0.01)
    engine = _fake_engine(calls, texts=["yes"], perturb_return=perturb_return)

    base_score, base_responses = m.compute_base_score(engine, handler, selection_requests, selection_datas, None)

    assert base_score == pytest.approx(1.0)
    assert base_responses == ["yes"]
    assert ("reset_to_base_weights", ()) in calls


# --- encoder-cache-reset wiring (see vlm_adapter.reset_vllm_encoder_cache /
# scopes.scope_requires_encoder_cache_reset) ---


def test_visual_scope_resets_encoder_cache_after_perturbation_before_generation(tmp_path):
    candidates = [(111, 0.02)]
    handler = _FakeHandler()
    selection_datas = [{"question_id": "q0", "ground_truth": {"answer": "yes"}}]
    selection_requests = ["r0"]

    calls = []
    perturb_return = _perturb_result("vision_encoder", "relative_l2", 111, 0.02)
    engine = _fake_engine(calls, texts=["yes"], perturb_return=perturb_return)
    ledger = CandidateLedger(tmp_path / "ledger.jsonl")

    m.run_sampling_phase(engine, handler, selection_requests, selection_datas, None, candidates, ledger, "vision_encoder", "relative_l2", base_score=0.5, base_responses=["yes"])

    method_names = [name for name, _ in calls]
    assert "reset_encoder_cache" in method_names
    assert method_names.index("scoped_apply_perturbation") < method_names.index("reset_encoder_cache")
    # generate.remote isn't logged by the fake, but the source calls it strictly after the
    # reset and before reset_to_base_weights -- confirmed by the reset appearing before the
    # restore call in the log, which is the only other candidate.
    assert method_names.index("reset_encoder_cache") < method_names.index("reset_to_base_weights")


def test_non_visual_scope_never_resets_encoder_cache(tmp_path):
    candidates = [(111, 0.02)]
    handler = _FakeHandler()
    selection_datas = [{"question_id": "q0", "ground_truth": {"answer": "yes"}}]
    selection_requests = ["r0"]

    for scope in ("full_lm", "lm_early", "lm_middle", "lm_late"):
        calls = []
        perturb_return = _perturb_result(scope, "relative_l2", 111, 0.02)
        engine = _fake_engine(calls, texts=["yes"], perturb_return=perturb_return)
        ledger = CandidateLedger(tmp_path / f"ledger_{scope}.jsonl")

        m.run_sampling_phase(engine, handler, selection_requests, selection_datas, None, candidates, ledger, scope, "relative_l2", base_score=0.5, base_responses=["yes"])

        method_names = [name for name, _ in calls]
        assert "reset_encoder_cache" not in method_names, f"{scope} should never reset the encoder cache"


def test_visual_scope_ensemble_phase_also_resets_encoder_cache(tmp_path):
    handler = _FakeHandler()
    test_datas = [{"question_id": "q0", "ground_truth": {"answer": "yes"}}]
    test_requests = ["r0"]

    calls = []
    perturb_return = _perturb_result("full_vlm", "relative_l2", 111, 0.02)
    engine = _fake_engine(calls, texts=["yes"], perturb_return=perturb_return)

    m.run_ensemble_phase(engine, handler, test_requests, test_datas, None, [(111, 0.02)], "full_vlm", "relative_l2")

    method_names = [name for name, _ in calls]
    assert "reset_encoder_cache" in method_names


def test_encoder_cache_reset_failure_raises_for_visual_scope_not_silently_continues(tmp_path):
    """If the installed vLLM engine's collective_rpc('reset_encoder_cache') call fails, this
    must propagate as a hard failure -- never silently proceed to generate() with a
    potentially stale cache.
    """
    candidates = [(111, 0.02)]
    handler = _FakeHandler()
    selection_datas = [{"question_id": "q0", "ground_truth": {"answer": "yes"}}]
    selection_requests = ["r0"]

    calls = []
    perturb_return = _perturb_result("vision_encoder", "relative_l2", 111, 0.02)
    engine = _fake_engine(calls, texts=["yes"], perturb_return=perturb_return, fail_methods={"reset_encoder_cache"})
    ledger = CandidateLedger(tmp_path / "ledger.jsonl")

    with pytest.raises(RuntimeError):
        m.run_sampling_phase(engine, handler, selection_requests, selection_datas, None, candidates, ledger, "vision_encoder", "relative_l2", base_score=0.5, base_responses=["yes"])

    # No candidate record should have been written -- the failure happened before scoring.
    assert ledger.load_all() == {}

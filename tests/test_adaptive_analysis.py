import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location("adaptive_analysis", Path(__file__).resolve().parents[1] / "scripts/analyze_adaptive_evaluation.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def entry(cost, recall, regret, success):
    return {"pair_fraction": {"mean": cost}, "topk": {"10": {"recall": {"mean": recall}}},
            "best_regret": {"mean": regret}, "joint_success_fraction": success}


def test_development_policy_selection_excludes_prefix_and_prefers_qualified_cost():
    protocol = {"feasibility_gate": {"minimum_joint_order_success_fraction": .9, "maximum_pair_fraction": .4}}
    table = {"prefix-004": entry(.02, 1, 0, 1), "subset-080": entry(.4, .95, 0, 1),
             "halving-016": entry(.3, .91, .001, .95)}
    chosen, qualified, names = module.choose_policy(table, protocol)
    assert chosen == "halving-016" and qualified and "prefix-004" not in names


def test_unqualified_development_fallback_cannot_masquerade_as_a_pass():
    protocol = {"feasibility_gate": {"minimum_joint_order_success_fraction": .9, "maximum_pair_fraction": .4}}
    table = {"subset-080": entry(.4, .7, .01, .1), "subset-100": entry(.5, .9, 0, 0),
             "halving-004": entry(.1, .2, .05, 0)}
    chosen, qualified, names = module.choose_policy(table, protocol)
    assert chosen == "subset-080" and not qualified and names == []


def test_signal_gate_uses_development_budgets_at_most_64():
    protocol = {"prompt_budgets": [4, 64, 80], "phase4_signal_gate": {
        "maximum_prompts": 64, "minimum_mean_spearman": .5, "maximum_mean_top10_miss_outside_top50": .1}}
    def point(rho, miss):
        return {"spearman": {"mean": rho}, "topk": {"10": {"miss_outside_top5k": {"mean": miss}}}}
    summary = {"subset-004": point(.2, .5), "subset-064": point(.6, .11), "subset-080": point(.8, .01)}
    assert not module.phase4_signal(summary, protocol)["pass"]
    summary["subset-064"] = point(.6, .09)
    assert module.phase4_signal(summary, protocol)["pass"]

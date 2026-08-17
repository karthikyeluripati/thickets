"""Unit tests for the vLLM-version comparison's statistics (pure math, no GPU/vllm needed).
compare()/main() themselves need load_gqa_handler() (the external clone), so only
mcnemar_exact_pvalue() -- the actual decision-rule logic -- is tested directly here.
"""
import math

from neural_thickets_repro.diagnostics.vllm_version_control.compare_results import (
    mcnemar_exact_pvalue,
)


def test_mcnemar_no_discordant_pairs_is_not_significant():
    assert mcnemar_exact_pvalue(0, 0) == 1.0


def test_mcnemar_symmetric_discordance_is_not_significant():
    # equal numbers of improvements and regressions -- no real effect
    assert mcnemar_exact_pvalue(10, 10) > 0.05


def test_mcnemar_strong_one_sided_effect_is_significant():
    # 30 examples flip wrong->correct, only 2 the other way -- should be very significant
    p = mcnemar_exact_pvalue(30, 2)
    assert p < 0.001


def test_mcnemar_matches_known_reference_value():
    # classic textbook McNemar example: b=21, c=2 (approx chi-sq statistic ~14.7 -> p<0.001
    # under the chi-square approximation; exact binomial sign test should agree in direction
    # and significance even if the exact p differs slightly from the chi-sq approximation)
    p = mcnemar_exact_pvalue(21, 2)
    assert p < 0.001


def test_mcnemar_borderline_case_not_significant_with_small_n():
    # 4 vs 1 discordant -- too few observations to reach p<0.05 even though all-one-direction
    p = mcnemar_exact_pvalue(4, 1)
    assert p > 0.05


def test_mcnemar_is_symmetric_in_direction():
    # p-value should be the same magnitude regardless of which side "improved"
    assert math.isclose(mcnemar_exact_pvalue(15, 3), mcnemar_exact_pvalue(3, 15))

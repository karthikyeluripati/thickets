from neural_thickets_repro.topk_voting import majority_vote, select_top_k


def test_select_top_k_basic():
    scores = {(1, 0.01): 0.9, (2, 0.01): 0.5, (3, 0.01): 0.7, (4, 0.01): 0.95}
    assert select_top_k(scores, k=2) == [(4, 0.01), (1, 0.01)]


def test_select_top_k_more_than_available_returns_all_sorted():
    scores = {(1, 0.01): 0.2, (2, 0.01): 0.8}
    assert select_top_k(scores, k=10) == [(2, 0.01), (1, 0.01)]


def test_select_top_k_ties_broken_by_insertion_order():
    scores = {(1, 0.01): 0.5, (2, 0.01): 0.5, (3, 0.01): 0.9}
    # (1, ...) was inserted before (2, ...); Python's stable sort preserves that on ties.
    assert select_top_k(scores, k=2) == [(3, 0.01), (1, 0.01)]


def test_majority_vote_simple_majority():
    predictions = [
        ["cat", "dog"],
        ["cat", "cat"],
        ["dog", "cat"],
    ]
    assert majority_vote(predictions, example_idx=0) == "cat"
    assert majority_vote(predictions, example_idx=1) == "cat"


def test_majority_vote_ignores_empty_predictions():
    predictions = [
        [""],
        ["dog"],
        [""],
    ]
    assert majority_vote(predictions, example_idx=0) == "dog"


def test_majority_vote_all_empty_returns_empty_string():
    predictions = [[""], [""]]
    assert majority_vote(predictions, example_idx=0) == ""


def test_majority_vote_tie_favors_higher_scoring_model():
    # models ordered highest-selection-score first; model 0 says "cat", model 1 says "dog" -- tie.
    predictions = [
        ["cat"],
        ["dog"],
    ]
    assert majority_vote(predictions, example_idx=0) == "cat"

from collections import Counter
import random

from thicket_runtime import perspective as pt
from thicket_runtime.visual_runtime import read


def upstream_vote(answers_by_model, gold, k):
    """Literal transcription of RandOpt randopt.py run_ensemble_evaluation voting."""
    correct = 0
    for idx, g in enumerate(gold):
        answers = [answers_by_model[m][idx] for m in range(k) if answers_by_model[m][idx]]
        if answers:
            final = Counter(answers).most_common(1)[0][0]
            correct += final == g
    return correct


def test_majority_matches_upstream_randopt_voting():
    rng = random.Random(0)
    for trial in range(300):
        n, m = 30, rng.choice([1, 2, 5, 10, 25, 50])
        gold = [rng.choice('ABCD') for _ in range(n)]
        answers = [[rng.choice(['A', 'B', 'C', 'D', 'A', '']) for _ in range(n)] for _ in range(m)]
        rows = [{'uid': str(i), 'answer': ord(g) - 65} for i, g in enumerate(gold)]
        outs = [[{'prediction': x, 'valid_letter': x != ''} for x in model] for model in answers]
        for k in range(1, m + 1):
            ours = pt.majority(outs, rows, k)
            assert sum(v['correct'] for v in ours) == upstream_vote(answers, gold, k)
            for i in range(n):
                valid = [answers[j][i] for j in range(k) if answers[j][i]]
                assert ours[i]['prediction'] == (Counter(valid).most_common(1)[0][0] if valid else '')


def test_tie_goes_to_first_ranked_answer_and_empty_is_wrong():
    rows = [{'uid': '0', 'answer': 1}, {'uid': '1', 'answer': 0}]
    o = lambda *p: [{'prediction': x, 'valid_letter': x != ''} for x in p]
    votes = pt.majority([o('B', ''), o('C', ''), o('C', ''), o('B', '')], rows, 4)
    assert votes[0]['prediction'] == 'B' and votes[0]['correct']   # 2-2 tie: B seen first
    assert votes[1]['prediction'] == '' and not votes[1]['correct']  # all invalid -> incorrect


def test_committee_is_search_prefix():
    c = read('results/perspective-taking-n5000-20261004/locks/committee.json')
    s = read('results/perspective-taking-n5000-20261004/locks/search.json')
    assert [r['candidate']['candidate_id'] for r in c['committee']] == s['ranked_ids'][:50]
    assert [r['search_rank'] for r in c['committee']] == list(range(1, 51))
    assert len(c['to_generate']) + len(c['existing_test_outputs']['candidate_ids']) == 50

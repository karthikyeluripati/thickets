from copy import deepcopy
import runpy
import sys

import numpy as np
import pytest

from thicket_runtime import omnispatial as om
from thicket_runtime.omnispatial_data import build_prompt, official_prompts, score
from thicket_runtime.visual_final import rank_hash
from thicket_runtime.visual_runtime import read

PATH = 'experiments/omnispatial_visual_expert_protocol.json'
P = read(PATH)


def test_frozen_benchmark_population_and_prompt():
    assert om.verify_inputs(P)['files_verified'] == 5
    sets = om.splits(P)
    assert {s: len(r) for s, r in sets.items()} == {'search': 705, 'validation': 475, 'test': 252}
    assert all(r['task_type'] == 'Complex_Logic' for rows in sets.values() for r in rows)
    assert {r['source_split'] for r in sets['search'] + sets['validation']} == {'train'}
    assert {r['source_split'] for r in sets['test']} == {'test'}
    assert {r['sub_task_type'] for r in sets['test']} == {'Geometric_Reasoning', 'Pattern_Recognition'}
    for s in ('search', 'validation'):  # stratified answer letters stay within one item of 60/40
        for letter in range(4):
            n = sum(r['answer'] == letter for r in sets['search'] + sets['validation'])
            share = sum(r['answer'] == letter for r in sets['search']) / n
            assert abs(share - .6) < .02
    p = official_prompts()
    r = sets['search'][0]
    assert build_prompt(r).startswith(p.DEFAULT_SYSTEM_PROMPT + '\n' + p.DIRECT_FORMAT + '\n\n' + r['question'])
    assert build_prompt(r).endswith('\nD. ' + r['options'][3])
    assert P['model_revision'] == 'cc594898137f460bfe9f0759e9844b3ce807cfb5' and P['tensor_parallel_size'] == 1


def test_official_direct_scorer_has_no_fallback_letter():
    assert score(' c', 2) == {'prediction': 'C', 'correct': True, 'valid_letter': True}
    assert score('', 0) == {'prediction': '', 'correct': False, 'valid_letter': False}
    assert score('Answer: A', 0)['correct'] is True  # first character 'A' — official behavior
    assert score('The answer is A', 0)['correct'] is False


def test_population_is_frozen():
    plan = om.candidate_plan(P)
    assert len({s for s, _ in plan}) == 400 and [s for _, s in plan].count(.001) == 100
    assert not {s for s, _ in plan} & set(P['previous_candidate_seeds'])
    bad = deepcopy(P); bad['candidates']['seed_start'] = 7300000
    with pytest.raises(ValueError, match='reused'): om.verify_inputs(bad)
    bad = deepcopy(P); bad['candidates']['sigma_mixture'] = [.001] * 4
    with pytest.raises(ValueError): om.candidate_plan(bad)
    assert sum(len(om.shard_plan(P, k)) for k in range(8)) == 400


def records(scores):
    return [{'candidate': {'seed': 9100000 + i, 'sigma': .001, 'candidate_id': f'{i:04d}'}, 'candidate_state_id': f's{i}',
             'split': 'search', 'examples': 705, 'correct_count': c} for i, c in enumerate(scores)]


def test_search_and_validation_ranking():
    scores = [200] * 400; scores[7] = 260; scores[3] = 250; scores[9] = 250
    ranked = om.rank_search(records(scores), P)
    assert ranked[0]['candidate']['candidate_id'] == '0007'
    assert [r['candidate']['candidate_id'] for r in ranked[1:3]] == sorted(['0003', '0009'], key=rank_hash)
    with pytest.raises(ValueError): om.rank_search(records(scores)[:399], P)
    top = ranked[:30]
    correct = {r['candidate']['candidate_id']: 100 for r in top}
    correct[top[5]['candidate']['candidate_id']] = 120
    order = om.rank_validation(top, correct)
    assert order[0] is top[5] and order[1] is top[0]


def outs(rows, correct, pred=None):
    return [{'uid': r['uid'], 'correct': bool(c), 'valid_letter': True,
             'prediction': (pred or chr(65 + r['answer'])) if not c else chr(65 + r['answer'])} for r, c in zip(rows, correct)]


def test_label_bias_audit_and_expert_rule():
    rows = om.splits(P)['test']
    base = outs(rows, [False] * len(rows), pred='A')
    # gains only on true letter B -> 100% share, fails the spread rule
    cand = outs(rows, [r['answer'] == 1 for r in rows], pred='A')
    audit = om.label_audit(base, cand, rows, .8)
    assert audit['largest_positive_gain_share'] == 1.0 and not audit['bias_spread_pass']
    # gains spread across letters pass
    spread = outs(rows, [i % 5 == 0 for i in range(len(rows))], pred='A')
    assert om.label_audit(base, spread, rows, .8)['bias_spread_pass']
    idx = np.random.default_rng(0).integers(0, len(rows), size=(2000, len(rows)))
    res = om.compare(base, spread, rows, idx, .8)
    v = om.expert(res, P, True)
    assert v['checks']['gain_at_least_5pp'] and v['checks']['bootstrap_lower_positive']
    assert v['checks']['positive_in_two_sub_tasks'] == all(g > 0 for g in res['sub_task_gain'].values())
    weak = dict(res, gain=.049)
    assert not om.expert(weak, P, True)['transferable_expert']
    d = om.decision([res] * 5, P, True)
    assert d['decision'] == ('GO_OMNISPATIAL_VISUAL_EXPERT' if v['transferable_expert'] else 'NO_GO_OMNISPATIAL_VISUAL_EXPERT')
    with pytest.raises(ValueError, match='operational'): om.decision([res] * 5, P, False)
    with pytest.raises(ValueError, match='operational'): om.decision([res] * 4, P, True)


def test_search_and_validation_lock_scripts(tmp_path, monkeypatch):
    import thicket_runtime.committee_validation as cv
    import thicket_runtime.omnispatial_audit as audit
    from thicket_runtime.visual_runtime import write
    monkeypatch.setattr(cv, 'committed', lambda path: None)
    sets = om.splits(P)
    rng = np.random.default_rng(1)
    root = tmp_path / 'study'
    (root / 'locks').mkdir(parents=True)
    base = {s: {'outputs': outs(rows, rng.random(len(rows)) < .3, pred='A')} for s, rows in sets.items()}
    write(root / 'baseline/base.json.gz', base)
    write(root / 'baseline/sha256.json', {'base.json.gz': __import__('thicket_runtime.line_tracing', fromlist=['sha']).sha(root / 'baseline/base.json.gz')})
    write(root / 'locks/baseline.json', {'protocol_sha256': '', 'base_id': 'b', 'output_identities': {},
          'summary': {s: om.summarize(v['outputs'], sets[s]) for s, v in base.items()}, 'preprocess': {}})
    def fake(run, protocol, path, locks, shard=None, base_reference=None):
        from pathlib import Path
        run = Path(run); run.mkdir(parents=True, exist_ok=True)
        for f in ('run-manifest.json', 'sha256.json'): (run / f).write_text('{}')
        if run.name.startswith('search'):
            recs, traces = [], {}
            for seed, sigma in om.shard_plan(P, shard):
                cid = f'c{seed}'
                o = outs(sets['search'], rng.random(705) < .3 + sigma * 10, pred='B')
                recs.append({'candidate': {'seed': seed, 'sigma': sigma, 'candidate_id': cid}, 'candidate_state_id': 's' + cid,
                             'split': 'search', 'examples': 705, 'correct_count': sum(x['correct'] for x in o)})
                traces[cid] = {'splits': {'search': {'outputs': o}}}
            return {'manifest': {'phase': 'search'}, 'records': recs, 'traces': traces}
        traces = {r['candidate']['candidate_id']: {'splits': {'validation': {'outputs': outs(sets['validation'], rng.random(475) < .32, pred='C')}}}
                  for r in locks['search']['top30']}
        return {'manifest': {'phase': 'validation'}, 'records': [], 'traces': traces}
    monkeypatch.setattr(audit, 'phase', fake)
    for lock in ('search', 'validation'):
        monkeypatch.setattr(sys, 'argv', ['x', lock, '--root', str(root), '--protocol', PATH])
        runpy.run_path('scripts/freeze_omnispatial_lock.py', run_name='__main__')
    s, v = read(root / 'locks/search.json'), read(root / 'locks/validation.json')
    assert len(s['records']) == 400 and s['top30'] == om.rank_search(s['records'], P)[:30]
    assert set(v['candidates']) == {r['candidate']['candidate_id'] for r in s['top30']}
    assert v['rank1'] == v['top5'][0] and len(v['top5']) == 5
    order = om.rank_validation(s['top30'], {c: x['summary']['correct_count'] for c, x in v['candidates'].items()})
    assert v['top5'] == order[:5]
    assert set(v['secondary_top30_counts']) == {'strictly_above_base', 'at_least_1pp', 'at_least_3pp', 'at_least_5pp'}

from copy import deepcopy
import runpy
import sys

import numpy as np
import pytest

from thicket_runtime import transfer_aware as ta
from thicket_runtime.transfer_aware_data import FOLDS, slots
from thicket_runtime.visual_final import rank_hash
from thicket_runtime.visual_runtime import read

PATH = 'experiments/transfer_aware_visual_protocol.json'
P = read(PATH)
BASE = {'pooled_accuracy': .30, 'fold_accuracy': {'A': .30, 'B': .30, 'C': .30}}


def rec(i, fold_correct, sigma=.0005):
    return {'candidate': {'seed': 7300000 + i, 'sigma': sigma, 'candidate_id': f'{i:04d}'},
            'candidate_state_id': f's{i}',
            'search': {'pooled_correct': sum(fold_correct), 'pooled_gain': sum(fold_correct) / 300 - .3,
                       'transfer_score': min(c / 100 - .3 for c in fold_correct),
                       'mean_fold_gain': sum(c / 100 - .3 for c in fold_correct) / 3}}


def population():
    return [rec(i, (30, 30, 30)) for i in range(400)]


def test_frozen_inputs_dataset_and_population():
    proof = ta.verify_inputs(P)
    assert proof['files_verified'] == 1606
    sets = ta.splits(P)
    assert {s: len(r) for s, r in sets.items()} == {'search': 300, 'validation': 300, 'test': 1000}
    for s, rows in sets.items():
        labels = [r['answer'] for r in rows]
        assert {labels.count(x) for x in '1234'} == {len(rows) // 4}
        assert max(sum(r['difficulty'] == l for r in rows) for l in ('easy', 'medium', 'hard')) - \
               min(sum(r['difficulty'] == l for r in rows) for l in ('easy', 'medium', 'hard')) <= 1
    index = ta.folds(P, sets['search'])
    assert sorted(i for f in FOLDS for i in index[f]) == list(range(300))
    for f in FOLDS:
        assert {[sets['search'][i]['answer'] for i in index[f]].count(x) for x in '1234'} == {25}
    plan = ta.candidate_plan(P)
    assert [s for _, s in plan].count(.002) == 100 and len({s for s, _ in plan}) == 400
    assert not {s for s, _ in plan} & set(P['previous_candidate_seeds'])
    assert P['model'] == 'Qwen/Qwen2.5-VL-7B-Instruct' and len(P['model_revision']) == 40
    assert P['tensor_parallel_size'] == 1 and P['dtype'] == 'bfloat16' and P['temperature'] == 0


def test_population_is_frozen():
    bad = deepcopy(P); bad['candidates']['sigma_mixture'] = [.0005] * 4
    with pytest.raises(ValueError): ta.candidate_plan(bad)
    bad = deepcopy(P); bad['candidates']['seed_start'] = 6100000
    with pytest.raises(ValueError, match='reused'): ta.verify_inputs(bad)
    assert [len(ta.shard_plan(P, k)) for k in range(8)] == [50] * 8
    with pytest.raises(ValueError): ta.shard_plan(P, 8)


def test_slots_balance_odd_sizes():
    s = slots(1000, 1)
    assert [sum(x[2] == e for x in s) for e in range(4)] == [250] * 4
    assert sorted(sum(x[0] == l for x in s) for l in ('easy', 'medium', 'hard')) == [333, 333, 334]


def test_fold_record_and_rankings():
    index = {'A': list(range(0, 100)), 'B': list(range(100, 200)), 'C': list(range(200, 300))}
    correct = [True] * 40 + [False] * 60 + [True] * 30 + [False] * 70 + [True] * 35 + [False] * 65
    r = ta.fold_record(correct, index, BASE)
    assert r['pooled_correct'] == 105 and r['transfer_score'] == pytest.approx(0.0)
    assert r['mean_fold_gain'] == pytest.approx(.05)
    pop = population()
    pop[5] = rec(5, (50, 30, 30))     # lucky single fold: best pooled
    pop[9] = rec(9, (36, 36, 36))     # consistent
    pop[11] = rec(11, (37, 36, 34))   # pooled 107, transfer score +4 pp
    v, t = ta.rank(pop, 'VANILLA'), ta.rank(pop, 'TRANSFER_MIN')
    assert v[0]['candidate']['candidate_id'] == '0005' and v[1]['candidate']['candidate_id'] == '0009'
    assert [x['candidate']['candidate_id'] for x in t[:2]] == ['0009', '0011']
    assert t.index(pop[5]) == 2  # transfer score 0 ties the crowd; mean fold gain breaks the tie
    ties = sorted([x for x in pop if x['search']['pooled_correct'] == 90], key=lambda x: rank_hash(x['candidate']['candidate_id']))
    assert [x['candidate']['candidate_id'] for x in v[3:6]] == [x['candidate']['candidate_id'] for x in ties[:3]]
    with pytest.raises(ValueError): ta.rank(pop[:399], 'VANILLA')
    with pytest.raises(ValueError): ta.rank(pop, 'TRANSFER_LAMBDA')


def lock_from(pop):
    return {'top': {m: ta.rank(pop, m)[:20] for m in ta.METHODS}}


def test_validation_gate_conditions():
    pop = population()
    for i in range(20): pop[i] = rec(i, (40, 30, 30))          # VANILLA top-20: lucky
    for i in range(20, 40): pop[i] = rec(i, (33, 33, 33))      # TRANSFER_MIN top-20: consistent
    lock = lock_from(pop)
    ids = lambda m: [r['candidate']['candidate_id'] for r in lock['top'][m]]
    assert set(ids('VANILLA')) == {f'{i:04d}' for i in range(20)}
    val = {r['candidate']['candidate_id']: 0. for r in pop}
    for k, cid in enumerate(ids('TRANSFER_MIN')[:10]): val[cid] = .04 if k < 3 else .0
    for cid in ids('VANILLA')[:10]: val[cid] = .025
    g = ta.validation_gate(lock, val, P)
    assert g['condition_b_pass'] and not g['condition_a_pass'] and g['pass']  # A: +4 is only 1.5 pp over +2.5
    val[ids('VANILLA')[0]] = .03
    g = ta.validation_gate(lock, val, P)
    assert g['condition_b_pass'] and g['methods']['VANILLA']['top10_counts']['at_least_3pp'] == 1
    val[ids('VANILLA')[1]] = .03
    g = ta.validation_gate(lock, val, P)
    assert not g['pass'] and g['decision'] == 'NO_GO_TRANSFER_AWARE_VISUAL_SEARCH'
    val[ids('TRANSFER_MIN')[15]] = .05 + 1e-15
    g = ta.validation_gate(lock, val, P)
    assert g['condition_a_pass']   # best T +5, best V +3: margin exactly 2 pp
    del val[ids('TRANSFER_MIN')[19]]
    with pytest.raises(ValueError, match='operational'): ta.validation_gate(lock, val, P)


def test_final_selection_dedup_and_order():
    pop = population()
    for i in range(20): pop[i] = rec(i, (40, 30, 30))
    for i in range(20, 40): pop[i] = rec(i, (33, 33, 33))
    pop[0] = rec(0, (40, 40, 40))  # in both top-20s
    lock = lock_from(pop)
    val = {r['candidate']['candidate_id']: 0. for r in pop}
    val['0000'] = .06
    chosen, union = ta.final_selection(lock, val)
    assert chosen['VANILLA']['best']['candidate']['candidate_id'] == '0000' == chosen['TRANSFER_MIN']['best']['candidate']['candidate_id']
    assert len(union) == 9 and len({u['candidate']['candidate_id'] for u in union}) == 9
    assert [r['candidate']['candidate_id'] for r in chosen['VANILLA']['top5'][1:]] == \
           [r['candidate']['candidate_id'] for r in lock['top']['VANILLA'] if r['candidate']['candidate_id'] != '0000'][:4]


def per_entry(gain, genuine):
    return {'comparison': {'gain': gain}, 'genuine_expert': genuine}


def selection(prefix):
    top = [{'candidate': {'candidate_id': f'{prefix}{k}'}} for k in range(5)]
    return {'best': top[0], 'top5': top}


def test_final_decision_routes():
    sel = {'VANILLA': selection('v'), 'TRANSFER_MIN': selection('t')}
    per = {f'v{k}': per_entry(.01, False) for k in range(5)} | {f't{k}': per_entry(.01, False) for k in range(5)}
    per['t0'] = per_entry(.06, True)
    assert ta.final_decision(sel, per, P, True)['decision'] == 'GO_TRANSFER_AWARE_VISUAL_SEARCH'
    per['v0'] = per_entry(.05, True)
    assert ta.final_decision(sel, per, P, True)['decision'] != 'GO_TRANSFER_AWARE_VISUAL_SEARCH'
    per['v0'] = per_entry(.04, True)  # genuine but t beats by exactly 2 pp
    assert ta.final_decision(sel, per, P, True)['decision'] == 'GO_TRANSFER_AWARE_VISUAL_SEARCH'
    per['t0'] = per_entry(.04, False); per['v0'] = per_entry(.0, False)
    for k in (1, 2): per[f't{k}'] = per_entry(.03, False)
    per['t3'] = per_entry(.03, False); per['t4'] = per_entry(.03, False)
    d = ta.final_decision(sel, per, P, True)
    assert d['decision'] == 'PROMISING_BUT_NOT_EXPERT'
    for k in range(5): per[f'v{k}'] = per_entry(.035, False)
    assert ta.final_decision(sel, per, P, True)['decision'] == 'NO_GO_TRANSFER_AWARE_VISUAL_SEARCH'
    with pytest.raises(ValueError, match='operational'): ta.final_decision(sel, per, P, False)


def test_expert_label_concentration_and_levels():
    base = {'difficulty': {l: {'accuracy': .3} for l in ('easy', 'medium', 'hard')}}
    metric = {'difficulty': {'easy': {'accuracy': .4}, 'medium': {'accuracy': .35}, 'hard': {'accuracy': .3}}}
    ok = ta.expert(metric, {'gain': .05, 'paired_bootstrap95': [.001, .1]}, base, {'label_spread_pass': True}, P)
    assert ok['genuine_expert']
    assert not ta.expert(metric, {'gain': .05, 'paired_bootstrap95': [0., .1]}, base, {'label_spread_pass': True}, P)['genuine_expert']
    assert not ta.expert(metric, {'gain': .05, 'paired_bootstrap95': [.01, .1]}, base, {'label_spread_pass': False}, P)['genuine_expert']


def test_correlation():
    c = ta.correlation([1, 2, 3, 4], [2, 4, 6, 9])
    assert c['spearman'] == pytest.approx(1.0) and c['pearson'] > .99
    assert ta.correlation([1, 1], [1, 2])['pearson'] is None


def test_ranking_and_validation_lock_scripts(tmp_path, monkeypatch):
    """Run the lock script end to end on fabricated audited phases."""
    import thicket_runtime.committee_validation as cv
    import thicket_runtime.transfer_aware_audit as audit
    monkeypatch.setattr(cv, 'committed', lambda path: None)
    sets = ta.splits(P)
    index = ta.folds(P, sets['search'])
    rng = np.random.default_rng(0)
    root = tmp_path / 'study'
    (root / 'locks').mkdir(parents=True)
    from thicket_runtime.visual_runtime import write
    base_correct = rng.random(300) < .3
    write(root / 'locks/baseline.json', {'protocol_sha256': '', 'base_id': 'b', 'base_path': 'baseline/base.json.gz',
          'output_identities': {}, 'search_base': {'pooled_correct': int(base_correct.sum()), 'pooled_accuracy': float(base_correct.mean()),
          'fold_accuracy': {f: float(base_correct[i].mean()) for f, i in index.items()}}})
    def out(correct, rows):
        return [{'id': r['id'], 'answer': r['answer'] if c else '', 'correct': bool(c), 'finish_reason': 'stop'} for r, c in zip(rows, correct)]
    states = {}
    def fake_phase(run, protocol, path, locks, shard=None):
        from pathlib import Path
        name = Path(run).name
        (Path(run)).mkdir(parents=True, exist_ok=True)
        for f in ('run-manifest.json', 'sha256.json'): (Path(run) / f).write_text('{}')
        if name.startswith('search'):
            recs, traces = [], {}
            for seed, sigma in ta.shard_plan(P, shard):
                cid = f'c{seed}'
                corr = rng.random(300) < .3 + (sigma * 20)
                states[cid] = 's' + cid
                recs.append({'candidate': {'seed': seed, 'sigma': sigma, 'candidate_id': cid}, 'candidate_state_id': states[cid],
                             'split': 'search', 'examples': 300, 'correct_count': int(corr.sum())})
                traces[cid] = {'splits': {'search': {'outputs': out(corr, sets['search'])}}}
            return {'manifest': {'phase': 'search'}, 'records': recs, 'traces': traces}
        traces = {r['candidate']['candidate_id']: {'splits': {'validation': {'outputs': out(rng.random(300) < .35, sets['validation'])}}}
                  for r in locks['ranking']['validation_union']}
        return {'manifest': {'phase': 'validation'}, 'records': [], 'traces': traces}
    monkeypatch.setattr(audit, 'phase', fake_phase)
    write(root / 'baseline/base.json.gz', {'validation': {'outputs': out(rng.random(300) < .3, sets['validation'])}})
    for lock in ('ranking', 'validation'):
        monkeypatch.setattr(sys, 'argv', ['x', lock, '--root', str(root), '--protocol', PATH])
        sys.modules.pop('thicket_runtime.transfer_aware_audit', None)
        sys.modules['thicket_runtime.transfer_aware_audit'] = audit
        runpy.run_path('scripts/freeze_transfer_aware_lock.py', run_name='__main__')
    ranking, validation = read(root / 'locks/ranking.json'), read(root / 'locks/validation.json')
    assert len(ranking['records']) == 400 and all(len(ranking['top'][m]) == 20 for m in ta.METHODS)
    assert ranking['top']['VANILLA'] == ta.rank(ranking['records'], 'VANILLA')[:20]
    assert len(ranking['validation_union']) == 40 - ranking['top_overlap']
    assert set(validation['candidates']) == {r['candidate']['candidate_id'] for r in ranking['validation_union']}
    assert validation['gate']['decision'] in ('PASS_TO_FINAL_TEST', 'NO_GO_TRANSFER_AWARE_VISUAL_SEARCH')
    assert (validation['final_union'] != []) == validation['gate']['pass']

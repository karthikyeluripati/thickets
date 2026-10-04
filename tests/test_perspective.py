from copy import deepcopy
from types import SimpleNamespace
import runpy
import sys

import numpy as np
import pytest
import torch

from thicket_runtime import fast_state, perspective as pt, vllm_audit
from thicket_runtime.visual_final import rank_hash
from thicket_runtime.visual_runtime import read

PATH = 'experiments/perspective_taking_n5000_protocol.json'
P = read(PATH)


def test_frozen_data_population_and_audit():
    pt.verify_inputs(P)
    sets = pt.splits(P)
    assert {s: len(r) for s, r in sets.items()} == {'search': 200, 'validation': 200, 'test': 561}
    assert all(r['task_type'] == 'Perspective_Taking' for rows in sets.values() for r in rows)
    for s in ('search', 'validation'):
        assert len({r['image_sha256'] for r in sets[s]}) == 200
    assert pt.summarize([{'uid': r['uid'], 'correct': False, 'valid_letter': True, 'prediction': 'A'} for r in sets['search']],
                        sets['search'])['sub_task'].keys() == {r['sub_task_type'] for r in sets['search']}
    rows = pt.manifest(P)
    assert len(rows) == 5000 and len({r['seed'] for r in rows}) == 5000
    assert all(sum(r['sigma'] == s for r in rows) == 1250 for s in P['candidates']['sigma_mixture'])
    assert sum(len(pt.shard_plan(P, k)) for k in range(50)) == 5000
    audit = P['density_audit']['indices']
    assert len(audit) == 500 and len(set(audit)) == 500
    assert all(sum(rows[i]['sigma'] == s for i in audit) == 125 for s in P['candidates']['sigma_mixture'])
    bad = deepcopy(P); bad['density_audit']['indices'][0] = (audit[0] + 1) % 5000
    with pytest.raises(ValueError, match='density'): pt.verify_inputs(bad)
    bad = deepcopy(P); bad['candidates']['seed_start'] = 9100000
    with pytest.raises(ValueError, match='reused'): pt.verify_inputs(bad)
    assert P['model'] == 'Qwen/Qwen3-VL-8B-Instruct' and P['dtype'] == 'bfloat16' and P['tensor_parallel_size'] == 1


def recs(scores):
    return [{'candidate': {'seed': 9500000 + i, 'sigma': .001, 'candidate_id': f'{i:05d}'}, 'candidate_state_id': f's{i}',
             'split': 'search', 'examples': 200, 'correct_count': c} for i, c in enumerate(scores)]


def test_rankings():
    scores = [80] * 5000; scores[42] = 100; scores[7] = scores[9] = 95
    ranked = pt.rank_search(recs(scores), P)
    assert ranked[0]['candidate']['candidate_id'] == '00042'
    assert [r['candidate']['candidate_id'] for r in ranked[1:3]] == sorted(['00007', '00009'], key=rank_hash)
    with pytest.raises(ValueError): pt.rank_search(recs(scores)[:4999], P)
    top = ranked[:50]
    val = {r['candidate']['candidate_id']: 90 for r in top}
    val[top[30]['candidate']['candidate_id']] = 99
    val[top[1]['candidate']['candidate_id']] = val[top[2]['candidate']['candidate_id']] = 95
    order = pt.rank_validation(top, val)
    assert order[0] is top[30] and {order[1]['candidate']['candidate_id'], order[2]['candidate']['candidate_id']} == {'00007', '00009'}
    assert order[3] is top[0]  # remaining ties fall back to SEARCH correct count


def res(gain, lo=.01, subs=(1, 1, 0), spread=True):
    return {'gain': gain, 'paired': {'paired_bootstrap95': [lo, .2]},
            'sub_task_gain': dict(zip(('Allocentric', 'Egocentric', 'Hypothetical'), subs)),
            'label_audit': {'bias_spread_pass': spread}}


def test_go_rules():
    good = res(.06)
    assert pt.decision([good] + [res(0)] * 9, P, True)['decision'] == 'GO_VISUAL_NEURAL_THICKET_PERSPECTIVE_TAKING'
    biased = res(.06, spread=False)
    assert pt.decision([biased] + [res(0)] * 9, P, True)['decision'] == 'NO_GO_VISUAL_NEURAL_THICKET_PERSPECTIVE_N5000'
    rep = [res(.02), res(.05), res(.04), res(.03)] + [res(.03)] * 6
    assert pt.decision(rep, P, True)['decision'] == 'GO_VISUAL_NEURAL_THICKET_PERSPECTIVE_TAKING_REPLICATED'
    rep[1] = res(.05, subs=(1, 0, 0))
    assert pt.decision(rep, P, True)['decision'] == 'NO_GO_VISUAL_NEURAL_THICKET_PERSPECTIVE_N5000'
    with pytest.raises(ValueError, match='operational'): pt.decision(rep, P, False)
    with pytest.raises(ValueError, match='operational'): pt.decision(rep[:9], P, True)


def test_majority_and_wilson():
    rows = [{'uid': 'a', 'answer': 1}, {'uid': 'b', 'answer': 0}, {'uid': 'c', 'answer': 2}]
    o = lambda *p: [{'prediction': x, 'valid_letter': x in 'ABCD' and x != ''} for x in p]
    votes = pt.majority([o('B', 'A', ''), o('C', 'B', ''), o('B', 'B', 'x')], rows, 3)
    assert [v['prediction'] for v in votes] == ['B', 'B', ''] and [v['correct'] for v in votes] == [True, False, False]
    tie = pt.majority([o('C', 'A', 'A'), o('B', 'B', 'A')], rows, 2)
    assert tie[0]['prediction'] == 'C'  # first occurrence in frozen order wins a tie
    lo, hi = pt.wilson(5, 100)
    assert 0.02 < lo < .05 < hi < .12
    d = pt.density({.001: [.0, .02, .06], .002: [.04]})
    assert d['all']['at_least_3pp']['count'] == 2 and d['0.001']['at_least_5pp']['count'] == 1


def test_fast_state_matches_original_controls():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.LayerNorm(32), torch.nn.Linear(32, 4)).to(torch.bfloat16)
    model.register_buffer('counter', torch.arange(5))
    worker = SimpleNamespace(model_runner=SimpleNamespace(model=model))
    worker._base_weights = {n: p.detach().clone() for n, p in model.named_parameters()}
    worker._thicket_base_buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
    strip = lambda d: {k: v for k, v in d.items() if k != 'slow_path_tensors'}
    assert strip(fast_state.drift(worker)) == vllm_audit.drift(worker)
    base = fast_state.fingerprint(worker)
    assert base == fast_state.fingerprint(worker)
    with torch.no_grad():
        model[0].weight[3, 2] += 0.5
    assert strip(fast_state.drift(worker)) == vllm_audit.drift(worker)
    assert not fast_state.drift(worker)['exact_base'] and fast_state.fingerprint(worker) != base
    with torch.no_grad():
        model[0].weight.copy_(worker._base_weights['0.weight'])
    assert fast_state.fingerprint(worker) == base and fast_state.drift(worker)['exact_base']


def outs(rows, correct, pred='A'):
    return [{'uid': r['uid'], 'correct': bool(c), 'valid_letter': True, 'prediction': chr(65 + r['answer']) if c else pred}
            for r, c in zip(rows, correct)]


def test_search_and_validation_lock_scripts(tmp_path, monkeypatch):
    import thicket_runtime.committee_validation as cv
    import thicket_runtime.perspective_audit as audit
    from thicket_runtime.line_tracing import sha
    from thicket_runtime.visual_runtime import write
    from thicket_runtime.perspective_runtime import validation_plan
    monkeypatch.setattr(cv, 'committed', lambda path: None)
    sets = pt.splits(P)
    rng = np.random.default_rng(3)
    root = tmp_path / 'study'
    (root / 'locks').mkdir(parents=True)
    base = {s: {'outputs': outs(rows, rng.random(len(rows)) < .4)} for s, rows in sets.items()}
    write(root / 'baseline/base.json.gz', base)
    write(root / 'baseline/sha256.json', {'base.json.gz': sha(root / 'baseline/base.json.gz')})
    write(root / 'locks/baseline.json', {'protocol_sha256': '', 'base_id': 'b', 'base_id_flat_sha256': 'f', 'output_identities': {},
          'summary': {s: pt.summarize(v['outputs'], sets[s]) for s, v in base.items()}, 'preprocess': {}})
    def fake(run, protocol, path, locks, shard=None, base_reference=None):
        run.mkdir(parents=True, exist_ok=True)
        for f in ('run-manifest.json', 'sha256.json'): (run / f).write_text('{}')
        if run.name.startswith('search'):
            records, traces = [], {}
            for seed, sigma in pt.shard_plan(P, shard):
                cid = f'c{seed}'
                o = outs(sets['search'], rng.random(200) < .4 + sigma * 20, pred='B')
                records.append({'candidate': {'seed': seed, 'sigma': sigma, 'candidate_id': cid}, 'candidate_state_id': 's' + cid,
                                'split': 'search', 'examples': 200, 'correct_count': sum(x['correct'] for x in o)})
                traces[cid] = {'splits': {'search': {'outputs': o}}}
            return {'manifest': {'phase': 'search'}, 'records': records, 'traces': traces}
        part = validation_plan(locks, P)[shard * 35:(shard + 1) * 35]
        return {'manifest': {'phase': 'validation'}, 'records': [],
                'traces': {r['candidate']['candidate_id']: {'splits': {'validation': {'outputs': outs(sets['validation'], rng.random(200) < .42, 'C')}}}
                           for r in part}}
    monkeypatch.setattr(audit, 'phase', fake)
    for lock in ('search', 'validation'):
        monkeypatch.setattr(sys, 'argv', ['x', lock, '--root', str(root), '--protocol', PATH])
        runpy.run_path('scripts/freeze_perspective_lock.py', run_name='__main__')
    s, v = read(root / 'locks/search.json'), read(root / 'locks/validation.json')
    assert len(s['records']) == 5000 and [r['index'] for r in s['records']] == list(range(5000))
    assert s['top50'] == pt.rank_search(s['records'], P)[:50]
    assert set(v['candidates']) == {r['candidate']['candidate_id'] for r in s['top50']}
    assert v['top10'][:5] == v['top5'] and v['rank1'] == v['top10'][0]
    assert sum(x['n'] for k, x in v['density_audit']['summary'].items() if k != 'all') == 500
    assert v['search_to_validation']['n'] == 50

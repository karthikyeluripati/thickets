from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from thicket_runtime.visual_final import (decision, evaluation_sets, label_audit,
    paired, population_id, rank_hash, rank_selection, verify_inputs)
from thicket_runtime.visual_runtime import read

P = read('experiments/visual_line_tracing_final_protocol.json')


def records():
    return [{'candidate':{'seed':P['search']['seed_start']+i,'sigma':.0005,'candidate_id':str(i)},
             'split':'selection','examples':150,'correct_count':30,
             'population_sha256':P['populations']['selection']['sha256']} for i in range(300)]


def individual(gains):
    return [{'comparison':{'gain':g,'paired_bootstrap95':[.01,.09]},
             'difficulty_gain':{'easy':g,'medium':g,'hard':0},
             'label_audit':{'label_spread_pass':True}} for g in gains]


def test_frozen_v1_dataset_reference_and_fresh_seeds():
    assert verify_inputs(P)['files_verified'] == 654
    assert P['dataset'] == 'examples/visual-line-tracing-v1'
    assert P['sigma'] == .0005
    assert P['diagnostic']['selected']['candidate']['seed'] == 3100000
    assert P['diagnostic']['selected']['correct_count'] == 50
    assert not set(range(6100000,6100300)) & set(P['previous_candidate_seeds'])
    bad = deepcopy(P); bad['search']['seed_start'] = 3100000
    with pytest.raises(ValueError,match='reused'): verify_inputs(bad)


def test_exact_v1_inference_and_upstream_executor_reused():
    from thicket_runtime import visual_runtime, visual_final_runtime
    old = read('experiments/visual_line_tracing_protocol.json')
    for key in ('model','model_revision','processor_revision','dtype','tensor_parallel_size','max_tokens',
                'temperature','sampling_seed','max_model_len','max_num_seqs','max_num_batched_tokens',
                'gpu_memory_utilization','image_pixels','perturbation','cache_contract'):
        assert P[key] == old[key]
    for name in ('VisualExecutor','evaluate_candidate','initialize_visual','launch'):
        assert getattr(visual_final_runtime,name) is getattr(visual_runtime,name)


def test_search_includes_all_v1_difficulties_but_no_heldout():
    sets = evaluation_sets(P,'search')
    assert list(sets) == ['selection']
    assert len(sets['selection']) == 150
    assert {r['difficulty'] for r in sets['selection']} == {'easy','medium','hard'}
    assert population_id(sets['selection']) == P['populations']['selection']['sha256']
    assert len(evaluation_sets(P,'heldout')['heldout']) == 500
    with pytest.raises(ValueError): evaluation_sets(P,'calibration')
    with pytest.raises(ValueError): evaluation_sets(P,'baseline')


@pytest.mark.parametrize('field,value',[('split','heldout'),('examples',500),('population_sha256','heldout')])
def test_rank_rejects_heldout_or_mixed_populations(field,value):
    r = records(); r[0][field] = value
    with pytest.raises(ValueError): rank_selection(r,P)


def test_full_search_ranking_is_deterministic_without_a_mean_gate():
    r = records()
    for item in r: item['correct_count'] = 0
    expected = sorted(r,key=lambda item:rank_hash(item['candidate']['candidate_id']))
    assert rank_selection(r,P) == expected
    assert rank_selection(r[::-1],P) == expected
    r[3]['correct_count'] = 1
    assert rank_selection(r,P)[0] == r[3]
    with pytest.raises(ValueError): rank_selection(r[:299],P)
    r[0]['candidate']['sigma'] = .001
    with pytest.raises(ValueError): rank_selection(r,P)


def test_rank1_effect_threshold_and_zero_lower_bound_are_strict():
    experts = individual([.05]+[0]*9)
    assert decision(experts,{'accuracy':.266},P,True)['decision'] == 'GO_VISUAL_THICKET_SINGLE_EXPERT'
    experts[0]['comparison']['paired_bootstrap95'][0] = 0
    result = decision(experts,{'accuracy':.266},P,True)
    assert not result['pass'] and result['effect_size_without_all_primary_checks']
    experts = individual([.048]+[0]*9)
    assert not decision(experts,{'accuracy':.266},P,True)['pass']


def test_primary_requires_difficulty_spread_and_label_spread():
    experts = individual([.05]+[0]*9)
    experts[0]['difficulty_gain']['medium'] = 0
    assert not decision(experts,{'accuracy':.266},P,True)['pass']
    experts = individual([.05]+[0]*9)
    experts[0]['label_audit']['label_spread_pass'] = False
    assert not decision(experts,{'accuracy':.266},P,True)['pass']


def test_replication_all_three_conditions_required_without_ensemble_rescue():
    experts = individual([.03,.03,.05]+[.03]*7)
    result = decision(experts,{'accuracy':.266},P,True)
    assert result['decision'] == 'GO_VISUAL_THICKET_REPLICATED_EXPERTS'
    assert result['top10_density']['at_least_3pp'] == 10
    experts = individual([.03,.03,.05]+[0]*7)
    assert not decision(experts,{'accuracy':.266},P,True)['pass']
    experts = individual([.03]*10)
    assert not decision(experts,{'accuracy':.266},P,True)['pass']
    with pytest.raises(ValueError,match='operational'): decision(experts,{'accuracy':.266},P,False)


def test_paired_counts_exact_binomial_and_bootstrap_identity():
    b = [{'correct':v} for v in [1,1,0,0,0,0]]
    c = [{'correct':v} for v in [1,0,1,1,1,0]]
    indices = np.tile(np.arange(6),(10000,1))
    result = paired(b,c,indices)
    assert result['base_only_correct'] == 1
    assert result['candidate_only_correct'] == 3
    assert result['both_correct'] == result['both_wrong'] == 1
    assert result['gain'] == pytest.approx(1/3)
    assert result['paired_bootstrap95'] == [1/3,1/3]
    assert result['mcnemar_exact_two_sided_p'] == .625
    same = paired(b,b,indices)
    assert same['mcnemar_exact_two_sided_p'] == 1
    assert same['paired_bootstrap95'] == [0,0]


def test_single_label_eighty_percent_boundary_and_confusion_counts():
    rows = [{'answer':str(i//10+1)} for i in range(40)]
    base = [{'correct':False,'answer':''} for r in rows]
    candidate = [{'correct':i in (0,1,2,3,10),'answer':r['answer'] if i in (0,1,2,3,10) else ''} for i,r in enumerate(rows)]
    result = label_audit(base,candidate,rows,.8)
    assert result['largest_positive_net_gain_share'] == .8
    assert not result['label_spread_pass']
    candidate[20] = {'correct':True,'answer':'3'}
    assert label_audit(base,candidate,rows,.8)['label_spread_pass']
    assert sum(result['candidate_prediction_counts'].values()) == 40


def test_ranking_entrypoint_has_no_heldout_or_diagnostic_input_argument():
    source = Path('scripts/freeze_visual_final_selection.py').read_text()
    assert "['phase'] != 'search'" in source
    assert '--heldout' not in source and '--diagnostic' not in source

from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from thicket_runtime.visual_runtime import read, output_identity
from thicket_runtime.visual_v21 import (
    PRIMARY, calibration_plan, calibrate, capability_gate, evaluation_sets, gate,
    population_id, rank_selection, summarize, verify_dataset,
)
from thicket_runtime.visual_v21_audit import outputs

PROTOCOL=read(Path('experiments/visual_line_tracing_v21_protocol.json'))


def metric(medium,hard,easy=0):
    return {'primary':{'accuracy':(167*medium+166*hard)/333},
            'difficulty':{'medium':{'accuracy':medium},'hard':{'accuracy':hard},'easy':{'accuracy':easy}}}


def selection_record(seed,sigma,correct):
    return {'candidate':{'seed':seed,'sigma':sigma,'candidate_id':str(seed)},
            'split':'selection','evaluation_set':'selection_primary','examples':100,
            'population_sha256':PROTOCOL['primary']['selection_population_sha256'],
            'correct_count':correct,'invalid_count':0}


def calibration_records():
    return [selection_record(seed,sigma,40) for seed,sigma in calibration_plan(PROTOCOL['calibration'])]


def test_v21_dataset_and_primary_population_are_frozen():
    assert verify_dataset(PROTOCOL)['files_verified']==654
    baseline=evaluation_sets(PROTOCOL,'baseline')
    assert {s:len(v) for s,v in baseline.items()}=={'selection_primary':100,'selection_easy':50,'heldout':500}
    assert sum(r['difficulty'] in PRIMARY for r in baseline['heldout'])==333
    for phase in ('calibration','search'):
        data=evaluation_sets(PROTOCOL,phase)
        assert list(data)==['selection_primary']
        assert all(r['difficulty'] in PRIMARY for r in data['selection_primary'])
        assert population_id(data['selection_primary'])==PROTOCOL['primary']['selection_population_sha256']
    bad=deepcopy(PROTOCOL); bad['dataset_pin']['manifest_sha256']='0'*64
    with pytest.raises(ValueError,match='manifest'): verify_dataset(bad)


def test_v21_uses_the_identical_inference_executor_and_settings():
    from thicket_runtime import visual_runtime,visual_v21_runtime
    for name in ('VisualExecutor','evaluate_candidate','initialize_visual','launch'):
        assert getattr(visual_v21_runtime,name) is getattr(visual_runtime,name)
    old=read('experiments/visual_line_tracing_v2_protocol.json')
    for key in ('model','model_revision','processor_revision','upstream_commit','dtype','tensor_parallel_size',
                'temperature','sampling_seed','max_tokens','max_model_len','max_num_seqs',
                'max_num_batched_tokens','gpu_memory_utilization','image_pixels','perturbation','cache_contract'):
        assert PROTOCOL[key]==old[key]


@pytest.mark.parametrize('correct,accepted',[(99,False),(100,True),(233,True),(234,False)])
def test_capability_gate_uses_exact_primary_bounds(correct,accepted):
    held=metric(correct/333,correct/333,easy=1)
    result=capability_gate({'heldout':held},PROTOCOL['baseline'])
    assert result['accepted'] is accepted
    held['difficulty']['easy']['accuracy']=0
    assert capability_gate({'heldout':held},PROTOCOL['baseline'])==result


def test_calibration_stop_boundary_and_no_signal_is_scientific_negative():
    records=calibration_records()
    for r in records[:3]: r['correct_count']=43
    result=calibrate(records,40,PROTOCOL)
    assert result['decision']=='NO_GO_no_search_signal' and result['sigma'] is None
    assert all(r['fraction_beating_base_by_at_least_3pp']==.05 for r in result['table'])
    records[1]['correct_count']=44
    assert calibrate(records,40,PROTOCOL)['sigma']==.001


def test_calibration_mean_tolerance_and_tie_break_are_frozen():
    records=calibration_records()
    for r in records: r['correct_count']=41
    assert not calibrate(records,40,PROTOCOL)['continue_search']
    for r in records: r['correct_count']=42
    assert calibrate(records,40,PROTOCOL)['sigma']==.0005
    records=calibration_records()
    for r in records[:3]: r['correct_count']=44
    records[4]['correct_count']=41
    assert calibrate(records,40,PROTOCOL)['sigma']==.001
    records[3]['correct_count']=41
    assert calibrate(records,40,PROTOCOL)['sigma']==.0005


@pytest.mark.parametrize('field,value',[('split','heldout'),('evaluation_set','selection_easy'),('examples',150),('population_sha256','wrong')])
def test_calibration_and_ranking_reject_nonprimary_records(field,value):
    records=calibration_records(); records[0][field]=value
    with pytest.raises(ValueError,match=r'Medium\+Hard'): calibrate(records,40,PROTOCOL)
    with pytest.raises(ValueError,match=r'Medium\+Hard'): rank_selection(records,PROTOCOL)


def test_search_ranking_hash_tie_break_is_order_independent():
    records=[selection_record(PROTOCOL['search']['seed_start']+i,.001,40) for i in range(300)]
    ranked=rank_selection(records,PROTOCOL)
    assert ranked==rank_selection(list(reversed(records)),PROTOCOL)
    assert [r['candidate']['candidate_id'] for r in ranked]==sorted([r['candidate']['candidate_id'] for r in records],key=lambda cid:hashlib.sha256(('visual-rank-v1:'+cid).encode()).hexdigest())


def test_heldout_gate_ignores_easy_and_requires_primary_gain():
    base=metric(.4,.4,0)
    methods={k:metric(.4,.4,1) for k in ('best','top5_vote','top10_vote')}
    assert not gate(base,methods,PROTOCOL['gate'],True,True)['pass']
    methods['best']=metric(.5,.4,0)
    assert gate(base,methods,PROTOCOL['gate'],True,True)['comparisons']['best']['pass']
    with pytest.raises(ValueError,match='controls'): gate(base,methods,PROTOCOL['gate'],False,True)


def test_ensemble_gate_allows_one_point_regression_but_not_more():
    base=metric(.4,.4)
    methods={k:base for k in ('best','top5_vote','top10_vote')}
    methods['top5_vote']=metric(.55,.39)
    assert gate(base,methods,PROTOCOL['gate'],True,True)['pass']
    methods['top5_vote']=metric(.56,.389)
    assert not gate(base,methods,PROTOCOL['gate'],True,True)['pass']
    methods['top10_vote']=metric(.53,.41)
    assert gate(base,methods,PROTOCOL['gate'],True,True)['pass']


def test_raw_audit_rescores_text_and_rejects_stale_embeddings():
    rows=evaluation_sets(PROTOCOL,'calibration')['selection_primary'][:1]
    r=rows[0]
    values=[{'id':r['id'],'image_sha256':r['image_sha256'],'text':r['answer'],'answer':r['answer'],
             'correct':True,'failure_category':'correct','prompt_token_ids':[1],'finish_reason':'stop'}]
    result={'outputs':values,'output_identity':output_identity(values),'cache_audit':{
        'fresh_images_encoded':1,'before':{'images_encoded':0,'cached_items_after':0},'after':{'images_encoded':1}}}
    assert outputs(result,rows)==values
    bad=deepcopy(result); bad['outputs'][0]['correct']=False; bad['output_identity']=output_identity(bad['outputs'])
    with pytest.raises(ValueError,match='score mismatch'): outputs(bad,rows)
    bad=deepcopy(result); bad['cache_audit']['after']['images_encoded']=0
    with pytest.raises(ValueError,match='vision'): outputs(bad,rows)


def test_primary_summary_is_invariant_to_easy_correctness():
    rows=evaluation_sets(PROTOCOL,'heldout')['heldout']
    values=[{'id':r['id'],'correct':i%2==0,'answer':'1','finish_reason':'stop'} for i,r in enumerate(rows)]
    before=summarize(values,rows)
    for o,r in zip(values,rows):
        if r['difficulty']=='easy': o['correct']=not o['correct']
    after=summarize(values,rows)
    assert before['primary']==after['primary']
    assert all(before['difficulty'][l]==after['difficulty'][l] for l in PRIMARY)

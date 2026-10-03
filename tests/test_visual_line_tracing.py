import hashlib
from pathlib import Path
import pytest
from thicket_runtime.line_tracing import load_split,parse_answer,topology,vote
from thicket_runtime.visual_decisions import calibrate,gate,rank_selection
from thicket_runtime.visual_runtime import evaluate_candidate,vision_hook


def test_frozen_graph_labels_and_disjoint_images():
    root=Path('examples/visual-line-tracing-v1')
    splits={s:load_split(root,s) for s in ('selection','heldout')}
    assert [len(r) for r in splits.values()]==[150,500]
    a,b=splits.values()
    assert not {r['seed'] for r in a}&{r['seed'] for r in b}
    assert len({r['image_sha256'] for r in a+b})==650
    for row in a+b:
        graph=row['topology']; lanes=list(range(4)); visits=[0]*4
        for crossing in graph['stages']:
            k=crossing['lane']; pair=lanes[k:k+2]
            assert pair==crossing['crossing_paths'] and crossing['over_path'] in pair
            for path in pair: visits[path]+=1
            lanes[k:k+2]=reversed(pair)
        assert lanes==graph['final_lane_to_path']
        assert lanes[int(row['answer'])-1]==graph['target_path']
        assert visits[graph['target_path']]>=row['style']['minimum_target_crossings']


def test_generator_reproduces_every_topology_and_one_render():
    from PIL import Image
    from thicket_runtime.line_tracing import draw_example
    root=Path('examples/visual-line-tracing-v1')
    rows=load_split(root,'selection')
    for r in rows:
        g=r['topology']
        assert topology(r['seed'],r['difficulty'],g['target_path'],g['target_end_lane'])==g
    assert draw_example(rows[0]).tobytes()==Image.open(root/rows[0]['image']).convert('RGB').tobytes()


def test_strict_answer_and_stable_vote_ties():
    assert parse_answer(' 3.\n')=='3'
    for bad in ('13','1 or 2','The answer is 3','', '5'):
        assert not parse_answer(bad)
    assert vote(['','2','1','1','2'])['answer']=='2'
    assert vote(['',''])['answer']==''


def test_gate_requires_gain_and_more_than_one_difficulty():
    def metric(x,levels): return {'accuracy':x,'difficulty':{k:{'accuracy':v} for k,v in zip(('easy','medium','hard'),levels)}}
    base=metric(.5,[.5,.5,.5]); m={k:base for k in ('best','top5_vote','top10_vote')}
    cfg={'single_expert_gain':.05,'ensemble_gain':.07}
    m['best']=metric(.55,[.65,.5,.5])
    assert not gate(base,m,cfg,True,True)['pass']
    m['best']=metric(.55,[.55,.60,.5])
    assert gate(base,m,cfg,True,True)['pass']
    assert not gate(base,m,cfg,False,True)['pass']


def test_ranking_rejects_heldout_records():
    with pytest.raises(ValueError,match='selection-only'):
        rank_selection([{'split':'heldout','examples':500}],{'candidates':1,'seed_start':1})


def test_candidate_state_mismatch_restores_without_generation():
    class Fake:
        restored=False
        def rebase(self): pass
        def drift(self): return {'exact_base':True}
        def memory(self,reset=False): return {}
        def apply(self,*a): pass
        def fingerprint(self): return 'different'
        def generate(self,*a): raise AssertionError('must not generate')
        def restore(self,*a): self.restored=True
    api=Fake()
    with pytest.raises(RuntimeError,match='BEFORE generation'):
        evaluate_candidate(api,{}, {'heldout':[]},expected_state='expected')
    assert api.restored


def test_vision_audit_counts_list_grids_as_used_by_pinned_vllm():
    class Visual:
        _thicket_images_encoded=0
        _thicket_encoder_calls=0
    visual=Visual()
    vision_hook(visual,(),{'grid_thw':[[1,32,32],[1,32,32]]})
    assert visual._thicket_images_encoded==2 and visual._thicket_encoder_calls==1


def test_calibration_locks_scale_and_stops_random_collapse():
    config={'sigmas':[.0005,.001,.002],'candidates_per_sigma':20}
    def records(counts):
        return [{'split':'selection','examples':150,'candidate':{'sigma':s},'correct_count':n,'invalid_count':0}
                for s,count in zip(config['sigmas'],counts) for n in count]
    result=calibrate(records([[70]*20,[70]*20,[68]*19+[75]]),.40,config)
    assert result['sigma']==.002 and result['scale_valid']
    result=calibrate(records([[38]*20]*3),.40,config)
    assert not result['scale_valid'] and result['sigma'] is None

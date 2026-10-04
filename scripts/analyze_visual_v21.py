"""Audit the executed v2.1 prefix and apply only the frozen terminal gates."""
import argparse
from collections import Counter
from pathlib import Path
import subprocess
import sys

import numpy as np

from thicket_runtime.cli import revision_info
from thicket_runtime.line_tracing import load_split,sha,vote
from thicket_runtime.visual_runtime import read,write
from thicket_runtime.visual_v21 import PRIMARY,baseline_summary,calibrate,capability_gate,gate,rank_selection,summarize,verify_dataset
from thicket_runtime.visual_v21_audit import phase


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,default=Path('experiments/visual_line_tracing_v21_protocol.json'))
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists(): raise FileExistsError(a.out)
    protocol=read(a.protocol)
    verify_dataset(protocol)
    rows={s:load_split(protocol['dataset'],s) for s in ('selection','heldout')}
    if {r['seed'] for r in rows['selection']} & {r['seed'] for r in rows['heldout']} or {r['image_sha256'] for r in rows['selection']} & {r['image_sha256'] for r in rows['heldout']}:
        raise ValueError('dataset splits overlap')
    locks=a.root/'locks'
    completed={}

    def audit(name,base=None):
        result=phase(a.root/(name+'-01'),protocol,a.protocol,base)
        if result['manifest']['phase']!=name: raise ValueError('phase identity changed')
        completed[name]=result
        return result

    def verify_lock(filename,name,result):
        path=locks/(filename+'.json'); lock=read(path)
        run=a.root/(name+'-01')
        expected={'phase':name,'protocol_sha256':sha(a.protocol),
                  'dataset_manifest_sha256':protocol['dataset_pin']['manifest_sha256'],
                  'run_manifest_sha256':sha(run/'run-manifest.json'),'run_checksum_sha256':sha(run/'sha256.json'),
                  'base_id':result['manifest']['base_id'],'controls_pass':True}
        if any(lock.get(k)!=v for k,v in expected.items()): raise ValueError('lock is not from the audited phase')
        return lock

    def prior_lock(name,filename):
        path=locks/(filename+'.json')
        copied=a.root/(name+'-01')/path.name
        if copied.read_bytes()!=path.read_bytes(): raise ValueError('pre-phase lock differs from final lock')
        source=completed[name]['manifest']['source']['commit']
        relative=path.resolve().relative_to(Path.cwd()).as_posix()
        if subprocess.check_output(['git','show',source+':'+relative])!=path.read_bytes():
            raise ValueError('lock was not committed in the execution revision')

    baseline=audit('baseline')
    base_lock=verify_lock('baseline','baseline',baseline)
    base_summary=baseline_summary(baseline['bases'],baseline['datasets'])
    acceptance=capability_gate(base_summary,protocol['baseline'])
    if any(base_lock.get(k)!=v for k,v in acceptance.items()) or base_lock['summary']!=base_summary:
        raise ValueError('baseline capability decision does not reproduce')
    if base_lock['output_identities']!={s:v['output_identity'] for s,v in baseline['bases'].items()}:
        raise ValueError('baseline output identities changed')
    artifacts={'baseline':base_summary,'capability_gate':acceptance}
    if not acceptance['accepted']:
        decision={'decision':'NO_GO_benchmark_capability_gate','execution_stop':'STOP_benchmark_capability_gate',
                  'scientific_search_result':None,'search_hypothesis_tested':False,
                  'reason':acceptance['reason'],'calibration_candidates':0,'search_candidates':0,'heldout_candidates':0}
    else:
        calibration=audit('calibration',base_lock)
        prior_lock('calibration','baseline')
        sigma_lock=verify_lock('sigma','calibration',calibration)
        choice=calibrate(calibration['records'],base_summary['selection']['primary']['correct_count'],protocol)
        if any(sigma_lock.get(k)!=v for k,v in choice.items()): raise ValueError('calibration rule does not reproduce')
        artifacts['calibration']=choice
        if not choice['continue_search']:
            decision={'decision':'NO_GO_no_search_signal','scientific_search_result':'NO-GO',
                      'search_hypothesis_tested':True,'calibration_candidates':60,'search_candidates':0,'heldout_candidates':0,
                      'scope':'Frozen three-sigma calibration after a valid capability gate; no held-out candidate claim'}
        else:
            search=audit('search',base_lock)
            prior_lock('search','baseline'); prior_lock('search','sigma')
            selection=verify_lock('selection','search',search)
            ranked=rank_selection(search['records'],protocol)
            if (selection['ranked_candidates']!=ranked or selection['best']!=ranked[0] or
                selection['top5']!=ranked[:5] or selection['top10']!=ranked[:10] or
                selection['easy_used'] is not False or selection['heldout_used'] is not False):
                raise ValueError('frozen primary-only ranking does not reproduce')
            if any(r['candidate']['sigma']!=choice['sigma'] for r in ranked): raise ValueError('search sigma changed')
            held=audit('heldout',base_lock)
            prior_lock('heldout','baseline'); prior_lock('heldout','selection')
            traces=held['traces']; top=ranked[:10]
            if set(traces)!={r['candidate']['candidate_id'] for r in top}: raise ValueError('held-out union differs from frozen top10')
            for r in top:
                raw=traces[r['candidate']['candidate_id']]
                if raw['candidate']!=r['candidate'] or raw['candidate_state_id']!=r['candidate_state_id']:
                    raise ValueError('held-out candidate fingerprint changed')
            expert_outputs=[traces[r['candidate']['candidate_id']]['splits']['heldout']['outputs'] for r in top]
            held_rows=held['datasets']['heldout']
            base_outputs=held['bases']['heldout']['outputs']
            base=summarize(base_outputs,held_rows)
            individual=[summarize(v,held_rows) for v in expert_outputs]
            methods={'best':individual[0]}; votes={}
            vectors={'base':[o['correct'] for o in base_outputs],'best':[o['correct'] for o in expert_outputs[0]]}
            for k in (5,10):
                values=[]
                for i,r in enumerate(held_rows):
                    v=vote([o[i]['answer'] for o in expert_outputs[:k]])
                    values.append({**v,'id':r['id'],'correct':v['answer']==r['answer'],'finish_reason':'vote'})
                methods[f'top{k}_vote']=summarize(values,held_rows)
                methods[f'top{k}_individual_mean']={
                    **{scope:{'accuracy':sum(v[scope]['accuracy'] for v in individual[:k])/k,'n':base[scope]['n']} for scope in ('primary','overall')},
                    'difficulty':{l:{'accuracy':sum(v['difficulty'][l]['accuracy'] for v in individual[:k])/k,'n':base['difficulty'][l]['n']} for l in ('easy','medium','hard')}}
                votes[str(k)]=values
                vectors[f'top{k}_vote']=[v['correct'] for v in values]
            decision=gate(base,methods,protocol['gate'],True,True)
            decision.update(scientific_search_result='GO' if decision['pass'] else 'NO-GO',search_hypothesis_tested=True,
                            calibration_candidates=60,search_candidates=300,heldout_candidates=10)
            base_correct=base_summary['selection']['primary']['correct_count']
            scores=[r['correct_count'] for r in ranked]
            distribution={'n':300,'examples_per_candidate':100,'base_primary_selection_accuracy':base_correct/100,
                          'mean_accuracy':float(np.mean(scores))/100,'best_accuracy':max(scores)/100,'minimum_accuracy':min(scores)/100,
                          'fraction_beating_base':sum(v>base_correct for v in scores)/300,
                          'expert_density':{str(d/100):sum(v>=base_correct+d for v in scores)/300 for d in (1,3,5)},
                          'histogram_correct_counts':dict(sorted(Counter(scores).items()))}
            primary=[i for i,r in enumerate(held_rows) if r['difficulty'] in PRIMARY]
            bootstrap=np.random.default_rng(20261003).integers(0,len(primary),size=(10000,len(primary)),dtype=np.int16)
            comparisons={}
            for name in ('best','top5_vote','top10_vote'):
                delta=np.array(vectors[name],dtype=int)[primary]-np.array(vectors['base'],dtype=int)[primary]
                comparisons[name]={'primary_gain':float(delta.mean()),'improved_primary_questions':int((delta>0).sum()),
                                   'worsened_primary_questions':int((delta<0).sum()),
                                   'descriptive_paired_bootstrap95':np.quantile(delta[bootstrap].mean(axis=1),[.025,.975]).tolist()}
            artifacts.update(distribution=distribution,heldout={'base':base,**methods},individual_experts=individual,
                             comparisons=comparisons,votes=votes,per_question={'ids':[r['id'] for r in held_rows],
                             'primary_indices':primary,'correct':vectors,'individual_correct':[[o['correct'] for o in v] for v in expert_outputs]})
    present={p.parent.name.removesuffix('-01') for p in a.root.glob('*/run-manifest.json')}
    if present!=set(completed): raise ValueError('executed phases extend beyond the declared stopping gate')
    decision.update(controls_pass=True,sets_disjoint=True,easy_used_for_decision=False,dataset_matches_6d30f60=True)
    a.out.mkdir(parents=True)
    for name,value in artifacts.items(): write(a.out/(name+('.json.gz' if name=='per_question' else '.json')),value)
    write(a.out/'gate.json',decision)
    write(a.out/'manifest.json',{'source':revision_info(),'command':[sys.executable,*sys.argv],
          'protocol_sha256':sha(a.protocol),'dataset_manifest_sha256':protocol['dataset_pin']['manifest_sha256'],
          'all_raw_generations_rescored':True,'executed_phases':list(completed),
          'phase_manifest_sha256':{name:sha(a.root/(name+'-01')/'run-manifest.json') for name in completed},
          'numpy':np.__version__})
    if 'distribution' in artifacts:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        axes[0].hist(scores,bins=np.arange(-.5,101.5,1),color='#4477aa')
        axes[0].axvline(base_correct,color='black',linestyle='--',label='Base primary selection')
        axes[0].set(xlabel='Medium+Hard selection accuracy (%)',ylabel='Candidates',title='300 frozen-sigma candidates')
        axes[0].legend(fontsize=8)
        keys=['base','best','top5_vote','top10_vote']; data=artifacts['heldout']
        axes[1].bar(keys,[100*data[k]['primary']['accuracy'] for k in keys],color=['#666666','#ee7733','#228833','#44aa99'])
        axes[1].set(ylabel='Medium+Hard held-out accuracy (%)',ylim=(0,100),title='333 primary held-out images')
        fig.savefig(a.out/'primary-results.png',dpi=180); fig.savefig(a.out/'primary-results.svg'); plt.close(fig)
    print(decision)


if __name__=='__main__': main()

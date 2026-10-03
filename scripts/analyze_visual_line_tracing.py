"""Audit the frozen visual study and report its predeclared terminal decision."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

from thicket_runtime.candidate import CandidateSpec
from thicket_runtime.cli import revision_info
from thicket_runtime.line_tracing import load_split,parse_answer,sha,vote
from thicket_runtime.visual_decisions import calibrate,gate,rank_selection,summarize
from thicket_runtime.visual_runtime import output_identity,read,write


def audit_outputs(result, rows, reference=None):
    outputs=result['outputs']
    if len(outputs)!=len(rows) or result['cache_audit']['fresh_images_encoded']!=len(rows):
        raise ValueError('incomplete generation or stale visual embeddings')
    if output_identity(outputs)!=result['output_identity']:
        raise ValueError('output identity mismatch')
    cache=result['cache_audit']
    if cache['after']['images_encoded']-cache['before']['images_encoded']!=len(rows) or cache['before']['cached_items_after']!=0:
        raise ValueError('encoder cache evidence differs from the fresh-image contract')
    for i,(o,r) in enumerate(zip(outputs,rows)):
        if (o['id']!=r['id'] or o['image_sha256']!=r['image_sha256'] or
            o['answer']!=parse_answer(o['text']) or o['correct']!=(o['answer']==r['answer'])):
            raise ValueError('raw generation/scoring mismatch')
        if reference and o['prompt_token_ids']!=reference[i]['prompt_token_ids']:
            raise ValueError('prompt tokenization changed')
    return outputs


def audit_phase(root, phase, protocol, datasets, base_id):
    manifest,controls=read(root/'run-manifest.json'),read(root/'controls.json')
    if manifest['status']!='complete' or manifest['phase']!=phase or not all(controls.values()):
        raise ValueError('operational failure is not a scientific NO')
    for name,expected in read(root/'sha256.json').items():
        if sha(root/name)!=expected: raise ValueError('run checksum mismatch: '+name)
    if manifest['base_id']!=base_id: raise ValueError('base state differs')
    if manifest['model_revision']!=protocol['model_revision'] or manifest['processor_revision']!=protocol['processor_revision']:
        raise ValueError('model or processor revision differs')
    bases=read(root/'base.json.gz')
    expected_splits={'selection','heldout'} if phase=='baseline' else ({'heldout'} if phase=='heldout' else {'selection'})
    if set(bases)!=expected_splits or set(manifest['generation_splits'])!=expected_splits:
        raise ValueError('phase generated an unauthorized split')
    for s,base in bases.items():
        audit_outputs(base,datasets[s])
        repeat=read(root/'base-repeat.json.gz')[s]
        audit_outputs(repeat,datasets[s],base['outputs'])
        if repeat['output_identity']!=base['output_identity']: raise ValueError('base repeat failed')
    if phase=='baseline':
        zero=read(root/'zero.json.gz')
        if zero['candidate_state_id']!=base_id or not zero['restoration']['exact_base']:
            raise ValueError('zero state control failed')
        for s,output in zero['splits'].items():
            audit_outputs(output,datasets[s],bases[s]['outputs'])
            if output['output_identity']!=bases[s]['output_identity']: raise ValueError('zero output control failed')
    records=read(root/'records.json')
    traces={}
    for record in records:
        c=record['candidate']; cid=c['candidate_id']
        spec=CandidateSpec(**{k:c[k] for k in ('base_id','seed','sigma','rng','sign')})
        raw=read(root/'candidates'/(cid+'.json.gz'))
        if c['base_id']!=base_id or cid in traces or spec.candidate_id!=cid or raw['candidate']!=c or raw['candidate_state_id']!=record['candidate_state_id'] or not raw['restoration']['exact_base']:
            raise ValueError('candidate identity or restoration mismatch')
        split=record['split']
        if set(raw['splits'])!=expected_splits or split not in expected_splits:
            raise ValueError('candidate used an unauthorized split')
        outputs=audit_outputs(raw['splits'][split],datasets[split],bases[split]['outputs'])
        if record['examples']!=len(outputs) or record['output_identity']!=raw['splits'][split]['output_identity'] or sum(o['correct'] for o in outputs)!=record['correct_count'] or sum(not o['answer'] for o in outputs)!=record['invalid_count']:
            raise ValueError('selection record does not reproduce')
        traces[cid]=raw
    if len(list((root/'candidates').glob('*.json.gz')))!=len(traces):
        raise ValueError('unrecorded candidate artifact')
    first={}
    for record in records:
        first.setdefault(record['candidate']['sigma'],record['candidate']['candidate_id'])
    repeated_ids=set()
    for path in root.glob('repeat-*.json.gz'):
        repeated=read(path)
        repeated_ids.add(repeated['candidate']['candidate_id'])
        original=traces[repeated['candidate']['candidate_id']]
        if repeated['candidate_state_id']!=original['candidate_state_id'] or not repeated['restoration']['exact_base']:
            raise ValueError('reconstruction control failed')
        for s,output in repeated['splits'].items():
            audit_outputs(output,datasets[s],bases[s]['outputs'])
            if output['output_identity']!=original['splits'][s]['output_identity']: raise ValueError('candidate output repeat failed')
    if repeated_ids!=set(first.values()):
        raise ValueError('required first-candidate repeat evidence is missing')
    return manifest,bases,records,traces


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,default=Path('experiments/visual_line_tracing_protocol.json'))
    for phase in ('baseline','calibration','search','heldout'):
        p.add_argument('--'+phase,type=Path,required=phase in ('baseline','calibration'))
    p.add_argument('--calibration-stop',action='store_true',help='Audit the predeclared all-scales-failed operational stop; never emit a scientific NO-GO.')
    p.add_argument('--locks',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.calibration_stop and (a.search or a.heldout):
        p.error('calibration stop cannot include search or held-out candidate phases')
    if not a.calibration_stop and not (a.search and a.heldout):
        p.error('the scientific gate requires both search and held-out phases')
    if a.out.exists(): raise FileExistsError(a.out)
    protocol=read(a.protocol)
    datasets={s:load_split(protocol['dataset'],s) for s in ('selection','heldout')}
    selected,held=datasets.values()
    disjoint=not {r['seed'] for r in selected}&{r['seed'] for r in held} and not {r['image_sha256'] for r in selected}&{r['image_sha256'] for r in held}
    baseline_lock,sigma_lock=[read(a.locks/(n+'.json')) for n in ('baseline','sigma')]
    selection_lock=read(a.locks/'selection.json') if not a.calibration_stop else None
    for lock in [baseline_lock,sigma_lock]+([selection_lock] if selection_lock else []):
        if lock['protocol_sha256']!=sha(a.protocol) or lock['dataset_manifest_sha256']!=sha(Path(protocol['dataset'])/'manifest.json'):
            raise ValueError('locks changed')
    phase_names=('baseline','calibration') if a.calibration_stop else ('baseline','calibration','search','heldout')
    phases={phase:audit_phase(getattr(a,phase),phase,protocol,datasets,baseline_lock['base_id']) for phase in phase_names}
    for phase,names in {'calibration':['baseline'],'search':['baseline','sigma'],'heldout':['baseline','selection']}.items():
        if phase not in phases: continue
        for name in names:
            if (getattr(a,phase)/(name+'.json')).read_bytes()!=(a.locks/(name+'.json')).read_bytes():
                raise ValueError('pre-phase frozen lock differs from final lock')
    for phase,(manifest,bases,_,_) in phases.items():
        if manifest['protocol_sha256']!=sha(a.protocol): raise ValueError('phase protocol changed')
        for s,value in bases.items():
            if value['output_identity']!=baseline_lock['output_identities'][s]: raise ValueError('cross-phase base changed')
    baseline_summary={s:summarize(v['outputs'],datasets[s]) for s,v in phases['baseline'][1].items()}
    if baseline_summary!=baseline_lock['summary'] or not baseline_lock['accepted']:
        raise ValueError('baseline acceptance does not reproduce')
    conf=protocol['calibration']
    expected={(conf['seed_start']+j*conf['candidates_per_sigma']+i,sigma) for i in range(conf['candidates_per_sigma']) for j,sigma in enumerate(conf['sigmas'])}
    if {(r['candidate']['seed'],r['candidate']['sigma']) for r in phases['calibration'][2]}!=expected:
        raise ValueError('calibration recipes differ from the frozen plan')
    calibration=calibrate(phases['calibration'][2],baseline_lock['summary']['selection']['accuracy'],protocol['calibration'])
    if any(calibration[k]!=sigma_lock[k] for k in calibration):
        raise ValueError('sigma was not frozen by declared rule')
    if a.calibration_stop:
        if calibration['scale_valid'] or selection_lock or (a.locks/'selection.json').exists():
            raise ValueError('operational stop requires all three scales failed and no selected experts')
        a.out.mkdir(parents=True)
        decision={'decision':'STOP_operational_scale_failure','scientific_go_no_go':None,
                  'search_candidates':0,'heldout_candidates':0,'controls_pass':True,'sets_disjoint':bool(disjoint),
                  'reason':'All three sigmas meet the predeclared scale-failure rule. The held-out transfer hypothesis is untested.'}
        write(a.out/'manifest.json',{'source':revision_info(),'command':[sys.executable,*sys.argv],
              'protocol_sha256':sha(a.protocol),'all_raw_generations_rescored':True,
              'phase_manifest_sha256':{phase:sha(getattr(a,phase)/'run-manifest.json') for phase in phases},
              'dataset_disjoint':bool(disjoint),'candidate_heldout_generated':False})
        write(a.out/'baseline.json',baseline_summary)
        write(a.out/'calibration.json',calibration)
        write(a.out/'gate.json',decision)
        print(json.dumps(decision))
        return
    if not calibration['scale_valid']:
        raise ValueError('all-scales-failed calibration must stop before search')
    ranked=rank_selection(phases['search'][2],protocol['search'])
    if ranked!=selection_lock['ranked_candidates'] or ranked[:10]!=selection_lock['top10']:
        raise ValueError('selection-only ranking lock changed')
    if any(r['candidate']['sigma']!=sigma_lock['sigma'] for r in ranked): raise ValueError('search sigma changed')
    top10=selection_lock['top10']
    traces=phases['heldout'][3]
    if set(traces)!={r['candidate']['candidate_id'] for r in top10}: raise ValueError('held-out union differs from top10')
    for r in top10:
        raw=traces[r['candidate']['candidate_id']]
        if raw['candidate_state_id']!=r['candidate_state_id']: raise ValueError('selected state changed')
    outputs=[traces[r['candidate']['candidate_id']]['splits']['heldout']['outputs'] for r in top10]
    base_outputs=phases['heldout'][1]['heldout']['outputs']
    base=summarize(base_outputs,held)
    individual=[summarize(o,held) for o in outputs]
    methods={'best':individual[0]}
    all_votes={}
    correct_vectors={'base':[o['correct'] for o in base_outputs],'best':[o['correct'] for o in outputs[0]]}
    for k in (5,10):
        detail=[vote([o[j]['answer'] for o in outputs[:k]]) for j in range(500)]
        votes=[dict(v,id=r['id'],correct=v['answer']==r['answer'],finish_reason='vote') for v,r in zip(detail,held)]
        methods[f'top{k}_vote']=summarize(votes,held)
        correct_vectors[f'top{k}_vote']=[v['correct'] for v in votes]
        methods[f'top{k}_individual_mean']={'accuracy':sum(r['accuracy'] for r in individual[:k])/k,
            'difficulty':{l:{'accuracy':sum(r['difficulty'][l]['accuracy'] for r in individual[:k])/k} for l in ('easy','medium','hard')}}
        all_votes[str(k)]=votes
    decision=gate(base,methods,protocol['gate'],True,bool(disjoint))
    base_selection=baseline_lock['summary']['selection']['accuracy']
    scores=[r['correct_count']/150 for r in ranked]
    density={str(delta):sum(s>=base_selection+delta-1e-12 for s in scores)/len(scores) for delta in (.01,.03,.05)}
    distribution={'n':len(scores),'base_selection_accuracy':base_selection,'mean_accuracy':float(np.mean(scores)),
        'best_accuracy':max(scores),'minimum_accuracy':min(scores),'fraction_strictly_beating_base':sum(s>base_selection+1e-12 for s in scores)/len(scores),
        'expert_density':density,'histogram_correct_counts':dict(sorted(Counter(r['correct_count'] for r in ranked).items()))}
    bootstrap=np.random.default_rng(20261003).integers(0,500,size=(10000,500),dtype=np.int16)
    comparisons={}
    for name in ('best','top5_vote','top10_vote'):
        delta=np.array(correct_vectors[name],dtype=int)-np.array(correct_vectors['base'],dtype=int)
        comparisons[name]={'heldout_gain':float(delta.mean()),'improved_questions':int(np.sum(delta>0)),
            'worsened_questions':int(np.sum(delta<0)),
            'paired_descriptive_bootstrap95':np.quantile(delta[bootstrap].mean(axis=1),[.025,.975]).tolist()}
    costs={}
    for k in (1,5,10):
        costs[str(k)]={'unique_experts':k,'generation_rpcs':k,'image_requests':500*k,
                      'generated_tokens':sum(len(o['token_ids']) for expert in outputs[:k] for o in expert)}
    a.out.mkdir(parents=True)
    write(a.out/'manifest.json',{'source':revision_info(),'command':[sys.executable,*sys.argv],
        'protocol_sha256':sha(a.protocol),'all_raw_generations_rescored':True,'ranking_reselected':False,
        'phase_manifest_sha256':{phase:sha(getattr(a,phase)/'run-manifest.json') for phase in phases},
        'dataset_disjoint':bool(disjoint),'numpy':np.__version__})
    for name,value in [('calibration',calibration),('distribution',distribution),('heldout',{'base':base,**methods}),
                       ('individual_experts',individual),('comparisons',comparisons),('gate',decision),('deployment_cost',costs),('votes',all_votes)]:
        write(a.out/(name+'.json'),value)
    write(a.out/'per_question.json.gz',{'image_ids':[r['id'] for r in held],'correct':correct_vectors,
                                     'individual_correct':[[o['correct'] for o in v] for v in outputs]})
    lines=['# Frozen visual study: derived tables','','## Calibration','',
           '| Sigma | Mean accuracy | Best accuracy | Beat base | Catastrophic |','|---|---:|---:|---:|---:|']
    for row in calibration['table']:
        lines.append('| '+str(row['sigma'])+' | '+' | '.join(f"{100*row[x]:.2f}%" for x in ('mean_accuracy','best_accuracy','fraction_beating_base','catastrophic_fraction'))+' |')
    lines.extend(['','## Held-out comparison','','| Method | Overall | Easy | Medium | Hard |','|---|---:|---:|---:|---:|'])
    for name,row in {'base':base,**methods}.items():
        lines.append('| '+name+' | '+f"{100*row['accuracy']:.2f}%"+' | '+' | '.join(f"{100*row['difficulty'][l]['accuracy']:.2f}%" for l in ('easy','medium','hard'))+' |')
    (a.out/'tables.md').write_bytes(('\n'.join(lines)+'\n').encode())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    axes[0].hist(np.array(scores)*100,bins=np.arange(0,102,2),color='#4477aa')
    axes[0].axvline(100*base_selection,color='black',linestyle='--',label='Base selection accuracy')
    axes[0].set(xlabel='Selection accuracy (%)',ylabel='Number of candidates',title='300 frozen-sigma candidates')
    axes[0].legend(fontsize=8)
    keys=['base','best','top5_vote','top10_vote']; data={'base':base,**methods}
    axes[1].bar(keys,[100*data[k]['accuracy'] for k in keys],color=['#666666','#ee7733','#228833','#44aa99'])
    axes[1].set(ylabel='Held-out accuracy (%)',ylim=(0,100),title='500 unseen line-tracing images')
    fig.savefig(a.out/'distribution-and-heldout.png',dpi=180)
    fig.savefig(a.out/'distribution-and-heldout.svg')
    plt.close(fig)
    print(json.dumps(decision))


if __name__=='__main__': main()

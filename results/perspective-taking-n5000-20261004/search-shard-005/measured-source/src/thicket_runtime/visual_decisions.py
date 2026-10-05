"""Predeclared selection-only decisions and the unchanged visual feasibility gate."""
import hashlib


def summarize(outputs, rows):
    if [o['id'] for o in outputs]!=[r['id'] for r in rows]:
        raise ValueError('output/example identity mismatch')
    def group(indices):
        n=len(indices)
        return {'n':n,'correct_count':sum(bool(outputs[i]['correct']) for i in indices),
                'accuracy':sum(bool(outputs[i]['correct']) for i in indices)/n,
                'invalid_count':sum(not outputs[i]['answer'] for i in indices),
                'wrong_endpoint_count':sum(bool(outputs[i]['answer']) and not outputs[i]['correct'] for i in indices),
                'capped_count':sum(outputs[i]['finish_reason']=='length' for i in indices)}
    return {**group(list(range(len(rows)))),'difficulty':{level:group([i for i,r in enumerate(rows) if r['difficulty']==level]) for level in ('easy','medium','hard')}}


def calibrate(records, base_accuracy, config):
    if any(r['split']!='selection' or r['examples']!=150 for r in records):
        raise ValueError('calibration can only use selection150')
    result=[]
    for sigma in config['sigmas']:
        rows=[r for r in records if r['candidate']['sigma']==sigma]
        if len(rows)!=config['candidates_per_sigma']:
            raise ValueError('incomplete sigma calibration')
        scores=[r['correct_count']/r['examples'] for r in rows]
        catastrophic=[s<=max(.15,base_accuracy-.20)+1e-12 or r['invalid_count']/r['examples']>=.50 for s,r in zip(scores,rows)]
        item={'sigma':sigma,'n':len(rows),'mean_accuracy':sum(scores)/len(scores),'best_accuracy':max(scores),
              'best_correct_count':max(r['correct_count'] for r in rows),
              'fraction_beating_base':sum(s>base_accuracy+1e-12 for s in scores)/len(scores),
              'catastrophic_fraction':sum(catastrophic)/len(rows)}
        item['scale_failed']=(item['mean_accuracy']<=.275+1e-12 and item['best_accuracy']<=.40+1e-12) or item['catastrophic_fraction']>=.80-1e-12
        result.append(item)
    valid=not all(r['scale_failed'] for r in result)
    chosen=min(result,key=lambda r:(-r['best_correct_count'],-r['fraction_beating_base'],r['sigma'])) if valid else None
    return {'scale_valid':valid,'sigma':chosen['sigma'] if chosen else None,'table':result,
            'decision':'freeze_one_sigma' if valid else 'STOP_operational_scale_failure'}


def rank_selection(records, config):
    if len(records)!=config['candidates'] or any(r['split']!='selection' or r['examples']!=150 for r in records):
        raise ValueError('ranking requires the complete selection-only search')
    expected=set(range(config['seed_start'],config['seed_start']+config['candidates']))
    if {r['candidate']['seed'] for r in records}!=expected or len({r['candidate']['candidate_id'] for r in records})!=len(records):
        raise ValueError('search seeds or candidate identities differ from plan')
    return sorted(records,key=lambda r:(-r['correct_count'],hashlib.sha256(('visual-rank-v1:'+r['candidate']['candidate_id']).encode()).hexdigest()))


def gate(base, methods, config, controls_pass, disjoint):
    results={}
    for name,threshold in [('best',config['single_expert_gain']),('top5_vote',config['ensemble_gain']),('top10_vote',config['ensemble_gain'])]:
        value=methods[name]
        delta=value['accuracy']-base['accuracy']
        by_level={l:value['difficulty'][l]['accuracy']-base['difficulty'][l]['accuracy'] for l in ('easy','medium','hard')}
        positive=[l for l,d in by_level.items() if d>1e-12]
        difficulty_ok=len(positive)>=2 and any(l in positive for l in ('medium','hard'))
        results[name]={'gain':delta,'threshold':threshold,'difficulty_gain':by_level,'difficulty_pass':difficulty_ok,
                       'pass':delta>=threshold-1e-12 and difficulty_ok and controls_pass and disjoint}
    passed=any(r['pass'] for r in results.values())
    return {'pass':passed,'decision':'GO_stop' if passed else 'NO-GO_close_model_task_setup',
            'comparisons':results,'controls_pass':controls_pass,'sets_disjoint':disjoint}

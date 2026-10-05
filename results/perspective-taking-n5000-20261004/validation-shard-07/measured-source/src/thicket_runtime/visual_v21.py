"""Frozen Medium+Hard-only decisions for the v2.1 visual experiment."""
import hashlib
import json
from pathlib import Path
import subprocess

from .line_tracing import load_split, sha

PRIMARY = ('medium', 'hard')


def population_id(rows):
    return hashlib.sha256(json.dumps([(r['id'],r['image_sha256']) for r in rows],separators=(',',':')).encode()).hexdigest()


def verify_dataset(protocol):
    root = Path(protocol['dataset'])
    pin = protocol['dataset_pin']
    entries = subprocess.check_output(['git','ls-tree','-rz',pin['commit'],'--',root.as_posix()]).split(b'\0')
    paths = set()
    for entry in filter(None,entries):
        meta,name = entry.decode().split('\t',1)
        mode,kind,oid = meta.split()
        if kind!='blob' or mode!='100644': raise ValueError('unexpected dataset object')
        path = Path(name); data = path.read_bytes()
        if hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()!=oid:
            raise ValueError('dataset differs from frozen commit: '+name)
        paths.add(path.resolve())
    if paths!={p.resolve() for p in root.rglob('*') if p.is_file()}:
        raise ValueError('frozen dataset file inventory changed')
    if len(paths)!=654 or sha(root/'manifest.json')!=pin['manifest_sha256']:
        raise ValueError('frozen dataset manifest changed')
    return {'commit':pin['commit'],'files_verified':len(paths),'manifest_sha256':pin['manifest_sha256']}


def evaluation_sets(protocol, phase):
    root = protocol['dataset']
    result = {}
    if phase in ('baseline','calibration','search'):
        rows = load_split(root,'selection')
        primary = [r for r in rows if r['difficulty'] in PRIMARY]
        if len(primary)!=100 or [r['id'] for r in primary]!=protocol['primary']['selection_ids'] or population_id(primary)!=protocol['primary']['selection_population_sha256']:
            raise ValueError('primary selection population changed')
        result['selection_primary'] = primary
        if phase=='baseline': result['selection_easy'] = [r for r in rows if r['difficulty']=='easy']
    if phase in ('baseline','heldout'):
        rows = load_split(root,'heldout')
        primary = [r for r in rows if r['difficulty'] in PRIMARY]
        if len(rows)!=500 or len(primary)!=333 or [r['id'] for r in primary]!=protocol['primary']['heldout_ids']:
            raise ValueError('held-out population changed')
        result['heldout'] = rows
    return result


def summarize(outputs, rows):
    if [o['id'] for o in outputs]!=[r['id'] for r in rows]: raise ValueError('example identity mismatch')
    def group(indices):
        n=len(indices)
        if not n: raise ValueError('empty evaluation group')
        correct=sum(bool(outputs[i]['correct']) for i in indices)
        return {'n':n,'correct_count':correct,'accuracy':correct/n,
                'invalid_count':sum(not outputs[i]['answer'] for i in indices),
                'wrong_endpoint_count':sum(bool(outputs[i]['answer']) and not outputs[i]['correct'] for i in indices),
                'capped_count':sum(outputs[i]['finish_reason']=='length' for i in indices)}
    levels={level:group([i for i,r in enumerate(rows) if r['difficulty']==level])
            for level in ('easy','medium','hard') if any(r['difficulty']==level for r in rows)}
    result={'overall':group(list(range(len(rows)))),'difficulty':levels}
    indices=[i for i,r in enumerate(rows) if r['difficulty'] in PRIMARY]
    if indices: result['primary']=group(indices)
    return result


def baseline_summary(bases, datasets):
    rows=datasets['selection_primary']+datasets['selection_easy']
    outputs=bases['selection_primary']['outputs']+bases['selection_easy']['outputs']
    return {'selection':summarize(outputs,rows),
            'heldout':summarize(bases['heldout']['outputs'],datasets['heldout'])}


def capability_gate(summary, config):
    held=summary['heldout']
    value=held['primary']['accuracy']
    lo,hi=config['primary_heldout_accuracy_range']
    individual=any(held['difficulty'][l]['accuracy']>=config['one_primary_level_minimum']-1e-12 for l in PRIMARY)
    accepted=lo-1e-12<=value<=hi+1e-12 and individual
    return {'accepted':accepted,'decision':'PASS_capability_gate' if accepted else 'STOP_benchmark_capability_gate',
            'primary_heldout_accuracy':value,'primary_range_pass':lo-1e-12<=value<=hi+1e-12,
            'individual_minimum_pass':individual,
            'reason':None if accepted else ('below_30_percent' if value<lo else 'above_70_percent' if value>hi else 'individual_minimum_failed')}


def validate_selection_records(records, protocol):
    expected=protocol['primary']['selection_population_sha256']
    if any(r['split']!='selection' or r['evaluation_set']!='selection_primary' or r['examples']!=100 or r['population_sha256']!=expected for r in records):
        raise ValueError('decision requires the frozen 100 Medium+Hard selection examples only')
    if len({r['candidate']['candidate_id'] for r in records})!=len(records):
        raise ValueError('duplicate candidate identities')


def calibration_plan(config):
    return [(config['seed_start']+j*config['candidates_per_sigma']+i,sigma)
            for i in range(config['candidates_per_sigma']) for j,sigma in enumerate(config['sigmas'])]


def calibrate(records, base_correct, protocol):
    validate_selection_records(records,protocol)
    conf=protocol['calibration']
    if len(records)!=60 or {(r['candidate']['seed'],r['candidate']['sigma']) for r in records}!=set(calibration_plan(conf)):
        raise ValueError('calibration does not match the exact frozen 60 recipes')
    base=base_correct/100
    table=[]
    for sigma in conf['sigmas']:
        rows=[r for r in records if r['candidate']['sigma']==sigma]
        scores=[r['correct_count']/100 for r in rows]
        catastrophic=[s<=max(.15,base-.20)+1e-12 or r['invalid_count']/100>=.50 for s,r in zip(scores,rows)]
        item={'sigma':sigma,'n':len(rows),'mean_accuracy':sum(scores)/len(scores),
              'best_accuracy':max(scores),'best_correct_count':max(r['correct_count'] for r in rows),
              'fraction_beating_base':sum(s>base+1e-12 for s in scores)/len(scores),
              'fraction_beating_base_by_at_least_3pp':sum(s>=base+.03-1e-12 for s in scores)/len(scores),
              'catastrophic_fraction':sum(catastrophic)/len(rows)}
        item['no_search_signal']=(item['best_accuracy']<=base+conf['stop_best_gain']+1e-12 and
                                  item['mean_accuracy']<=base+conf['mean_equal_tolerance']+1e-12)
        table.append(item)
    stop=all(r['no_search_signal'] for r in table)
    chosen=min(table,key=lambda r:(-r['best_correct_count'],-r['fraction_beating_base'],r['sigma'])) if not stop else None
    return {'decision':'NO_GO_no_search_signal' if stop else 'freeze_one_sigma',
            'continue_search':not stop,'sigma':chosen['sigma'] if chosen else None,
            'base_primary_selection_correct':base_correct,'table':table,'easy_used':False,'heldout_used':False}


def rank_selection(records, protocol):
    validate_selection_records(records,protocol)
    conf=protocol['search']
    if len(records)!=300 or {r['candidate']['seed'] for r in records}!=set(range(conf['seed_start'],conf['seed_start']+300)):
        raise ValueError('ranking requires exactly 300 frozen new seeds')
    return sorted(records,key=lambda r:(-r['correct_count'],hashlib.sha256(('visual-rank-v1:'+r['candidate']['candidate_id']).encode()).hexdigest()))


def gate(base, methods, config, controls_pass, disjoint):
    if not controls_pass or not disjoint: raise ValueError('invalid controls cannot support a scientific decision')
    comparisons={}
    for name,threshold in [('best',config['single_expert_gain']),('top5_vote',config['ensemble_gain']),('top10_vote',config['ensemble_gain'])]:
        value=methods[name]
        delta=value['primary']['accuracy']-base['primary']['accuracy']
        levels={l:value['difficulty'][l]['accuracy']-base['difficulty'][l]['accuracy'] for l in PRIMARY}
        if name=='best':
            difficulty_ok=any(gain>1e-12 for gain in levels.values())
        else:
            difficulty_ok=all(gain>1e-12 for gain in levels.values()) or any(
                levels[l]>=config['substantial_gain']-1e-12 and levels[other]>=-config['maximum_other_regression']-1e-12
                for l,other in [('medium','hard'),('hard','medium')])
        comparisons[name]={'primary_gain':delta,'threshold':threshold,'difficulty_gain':levels,
                           'difficulty_pass':difficulty_ok,'pass':delta>=threshold-1e-12 and difficulty_ok}
    passed=any(v['pass'] for v in comparisons.values())
    return {'decision':'GO_visual_neural_thickets' if passed else 'NO_GO_close_model_task',
            'pass':passed,'comparisons':comparisons,'controls_pass':True,'sets_disjoint':True,'easy_used':False}

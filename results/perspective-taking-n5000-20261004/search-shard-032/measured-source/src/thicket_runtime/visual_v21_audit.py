"""Independent raw-output and phase-contract audits before v2.1 decisions."""
from pathlib import Path

from .candidate import CandidateSpec
from .line_tracing import parse_answer, sha
from .visual_runtime import output_identity, read
from .visual_v21 import PRIMARY, evaluation_sets, population_id, summarize


def outputs(result, rows, reference=None):
    values=result['outputs']
    cache=result['cache_audit']
    if len(values)!=len(rows) or output_identity(values)!=result['output_identity']:
        raise ValueError('output cardinality or identity mismatch')
    if cache['fresh_images_encoded']!=len(rows) or cache['after']['images_encoded']-cache['before']['images_encoded']!=len(rows) or cache['before']['cached_items_after']!=0:
        raise ValueError('stale or missing vision encoding')
    for i,(o,r) in enumerate(zip(values,rows)):
        answer=parse_answer(o['text'])
        category='correct' if answer==r['answer'] else 'wrong_endpoint' if answer else 'invalid_format'
        if o['id']!=r['id'] or o['image_sha256']!=r['image_sha256'] or o['answer']!=answer or o['correct']!=(answer==r['answer']) or o['failure_category']!=category:
            raise ValueError('raw text, example identity or score mismatch')
        if reference and o['prompt_token_ids']!=reference[i]['prompt_token_ids']:
            raise ValueError('tokenized prompt changed')
    return values


def exact_restore(raw):
    r=raw['restoration']
    if not r['exact_base'] or not r['finite'] or r['changed_values']!=0 or r['changed_buffer_values']!=0 or r['max_abs']!=0:
        raise ValueError('bitwise restoration control failed')


def phase(root, protocol, protocol_path, baseline_lock=None):
    root=Path(root)
    manifest=read(root/'run-manifest.json')
    name=manifest['phase']
    if manifest['status']!='complete': raise ValueError('incomplete/invalid execution is operational failure')
    index=read(root/'sha256.json')
    if set(index)!={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p!=root/'sha256.json'}:
        raise ValueError('phase artifact inventory changed')
    for file,expected in index.items():
        if sha(root/file)!=expected: raise ValueError('phase artifact changed: '+file)
    if manifest['protocol_sha256']!=sha(protocol_path) or manifest['dataset_manifest_sha256']!=protocol['dataset_pin']['manifest_sha256']:
        raise ValueError('phase protocol or dataset differs')
    if manifest['versions']!=protocol['required_versions'] or manifest['model_revision']!=protocol['model_revision'] or manifest['processor_revision']!=protocol['processor_revision']:
        raise ValueError('model/environment differs')
    if manifest['dataset_proof']!={'commit':protocol['dataset_pin']['commit'],'files_verified':654,'manifest_sha256':protocol['dataset_pin']['manifest_sha256']}:
        raise ValueError('dataset pin was not verified')
    datasets=evaluation_sets(protocol,name)
    expected_sets={k:{'examples':len(v),'population_sha256':population_id(v)} for k,v in datasets.items()}
    if manifest['evaluation_sets']!=expected_sets: raise ValueError('unauthorized evaluation population')
    native=read(root/'native-state.json')
    if native['base_id']!=manifest['base_id'] or native['base_id']!=protocol['expected_base_id'] or not native['all_parameters_perturbed'] or native['perturb_visual']!='1':
        raise ValueError('native state or parameter scope mismatch')
    if baseline_lock and native['base_id']!=baseline_lock['base_id']: raise ValueError('cross-phase base state changed')
    controls=read(root/'controls.json')
    required={'base_state_unchanged','final_base_exact'} | {s+'_base_repeat_equal' for s in datasets}
    bases,repeated=read(root/'base.json.gz'),read(root/'base-repeat.json.gz')
    if set(bases)!=set(datasets) or set(repeated)!=set(datasets): raise ValueError('wrong base evaluation sets')
    for s,rows in datasets.items():
        outputs(bases[s],rows)
        outputs(repeated[s],rows,bases[s]['outputs'])
        if bases[s]['output_identity']!=repeated[s]['output_identity']: raise ValueError('base repeat differs')
        if baseline_lock:
            required.add('base_'+s+'_matches_baseline')
            if bases[s]['output_identity']!=baseline_lock['output_identities'][s]: raise ValueError('cross-phase base outputs differ')
    if name=='baseline':
        raw=read(root/'zero.json.gz')
        zero=CandidateSpec(native['base_id'],5100000,0.)
        if raw['candidate']['candidate_id']!=zero.candidate_id or raw['candidate_state_id']!=native['base_id'] or set(raw['splits'])!=set(datasets):
            raise ValueError('zero recipe/state differs')
        exact_restore(raw)
        for s,rows in datasets.items():
            outputs(raw['splits'][s],rows,bases[s]['outputs'])
            if raw['splits'][s]['output_identity']!=bases[s]['output_identity']: raise ValueError('zero output differs')
        required|={s+'_zero_equal' for s in datasets}|{'zero_restore_exact'}
    records=read(root/'records.json')
    traces,first={},{}
    if manifest['completed_candidates']!=len(records): raise ValueError('completed-candidate count mismatch')
    if name=='baseline' and records: raise ValueError('nonzero candidate before capability gate')
    for record in records:
        c=record['candidate']; cid=c['candidate_id']
        spec=CandidateSpec(**{k:c[k] for k in ('base_id','seed','sigma','rng','sign')})
        if c['base_id']!=native['base_id'] or c['rng']!='randopt-per-tensor-v1' or c['sign']!=1 or c['parameter_mask']!='all-parameters-including-vision' or c['parameter_mask_sha256']!=native['parameter_mask_sha256']:
            raise ValueError('candidate recipe scope changed')
        raw=read(root/'candidates'/(cid+'.json.gz'))
        if spec.candidate_id!=cid or cid in traces or raw['candidate']!=c or raw['candidate_state_id']!=record['candidate_state_id'] or set(raw['splits'])!=set(datasets):
            raise ValueError('candidate identity or population mismatch')
        exact_restore(raw)
        s=next(iter(datasets)); rows=datasets[s]
        values=outputs(raw['splits'][s],rows,bases[s]['outputs'])
        metric=summarize(values,rows)['primary']
        primary=[r for r in rows if r['difficulty'] in PRIMARY]
        if (record['evaluation_set']!=s or record['split']!=('heldout' if name=='heldout' else 'selection') or
            record['examples']!=metric['n'] or record['correct_count']!=metric['correct_count'] or
            record['invalid_count']!=metric['invalid_count'] or record['population_sha256']!=population_id(primary) or
            record['output_identity']!=raw['splits'][s]['output_identity']):
            raise ValueError('primary score record does not reproduce')
        traces[cid]=raw
        first.setdefault(c['sigma'],cid)
    if len(list((root/'candidates').glob('*.json.gz')))!=len(traces): raise ValueError('unrecorded candidate file')
    repeats=[]
    for path in root.glob('repeat-*.json.gz'):
        raw=read(path); cid=raw['candidate']['candidate_id']; repeats.append(cid)
        original=traces[cid]
        if raw['candidate']!=original['candidate'] or raw['candidate_state_id']!=original['candidate_state_id'] or set(raw['splits'])!=set(datasets):
            raise ValueError('repeated state/recipe differs')
        exact_restore(raw)
        for s,rows in datasets.items():
            outputs(raw['splits'][s],rows,bases[s]['outputs'])
            if raw['splits'][s]['output_identity']!=original['splits'][s]['output_identity']: raise ValueError('candidate output repeat differs')
    if len(repeats)!=len(first) or set(repeats)!=set(first.values()): raise ValueError('required candidate repeats missing')
    required|={prefix+str(sigma) for sigma in first for prefix in ('same_state_','same_outputs_')}
    if set(controls)!=required or not all(value is True for value in controls.values()):
        raise ValueError('required control missing or failed')
    return {'manifest':manifest,'datasets':datasets,'bases':bases,'records':records,'traces':traces}

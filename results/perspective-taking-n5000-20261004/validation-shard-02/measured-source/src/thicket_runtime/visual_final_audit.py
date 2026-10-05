"""Reproduce final-study records and controls from raw generations."""
from dataclasses import asdict
from pathlib import Path

from .candidate import CandidateSpec
from .line_tracing import sha
from .visual_runtime import read
from .visual_v21_audit import outputs, exact_restore
from .visual_final import evaluation_sets, population_id, rank_selection, summarize


def phase(root, protocol, protocol_path, selected=None):
    root = Path(root)
    manifest = read(root/'run-manifest.json')
    name = manifest['phase']
    if manifest['status'] != 'complete': raise ValueError('operational failure: phase incomplete')
    index = read(root/'sha256.json')
    if set(index) != {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p != root/'sha256.json'}:
        raise ValueError('phase inventory changed')
    for file, expected in index.items():
        if sha(root/file) != expected: raise ValueError('phase checksum mismatch: '+file)
    if manifest['protocol_sha256'] != sha(protocol_path) or manifest['dataset_manifest_sha256'] != protocol['dataset_pin']['manifest_sha256']:
        raise ValueError('protocol/dataset changed')
    if manifest['versions'] != protocol['required_versions'] or manifest['model_revision'] != protocol['model_revision'] or manifest['processor_revision'] != protocol['processor_revision']:
        raise ValueError('environment/model changed')
    if manifest['dataset_proof'] != {'commit':protocol['dataset_pin']['commit'],'files_verified':654,'manifest_sha256':protocol['dataset_pin']['manifest_sha256']}:
        raise ValueError('dataset proof missing')
    if manifest['upstream']['commit'] != protocol['upstream_commit']: raise ValueError('upstream changed')
    datasets = evaluation_sets(protocol,name)
    if manifest['evaluation_sets'] != {s:{'examples':len(rows),'population_sha256':population_id(rows)} for s,rows in datasets.items()}:
        raise ValueError('unauthorized evaluation population')
    native = read(root/'native-state.json')
    if native['base_id'] != manifest['base_id'] or native['base_id'] != protocol['expected_base_id'] or not native['all_parameters_perturbed'] or native['perturb_visual'] != '1':
        raise ValueError('native state/parameter scope changed')
    historical = read(Path(protocol['reference']['baseline'])/'base.json.gz')
    controls = read(root/'controls.json')
    required = {'initial_base_exact','final_base_exact'}
    if name in ('search','heldout'):
        required |= {'zero_matches_committed_base','zero_restore_exact','repeated_state_equal','repeated_outputs_equal'}
        zero = read(root/'zero.json.gz')
        spec = CandidateSpec(native['base_id'],5100000,0.)
        if zero['candidate'] != asdict(spec)|{'candidate_id':spec.candidate_id} or zero['candidate_state_id'] != native['base_id'] or set(zero['splits']) != set(datasets):
            raise ValueError('zero recipe/state changed')
        exact_restore(zero)
        for s, rows in datasets.items():
            outputs(historical[s],rows)
            outputs(zero['splits'][s],rows,historical[s]['outputs'])
            if zero['splits'][s]['output_identity'] != historical[s]['output_identity']:
                raise ValueError('zero differs from committed base')
    records = read(root/'records.json')
    expected_records = [protocol['diagnostic']['selected']] if name == 'diagnostic' else selected['top10'] if name == 'heldout' and selected else None
    expected = {r['candidate']['candidate_id']:r for r in expected_records} if expected_records else {}
    if name == 'search': rank_selection(records,protocol)
    elif expected_records is None or [r['candidate'] for r in records] != [r['candidate'] for r in expected_records]:
        raise ValueError('diagnostic/heldout recipes differ from frozen selection')
    if manifest['completed_candidates'] != len(records): raise ValueError('candidate count mismatch')
    traces = {}
    for record in records:
        c = record['candidate']; cid = c['candidate_id']
        spec = CandidateSpec(native['base_id'],c['seed'],protocol['sigma'])
        recipe = asdict(spec)|{'candidate_id':spec.candidate_id,'parameter_mask':'all-parameters-including-vision','parameter_mask_sha256':native['parameter_mask_sha256']}
        raw = read(root/'candidates'/(cid+'.json.gz'))
        if c != recipe or raw['candidate'] != c or raw['candidate_state_id'] != record['candidate_state_id'] or cid in traces or set(raw['splits']) != set(datasets):
            raise ValueError('candidate identity/state/population changed')
        if expected and (raw['candidate_state_id'] != expected[cid]['candidate_state_id'] or raw['expected_state_id'] != expected[cid]['candidate_state_id'] or raw['fingerprint_checked_before_generation'] is not True):
            raise ValueError('pre-generation reconstruction control failed')
        exact_restore(raw)
        s = next(iter(datasets)); rows = datasets[s]
        values = outputs(raw['splits'][s],rows,historical[s]['outputs'])
        metric = summarize(values,rows)
        if record['split'] != s or record['examples'] != len(rows) or record['population_sha256'] != population_id(rows) or record['correct_count'] != metric['correct_count'] or record['invalid_count'] != metric['invalid_count'] or record['output_identity'] != raw['splits'][s]['output_identity']:
            raise ValueError('score record does not reproduce from raw text')
        traces[cid] = raw
    if len(list((root/'candidates').glob('*.json.gz'))) != len(traces): raise ValueError('unrecorded candidate file')
    if name in ('search','heldout'):
        repeated = read(root/'repeat.json.gz'); original = traces[records[0]['candidate']['candidate_id']]
        if repeated['candidate'] != original['candidate'] or repeated['candidate_state_id'] != original['candidate_state_id'] or set(repeated['splits']) != set(datasets):
            raise ValueError('repeated candidate identity/state differs')
        exact_restore(repeated)
        for s,rows in datasets.items():
            outputs(repeated['splits'][s],rows,historical[s]['outputs'])
            if repeated['splits'][s]['output_identity'] != original['splits'][s]['output_identity']:
                raise ValueError('repeated candidate outputs differ')
    if set(controls) != required or not all(v is True for v in controls.values()):
        raise ValueError('required control missing/failed')
    return {'manifest':manifest,'records':records,'traces':traces,'datasets':datasets}

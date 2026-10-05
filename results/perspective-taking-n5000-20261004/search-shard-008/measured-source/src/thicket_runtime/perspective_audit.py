"""Re-derive every Perspective-Taking N=5000 phase record and control from raw generations."""
from pathlib import Path

from .line_tracing import sha
from .visual_runtime import read
from .perspective import PHASE_SPLITS, ident, splits
from .omnispatial_data import score
from .perspective_runtime import ZERO_SEED, output_identity, planned, recipe


def outputs(result, rows, reference=None):
    """Official direct scorer re-applied to raw text; fresh encoding; prompt identity."""
    values = result['outputs']
    cache = result['cache_audit']
    if len(values) != len(rows) or output_identity(values) != result['output_identity']:
        raise ValueError('output cardinality or identity mismatch')
    if cache['fresh_images_encoded'] != len(rows) or cache['before']['cached_items_after'] != 0:
        raise ValueError('stale or missing vision encoding')
    for i, (o, r) in enumerate(zip(values, rows)):
        if o['uid'] != r['uid'] or o['image_sha256'] != r['image_sha256'] or {k: o[k] for k in ('prediction', 'correct', 'valid_letter')} != score(o['text'], r['answer']):
            raise ValueError('raw text, example identity or score mismatch')
        if reference and o['prompt_token_ids_sha256'] != reference[i]['prompt_token_ids_sha256']:
            raise ValueError('tokenized prompt changed')
    return values


def exact_restore(raw):
    r = raw['restoration']
    if not r['exact_base'] or not r['finite'] or r['changed_values'] != 0 or r['changed_buffer_values'] != 0 or r['max_abs'] != 0:
        raise ValueError('bitwise restoration control failed')


def phase(root, protocol, protocol_path, locks, shard=None, base_reference=None):
    """base_reference: {split: base outputs} from the baseline phase (prompt identity)."""
    root = Path(root)
    m = read(root/'run-manifest.json')
    name = m['phase']
    if m['status'] != 'complete': raise ValueError('operational failure: phase incomplete: '+str(root))
    if m['shard'] != shard: raise ValueError('wrong search shard')
    index = read(root/'sha256.json')
    if set(index) != {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p != root/'sha256.json'}:
        raise ValueError('phase inventory changed')
    for file, expected in index.items():
        if sha(root/file) != expected: raise ValueError('phase checksum mismatch: '+file)
    if m['protocol_sha256'] != sha(protocol_path) or m['dataset_proof']['manifest_sha256'] != protocol['dataset_pin']['manifest_sha256']:
        raise ValueError('protocol/dataset changed')
    if m['versions'] != protocol['required_versions'] or m['model_revision'] != protocol['model_revision'] or m['upstream']['commit'] != protocol['upstream_commit']:
        raise ValueError('environment/model/upstream changed')
    sets = splits(protocol)
    datasets = {s: sets[s] for s in PHASE_SPLITS[name]}
    if m['evaluation_sets'] != {s: {'examples': len(r), 'population_sha256': ident(r)} for s, r in datasets.items()}:
        raise ValueError('unauthorized evaluation population')
    native = read(root/'native-state.json')
    base_id = native['base_id']
    if base_id != m['base_id'] or not native['all_parameters_perturbed'] or native['perturb_visual'] != '1':
        raise ValueError('native state/parameter scope changed')
    if native.get('fingerprint_contract') != 'state-sha256-tree-v1': raise ValueError('fingerprint contract changed')
    if 'baseline' in locks and (base_id != locks['baseline']['base_id'] or native['base_id_flat_sha256'] != locks['baseline']['base_id_flat_sha256']):
        raise ValueError('base changed across phases')
    for lock in locks:
        if read(root/('lock-'+lock+'.json')) != locks[lock]: raise ValueError('phase used a different lock: '+lock)
    preprocess = read(root/'preprocess.json')
    if set(preprocess) != {r['uid'] for rows in datasets.values() for r in rows}: raise ValueError('preprocessing record incomplete')
    if 'baseline' in locks and any(preprocess[u] != locks['baseline']['preprocess'][u] for u in preprocess):
        raise ValueError('image preprocessing differs from baseline')
    if name == 'baseline':
        base = read(root/'base.json.gz')
        prompts = {s: outputs(base[s], rows) for s, rows in datasets.items()}
        reference = {s: base[s]['output_identity'] for s in datasets}
        repeat = read(root/'base-repeat.json.gz')
        for s, rows in datasets.items():
            outputs(repeat[s], rows, prompts[s])
            if repeat[s]['output_identity'] != reference[s]: raise ValueError('base repeat differs')
    else:
        prompts = base_reference
        reference = {s: locks['baseline']['output_identities'][s] for s in datasets}
    zero = read(root/'zero.json.gz')
    if zero['candidate'] != recipe(base_id, ZERO_SEED, 0., native['parameter_mask_sha256']) or zero['candidate_state_id'] != base_id:
        raise ValueError('zero recipe/state changed')
    exact_restore(zero)
    for s, rows in datasets.items():
        outputs(zero['splits'][s], rows, prompts[s] if prompts else None)
        if zero['splits'][s]['output_identity'] != reference[s]: raise ValueError('zero differs from base')
    plan = planned(name, protocol, locks, shard)
    records = read(root/'records.json')
    if len(records) != len(plan) or m['completed_candidates'] != len(plan): raise ValueError('candidate count mismatch')
    traces = {}
    for record, (seed, sigma, expected) in zip(records, plan):
        c = recipe(base_id, seed, sigma, native['parameter_mask_sha256'])
        cid = c['candidate_id']
        raw = read(root/'candidates'/(cid+'.json.gz'))
        if record['candidate'] != c or raw['candidate'] != c or raw['candidate_state_id'] != record['candidate_state_id'] or cid in traces:
            raise ValueError('candidate identity/state changed')
        if expected and (expected['candidate'] != c or raw['candidate_state_id'] != expected['candidate_state_id'] or
                         raw['expected_state_id'] != expected['candidate_state_id'] or raw['fingerprint_checked_before_generation'] is not True):
            raise ValueError('candidate fingerprint does not match its search state')
        if raw['candidate_state_id'] == base_id: raise ValueError('nonzero candidate equals base state')
        exact_restore(raw)
        s = next(iter(datasets))
        values = outputs(raw['splits'][s], datasets[s], prompts[s] if prompts else None)
        if (record['split'] != s or record['examples'] != len(values) or record['correct_count'] != sum(o['correct'] for o in values) or
                record['invalid_count'] != sum(not o['valid_letter'] for o in values) or record['output_identity'] != raw['splits'][s]['output_identity']):
            raise ValueError('score record does not reproduce from raw text')
        traces[cid] = raw
    if len(list((root/'candidates').glob('*.json.gz'))) != len(traces): raise ValueError('unrecorded candidate file')
    controls = read(root/'controls.json')
    required = {'initial_base_exact', 'zero_reproduces_base', 'zero_restore_exact', 'final_base_exact'}
    if name == 'baseline':
        required |= {'base_repeat_equal', 'fast_controls_equivalent'}
        if not all(read(root/'fast-control-equivalence.json')['checks'].values()): raise ValueError('fast controls not proven equivalent')
    else:
        required |= {'repeated_state_equal', 'repeated_outputs_equal', 'preprocessing_matches_baseline'}
        rep, orig = read(root/'repeat.json.gz'), traces[records[0]['candidate']['candidate_id']]
        if rep['candidate'] != orig['candidate'] or rep['candidate_state_id'] != orig['candidate_state_id']:
            raise ValueError('repeated candidate state differs')
        exact_restore(rep)
        for s, rows in datasets.items():
            outputs(rep['splits'][s], rows)
            if rep['splits'][s]['output_identity'] != orig['splits'][s]['output_identity']: raise ValueError('repeat outputs differ')
    if set(controls) != required or not all(v is True for v in controls.values()):
        raise ValueError('required control missing/failed')
    return {'manifest': m, 'native': native, 'records': records, 'traces': traces, 'datasets': datasets,
            'preprocess': preprocess, 'base': read(root/'base.json.gz') if name == 'baseline' else None}

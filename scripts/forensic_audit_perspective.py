"""Independent forensic audit of the Perspective-Taking N=5000 study.

Deliberately imports NOTHING from thicket_runtime. It reads raw artifacts with the
standard library (json/gzip/hashlib/csv), re-parses every raw model response with
its own parser, rehashes images from the original PNG files, and recomputes
candidate identities from first principles. Output goes to a NEW directory.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re

ROOT = Path('results/perspective-taking-n5000-20261004')
DATA = Path('examples/omnispatial-perspective-taking')
OUT = ROOT / 'forensic-audit'
SEED, SIGMA = 9504111, 0.002
LETTERS = 'ABCD'


def load(path):
    raw = Path(path).read_bytes()
    return json.loads(gzip.decompress(raw) if str(path).endswith('.gz') else raw)


def rows(split):
    return [json.loads(l) for l in (DATA / f'{split}.jsonl').read_text(encoding='utf-8').splitlines()]


def parse(text):
    """Independent re-implementation of the frozen 'direct' rule from its written
    specification: skip leading whitespace, take the first character, upper-case
    it; valid only if it is one of A-D. Implemented with a regex, not str.strip."""
    m = re.match(r'\s*(\S)', text)
    ch = m.group(1).upper() if m else ''
    return ch if ch in LETTERS else ''


def find_candidate_file(phase_dir, seed):
    hits = []
    for f in sorted((ROOT / phase_dir / 'candidates').glob('*.json.gz')):
        raw = load(f)
        if raw['candidate']['seed'] == seed:
            hits.append((f, raw))
    if len(hits) != 1:
        raise SystemExit(f'expected exactly one file for seed {seed} in {phase_dir}, found {len(hits)}')
    return hits[0]


def transition(b, c):
    return {(True, True): 'both_correct', (False, True): 'base_wrong->cand_correct',
            (True, False): 'base_correct->cand_wrong', (False, False): 'both_wrong'}[(b, c)]


def phase_a(images_root):
    OUT.mkdir(parents=True, exist_ok=True)
    base = load(ROOT / 'baseline/base.json.gz')
    where = {'search': 'search-shard-041', 'validation': 'validation-shard-01', 'test': 'test'}
    reported = {'validation': (79, 95), 'test': (259, 244)}
    summary, all_rows, files = {}, [], {}
    for split, phase_dir in where.items():
        recs = rows(split)
        f, cand = find_candidate_file(phase_dir, SEED)
        files[split] = {'file': f.as_posix(), 'sha256': hashlib.sha256(f.read_bytes()).hexdigest(),
                        'candidate': cand['candidate'], 'state': cand['candidate_state_id']}
        bo, co = base[split]['outputs'], cand['splits'][split]['outputs']
        if len(bo) != len(recs) or len(co) != len(recs):
            raise SystemExit('cardinality mismatch in ' + split)
        nb = nc = 0
        for r, b, c in zip(recs, bo, co):
            if b['uid'] != r['uid'] or c['uid'] != r['uid']:
                raise SystemExit('row/output identity mismatch in ' + split)
            img = images_root / r['source_split'] / r['image_member'].rsplit('/', 1)[1]
            h = hashlib.sha256(img.read_bytes()).hexdigest()
            if h != r['image_sha256'] or b['image_sha256'] != h or c['image_sha256'] != h:
                raise SystemExit('image hash mismatch for ' + r['uid'])
            gold = LETTERS[r['answer']]
            pb, pc = parse(b['text']), parse(c['text'])
            bc, cc = pb == gold, pc == gold
            nb += bc; nc += cc
            all_rows.append({'split': split, 'instance_id': r['uid'], 'image_id': r['image_member'].rsplit('/', 1)[1],
                             'image_sha256': h, 'subtask': r['sub_task_type'], 'options': json.dumps(r['options'], ensure_ascii=False),
                             'correct_option': gold, 'raw_base_output': b['text'], 'parsed_base_option': pb, 'base_correct': bc,
                             'raw_candidate_output': c['text'], 'parsed_candidate_option': pc, 'candidate_correct': cc,
                             'transition': transition(bc, cc),
                             'orig_scorer_base_correct': b['correct'], 'orig_scorer_candidate_correct': c['correct'],
                             'prompt_hash_equal': b['prompt_token_ids_sha256'] == c['prompt_token_ids_sha256']})
        summary[split] = {'n': len(recs), 'base_correct': nb, 'candidate_correct': nc,
                          'base_accuracy': nb / len(recs), 'candidate_accuracy': nc / len(recs),
                          'gain_pp': 100 * (nc - nb) / len(recs)}
    mismatches = {s: (summary[s]['base_correct'], summary[s]['candidate_correct'], reported[s])
                  for s in reported if (summary[s]['base_correct'], summary[s]['candidate_correct']) != reported[s]}
    scorer_disagreements = [r['instance_id'] for r in all_rows
                            if r['base_correct'] != r['orig_scorer_base_correct'] or r['candidate_correct'] != r['orig_scorer_candidate_correct']]
    status = 'AUDIT_BUG_SCORING_OR_ARTIFACT_MISMATCH' if mismatches or scorer_disagreements else 'PHASE_A_REPRODUCED'
    with open(OUT / 'phaseA_per_question.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
        w.writeheader(); w.writerows(all_rows)
    result = {'status': status, 'summary': summary, 'reported': reported, 'mismatches': mismatches,
              'independent_vs_original_scorer_disagreements': scorer_disagreements,
              'prompt_hash_equal_all': all(r['prompt_hash_equal'] for r in all_rows),
              'images_rehashed_from_original_png': len(all_rows), 'candidate_files': files}
    (OUT / 'phaseA_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('status', 'summary', 'mismatches', 'prompt_hash_equal_all')}, indent=1))
    print('scorer disagreements:', len(scorer_disagreements))


def template(q):
    """Coarse question template: first five lower-cased words with object nouns kept."""
    return ' '.join(re.sub(r'[^a-z ]', ' ', q.lower()).split()[:5])


def phase_b():
    from collections import Counter, defaultdict
    with open(OUT / 'phaseA_per_question.csv', encoding='utf-8') as fh:
        rows_ = list(csv.DictReader(fh))
    meta = {r['uid']: r for s in ('search', 'validation', 'test') for r in rows(s)}
    report = {}
    for split in ('search', 'validation', 'test'):
        rs = [r for r in rows_ if r['split'] == split]
        for r in rs:
            m = meta[r['instance_id']]
            r['question_template'] = template(m['question'])
            r['image_numeric_id'] = int(m['id'].split('_')[0])
            r['image_id_bin'] = f"{(r['image_numeric_id'] // 100) * 100:04d}-{(r['image_numeric_id'] // 100) * 100 + 99:04d}"
        with open(OUT / f'candidate_9504111_{split}_transitions.csv', 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rs[0])); w.writeheader(); w.writerows(rs)
        def tab(key):
            t = defaultdict(Counter)
            for r in rs: t[r[key]][r['transition']] += 1
            return {k: dict(v, net=v['base_wrong->cand_correct'] - v['base_correct->cand_wrong'], n=sum(v.values())) for k, v in sorted(t.items())}
        report[split] = {'overall': dict(Counter(r['transition'] for r in rs)),
                         'by_subtask': tab('subtask'), 'by_true_answer': tab('correct_option'),
                         'by_base_prediction': tab('parsed_base_option'), 'by_candidate_prediction': tab('parsed_candidate_option'),
                         'by_image_id_bin': tab('image_id_bin'), 'by_question_template': tab('question_template'),
                         'flow_base_pred_to_candidate_pred': dict(Counter(f"{r['parsed_base_option'] or '-'}->{r['parsed_candidate_option'] or '-'}" for r in rs)),
                         'gains_by_flow': dict(Counter(f"{r['parsed_base_option']}->{r['parsed_candidate_option']} (gold {r['correct_option']})"
                                                       for r in rs if r['transition'] == 'base_wrong->cand_correct')),
                         'losses_by_flow': dict(Counter(f"{r['parsed_base_option']}->{r['parsed_candidate_option']} (gold {r['correct_option']})"
                                                        for r in rs if r['transition'] == 'base_correct->cand_wrong'))}
    (OUT / 'phaseB_transitions.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
    for split in ('validation', 'test'):
        r = report[split]
        print('==', split, r['overall'])
        for k in ('by_subtask', 'by_true_answer', 'by_candidate_prediction', 'by_base_prediction'):
            print(' ', k, {a: (b.get('base_wrong->cand_correct', 0), b.get('base_correct->cand_wrong', 0), b['net']) for a, b in r[k].items()})
        print('  base->cand flow', r['flow_base_pred_to_candidate_pred'])
        print('  top template nets', sorted(((b['net'], a, b['n']) for a, b in r['by_question_template'].items()), reverse=True)[:6])
        print('  worst template nets', sorted(((b['net'], a, b['n']) for a, b in r['by_question_template'].items()))[:4])


def tvd(a, b):
    keys = set(a) | set(b)
    na, nb = sum(a.values()), sum(b.values())
    return 0.5 * sum(abs(a.get(k, 0) / na - b.get(k, 0) / nb) for k in keys)


def chi2(tables):
    from scipy.stats import chi2_contingency
    keys = sorted(set().union(*tables))
    keep = [k for k in keys if any(t.get(k, 0) for t in tables)]
    stat, pv, dof, _ = chi2_contingency([[t.get(k, 0) for k in keep] for t in tables])
    return {'chi2': float(stat), 'dof': int(dof), 'p': float(pv)}


FEATURES = {'subtask': lambda r: r['sub_task_type'], 'answer': lambda r: LETTERS[r['answer']],
            'subtask_x_answer': lambda r: r['sub_task_type'] + '|' + LETTERS[r['answer']],
            'template': lambda r: template(r['question'])}


def phase_c():
    from collections import Counter
    base = load(ROOT / 'baseline/base.json.gz')
    sp = {s: rows(s) for s in ('search', 'validation', 'test')}
    rep = {}
    for s, rs in sp.items():
        out = base[s]['outputs']
        subs = sorted({r['sub_task_type'] for r in rs})
        rep[s] = {'n': len(rs), 'unique_images': len({r['image_sha256'] for r in rs}),
                  'questions_per_image': dict(Counter(Counter(r['image_sha256'] for r in rs).values())),
                  'subtask': dict(Counter(r['sub_task_type'] for r in rs)),
                  'answer': dict(Counter(LETTERS[r['answer']] for r in rs)),
                  'subtask_x_answer': {st: dict(Counter(LETTERS[r['answer']] for r in rs if r['sub_task_type'] == st)) for st in subs},
                  'source_split': dict(Counter(r['source_split'] for r in rs)),
                  'image_mode': dict(Counter(r['image_mode'] for r in rs)),
                  'image_pixels_median': sorted(r['image_size'][0] * r['image_size'][1] for r in rs)[len(rs) // 2],
                  'option_count': dict(Counter(len(r['options']) for r in rs)),
                  'templates_top': Counter(template(r['question']) for r in rs).most_common(12),
                  'distinct_templates': len({template(r['question']) for r in rs}),
                  'base_accuracy_by_subtask': {st: sum(parse(o['text']) == LETTERS[r['answer']] for r, o in zip(rs, out) if r['sub_task_type'] == st)
                                               / sum(r['sub_task_type'] == st for r in rs) for st in subs},
                  'base_prediction': dict(Counter(parse(o['text']) or 'invalid' for o in out))}
    comp = {}
    for a, b in (('search', 'validation'), ('search', 'test'), ('validation', 'test')):
        ta = {f: Counter(fn(r) for r in sp[a]) for f, fn in FEATURES.items()}
        tb = {f: Counter(fn(r) for r in sp[b]) for f, fn in FEATURES.items()}
        c = {f: {'tvd': tvd(ta[f], tb[f]), **(chi2([ta[f], tb[f]]) if f != 'template' else {})} for f in FEATURES}
        tA, tB = set(ta['template']), set(tb['template'])
        c['template_overlap'] = {'shared_templates': len(tA & tB), f'{a}_templates': len(tA), f'{b}_templates': len(tB),
                                 f'{b}_questions_with_template_seen_in_{a}': sum(template(r['question']) in tA for r in sp[b])}
        comp[f'{a}_vs_{b}'] = c
    (OUT / 'phaseC_splits.json').write_text(json.dumps({'splits': rep, 'comparisons': comp}, indent=1, default=str), encoding='utf-8')
    for s, r in rep.items():
        print(s, r['n'], 'imgs', r['unique_images'], r['subtask'], r['answer'], 'base_acc',
              {k: round(100 * v, 1) for k, v in r['base_accuracy_by_subtask'].items()}, 'px', r['image_pixels_median'], 'templates', r['distinct_templates'])
    for k, v in comp.items():
        print(k, {f: (round(x['tvd'], 3), round(x.get('p', float('nan')), 4)) for f, x in v.items() if f != 'template_overlap'}, v['template_overlap'])


def norm(q):
    return ' '.join(re.sub(r'[^a-z0-9 ]', ' ', q.lower()).split())


def phase_d(src):
    from collections import Counter
    sp = {s: rows(s) for s in ('search', 'validation', 'test')}
    allrec = [r for rs in sp.values() for r in rs]
    out = {'exact_duplicate_qa_records': sum(n - 1 for n in Counter(json.dumps([r['question'], r['options'], r['answer'], r['image_sha256']]) for r in allrec).values()),
           'duplicate_uids': sum(n - 1 for n in Counter(r['uid'] for r in allrec).values())}
    for a, b in (('search', 'validation'), ('search', 'test'), ('validation', 'test')):
        A, B = sp[a], sp[b]
        key = lambda f, R: {f(r) for r in R}
        out[f'{a}_vs_{b}'] = {'shared_uids': len(key(lambda r: r['uid'], A) & key(lambda r: r['uid'], B)),
                              'shared_image_hashes': len(key(lambda r: r['image_sha256'], A) & key(lambda r: r['image_sha256'], B)),
                              'shared_question_texts': len(key(lambda r: r['question'], A) & key(lambda r: r['question'], B)),
                              'shared_normalized_questions': len(key(lambda r: norm(r['question']), A) & key(lambda r: norm(r['question']), B)),
                              'shared_question_plus_options': len(key(lambda r: (r['question'], tuple(r['options'])), A) & key(lambda r: (r['question'], tuple(r['options'])), B)),
                              'shared_option_sets': len(key(lambda r: tuple(sorted(r['options'])), A) & key(lambda r: tuple(sorted(r['options'])), B))}
    out['within_split'] = {s: {'duplicate_image_hashes': sum(n - 1 for n in Counter(r['image_sha256'] for r in rs).values()),
                               'same_question_text_repeats': sum(n - 1 for n in Counter(r['question'] for r in rs).values())} for s, rs in sp.items()}
    # Full official PT train/test, from the pinned official data.json files and the image-hash table.
    meta = json.loads(Path(src, 'images.json').read_text())
    full = {}
    for s in ('train', 'test'):
        recs = [r for r in json.loads(Path(src, f'{s}-data.json').read_text(encoding='utf-8')) if r['task_type'] == 'Perspective_Taking']
        for r in recs:
            r['h'] = meta[f"{s}/{r['id'].split('_')[0]}.png"]['sha256']
        full[s] = recs
    tr, te = full['train'], full['test']
    out['official_train_vs_test'] = {
        'train_records': len(tr), 'test_records': len(te),
        'shared_image_hashes': len({r['h'] for r in tr} & {r['h'] for r in te}),
        'shared_question_texts': len({r['question'] for r in tr} & {r['question'] for r in te}),
        'shared_normalized_questions': len({norm(r['question']) for r in tr} & {norm(r['question']) for r in te}),
        'shared_question_plus_options': len({(r['question'], tuple(r['options'])) for r in tr} & {(r['question'], tuple(r['options'])) for r in te}),
        'train_images_with_multiple_questions': sum(n > 1 for n in Counter(r['h'] for r in tr).values()),
        'train_duplicate_image_files_by_hash': len({r['id'].split('_')[0] for r in tr}) - len({r['h'] for r in tr}),
        'test_duplicate_image_files_by_hash': len({r['id'].split('_')[0] for r in te}) - len({r['h'] for r in te}),
        'train_same_question_text_on_different_images': sum(len({r['h'] for r in tr if r['question'] == q}) > 1 for q in {r['question'] for r in tr})}
    (OUT / 'phaseD_duplicates.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(json.dumps(out, indent=1))


def canonical(v):
    return json.dumps(v, sort_keys=True, separators=(',', ':'), allow_nan=False)


def candidate_id(base_id, seed, sigma, rng='randopt-per-tensor-v1', sign=1):
    """First-principles recomputation of the identity hash (fields as recorded in the protocol)."""
    return hashlib.sha256(canonical({'base_id': base_id, 'seed': seed, 'sigma': float(sigma), 'rng': rng, 'sign': sign}).encode()).hexdigest()


def index_candidate_files():
    """candidate_id -> list of (phase_dir, path) for every raw candidate file in the study."""
    idx = {}
    for d in sorted(ROOT.iterdir()):
        if d.is_dir() and (d / 'candidates').is_dir():
            for f in (d / 'candidates').glob('*.json.gz'):
                idx.setdefault(f.name[:-8], []).append((d.name, f))
    return idx


def phase_e():
    proto = load('experiments/perspective_taking_n5000_protocol.json')
    c = proto['candidates']
    i = 4111
    seed, sigma, shard = c['seed_start'] + i, c['sigma_mixture'][i % 4], i // c['shard_size']
    chain = {'manifest': {'index': i, 'seed': seed, 'sigma': sigma, 'shard': shard, 'rule': c['rule']}}
    phases = {}
    for d in sorted(ROOT.iterdir()):
        if d.is_dir() and (d / 'native-state.json').exists():
            n, m, e = load(d / 'native-state.json'), load(d / 'run-manifest.json'), load(d / 'engine-config.json')
            phases[d.name] = {'base_id': n['base_id'], 'base_id_flat': n['base_id_flat_sha256'], 'perturb_visual': n['perturb_visual'],
                              'all_parameters_perturbed': n['all_parameters_perturbed'], 'mask': n['parameter_mask_sha256'],
                              'layout': n['layout_sha256'], 'model_revision': m['model_revision'], 'dtype': e['dtype'],
                              'model_path_revision': Path(e['model']).name, 'tokenizer_path_revision': Path(e['tokenizer']).name,
                              'versions': m['versions'], 'status': m['status'], 'fingerprint_contract': n.get('fingerprint_contract')}
    for key in ('base_id', 'base_id_flat', 'perturb_visual', 'all_parameters_perturbed', 'mask', 'layout', 'model_revision', 'dtype',
                'model_path_revision', 'tokenizer_path_revision', 'fingerprint_contract'):
        vals = {json.dumps(v[key]) for v in phases.values()}
        chain.setdefault('consistent_across_all_phases', {})[key] = (len(vals) == 1, sorted(vals)[0] if len(vals) == 1 else sorted(vals))
    chain['versions_consistent'] = len({json.dumps(v['versions'], sort_keys=True) for v in phases.values()}) == 1
    chain['phases_audited'] = len(phases)
    base_id = next(iter(phases.values()))['base_id']
    cid = candidate_id(base_id, seed, sigma)
    chain['recomputed_candidate_id'] = cid
    files = index_candidate_files()[cid]
    chain['raw_files'] = []
    for dname, f in files:
        raw = load(f)
        chain['raw_files'].append({'phase_dir': dname, 'file': f.as_posix(), 'seed': raw['candidate']['seed'], 'sigma': raw['candidate']['sigma'],
                                   'rng': raw['candidate']['rng'], 'sign': raw['candidate']['sign'], 'mask': raw['candidate']['parameter_mask_sha256'],
                                   'candidate_id_in_file': raw['candidate']['candidate_id'], 'state': raw['candidate_state_id'],
                                   'expected_state_checked_before_generation': raw.get('fingerprint_checked_before_generation'),
                                   'expected_state_id': raw.get('expected_state_id'), 'restore_exact': raw['restoration']['exact_base']})
    sl, vl = load(ROOT / 'locks/search.json'), load(ROOT / 'locks/validation.json')
    rec = [r for r in sl['records'] if r['candidate']['candidate_id'] == cid]
    chain['search_lock_record'] = {'count': len(rec), 'index': rec[0]['index'], 'shard': rec[0]['shard'], 'state': rec[0]['candidate_state_id'],
                                   'search_correct': rec[0]['correct_count'], 'search_rank': sl['ranked_ids'].index(cid) + 1}
    chain['in_top50'] = [r['candidate']['candidate_id'] for r in sl['top50']].index(cid) + 1
    chain['validation_lock'] = {'rank1_is_candidate': vl['rank1']['candidate']['candidate_id'] == cid,
                                'top10_position': [r['candidate']['candidate_id'] for r in vl['top10']].index(cid) + 1,
                                'state_in_lock': vl['rank1']['candidate_state_id'], 'validation_correct': vl['candidates'][cid]['summary']['correct_count']}
    states = {f['state'] for f in chain['raw_files']} | {chain['search_lock_record']['state'], chain['validation_lock']['state_in_lock']}
    ids = {f['candidate_id_in_file'] for f in chain['raw_files']} | {cid}
    chain['verdict'] = {'single_state_fingerprint_everywhere': len(states) == 1, 'single_candidate_id_everywhere': len(ids) == 1,
                        'seed_sigma_exact_everywhere': all(f['seed'] == seed and f['sigma'] == sigma and isinstance(f['seed'], int) for f in chain['raw_files']),
                        'index_shard_match_manifest': chain['search_lock_record']['index'] == i and chain['search_lock_record']['shard'] == shard,
                        'checked_before_generation_in_validation_and_test': all(f['expected_state_checked_before_generation'] for f in chain['raw_files'] if not f['phase_dir'].startswith('search'))}
    # Population-wide identity check: every search record's id recomputes from its manifest (seed, sigma).
    bad = [r['index'] for r in sl['records'] if r['candidate']['candidate_id'] != candidate_id(base_id, c['seed_start'] + r['index'], c['sigma_mixture'][r['index'] % 4])
           or r['candidate']['seed'] != c['seed_start'] + r['index'] or r['shard'] != r['index'] // c['shard_size']]
    chain['population_identity_mismatches'] = bad
    (OUT / 'phaseE_provenance.json').write_text(json.dumps(chain, indent=1), encoding='utf-8')
    print(json.dumps({k: chain[k] for k in ('manifest', 'recomputed_candidate_id', 'search_lock_record', 'in_top50', 'validation_lock', 'verdict', 'versions_consistent', 'phases_audited')}, indent=1))
    print('consistent:', {k: v[0] for k, v in chain['consistent_across_all_phases'].items()}, 'population id mismatches:', len(bad))
    for f in chain['raw_files']: print(' ', f['phase_dir'], f['state'][:16], f['expected_state_checked_before_generation'], f['restore_exact'])


def phase_jkl():
    from collections import Counter
    files = index_candidate_files()
    base = load(ROOT / 'baseline/base.json.gz')
    meta = {s: rows(s) for s in ('search', 'validation', 'test')}
    gold = {r['uid']: LETTERS[r['answer']] for rs in meta.values() for r in rs}
    prompt_hash, shapes, finish, disagreements, n_out = {}, Counter(), Counter(), [], 0
    def scan(outs):
        nonlocal n_out
        for o in outs:
            n_out += 1
            t = o['text']
            shape = ('single_letter' if re.fullmatch(r'[ABCD]', t) else 'letter_with_whitespace' if re.fullmatch(r'\s*[ABCD]\s*', t)
                     else 'lowercase_letter' if re.fullmatch(r'\s*[abcd]\s*', t) else 'letter_then_text' if re.match(r'\s*[ABCDabcd]\W', t) else 'other')
            shapes[shape] += 1
            finish[o['finish_reason']] += 1
            prompt_hash.setdefault(o['uid'], set()).add(o['prompt_token_ids_sha256'])
            mine = parse(t)
            if (mine == gold[o['uid']]) != o['correct'] or (mine != (o['prediction'] if o['valid_letter'] else '')):
                disagreements.append({'uid': o['uid'], 'text': t, 'independent': mine, 'original': o['prediction']})
            if shape == 'other' and len(disagreements) < 50:
                disagreements.append({'uid': o['uid'], 'text': t, 'note': 'non-letter-shaped output', 'independent': mine})
    for s in base: scan(base[s]['outputs'])
    search_scores = {}
    for cid, lst in files.items():
        for dname, f in lst:
            raw = load(f)
            for s, v in raw['splits'].items():
                scan(v['outputs'])
                if dname.startswith('search-shard'):
                    search_scores[cid] = (sum(parse(o['text']) == gold[o['uid']] for o in v['outputs']), raw['candidate'])
    J = {'outputs_scanned': n_out, 'output_shapes': dict(shapes), 'finish_reasons': dict(finish),
         'independent_vs_original_disagreements': [d for d in disagreements if 'note' not in d], 'other_shaped_examples': [d for d in disagreements if 'note' in d][:20]}
    K = {'uids': len(prompt_hash), 'uids_with_more_than_one_prompt_hash': sum(len(v) > 1 for v in prompt_hash.values())}
    engine = {d.name: load(d / 'engine-config.json') for d in ROOT.iterdir() if (d / 'engine-config.json').exists()}
    keyset = {canonical({k: v for k, v in e.items()}) for e in engine.values()}
    K['engine_configs_distinct'] = len(keyset); K['engine_config_example'] = next(iter(engine.values()))
    # Independent ranking (L)
    sl = load(ROOT / 'locks/search.json')
    if len(search_scores) != 5000: raise SystemExit(f'expected 5000 search candidates, found {len(search_scores)}')
    ranked = sorted(search_scores, key=lambda c: (-search_scores[c][0], hashlib.sha256(('visual-rank-v1:' + c).encode()).hexdigest()))
    cid = next(c for c, (_, cand) in search_scores.items() if cand['seed'] == SEED)
    score = search_scores[cid][0]
    tie = [c for c in ranked if search_scores[c][0] == score]
    L = {'unique_candidates': len(search_scores), 'per_sigma': dict(Counter(cand['sigma'] for _, cand in search_scores.values())),
         'unique_seeds': len({cand['seed'] for _, cand in search_scores.values()}),
         'ranking_equals_lock': ranked == sl['ranked_ids'], 'top50_equals_lock': ranked[:50] == [r['candidate']['candidate_id'] for r in sl['top50']],
         'scores_equal_lock': all(search_scores[r['candidate']['candidate_id']][0] == r['correct_count'] for r in sl['records']),
         'candidate_search_score': score, 'candidate_search_rank': ranked.index(cid) + 1,
         'tie_group_size_at_score': len(tie), 'tie_group_rank_span': [ranked.index(tie[0]) + 1, ranked.index(tie[-1]) + 1],
         'top50_score_histogram': dict(sorted(Counter(search_scores[c][0] for c in ranked[:50]).items(), reverse=True)),
         'members_of_top50_from_77_tie': sum(search_scores[c][0] == 77 for c in ranked[:50]), 'candidates_with_77': sum(v[0] == 77 for v in search_scores.values())}
    vl = load(ROOT / 'locks/validation.json')
    vcorrect = {}
    for c in [r['candidate']['candidate_id'] for r in sl['top50']]:
        f = next(f for d, f in files[c] if d.startswith('validation-shard'))
        vcorrect[c] = sum(parse(o['text']) == gold[o['uid']] for o in load(f)['splits']['validation']['outputs'])
    vr = sorted(vcorrect, key=lambda c: (-vcorrect[c], -search_scores[c][0], hashlib.sha256(('visual-rank-v1:' + c).encode()).hexdigest()))
    L['validation_ranking_top10_equals_lock'] = vr[:10] == [r['candidate']['candidate_id'] for r in vl['top10']]
    L['candidate_validation_correct'] = vcorrect[cid]; L['candidate_validation_rank'] = vr.index(cid) + 1
    L['validation_scores_top50_sorted'] = sorted(vcorrect.values(), reverse=True)
    for name, v in (('J_parser', J), ('K_prompts', K), ('L_ranking', L)):
        (OUT / f'phase{name}.json').write_text(json.dumps(v, indent=1), encoding='utf-8')
    print('J', {k: J[k] for k in ('outputs_scanned', 'output_shapes', 'finish_reasons')}, 'disagreements', len(J['independent_vs_original_disagreements']))
    print('J other examples', [d['text'][:40] for d in J['other_shaped_examples'][:8]])
    print('K', {k: K[k] for k in ('uids', 'uids_with_more_than_one_prompt_hash', 'engine_configs_distinct')})
    print('L', {k: v for k, v in L.items() if k != 'validation_scores_top50_sorted'})


def correct_vector(outs, gold):
    return [parse(o['text']) == gold[o['uid']] for o in outs]


def phase_mno():
    import numpy as np
    from collections import Counter
    files = index_candidate_files()
    meta = {s: rows(s) for s in ('search', 'validation', 'test')}
    gold = {r['uid']: LETTERS[r['answer']] for rs in meta.values() for r in rs}
    base = load(ROOT / 'baseline/base.json.gz')
    bvec = {s: np.array(correct_vector(base[s]['outputs'], gold)) for s in meta}
    bpred = {s: [parse(o['text']) for o in base[s]['outputs']] for s in meta}
    proto, sl = load('experiments/perspective_taking_n5000_protocol.json'), load(ROOT / 'locks/search.json')
    by_index = {r['index']: r for r in sl['records']}
    def outputs(cid, prefix, split):
        f = next(f for d, f in files[cid] if d.startswith(prefix))
        return load(f)['splits'][split]['outputs']
    # Random density audit: SEARCH and VALIDATION for all 500 (unselected).
    audit = []
    for i in proto['density_audit']['indices']:
        r = by_index[i]; cid = r['candidate']['candidate_id']
        so, vo = outputs(cid, 'search-shard', 'search'), outputs(cid, 'validation-shard', 'validation')
        sv, vv = np.array(correct_vector(so, gold)), np.array(correct_vector(vo, gold))
        audit.append({'index': i, 'sigma': r['candidate']['sigma'], 'search_gain': float(sv.mean() - bvec['search'].mean()),
                      'val_gain': float(vv.mean() - bvec['validation'].mean()), 'val_vec': vv, 'search_vec': sv,
                      'val_pred': [parse(o['text']) for o in vo], 'search_pred': [parse(o['text']) for o in so],
                      'val_wins': int((vv & ~bvec['validation']).sum()), 'val_losses': int((~vv & bvec['validation']).sum())})
    # ---------- M: winner's curse ----------
    rng = np.random.default_rng(20261005)
    M = {}
    for label, pool in (('all_sigmas', audit), ('sigma_0.002', [a for a in audit if a['sigma'] == .002])):
        g = np.array([a['val_gain'] for a in pool]) * 100
        res = {'n': len(g), 'mean_pp': float(g.mean()), 'sd_pp': float(g.std(ddof=1)), 'max_pp': float(g.max()),
               'quantiles_pp': {q: float(np.quantile(g, q)) for q in (.5, .9, .95, .99)}}
        for n in (10, 50, 125, 500):
            mx = np.array([rng.choice(g, n, replace=True).max() for _ in range(10000)])
            res[f'max_of_{n}'] = {'mean_pp': float(mx.mean()), 'p95_pp': float(np.quantile(mx, .95)), 'P_max_ge_8pp': float((mx >= 8 - 1e-9).mean())}
        M[label] = res
    # Null: no true per-candidate effect -> each discordant pair is a fair coin (sign-flip), candidate-specific discordance.
    disc = np.array([a['val_wins'] + a['val_losses'] for a in audit])
    null = {}
    for n in (10, 50, 125, 500):
        mx = []
        for _ in range(10000):
            d = rng.choice(disc, n, replace=True)
            w = rng.binomial(d, .5)
            mx.append(((2 * w - d) / 200 * 100).max())
        mx = np.array(mx)
        null[f'max_of_{n}'] = {'mean_pp': float(mx.mean()), 'p95_pp': float(np.quantile(mx, .95)), 'P_max_ge_8pp': float((mx >= 8 - 1e-9).mean())}
    M['null_sign_flip_no_true_effect'] = {'discordant_pairs_median': float(np.median(disc)), **null}
    # Top-50 committee: search, validation and test gains for all 50 (test now available for all).
    vl = load(ROOT / 'locks/validation.json')
    t50 = []
    for r in sl['top50']:
        cid = r['candidate']['candidate_id']
        tf = next(f for d, f in files[cid] if d == 'test' or d.startswith('committee-test'))
        tv = np.array(correct_vector(load(tf)['splits']['test']['outputs'], gold))
        t50.append({'seed': r['candidate']['seed'], 'sigma': r['candidate']['sigma'], 'search_gain': r['correct_count'] / 200 - bvec['search'].mean(),
                    'val_gain': vl['candidates'][cid]['gain'], 'test_gain': float(tv.mean() - bvec['test'].mean())})
    sg, vg, tg = (np.array([x[k] for x in t50]) for k in ('search_gain', 'val_gain', 'test_gain'))
    corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    M['top50_gain_correlations'] = {'search_vs_val': corr(sg, vg), 'search_vs_test': corr(sg, tg), 'val_vs_test': corr(vg, tg),
                                    'val_gain_mean_pp': float(vg.mean() * 100), 'test_gain_mean_pp': float(tg.mean() * 100),
                                    'test_slope_on_val': float(np.polyfit(vg, tg, 1)[0])}
    M['top50'] = t50
    # ---------- N: sigma = 0.002 anomaly ----------
    N = {}
    for sig in (.00025, .0005, .001, .002):
        pool = [a for a in audit if a['sigma'] == sig]
        out = {'n': len(pool), 'mean_search_gain_pp': 100 * float(np.mean([a['search_gain'] for a in pool])),
               'mean_val_gain_pp': 100 * float(np.mean([a['val_gain'] for a in pool]))}
        for split, key in (('search', 'search_pred'), ('validation', 'val_pred')):
            bm = Counter(bpred[split]); n = len(bpred[split])
            shift = {L: float(np.mean([Counter(a[key])[L] for a in pool]) - bm[L]) / n * 100 for L in LETTERS}
            out[f'{split}_prediction_shift_pp'] = shift
            st = {}
            for s in sorted({r['sub_task_type'] for r in meta[split]}):
                m = np.array([r['sub_task_type'] == s for r in meta[split]])
                vec = 'search_vec' if split == 'search' else 'val_vec'
                st[s] = 100 * float(np.mean([a[vec][m].mean() for a in pool]) - bvec[split][m].mean())
            out[f'{split}_subtask_gain_pp'] = st
        N[str(sig)] = out
    # Regression of validation gain on prediction-share shifts (sigma 0.002 audit).
    pool = [a for a in audit if a['sigma'] == .002]
    bm = Counter(bpred['validation'])
    X = np.array([[Counter(a['val_pred'])[L] - bm[L] for L in LETTERS] for a in pool], float) / 200 * 100
    y = np.array([a['val_gain'] for a in pool]) * 100
    N['val_gain_vs_prediction_shift_sigma_0.002'] = {L: corr(X[:, i], y) for i, L in enumerate(LETTERS)}
    Xs = np.array([[Counter(a['search_pred'])[L] - Counter(bpred['search'])[L] for L in LETTERS] for a in pool], float) / 200 * 100
    ys = np.array([a['search_gain'] for a in pool]) * 100
    N['search_gain_vs_prediction_shift_sigma_0.002'] = {L: corr(Xs[:, i], ys) for i, L in enumerate(LETTERS)}
    # Per-question: which VALIDATION questions do sigma=0.002 random candidates systematically flip?
    vr = meta['validation']
    gain_rate = np.mean([a['val_vec'] for a in pool], axis=0) - bvec['validation']
    order = np.argsort(-gain_rate)
    N['val_questions_most_often_gained_by_random_sigma_0.002'] = [
        {'uid': vr[i]['uid'], 'subtask': vr[i]['sub_task_type'], 'template': template(vr[i]['question']), 'gold': LETTERS[vr[i]['answer']],
         'base_pred': bpred['validation'][i], 'frac_random_0.002_correct': float(np.mean([a['val_vec'][i] for a in pool]))} for i in order[:20]]
    N['val_net_gain_by_subtask_template_sigma_0.002'] = {}
    for key in ('subtask', 'template'):
        agg = {}
        for i, r in enumerate(vr):
            k = r['sub_task_type'] if key == 'subtask' else template(r['question'])
            agg.setdefault(k, []).append(gain_rate[i])
        N['val_net_gain_by_subtask_template_sigma_0.002'][key] = sorted(((float(np.sum(v)), k, len(v)) for k, v in agg.items()), reverse=True)[:10]
    # Same for SEARCH, to show why the population mean is worse there.
    sr = meta['search']
    srate = np.mean([a['search_vec'] for a in pool], axis=0) - bvec['search']
    agg = {}
    for i, r in enumerate(sr):
        agg.setdefault(r['sub_task_type'], []).append(srate[i])
    N['search_net_gain_by_subtask_sigma_0.002'] = {k: float(np.sum(v)) for k, v in agg.items()}
    # Does candidate 9504111 gain on the same questions random sigma=0.002 candidates gain on?
    cand_val = np.array(correct_vector(outputs(next(c for c in files if load(files[c][0][1])['candidate']['seed'] == SEED), 'validation-shard', 'validation'), gold))
    cgain = cand_val.astype(int) - bvec['validation'].astype(int)
    N['candidate_vs_random_0.002_per_question_gain_correlation'] = corr(cgain, gain_rate)
    N['candidate_gained_questions_mean_random_gain_rate'] = float(gain_rate[cgain > 0].mean())
    N['other_questions_mean_random_gain_rate'] = float(gain_rate[cgain == 0].mean())
    # ---------- O: transition visual ----------
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(12, 4.2), gridspec_kw={'height_ratios': [1, 2.8]})
    colors = {'both_correct': '#d9d8d2', 'both_wrong': '#f3f2ee', 'base_wrong->cand_correct': '#2a78d6', 'base_correct->cand_wrong': '#eb6834'}
    with open(OUT / 'phaseA_per_question.csv', encoding='utf-8') as fh:
        pq = list(csv.DictReader(fh))
    for ax, split in zip(axes, ('validation', 'test')):
        rs = sorted([r for r in pq if r['split'] == split], key=lambda r: (r['subtask'], r['correct_option'], r['transition'], r['image_id']))
        ax.bar(range(len(rs)), [1] * len(rs), width=1.0, color=[colors[r['transition']] for r in rs], linewidth=0)
        bounds = [i for i in range(1, len(rs)) if rs[i]['subtask'] != rs[i - 1]['subtask']]
        for b in bounds: ax.axvline(b - .5, color='#0b0b0b', lw=1)
        starts = [0] + bounds
        for s0, s1 in zip(starts, bounds + [len(rs)]):
            ax.text((s0 + s1) / 2, 1.05, rs[s0]['subtask'] + f' (n={s1 - s0})', ha='center', va='bottom', fontsize=8)
        ax.set_xlim(-.5, len(rs) - .5); ax.set_ylim(0, 1.35); ax.set_yticks([]); ax.set_ylabel(split.upper(), rotation=0, ha='right', va='center')
        ax.spines[:].set_visible(False); ax.set_xticks([])
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[k]) for k in colors]
    fig.legend(handles, ['both correct', 'both wrong', 'gained (base wrong → candidate correct)', 'lost (base correct → candidate wrong)'],
               loc='lower center', ncol=4, frameon=False, fontsize=8)
    fig.suptitle('Candidate 9504111 vs base, per question (sorted by sub-task, gold answer, transition)', x=.01, ha='left', fontsize=10)
    fig.tight_layout(rect=(0, .08, 1, 1)); fig.savefig(OUT / 'phaseO_transitions.png', dpi=150); plt.close(fig)
    for name, v in (('M_winners_curse', M), ('N_sigma_0.002', N)):
        (OUT / f'phase{name}.json').write_text(json.dumps(v, indent=1, default=float), encoding='utf-8')
    print('M', json.dumps({k: v for k, v in M.items() if k != 'top50'}, indent=1))
    print('N', json.dumps({k: v for k, v in N.items() if not k.startswith('val_questions')}, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['A', 'B', 'C', 'D', 'E', 'JKL', 'MNO'])
    p.add_argument('--images', type=Path)
    p.add_argument('--src', type=Path)
    a = p.parse_args()
    {'A': lambda: phase_a(a.images), 'B': phase_b, 'C': phase_c, 'D': lambda: phase_d(a.src), 'E': phase_e, 'JKL': phase_jkl, 'MNO': phase_mno}[a.phase]()

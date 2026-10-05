"""Part A: one row per candidate x example x phase, built only from stored raw outputs.

Phase names follow the paper: SEARCH (SEARCH200), RERANK (historical VALIDATION200,
used for second-stage selection), TEST (official TEST561). Historical files are not
renamed. Base rows use candidate_id='BASE'. Independent parser (first non-space
character, upper-cased, valid if A-D). Output: results/paper-analysis/.
"""
import gzip
import json
from pathlib import Path
import re

import pandas as pd

ROOT = Path('results/perspective-taking-n5000-20261004')
DATA = Path('examples/omnispatial-perspective-taking')
OUT = Path('results/paper-analysis')
LETTERS = 'ABCD'
PHASE_OF = {'search': 'SEARCH', 'validation': 'RERANK', 'test': 'TEST'}


def load(p):
    raw = Path(p).read_bytes()
    return json.loads(gzip.decompress(raw) if str(p).endswith('.gz') else raw)


def parse(t):
    m = re.match(r'\s*(\S)', t)
    c = m.group(1).upper() if m else ''
    return c if c in LETTERS else ''


def form(q):
    """Normalized question form: lower-case, punctuation/digits removed, first five words.
    A coarse family key for recurring phrasings; not a claim about how the benchmark was authored."""
    return ' '.join(re.sub(r'[^a-z ]', ' ', q.lower()).split()[:5])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {s: [json.loads(l) for l in (DATA / f'{s}.jsonl').read_text(encoding='utf-8').splitlines()] for s in PHASE_OF}
    base = load(ROOT / 'baseline/base.json.gz')
    base_pred = {s: [parse(o['text']) for o in base[s]['outputs']] for s in rows}
    sl, vl = load(ROOT / 'locks/search.json'), load(ROOT / 'locks/validation.json')
    search_rank = {cid: i + 1 for i, cid in enumerate(sl['ranked_ids'])}
    rerank_rank = {cid: i + 1 for i, cid in enumerate(vl['validation_ranked_ids'])}
    audit = set(load('experiments/perspective_taking_n5000_protocol.json')['density_audit']['indices'])
    index_of = {r['candidate']['candidate_id']: r['index'] for r in sl['records']}
    cols = {k: [] for k in ('candidate_id', 'seed', 'sigma', 'manifest_index', 'in_density_audit', 'search_rank', 'rerank_rank', 'phase',
                            'source_phase_dir', 'example_id', 'image_id', 'image_sha256', 'subtask', 'question_form', 'question_text',
                            'true_option', 'base_prediction', 'candidate_prediction', 'base_correct', 'candidate_correct',
                            'transition', 'parameter_mask', 'candidate_state_id')}

    def add(cid, seed, sigma, mask, state, split, dirname, preds):
        ph = PHASE_OF[split]
        for r, bp, cp in zip(rows[split], base_pred[split], preds):
            g = LETTERS[r['answer']]; bc, cc = bp == g, cp == g
            for k, v in (('candidate_id', cid), ('seed', seed), ('sigma', sigma), ('manifest_index', index_of.get(cid, -1)),
                         ('in_density_audit', index_of.get(cid, -1) in audit), ('search_rank', search_rank.get(cid, -1)),
                         ('rerank_rank', rerank_rank.get(cid, -1)), ('phase', ph), ('source_phase_dir', dirname),
                         ('example_id', r['uid']), ('image_id', r['image_member'].rsplit('/', 1)[1]), ('image_sha256', r['image_sha256']),
                         ('subtask', r['sub_task_type']), ('question_form', form(r['question'])), ('question_text', r['question']),
                         ('true_option', g), ('base_prediction', bp), ('candidate_prediction', cp), ('base_correct', bc),
                         ('candidate_correct', cc),
                         ('transition', 'both_correct' if bc and cc else 'base_only_correct' if bc else 'candidate_only_correct' if cc else 'both_wrong'),
                         ('parameter_mask', mask), ('candidate_state_id', state)):
                cols[k].append(v)

    for s in rows:
        add('BASE', -1, 0.0, 'none', 'base', s, 'baseline', base_pred[s])
    seen = set()
    for d in sorted(ROOT.iterdir()):
        if not (d / 'candidates').is_dir() or '.failed' in d.name:
            continue
        for f in sorted((d / 'candidates').glob('*.json.gz')):
            raw = load(f)
            c = raw['candidate']
            for split, v in raw['splits'].items():
                key = (c['candidate_id'], split)
                if key in seen:
                    raise SystemExit(f'duplicate candidate/phase {key} in {d.name}')
                seen.add(key)
                add(c['candidate_id'], c['seed'], c['sigma'], c['parameter_mask'], raw['candidate_state_id'], split, d.name,
                    [parse(o['text']) for o in v['outputs']])
    df = pd.DataFrame(cols)
    for c in ('phase', 'source_phase_dir', 'subtask', 'question_form', 'true_option', 'base_prediction', 'candidate_prediction',
              'transition', 'parameter_mask', 'candidate_id', 'candidate_state_id', 'example_id', 'image_id', 'image_sha256', 'question_text'):
        df[c] = df[c].astype('category')
    df.to_parquet(OUT / 'paper_master_predictions.parquet', index=False, compression='zstd')
    # Sanity checks against committed reports.
    acc = df.groupby(['candidate_id', 'phase'], observed=True)['candidate_correct'].sum()
    checks = {
        'rows': len(df), 'candidate_phase_pairs': int(acc.size),
        'base_search_72': int(acc[('BASE', 'SEARCH')]) == 72, 'base_rerank_79': int(acc[('BASE', 'RERANK')]) == 79,
        'base_test_259': int(acc[('BASE', 'TEST')]) == 259,
        'search_candidates_5000': int((df[df.phase == 'SEARCH'].candidate_id.nunique()) - 1) == 5000,
        'rerank_candidates_543': int(df[df.phase == 'RERANK'].candidate_id.nunique() - 1) == 543,
        'test_candidates_50': int(df[df.phase == 'TEST'].candidate_id.nunique() - 1) == 50,
        'search_scores_match_lock': all(int(acc[(r['candidate']['candidate_id'], 'SEARCH')]) == r['correct_count'] for r in sl['records']),
        'rerank_scores_match_lock': all(int(acc[(c, 'RERANK')]) == v['summary']['correct_count'] for c, v in vl['candidates'].items()),
    }
    cand = '25a60f0b59f103ebf782a31f94bb569d897a652a0bc1bd95c5cdc5521418b843'
    checks['candidate_9504111'] = {p: int(acc[(cand, p)]) for p in ('SEARCH', 'RERANK', 'TEST')}
    checks['candidate_9504111_expected'] = checks['candidate_9504111'] == {'SEARCH': 77, 'RERANK': 95, 'TEST': 244}
    summ = df[df.candidate_id != 'BASE'].groupby('phase', observed=True).agg(candidates=('candidate_id', 'nunique'), rows=('example_id', 'size'))
    summ.to_csv(OUT / 'paper_master_summary.csv')
    (OUT / 'paper_master_checks.json').write_text(json.dumps(checks, indent=1))
    print(json.dumps(checks, indent=1)); print(summ)


if __name__ == '__main__':
    main()

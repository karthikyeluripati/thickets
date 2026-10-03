"""Select committees using only the existing 200 selection columns and freeze their union."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from thicket_runtime.cli import revision_info, write_json
from thicket_runtime.committee_selection import greedy_committee, matched_random, voting_details
from thicket_runtime.evaluation_matrix import reward_and_extractor
from thicket_runtime.partial_evaluation import order_scores, tie_keys
from thicket_runtime.candidate import CandidateSpec
from thicket_runtime.shared_speculative import sha256


def committed_bytes(path):
    name = path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    if subprocess.check_output(['git', 'show', 'HEAD:' + name]) != path.read_bytes():
        raise ValueError('protocol/inputs must be committed before offline selection: ' + name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, default=Path('experiments/complementarity_protocol.json'))
    p.add_argument('--upstream-root', required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    protocol = json.loads(args.protocol.read_bytes())
    committed_bytes(args.protocol)
    for key in ('path', 'provenance_path'):
        committed_bytes(Path(protocol['fresh_dataset'][key]))
    reward, _ = reward_and_extractor(args.upstream_root)
    reference = Path(protocol['reference'])
    checksums = json.loads((reference / 'sha256.json').read_bytes())
    args.out.mkdir(parents=True)
    manifest = {'source': revision_info(), 'command': [sys.executable, *sys.argv],
                'numpy': np.__version__, 'python': sys.version,
                'protocol_sha256': sha256(args.protocol), 'selection_columns_only': 200,
                'old_test40_evaluated': False, 'new_test_outputs_available': False,
                'rule_selection': 'specified and committed a priori; development first; no performance-based method or tie-seed choice'}
    write_json(args.out / 'manifest.json', manifest)
    populations, union_by_pop = {}, {}
    for pop in protocol['populations']:
        path = reference / ('matrix-' + pop + '.json')
        if sha256(path) != checksums[path.name]:
            raise ValueError('existing matrix checksum mismatch')
        raw = json.loads(path.read_bytes())
        n = protocol['selection_columns']
        if raw['selection_count'] != n:
            raise ValueError('selection split mismatch')
        # Do not compute or inspect old test40 performance. Select columns first.
        gold = raw['answers'][:n]
        answers = [row['voting_answers'][:n] for row in raw['rows']]
        scores = [int(sum(row['rewards'][:n])) for row in raw['rows']]
        ids = [row['candidate']['candidate_id'] for row in raw['rows']]
        for row in raw['rows']:
            c = row['candidate']
            recipe = CandidateSpec(**{k: c[k] for k in ('base_id', 'seed', 'sigma', 'rng', 'sign')})
            if recipe.candidate_id != c['candidate_id'] or c['population'] != pop:
                raise ValueError('candidate identity mismatch')
        standard = order_scores(scores, tie_keys(ids))
        sequences = {'standard': standard[:max(protocol['offline_k'])]}
        histories, timings = {}, {}
        for seed in [protocol['primary_tie_seed'], *protocol['diagnostic_tie_seeds']]:
            started = time.perf_counter()
            sequence, history = greedy_committee(answers, gold, scores, ids, reward, max(protocol['offline_k']), seed)
            sequences[f'greedy-{seed}'], histories[str(seed)] = sequence, history
            timings[f'greedy-{seed}'] = time.perf_counter() - started
        for seed in protocol['random_control_seeds']:
            sequences[f'random-{seed}'] = matched_random(sequences['greedy-0'], scores, ids, seed)
        committees, selected_union = {}, set()
        for method, sequence in sequences.items():
            for k in protocol['offline_k']:
                committee = sequence[:k]
                votes = voting_details(answers, committee, gold, reward)
                required = k in (protocol['gpu_standard_k'] if method == 'standard' else protocol['gpu_greedy_k'])
                if required:
                    selected_union.update(committee)
                key = f'{method}-k{k}'
                committees[key] = {'method': method, 'k': k, 'indices': committee,
                    'candidate_ids': [ids[i] for i in committee], 'individual_selection_correct': [scores[i] for i in committee],
                    'selection_correct': sum(v['correct'] for v in votes), 'selection_accuracy': sum(v['correct'] for v in votes) / n,
                    'selection_votes': votes, 'required_for_gpu': required,
                    'overlap_standard_same_k': len(set(committee) & set(standard[:k]))}
                if method.startswith('greedy-'):
                    if committees[key]['selection_correct'] != histories[method.split('-')[1]][k-1]['selection_correct']:
                        raise AssertionError('optimized append objective differs from original Counter voting')
                if method.startswith('random-') and [scores[i] for i in committee] != [scores[i] for i in sequences['greedy-0'][:k]]:
                    raise AssertionError('random control is not exactly quality matched')
        populations[pop] = {'matrix_sha256': sha256(path), 'candidate_ids': ids,
            'selection_prompt_ids': raw['prompt_ids'][:n], 'committees': committees,
            'greedy_history': histories, 'selection_seconds': timings,
            'individual_correct_counts': scores, 'standard_ranking': standard,
            'selected_union_count': len(selected_union)}
        union_by_pop[pop] = [{'candidate': raw['rows'][i]['candidate'], 'expected_state_id': raw['rows'][i]['state_id']}
                            for i in sorted(selected_union, key=lambda i: ids[i])]
        write_json(args.out / (pop + '.json'), populations[pop])
        print(json.dumps({'population': pop, 'union_candidates': len(selected_union), 'greedy_seconds': timings}), flush=True)
    # Round-robin population order, fixed by ID before any new generation.
    union = [union_by_pop[pop][i] for i in range(max(map(len, union_by_pop.values())))
             for pop in protocol['populations'] if i < len(union_by_pop[pop])]
    if len(union) >= 504 or len({r['candidate']['candidate_id'] for r in union}) != len(union):
        raise ValueError('must collect only the unique selected union, not all 504 candidates')
    write_json(args.out / 'union.json', union)
    lock = {'schema': 'committee-lock-v1', 'source': revision_info(), 'protocol_sha256': sha256(args.protocol),
            'fresh_inputs_sha256': {key: sha256(protocol['fresh_dataset'][key]) for key in ('path', 'provenance_path')},
            'selection_files_sha256': {pop: sha256(args.out / (pop + '.json')) for pop in protocol['populations']},
            'union_sha256': sha256(args.out / 'union.json'), 'union_candidates': len(union),
            'primary_tie_seed': 0, 'new_heldout_outputs_inspected': False,
            'populations': {pop: {key: {'candidate_ids': c['candidate_ids'], 'k': c['k'], 'method': c['method']}
                          for key, c in value['committees'].items() if c['required_for_gpu']}
                          for pop, value in populations.items()}}
    write_json(args.out / 'committee-lock.json', lock)
    print(json.dumps({'total_selected_union': len(union), 'full_matrix_regeneration': False}))


if __name__ == '__main__':
    main()

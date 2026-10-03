"""Evaluate frozen committees on the one fresh selected-union GPU experiment."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from thicket_runtime.agreement import distribution
from thicket_runtime.cli import revision_info, write_json
from thicket_runtime.committee_selection import frozen_gate, voting_details
from thicket_runtime.committee_validation import committed
from thicket_runtime.evaluation_matrix import reward_and_extractor, write_gzip
from thicket_runtime.shared_speculative import sha256


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def pairwise(answers, rewards, ids):
    answers, rewards = np.asarray(answers), np.asarray(rewards, dtype=float)
    errors = 1 - rewards
    pairs = []
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            x, y = errors[a], errors[b]
            dx, dy = x - x.mean(), y - y.mean()
            denom = float(np.sqrt(np.sum(dx*dx) * np.sum(dy*dy)))
            pairs.append({'a': ids[a], 'b': ids[b],
                'error_correlation': float(np.sum(dx*dy) / denom) if denom else None,
                'both_wrong': float(np.mean((x > 0) & (y > 0))),
                'reward_disagreement': float(np.mean(x != y)),
                'answer_disagreement': float(np.mean(answers[a] != answers[b]))})
    return {'summary': {k: distribution(p[k] for p in pairs) for k in
                       ('error_correlation', 'both_wrong', 'reward_disagreement', 'answer_disagreement')}, 'pairs': pairs}


def comparison(proposed, baseline, bootstrap):
    a, b = np.array(proposed['correct'], dtype=int), np.array(baseline['correct'], dtype=int)
    delta = a - b
    samples = delta[bootstrap].mean(axis=1)
    return {'selection_accuracy_delta': proposed['selection_accuracy'] - baseline['selection_accuracy'],
        'heldout_accuracy_delta': float(delta.mean()), 'improved_questions': int(np.sum(delta > 0)),
        'worsened_questions': int(np.sum(delta < 0)), 'both_correct': int(np.sum((a == 1) & (b == 1))),
        'both_wrong': int(np.sum((a == 0) & (b == 0))),
        'descriptive_paired_bootstrap95': np.quantile(samples, [.025, .975]).tolist(),
        'expert_pass_ratio_baseline_over_proposed': baseline['k'] / proposed['k'],
        'expert_pass_reduction': 1 - proposed['k'] / baseline['k'],
        'token_reduction': 1 - proposed['generated_tokens'] / baseline['generated_tokens'],
        'selection_gain_without_positive_heldout_gain': proposed['selection_accuracy'] > baseline['selection_accuracy'] and delta.mean() <= 0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, default=Path('experiments/complementarity_protocol.json'))
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--upstream-root', required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    protocol, lock = read(args.protocol), read(args.selection / 'committee-lock.json')
    committed(args.protocol)
    committed(args.selection / 'committee-lock.json')
    checksums = read(args.run / 'sha256.json')
    for name, expected in checksums.items():
        if sha256(args.run / name) != expected:
            raise ValueError('GPU artifact checksum mismatch: ' + name)
    manifest, controls = read(args.run / 'manifest.json'), read(args.run / 'controls.json')
    if manifest['status'] != 'complete' or not all(controls.values()):
        raise ValueError('complete valid GPU test required; operational failure is not a scientific NO')
    if manifest['committee_lock_sha256'] != sha256(args.selection / 'committee-lock.json') or lock['protocol_sha256'] != sha256(args.protocol):
        raise ValueError('GPU test differs from frozen selector or gate')
    reward, extract = reward_and_extractor(args.upstream_root)
    matrix = read(args.run / 'matrix.json')
    gold = matrix['answers']
    if len(gold) != 300 or len(set(matrix['prompt_ids'])) != 300:
        raise ValueError('incomplete fresh test')
    expected_inputs = [json.loads(line) for line in Path(protocol['fresh_dataset']['path']).read_text(encoding='utf-8').splitlines()]
    if matrix['prompt_ids'] != [r['id'] for r in expected_inputs] or gold != [str(r['answer']) for r in expected_inputs]:
        raise ValueError('test prompt identity mismatch')
    plan = {p['candidate']['candidate_id']: p for p in read(args.selection / 'union.json')}
    rows, traces = {}, {}
    for row in matrix['rows']:
        cid = row['candidate']['candidate_id']
        if cid in rows or row['candidate'] != plan[cid]['candidate'] or row['state_id'] != plan[cid]['expected_state_id']:
            raise ValueError('candidate identity/fingerprint mismatch')
        raw = read(args.run / 'candidates' / (cid + '.json.gz'))
        if not raw['fingerprint_checked_before_generation'] or not raw['restoration']['exact_base'] or raw['candidate_state_id'] != row['state_id']:
            raise ValueError('candidate audit failed')
        outputs = raw['outputs']
        if [o['prompt_id'] for o in outputs] != matrix['prompt_ids']:
            raise ValueError('output prompt identity mismatch')
        for j, o in enumerate(outputs):
            if (o['reward'] != reward(o['text'], gold[j]) or o['voting_answer'] != extract(o['text'])
                or o['reward'] != row['rewards'][j] or o['voting_answer'] != row['voting_answers'][j]
                or len(o['token_ids']) != row['tokens'][j] or (o['finish_reason'] == 'length') != row['capped'][j]):
                raise ValueError('raw generation/scoring matrix mismatch')
        rows[cid], traces[cid] = row, raw
    if set(rows) != set(plan):
        raise ValueError('selected union coverage mismatch')
    args.out.mkdir(parents=True)
    bootstrap = np.random.default_rng(20261003).integers(0, 300, size=(10000, 300), dtype=np.int16)
    metadata = {'source': revision_info(), 'command': [sys.executable, *sys.argv], 'python': sys.version,
                'numpy': np.__version__, 'protocol_sha256': sha256(args.protocol),
                'committee_lock_sha256': sha256(args.selection / 'committee-lock.json'),
                'run_manifest_sha256': sha256(args.run / 'manifest.json'),
                'all_raw_generations_rescored': True, 'bootstrap_seed': 20261003, 'bootstrap_replicates': 10000,
                'gate_reselected': False, 'previous_test40_used': False}
    write_json(args.out / 'manifest.json', metadata)
    all_summary, all_details, all_comparisons, gate_input = {}, {}, {}, {}
    reference = Path(protocol['reference'])
    for pop in protocol['populations']:
        offline_path = args.selection / (pop + '.json')
        if sha256(offline_path) != lock['selection_files_sha256'][pop]:
            raise ValueError('selection results changed after locking')
        offline = read(offline_path)
        reference_path = reference / ('matrix-' + pop + '.json')
        if sha256(reference_path) != offline['matrix_sha256']:
            raise ValueError('original selection matrix changed')
        original = read(reference_path)
        original_by_id = {r['candidate']['candidate_id']: r for r in original['rows']}
        detail, summary = {}, {}
        for name, committee in lock['populations'][pop].items():
            ids = committee['candidate_ids']
            if len(set(ids)) != committee['k']:
                raise ValueError('repeated deployed expert')
            selected = [rows[cid] for cid in ids]
            answers = [r['voting_answers'] for r in selected]
            rewards = [r['rewards'] for r in selected]
            votes = voting_details(answers, list(range(len(ids))), gold, reward)
            old_answers = [original_by_id[cid]['voting_answers'][:200] for cid in ids]
            old_rewards = [original_by_id[cid]['rewards'][:200] for cid in ids]
            selection_votes = voting_details(old_answers, list(range(len(ids))), original['answers'][:200], reward)
            recorded = offline['committees'][name]
            if selection_votes != recorded['selection_votes']:
                raise ValueError('original Counter selection votes do not reproduce')
            result = {**committee, 'correct': [v['correct'] for v in votes], 'votes': votes,
                'correct_count': sum(v['correct'] for v in votes), 'accuracy': sum(v['correct'] for v in votes) / 300,
                'selection_accuracy': recorded['selection_accuracy'],
                'selection_pairwise': pairwise(old_answers, old_rewards, ids), 'heldout_pairwise': pairwise(answers, rewards, ids),
                'individual_selection_reward': distribution(float(np.mean(r)) for r in old_rewards),
                'individual_heldout_reward': distribution(float(np.mean(r)) for r in rewards),
                'unique_experts_deployed': len(ids), 'generation_rpcs_per_full_test': len(ids), 'generated_request_sequences': 300 * len(ids),
                'generated_tokens': sum(sum(r['tokens']) for r in selected),
                'trace_inference_seconds_sum': sum(traces[cid]['descriptive_costs']['workloads']['fresh300']['inference_seconds'] for cid in ids),
                'capped_fraction': float(np.mean([r['capped'] for r in selected])),
                'overlap_standard_same_k': recorded['overlap_standard_same_k'],
                'winning_vote_margin': distribution(v['margin'] for v in votes),
                'vote_tie_fraction': float(np.mean([v['winning_tie_count'] > 1 for v in votes])),
                'exact_individual_score_group_sizes': [sum(s == value for s in offline['individual_correct_counts']) for value in recorded['individual_selection_correct']]}
            detail[name] = result
            summary[name] = {k: v for k, v in result.items() if k not in ('votes', 'correct', 'selection_pairwise', 'heldout_pairwise')}
            summary[name]['selection_pairwise'] = result['selection_pairwise']['summary']
            summary[name]['heldout_pairwise'] = result['heldout_pairwise']['summary']
        comparisons = {}
        for name, result in detail.items():
            if not name.startswith('standard-'):
                baseline = 'standard-k' + str(result['k'])
                comparisons[name + '_vs_' + baseline] = comparison(result, detail[baseline], bootstrap)
        for small in protocol['gate']['efficiency_complementary_k']:
            for large in protocol['gate']['efficiency_standard_k']:
                for method in ('greedy-0', 'standard'):
                    a, b = f'{method}-k{small}', f'standard-k{large}'
                    comparisons[a + '_vs_' + b] = comparison(detail[a], detail[b], bootstrap)
        for k in protocol['gpu_greedy_k']:
            for seed in protocol['random_control_seeds']:
                a, b = f'greedy-0-k{k}', f'random-{seed}-k{k}'
                comparisons[a + '_vs_' + b] = comparison(detail[a], detail[b], bootstrap)
        all_summary[pop], all_details[pop], all_comparisons[pop] = summary, detail, comparisons
        gate_input[pop] = {name: r['accuracy'] for name, r in detail.items()}
    gate = frozen_gate(gate_input, protocol['gate'])
    gate['primary_seed'] = protocol['primary_tie_seed']
    gate['diagnostic_seed_gates_not_eligible'] = {}
    for seed in protocol['diagnostic_tie_seeds']:
        alternative = {pop: {**values, **{f'greedy-0-k{k}': values[f'greedy-{seed}-k{k}'] for k in protocol['gpu_greedy_k']}}
                       for pop, values in gate_input.items()}
        gate['diagnostic_seed_gates_not_eligible'][str(seed)] = frozen_gate(alternative, protocol['gate'])
    control_tokens = 0
    for name in ('base.json.gz', 'zero-control.json.gz', 'base-repeat.json.gz', 'repeat-development.json.gz', 'repeat-validation_a.json.gz', 'repeat-validation_b.json.gz'):
        outputs = read(args.run / name)['outputs']
        if isinstance(outputs, dict):
            outputs = outputs['fresh300']
        control_tokens += sum(len(o['token_ids']) for o in outputs)
    collection = {'unique_selected_experts': len(rows), 'matrix_candidates_regenerated': False,
        'collection_generation_rpcs': len(rows), 'collection_request_sequences': 300 * len(rows),
        'collection_generated_tokens': sum(sum(r['tokens']) for r in rows.values()),
        'control_generation_rpcs': manifest['control_generation_rpcs'], 'control_generated_tokens': control_tokens,
        'sweep_seconds': manifest['sweep_seconds'], 'capped_fraction': float(np.mean([r['capped'] for r in rows.values()])),
        'memory_bytes': {k: distribution(r['memory'][k] for r in traces.values()) for k in next(iter(traces.values()))['memory']}}
    write_json(args.out / 'summary.json', all_summary)
    write_gzip(args.out / 'committee-details.json.gz', all_details)
    write_json(args.out / 'comparisons.json', all_comparisons)
    write_json(args.out / 'gate.json', gate)
    write_json(args.out / 'collection.json', collection)
    print(json.dumps({'gate_pass': gate['pass'], 'decision': gate['decision'], 'selected_experts': len(rows)}))


if __name__ == '__main__':
    main()

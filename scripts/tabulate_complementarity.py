"""Render complete report tables from the frozen committee analysis."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from thicket_runtime.cli import revision_info, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    inputs = [args.root / 'analysis' / name for name in ('summary.json', 'comparisons.json', 'gate.json')]
    inputs += [args.root / 'offline' / (pop + '.json') for pop in ('development', 'validation_a', 'validation_b')]
    summary, comparisons, gate = [json.loads(f.read_bytes()) for f in inputs[:3]]
    lines = ['# Frozen complementarity study: complete derived tables', '',
             'Accuracy values are percentages; deltas and confidence intervals are percentage points.',
             'Seed 0 is primary. Other seeds and random controls are diagnostics only.',
             'Inference calls count one full 300-question generation RPC per expert.',
             'Expert-pass ratios are logical counts, not measured wall-clock speedups.', '']

    def table(title, headers, rows):
        lines.extend(['## ' + title, '', '| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'])
        lines.extend('| ' + ' | '.join(map(str, row)) + ' |' for row in rows)
        lines.append('')

    percent = lambda value: f'{100 * value:.2f}'
    delta = lambda value: f'{100 * value:+.2f}'
    primary, costs, diagnostics, diversity, efficiency, controls = [], [], [], [], [], []
    for pop in ('development', 'validation_a', 'validation_b'):
        results = summary[pop]
        selection = json.loads((args.root / 'offline' / (pop + '.json')).read_bytes())
        for k in (3, 5, 10, 20, 50):
            standard = results[f'standard-k{k}']
            old_greedy = selection['committees'][f'greedy-0-k{k}']
            greedy = results.get(f'greedy-0-k{k}')
            primary.append([pop, k, percent(standard['selection_accuracy']), percent(old_greedy['selection_accuracy']),
                            percent(standard['accuracy']), percent(greedy['accuracy']) if greedy else 'not collected',
                            delta(old_greedy['selection_accuracy'] - standard['selection_accuracy']),
                            delta(greedy['accuracy'] - standard['accuracy']) if greedy else 'not collected'])
            for method in ('standard', 'greedy-0'):
                item = results.get(f'{method}-k{k}')
                if item is None:
                    continue
                costs.append([pop, method, k, item['correct_count'], item['unique_experts_deployed'],
                              item['generation_rpcs_per_full_test'], item['generated_request_sequences'], item['generated_tokens'],
                              percent(item['capped_fraction']), item['overlap_standard_same_k']])
                diversity.append([pop, method, k,
                    percent(item['individual_selection_reward']['mean']), percent(item['individual_heldout_reward']['mean']),
                    f"{item['selection_pairwise']['error_correlation']['mean']:.3f}",
                    f"{item['heldout_pairwise']['error_correlation']['mean']:.3f}",
                    percent(item['heldout_pairwise']['answer_disagreement']['mean']),
                    percent(item['heldout_pairwise']['both_wrong']['mean']),
                    f"{item['winning_vote_margin']['mean']:.2f}", percent(item['vote_tie_fraction'])])
        for k in (3, 5, 10, 20):
            primary_item = results[f'greedy-0-k{k}']
            sizes = primary_item['exact_individual_score_group_sizes']
            random = [results[f'random-{seed}-k{k}'] for seed in (101, 102, 103)]
            diagnostics.append([pop, k, *[percent(results[f'greedy-{seed}-k{k}']['accuracy']) for seed in (0, 1, 2)],
                                *[percent(r['accuracy']) for r in random], sum(s == 1 for s in sizes), min(sizes), max(sizes)])
            for seed, item in zip((101, 102, 103), random):
                comp = comparisons[pop][f'greedy-0-k{k}_vs_random-{seed}-k{k}']
                controls.append([pop, k, seed, percent(item['selection_accuracy']), percent(item['accuracy']),
                                 delta(comp['heldout_accuracy_delta']),
                                 f"{item['heldout_pairwise']['error_correlation']['mean']:.3f}",
                                 percent(item['heldout_pairwise']['answer_disagreement']['mean'])])
        for small in (5, 10):
            for large in (20, 50):
                for method in ('greedy-0', 'standard'):
                    name = f'{method}-k{small}_vs_standard-k{large}'
                    c = comparisons[pop][name]
                    efficiency.append([pop, name, delta(c['selection_accuracy_delta']), delta(c['heldout_accuracy_delta']),
                                       '[' + ', '.join(delta(x) for x in c['descriptive_paired_bootstrap95']) + ']',
                                       f"{c['expert_pass_ratio_baseline_over_proposed']:g}", percent(c['expert_pass_reduction']),
                                       percent(c['token_reduction']), c['improved_questions'], c['worsened_questions']])
    table('Primary selection to held-out transfer', ['Population','K','Standard selection','Greedy selection','Standard fresh','Greedy fresh','Selection delta','Fresh delta'], primary)
    table('Exact deployment costs and committee overlap', ['Population','Method','K','Correct / 300','Unique experts','Inference calls','Request sequences','Generated tokens','Capped %','Overlap with top-K'], costs)
    table('Primary quality and diversity', ['Population','Method','K','Mean individual selection','Mean individual fresh','Selection error correlation','Fresh error correlation','Fresh answer disagreement','Fresh both wrong','Mean vote margin','Vote tie %'], diversity)
    table('Frozen tie seeds and matched controls', ['Population','K','Greedy 0','Greedy 1','Greedy 2','Random 101','Random 102','Random 103','Singleton positions','Min score-group size','Max score-group size'], diagnostics)
    table('Exact-quality random diversity controls', ['Population','K','Seed','Selection accuracy','Fresh accuracy','Primary greedy minus random','Fresh error correlation','Fresh answer disagreement'], controls)
    table('Efficiency and ordinary-small-K checks', ['Population','Comparison','Selection delta','Fresh delta','Paired descriptive 95% interval','Expert-pass ratio','Pass reduction %','Token reduction %','Improved questions','Worsened questions'], efficiency)
    lines.extend(['## Frozen gate', '', '```json', json.dumps(gate, indent=2), '```', '',
                  'Per-question votes/margins, full individual score distributions, and every pair of expert errors are preserved in `analysis/committee-details.json.gz`.', ''])
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / 'manifest.json', {'source': revision_info(), 'command': [sys.executable, *sys.argv],
        'inputs_sha256': {f.as_posix(): hashlib.sha256(f.read_bytes()).hexdigest() for f in inputs},
        'changes_to_selector_or_gate': False})
    (args.out / 'tables.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'tables': 6, 'output': str(args.out / 'tables.md')}))


if __name__ == '__main__':
    main()

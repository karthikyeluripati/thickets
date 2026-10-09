"""EXPLORATORY: does selection-set gain transfer to test? For every N = 5000 search: mean selection reward of the top 50
minus the base model's selection reward (selection gain), vs the top 50's mean single-model test accuracy minus the
base model's test accuracy (test gain). Same scorer as each run. Output: results/paper-analysis/transfer/."""
import json
from pathlib import Path
import re
import sys

import numpy as np

A = Path('results/paper-analysis'); OUT = A / 'transfer'


def log_rewards(p):
    t = Path(p).read_text(encoding='utf-8', errors='replace')
    return [float(x.strip().strip("'")) for m in re.finditer(r'Batch \d+ \| \d+/5000 \| \[(.*?)\]', t) for x in m.group(1).split(',')]


def jsonl_rewards(d):
    return [json.loads(l)['reward'] for f in sorted(Path(d).glob('select_*.jsonl')) for l in f.read_text().splitlines() if l.strip()]


def members_test(ens=None, d=None):
    if ens is not None:  # randopt.py dump: answers per model; correctness recomputed with RandOpt's scorer
        sys.path.insert(0, 'C:/Users/karth/AppData/Local/Temp/claude/c--Users-karth-OneDrive-Desktop-projects-thickets/f30c28bf-5da4-45f4-8bca-2f8edaf9c506/scratchpad/RandOpt')
        from data_handlers.gsm8k import GSM8KHandler
        h = GSM8KHandler(); up = Path(sys.path[0]) / 'data/gsm8k/test.parquet'
        gt = [t['ground_truth'] for t in h.load_data(str(up), split='test')]
        E = json.loads(Path(ens).read_text())
        return 100 * float(np.mean([[bool(v) and bool(h.is_answer_correct(h.format_answer_for_check(v), gt[i])) for i, v in enumerate(E[m])] for m in range(50)]))
    return 100 * float(np.mean([json.loads((Path(d) / f'test_rank{r}.json').read_text())['correct'] for r in range(50)]))


def main():
    OUT.mkdir(exist_ok=True)
    gd_out = A / 'gd-gqa-direct-search/pod/gd/out'
    rows = [  # name, prompt, selection rewards, base selection reward, members test %, base test %
        ('GSM8K Qwen2.5-1.5B (C)', "RandOpt's", log_rewards(A / 'c-sameRun/pod6/c/out/randopt.log'), 0.730,
         members_test(ens=A / 'c-sameRun/pod6/c/out/ensemble_answers.json'), 60.27),
        ('GSM8K Qwen2.5-3B (C3B)', "RandOpt's", log_rewards(A / 'c3b-sameRun/pod/c3b/out/randopt.log'), 0.855,
         members_test(ens=A / 'c3b-sameRun/pod/c3b/out/ensemble_answers.json'), 80.67),
        ('GSM8K OLMo-2-1B (O1)', "RandOpt's", log_rewards(A / 'o1-olmo-sameRun/pod/o1/out/randopt.log'), 0.415,
         members_test(ens=A / 'o1-olmo-sameRun/pod/o1/out/ensemble_answers.json'), 35.25),
        ('GQA Qwen2.5-VL-3B (G2)', 'CoT', jsonl_rewards(A / 'g2-sameRun/pod/g2/out'), json.loads((A / 'g2-sameRun/pod/g2/out/base_select.json').read_text())['reward'],
         members_test(d=A / 'g2-sameRun/pod/g2/out'), 53.39),
        ('GQA Qwen2.5-VL-3B (G2R)', 'CoT', jsonl_rewards(A / 'g2r-seed/pod/g2r/out'), json.loads((A / 'g2r-seed/pod/g2r/out/base_select.json').read_text())['reward'],
         members_test(d=A / 'g2r-seed/pod/g2r/out'), 53.39),
        ('GQA Qwen2.5-VL-3B (GD)', 'direct', jsonl_rewards(gd_out), json.loads((gd_out / 'base_select.json').read_text())['reward'],
         members_test(d=A / 'gd-gqa-direct-search/pod2/gd/out') if (A / 'gd-gqa-direct-search/pod2/gd/out/test_rank49.json').exists() else None, 64.70),
        ('GSM8K OLMo-2-1B (GB)', 'boxed', jsonl_rewards(A / 'gb-gsm8k-boxed-search/pod/gb/olmo/out'),
         json.loads((A / 'gb-gsm8k-boxed-search/pod/gb/olmo/out/base_select.json').read_text())['reward'],
         members_test(d=A / 'gb-gsm8k-boxed-search/pod/gb/olmo/out'), 67.85)]
    res = []
    for name, prompt, rew, b, mt, bt in rows:
        rew = np.array(rew); top = np.sort(rew)[-50:]
        res.append({'search': name, 'prompt': prompt, 'n': int(len(rew)), 'base_selection': 100 * b, 'population_mean_selection': 100 * float(rew.mean()),
                    'share_above_base': 100 * float((rew > b).mean()), 'top50_selection_mean': 100 * float(top.mean()),
                    'selection_gain': 100 * float(top.mean() - b), 'members_test': mt, 'base_test': bt,
                    'test_gain': None if mt is None else mt - bt,
                    'transfer_ratio': None if mt is None else (mt - bt) / (100 * float(top.mean() - b))})
    (OUT / 'transfer_results.json').write_text(json.dumps(res, indent=1))
    md = ['| Search | Prompt | Selection gain (top-50 mean − base, pp) | Test gain (members − base, pp) | Transfer ratio | Share above base | Population mean − base |',
          '|---|---|---|---|---|---|---|']
    for r in res:
        tg = '–' if r['test_gain'] is None else f"{r['test_gain']:+.1f}"; tr = '–' if r['transfer_ratio'] is None else f"{r['transfer_ratio']:.2f}"
        md.append(f"| {r['search']} | {r['prompt']} | {r['selection_gain']:+.1f} | {tg} | {tr} | {r['share_above_base']:.1f}% | {r['population_mean_selection'] - r['base_selection']:+.1f} |")
    (OUT / 'transfer_table.md').write_text('\n'.join(md) + '\n\nEXPLORATORY. Selection: 200 questions; test: 1319 (GSM8K) / 1238 (GQA).\n', encoding='utf-8')
    print('\n'.join(md))


if __name__ == '__main__':
    main()

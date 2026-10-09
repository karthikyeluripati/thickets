"""RV-1 template choice on the pod (results/paper-analysis/rv-reviewer-round/plan_lock.md, "Rule"): among the NEW public
templates only, the one with the highest greedy BASE accuracy on RandOpt's 200 selection questions; ties by listed order.
GSM8K rows rescore the saved answers with RandOpt's scorer (o2_eval's stored 'c' is not used); the GQA row uses g3_eval's
'c' (RandOpt's GQA scorer on the text). Prints the choice and writes <out>/choice.json."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

GSM_NEW = ('harness_cot', 'harness_plain', 'simple_evals')
GQA_NEW = ('llava', 'blip')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--task', choices=['gsm8k', 'gqa'], required=True)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--upstream', type=Path, default=Path('/workspace/RandOpt'))
    a = ap.parse_args(); sys.path.insert(0, str(a.upstream))
    if a.task == 'gsm8k':
        from data_handlers.gsm8k import GSM8KHandler
        h = GSM8KHandler(); gts = [t['ground_truth'] for t in h.load_data(str(a.upstream / 'data/gsm8k/train.parquet'), split='train', max_samples=200)]
        acc = {p: 100 * float(np.mean([bool(x['a']) and bool(h.is_answer_correct(h.format_answer_for_check(x['a']), gts[i]))
                                        for i, x in enumerate(json.loads((a.out / f'sel_{p}_base.json').read_text()))])) for p in GSM_NEW}
        order = GSM_NEW
    else:
        acc = {p: 100 * float(np.mean([x['c'] for x in json.loads((a.out / p / 'sel_b256_base.json').read_text())])) for p in GQA_NEW}
        order = GQA_NEW
    chosen = max(order, key=lambda p: (acc[p], -order.index(p)))
    (a.out / 'choice.json').write_text(json.dumps({'selection_acc': acc, 'chosen': chosen}))
    print(chosen)


if __name__ == '__main__':
    main()

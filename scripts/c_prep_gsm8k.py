"""Session C (results/paper-analysis/c-sameRun/plan_lock.md): openai/gsm8k main -> RandOpt's verl-format parquet
(data/gsm8k/{train,test}.parquet) in original order. Run from the RandOpt checkout."""
import os

import datasets
import pandas as pd

INSTR = 'Let\'s think step by step and output the final answer after "####".'
os.makedirs('data/gsm8k', exist_ok=True)
for split in ('train', 'test'):
    ds = datasets.load_dataset('openai/gsm8k', 'main', split=split)
    rows = [{'data_source': 'openai/gsm8k', 'prompt': [{'role': 'user', 'content': r['question'] + ' ' + INSTR}],
             'reward_model': {'style': 'rule', 'ground_truth': r['answer'].split('####')[-1].strip().replace(',', '')}} for r in ds]
    pd.DataFrame(rows).to_parquet(f'data/gsm8k/{split}.parquet')
    print(split, len(rows), ds._fingerprint, flush=True)

"""EXPLORATORY: are RandOpt's selected 'experts' prompt-specific, and do they implement the prompt fix?
(1) Same 5000 perturbations scored under two prompts (O1 vs GB on OLMo; G2 vs GD on GQA): rank correlation of selection
rewards, top-50 overlap, and the percentile of one prompt's top 50 under the other. (2) Answer agreement of the
selected members (searched under RandOpt's prompt) with the base model under RandOpt's prompt vs under the good prompt.
Output: results/paper-analysis/expert-specificity/expert_specificity.json"""
import io
import json
from pathlib import Path
import re
import tarfile

import numpy as np
from scipy.stats import spearmanr

A = Path('results/paper-analysis'); OUT = A / 'expert-specificity'


def log_rewards(p):
    t = Path(p).read_text(encoding='utf-8', errors='replace')
    return np.array([float(x.strip().strip("'")) for m in re.finditer(r'Batch \d+ \| \d+/5000 \| \[(.*?)\]', t) for x in m.group(1).split(',')])


def jsonl(d):
    r = {}
    for f in Path(d).glob('select_*.jsonl'):
        for l in f.read_text().splitlines():
            if l.strip():
                x = json.loads(l); r[x['k']] = x['reward']
    return np.array([r[k] for k in range(5000)])


def main():
    OUT.mkdir(exist_ok=True); res = {'note': 'EXPLORATORY; not pre-registered'}
    for name, a, b in (('olmo_randopt_vs_boxed', log_rewards(A / 'o1-olmo-sameRun/pod/o1/out/randopt.log'), jsonl(A / 'gb-gsm8k-boxed-search/pod/gb/olmo/out')),
                       ('gqa_cot_vs_direct', jsonl(A / 'g2-sameRun/pod/g2/out'), jsonl(A / 'gd-gqa-direct-search/pod/gd/out'))):
        ta, tb = set(np.argsort(-a, kind='stable')[:50]), set(np.argsort(-b, kind='stable')[:50])
        res[name] = {'spearman_5000': float(spearmanr(a, b).correlation), 'top50_overlap': len(ta & tb),
                     'mean_midrank_percentile_of_first_top50_under_second': float(100 * np.mean([((b < b[k]).sum() + 0.5 * (b == b[k]).sum()) / 5000 for k in ta]))}
    agree = lambda u, v: float(np.mean([x == y for x, y in zip(u, v)]))
    tf = tarfile.open(A / 'g3-termination/pod/g3_out_nontext.tgz'); g3 = lambda nm: json.load(io.TextIOWrapper(tf.extractfile(f'out/b256_{nm}.json')))
    cot = [x['key'] for x in g3('base')]; mem = [[x['key'] for x in g3(f'rank{r}')] for r in range(50)]
    direct = [x['key'] for x in json.loads((A / 'g4-direct-prompt/pod/g4/out/direct/b256_base.json').read_text())]
    dis = [i for i in range(len(cot)) if cot[i] != direct[i]]
    res['gqa_members_vs_bases'] = {'agree_cot_base': 100 * np.mean([agree(m, cot) for m in mem]), 'agree_direct_base': 100 * np.mean([agree(m, direct) for m in mem]),
                                   'cot_vs_direct_base': 100 * agree(cot, direct), 'n_disagreements': len(dis),
                                   'on_disagreements_side_with_direct': 100 * float(np.mean([[m[i] == direct[i] for i in dis] for m in mem])),
                                   'on_disagreements_side_with_cot': 100 * float(np.mean([[m[i] == cot[i] for i in dis] for m in mem]))}
    E = json.loads((A / 'o1-olmo-sameRun/pod/o1/out/ensemble_answers.json').read_text())
    ro = [x['a'] for x in json.loads((A / 'o2-olmo-prompt/pod/o2/out/randopt_base.json').read_text())]
    for p in ('plain', 'boxed'):
        good = [x['a'] for x in json.loads((A / f'o2-olmo-prompt/pod/o2/out/{p}_base.json').read_text())]
        dis = [i for i in range(len(ro)) if ro[i] != good[i]]
        res[f'olmo_members_vs_{p}'] = {'agree_randopt_base': 100 * np.mean([agree(m, ro) for m in E]), f'agree_{p}_base': 100 * np.mean([agree(m, good) for m in E]),
                                       'n_disagreements': len(dis), f'on_disagreements_side_with_{p}': 100 * float(np.mean([[m[i] == good[i] for i in dis] for m in E])),
                                       'on_disagreements_side_with_randopt': 100 * float(np.mean([[m[i] == ro[i] for i in dis] for m in E]))}
    (OUT / 'expert_specificity.json').write_text(json.dumps(res, indent=1, default=float)); print(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main()

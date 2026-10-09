"""Locked analysis for OS (results/paper-analysis/os-prompt-control/plan_lock.md). CPU.
Prompt chosen on SEARCH200 by BASE accuracy (ties: direct, zeroshot_cot, manual_cot). Primary OS-1: winner (direct
prompt, as M1) minus BASE under the chosen prompt on the M1 hold-out. Key secondary OS-2: the winner's gain with the
prompt held fixed (chosen prompt). Image-cluster bootstrap exactly as M1 (2000 resamples, seed 20261006)."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

ORDER = ('direct', 'zeroshot_cot', 'manual_cot')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--d', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    ld = lambda n: json.loads(gzip.decompress((a.d / f'{n}.json.gz').read_bytes()))
    rows = [json.loads(l) for l in open('results/paper-analysis/m1/pod/holdout.jsonl', encoding='utf-8') if l.strip()]
    m1b = json.loads(gzip.decompress(Path('results/paper-analysis/m1/pod/base.json.gz').read_bytes()))
    m1w = json.loads(gzip.decompress(Path('results/paper-analysis/m1/pod/cand_9504111.json.gz').read_bytes()))['HOLDOUT']
    gold = np.array(['ABCD'[r['answer']] for r in rows]); img = np.array([r['image_sha256'] for r in rows])
    ui = np.unique(img); groups = [np.flatnonzero(img == u) for u in ui]; rng = np.random.default_rng(20261006)
    draws = [np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))]) for _ in range(2000)]
    boot = lambda d: [float(np.quantile([100 * d[ix].mean() for ix in draws], q)) for q in (.025, .975)]
    acc = lambda x: np.array([r['parsed'] for r in x]) == gold
    sel = {p: 100 * float(np.mean([r['correct'] for r in ld(f'base_search_{p}')])) for p in ORDER}
    chosen = max(ORDER, key=lambda p: (sel[p], -ORDER.index(p)))
    B = {p: acc(ld(f'base_holdout_{p}')) for p in ORDER}; W = {p: acc(ld(f'winner_holdout_{p}')) for p in ORDER}
    pf = {f'{m}_{p}': int(sum(r['parsed'] == '' for r in ld(f'{m}_holdout_{p}'))) for m in ('base', 'winner') for p in ORDER}
    res = {'selection_accuracy_search200': sel, 'chosen': chosen, 'n_items': len(rows), 'n_images': int(len(ui)),
           'holdout_accuracy': {f'{m}_{p}': 100 * float(X[p].mean()) for m, X in (('base', B), ('winner', W)) for p in ORDER},
           'parse_failures': pf,
           'env': {'base_direct_vs_m1': [int(B['direct'].sum()), int((np.array([x['parsed'] for x in m1b]) == gold).sum())],
                   'winner_direct_vs_m1': [int(W['direct'].sum()), int((np.array([x['parsed'] for x in m1w]) == gold).sum())],
                   'base_direct_answer_agreement_with_m1': float(np.mean([x['parsed'] == y['parsed'] for x, y in zip(ld('base_holdout_direct'), m1b)])),
                   'winner_direct_answer_agreement_with_m1': float(np.mean([x['parsed'] == y['parsed'] for x, y in zip(ld('winner_holdout_direct'), m1w)]))}}
    e = res['env']; res['valid_env'] = bool(abs(e['base_direct_vs_m1'][0] - e['base_direct_vs_m1'][1]) <= 3 and abs(e['winner_direct_vs_m1'][0] - e['winner_direct_vs_m1'][1]) <= 3)
    d = W['direct'].astype(float) - B[chosen].astype(float); c = boot(d)
    res['OS_1'] = {'D_winner_direct_minus_base_chosen': 100 * float(d.mean()), 'ci95_image_cluster': c,
                   'outcome': 'WINNER AHEAD' if c[0] > 0 else ('PROMPT AHEAD' if c[1] < 0 else 'NO DIFFERENCE DETECTED'),
                   'equivalent_2pp': bool(c[0] >= -2 and c[1] <= 2)}
    d = W[chosen].astype(float) - B[chosen].astype(float); c = boot(d)
    res['OS_2_winner_gain_prompt_held_fixed'] = {'prompt': chosen, 'D': 100 * float(d.mean()), 'ci95_image_cluster': c}
    res['winner_gain_by_prompt'] = {p: {'D': 100 * float((W[p].astype(float) - B[p].astype(float)).mean()), 'ci95_image_cluster': boot(W[p].astype(float) - B[p].astype(float))} for p in ORDER}
    if not res['valid_env']: res['OS_1']['outcome'] = 'INVALID-ENV (' + res['OS_1']['outcome'] + ')'
    txt = json.dumps(res, indent=1, default=float); a.out.write_text(txt); print(txt)


if __name__ == '__main__':
    main()

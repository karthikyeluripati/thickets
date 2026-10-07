"""Locked analysis for P2 (results/paper-analysis/p2/plan_lock.md)."""
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

import numpy as np

D = Path('results/paper-analysis/p2'); SRC = D / 'pod'
RANDOPT = Path('C:/Users/karth/AppData/Local/Temp/claude/c--Users-karth-OneDrive-Desktop-projects-thickets/f30c28bf-5da4-45f4-8bca-2f8edaf9c506/scratchpad/RandOpt')
PAPER = {('gsm8k', 'q05'): {'base': 39.9, 'ttmv': 41.0, 'randopt': 54.1}, ('gsm8k', 'q15'): {'base': 58.8, 'ttmv': 69.1, 'randopt': 76.4},
         ('gqa', 'vl3'): {'base': 56.6, 'ttmv': None, 'randopt': 69.0}}


def ld(f):
    return json.loads(gzip.decompress(Path(f).read_bytes()))


def main():
    sys.path.insert(0, str(RANDOPT))
    from data_handlers.gqa import GQAHandler
    gq = GQAHandler(); R = {}; rng = np.random.default_rng(20261007)
    for (task, tag), P in PAPER.items():
        g = SRC / f'{task}_{tag}_greedy.json.gz'
        if not g.exists(): R[f'{task}_{tag}'] = 'missing'; continue
        G = ld(g); gt = G['gt']; ids = list(gt)
        greedy = 100 * np.mean([G['res'][i][0]['c'] for i in ids])
        row = {'n_items': len(ids), 'greedy': float(greedy), 'paper': P, 'gate_pass': abs(greedy - P['base']) <= (2.5 if task == 'gqa' else 2.0)}

        def vote_ok(samples, i):
            v = Counter(s['a'] for s in samples).most_common(1)[0][0]
            if task == 'gsm8k':
                return v.replace(',', '') == gt[i]
            return gq._match_answer(gq._normalize_answer(v), gq._normalize_answer(gt[i]))
        for T in ('0.7', '0.3'):
            f = SRC / f'{task}_{tag}_T{T}.json.gz'
            if not f.exists(): continue
            S = ld(f)['res']; curve = {}
            for K in (1, 5, 10, 20, 50):
                ok = np.array([vote_ok(S[i][:K], i) for i in ids], float); curve[K] = float(100 * ok.mean())
                if K == 50: ok50 = ok
            bs = [100 * ok50[rng.integers(0, len(ok50), len(ok50))].mean() for _ in range(2000)]
            row[f'SC_T{T}'] = {'curve': curve, 'sc50_ci95': [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                               'mean_single_sample_acc': float(100 * np.mean([s['c'] for i in ids for s in S[i]]))}
        if 'SC_T0.7' in row:
            sc = row['SC_T0.7']['curve'][50]; ro = P['randopt']
            row['decision_vs_randopt'] = ('NOT COMPARABLE (gate)' if not row['gate_pass'] else 'MATCHES' if sc >= ro - 1.0 else 'RANDOPT AHEAD' if sc <= ro - 3.0 else 'CLOSE')
            row['sc50_minus_randopt'] = float(sc - ro)
            if P['ttmv'] is not None: row['sc50_minus_paper_ttmv'] = float(sc - P['ttmv'])
        R[f'{task}_{tag}'] = row
    (D / 'p2_results.json').write_text(json.dumps(R, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
    return R


if __name__ == '__main__':
    print(json.dumps(main(), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))

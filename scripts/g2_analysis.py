"""Locked analysis for G2 (results/paper-analysis/g2-sameRun/plan_lock.md). CPU. Needs RandOpt (--upstream) for GQAHandler."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

import numpy as np

P2 = Path('results/paper-analysis/p2/pod')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--upstream', type=Path, required=True); ap.add_argument('--d', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True); a = ap.parse_args()
    sys.path.insert(0, str(a.upstream))
    from data_handlers.gqa import GQAHandler
    h = GQAHandler()

    def key(ans):  # == extract_answer_for_voting applied after extract_answer
        n = h._normalize_answer(ans)
        return ' '.join(h._canonicalize(h._singularize(w)) for w in n.split()) if n else ''
    items = json.loads((a.d / 'items.json').read_text())['test']; ids = [r['id'] for r in items]; n = len(ids)
    gt = [{'answer': r['answer'], 'full_answer': r['fullAnswer']} for r in items]; img = [r['imageId'] for r in items]
    topk = json.loads((a.d / 'out/topk.json').read_text()); K = len(topk)
    E = [json.loads((a.d / f'out/test_rank{i}.json').read_text())['ans'] for i in range(K)]
    M = np.array([json.loads((a.d / f'out/test_rank{i}.json').read_text())['correct'] for i in range(K)], float)
    base = np.array(json.loads((a.d / 'out/test_base.json').read_text())['correct'], float)
    sc = json.loads(gzip.decompress((P2 / 'gqa_vl3_T0.7.json.gz').read_bytes()))['res']
    p2g = json.loads(gzip.decompress((P2 / 'gqa_vl3_greedy.json.gz').read_bytes()))['res']

    def vote(keys, i):
        keys = [x for x in keys if x]
        return bool(keys) and bool(h.is_voted_answer_correct(Counter(keys).most_common(1)[0][0], gt[i]))
    p2_greedy = np.array([p2g[i][0]['c'] for i in ids], float)
    res = {'n_test': n, 'K': K, 'base_g2': 100 * base.mean(), 'base_p2_same_items': 100 * p2_greedy.mean(),
           'validity_env': bool(abs(base.mean() - p2_greedy.mean()) * 100 <= 1.0),
           'members_mean': 100 * M.mean(), 'members_range': [100 * M.mean(1).min(), 100 * M.mean(1).max()],
           'sigma_top': dict(Counter(str(t['sigma']) for t in topk)), 'top_reward': [topk[0]['reward'], topk[-1]['reward']]}
    base_sel = a.d / 'out/base_select.json'
    if base_sel.exists(): res['base_select_reward'] = json.loads(base_sel.read_text())['reward']
    rng = np.random.default_rng(0); uimg = sorted(set(img)); idx_by_img = {u: [i for i in range(n) if img[i] == u] for u in uimg}
    for k in (50, 10):
        if k > K: continue
        R = np.array([vote([E[m][i] for m in range(k)], i) for i in range(n)], float)
        S = np.array([vote([key(s['a']) for s in sc[ids[i]][:k]], i) for i in range(n)], float)
        d = R - S; bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(10000)]) * 100; lo, hi = np.percentile(bs, [2.5, 97.5])
        cl = []
        for _ in range(2000):
            pick = rng.integers(0, len(uimg), len(uimg)); ii = [j for p in pick for j in idx_by_img[uimg[p]]]; cl.append(d[ii].mean() * 100)
        res[f'K{k}'] = {'randopt': 100 * R.mean(), 'sc': 100 * S.mean(), 'D': 100 * d.mean(), 'D_ci95': [float(lo), float(hi)],
                        'D_ci95_image_cluster': [float(np.percentile(cl, 2.5)), float(np.percentile(cl, 97.5))],
                        'outcome': 'RANDOPT AHEAD' if lo > 0 else ('SC AHEAD' if hi < 0 else 'NO DIFFERENCE DETECTED'),
                        'equivalent_2pp': bool(lo >= -2 and hi <= 2), 'n_randopt_only': int(((R == 1) & (S == 0)).sum()), 'n_sc_only': int(((R == 0) & (S == 1)).sum())}
    res['G2_1_valid'] = res['validity_env']
    a.out.write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()

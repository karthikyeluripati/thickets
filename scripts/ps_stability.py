"""EXPLORATORY: how stable is prompt selection (PS, ps-prompt-selection/plan_lock.md)? CPU only; no new generations.
(1) Bootstrap the 200 selection questions (10,000 resamples, seed 0): how often does each candidate win the rule
    (highest greedy BASE accuracy; ties by candidate order)? (2) The same with smaller selection sets (25, 50, 100
    questions drawn without replacement). (3) Regret: D = RandOpt K = 50 - SC@50 under EVERY candidate on test, so the
    cost of choosing a non-winning prompt is visible; and the expected D under the selection distribution.
Writes results/paper-analysis/ps-stability/ps_stability.json and .md. Needs RandOpt (--upstream) for GSM8K data."""
import argparse
from collections import Counter
import gzip
import io
import json
from pathlib import Path
import sys
import tarfile

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ps_analysis import A, GSM, GQA_PROMPTS  # noqa: E402

GSM_C = ('randopt', 'plain', 'boxed')
OUT = A / 'ps-stability'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--upstream', type=Path, required=True)
    ap.add_argument('--ps', type=Path, default=A / 'ps-prompt-selection/pod/ps/out'); a = ap.parse_args()
    sys.path.insert(0, str(a.upstream))
    from data_handlers.gqa import GQAHandler
    from data_handlers.gsm8k import GSM8KHandler

    def load(p):
        p = a.ps / p[3:] if isinstance(p, str) and p.startswith('PS:') else Path(p)
        if p.suffix == '.gz':
            r = json.loads(gzip.decompress(p.read_bytes()))['res']; return [r[str(i)] for i in range(len(r))]
        return json.loads(p.read_text())
    h = GSM8KHandler(); up = a.upstream / 'data/gsm8k'
    gt = [t['ground_truth'] for t in h.load_data(str(up / 'test.parquet'), split='test')]
    gts = [t['ground_truth'] for t in h.load_data(str(up / 'train.parquet'), split='train', max_samples=200)]
    ok = lambda v, g: bool(v) and bool(h.is_answer_correct(h.format_answer_for_check(v), g))

    def vote(ans, g):
        ans = [x for x in ans if x]
        return ok(Counter(ans).most_common(1)[0][0], g) if ans else False
    rng = np.random.default_rng(0)

    def choice_freq(S, cands, n, B=10000, replace=True):
        """S: candidates x 200 correctness matrix. Fraction of resamples in which each candidate wins (ties by order)."""
        m = S.shape[1]; wins = Counter()
        for _ in range(B):
            idx = rng.integers(0, m, n) if replace else rng.choice(m, n, replace=False)
            acc = S[:, idx].mean(1)
            best = max(range(len(cands)), key=lambda c: (acc[c], -c)); wins[cands[best]] += 1
        return {c: wins[c] / B for c in cands}

    res = {'note': 'EXPLORATORY; selection-set bootstrap and subsampling; D = RandOpt K=50 - SC@50 on test per candidate'}
    for row, (ens, scf, basef) in GSM.items():
        S = np.array([[ok(x['a'], gts[i]) for i, x in enumerate(load(f'PS:{row}/sel_{p}_base.json'))] for p in GSM_C], float)
        E = json.loads(Path(ens).read_text()); n = len(gt)
        R = np.array([vote([E[m][i] for m in range(50)], gt[i]) for i in range(n)], float)
        D = {}
        for p in GSM_C:
            sc = load(scf[p]); Sc = np.array([vote([x['a'] for x in sc[i][:50]], gt[i]) for i in range(n)], float)
            D[p] = 100 * float((R - Sc).mean())
        res[row] = {'selection_acc': {p: 100 * float(S[k].mean()) for k, p in enumerate(GSM_C)}, 'D_by_candidate': D}
    g = GQAHandler(); items = json.loads((A / 'g2-sameRun/pod/g2/items.json').read_text())
    gtq = [{'answer': r['answer'], 'full_answer': r['fullAnswer']} for r in items['test']]; nq = len(gtq)
    tf = tarfile.open(A / 'g3-termination/pod/g3_out_nontext.tgz'); g3 = lambda name: json.load(io.TextIOWrapper(tf.extractfile(f'out/b256_{name}.json')))

    def vq(keys, i):
        keys = [x for x in keys if x]
        return bool(keys) and bool(g.is_voted_answer_correct(Counter(keys).most_common(1)[0][0], gtq[i]))
    mem = [g3(f'rank{r}') for r in range(50)]; Rq = np.array([vq([mem[m][i]['key'] for m in range(50)], i) for i in range(nq)], float)
    S = np.array([[x['c'] for x in load(f'PS:gqa/{p}/sel_b256_base.json')] for p in GQA_PROMPTS], float)
    D = {}
    for p in GQA_PROMPTS:
        scq = g3('sc') if p == 'cot' else json.loads((A / f'g4-direct-prompt/pod/g4/out/{p}/b256_sc.json').read_text())
        Sq = np.array([vq([s['key'] for s in scq[i][:50]], i) for i in range(nq)], float); D[p] = 100 * float((Rq - Sq).mean())
    res['gqa'] = {'selection_acc': {p: 100 * float(S[k].mean()) for k, p in enumerate(GQA_PROMPTS)}, 'D_by_candidate': D}

    for row in ('q15', 'q3', 'olmo', 'gqa'):
        cands = GQA_PROMPTS if row == 'gqa' else GSM_C
        if row == 'gqa':
            S = np.array([[x['c'] for x in load(f'PS:gqa/{p}/sel_b256_base.json')] for p in cands], float)
        else:
            S = np.array([[ok(x['a'], gts[i]) for i, x in enumerate(load(f'PS:{row}/sel_{p}_base.json'))] for p in cands], float)
        r = res[row]; r['bootstrap_200'] = choice_freq(S, cands, 200)
        r['subsample'] = {str(n): choice_freq(S, cands, n, B=5000, replace=False) for n in (25, 50, 100)}
        r['expected_D_bootstrap_200'] = sum(r['bootstrap_200'][p] * r['D_by_candidate'][p] for p in cands)
        r['expected_D_subsample_50'] = sum(r['subsample']['50'][p] * r['D_by_candidate'][p] for p in cands)
        r['chosen'] = max(cands, key=lambda p: (r['selection_acc'][p], -cands.index(p)))
    OUT.mkdir(exist_ok=True)
    (OUT / 'ps_stability.json').write_text(json.dumps(res, indent=1))
    md = ['| Row | Selection accuracy (candidates) | Chosen | P(same choice), bootstrap of 200 | P(same), 50 questions | P(same), 25 questions | D under each candidate (RandOpt − SC@50) | Expected D, 200 / 50 |',
          '|---|---|---|---|---|---|---|---|']
    for row in ('q15', 'q3', 'olmo', 'gqa'):
        r = res[row]; c = r['chosen']
        md.append(f"| {row} | {' / '.join(f'{p} {v:.1f}' for p, v in r['selection_acc'].items())} | {c} | {r['bootstrap_200'][c]:.3f} | "
                  f"{r['subsample']['50'][c]:.3f} | {r['subsample']['25'][c]:.3f} | {' / '.join(f'{p} {v:+.2f}' for p, v in r['D_by_candidate'].items())} | "
                  f"{r['expected_D_bootstrap_200']:+.2f} / {r['expected_D_subsample_50']:+.2f} |")
    (OUT / 'ps_stability.md').write_text('\n'.join(md) + '\n\nEXPLORATORY. No new generations; PS selection outputs and existing test-set SC@50 runs.\n', encoding='utf-8')
    print('\n'.join(md))


if __name__ == '__main__':
    main()

"""Locked analysis for PS (results/paper-analysis/ps-prompt-selection/plan_lock.md). CPU. Needs RandOpt (--upstream) with
data/gsm8k/{train,test}.parquet. For each same-run row, choose the prompt with the highest greedy BASE accuracy on
RandOpt's selection set (ties: candidate order), then compare RandOpt K = 50 with SC@50 under the chosen prompt on test.
--ps: PS outputs (<row>/sel_<prompt>_base.json for GSM8K; gqa/<prompt>/sel_b256_base.json; q15|q3/boxed_sc.json)."""
import argparse
from collections import Counter
import gzip
import io
import json
from pathlib import Path
import sys
import tarfile

import numpy as np

A = Path('results/paper-analysis')
GSM = {  # row: (RandOpt dump, {prompt: SC@50 test file}, {prompt: BASE test file})
    'q15': (A / 'c-sameRun/pod6/c/out/ensemble_answers.json',
            {'randopt': A / 'c-sameRun/pod/c/out/gsm8k_c15_T0.7.json.gz', 'plain': A / 'q2-qwen-prompt/pod/q2/out/q15/plain_sc.json', 'boxed': 'PS:q15/boxed_sc.json'},
            {p: A / f'q2-qwen-prompt/pod/q2/out/q15/{p}_base.json' for p in ('randopt', 'plain', 'boxed')}),
    'q3': (A / 'c3b-sameRun/pod/c3b/out/ensemble_answers.json',
           {'randopt': A / 'c3b-sameRun/pod/c3b/out/gsm8k_c3b_T0.7.json.gz', 'plain': A / 'q2-qwen-prompt/pod/q2/out/q3/plain_sc.json', 'boxed': 'PS:q3/boxed_sc.json'},
           {p: A / f'q2-qwen-prompt/pod/q2/out/q3/{p}_base.json' for p in ('randopt', 'plain', 'boxed')}),
    'olmo': (A / 'o1-olmo-sameRun/pod/o1/out/ensemble_answers.json',
             {'randopt': A / 'o1-olmo-sameRun/pod/o1/sc/gsm8k_o1_T0.7.json.gz', 'plain': A / 'o2-olmo-prompt/pod/o2/out/plain_sc.json',
              'boxed': A / 'o2-olmo-prompt/pod/o2/out/boxed_sc.json'},
             {p: A / f'o2-olmo-prompt/pod/o2/out/{p}_base.json' for p in ('randopt', 'plain', 'boxed')})}
GQA_PROMPTS = ('cot', 'direct', 'short')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--upstream', type=Path, required=True); ap.add_argument('--ps', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True); a = ap.parse_args()
    sys.path.insert(0, str(a.upstream))
    from data_handlers.gqa import GQAHandler
    from data_handlers.gsm8k import GSM8KHandler
    boot = lambda n: np.random.default_rng(0).integers(0, n, (10000, n))

    def ci(x, b):
        return [float(v) for v in np.percentile(x[b].mean(1) * 100, [2.5, 97.5])]

    def outcome(c):
        return 'RANDOPT AHEAD' if c[0] > 0 else ('PROMPT-SELECTED SC AHEAD' if c[1] < 0 else 'NO DIFFERENCE DETECTED')

    def load(p):
        p = a.ps / p[3:] if isinstance(p, str) and p.startswith('PS:') else Path(p)
        if p.suffix == '.gz':  # p2_sc format {'res': {id: [{'a', 'c'}]}}
            r = json.loads(gzip.decompress(p.read_bytes()))['res']; return [r[str(i)] for i in range(len(r))]
        return json.loads(p.read_text())
    res = {'rule': 'argmax greedy BASE accuracy on the selection set; ties by candidate order'}
    # GSM8K rows
    h = GSM8KHandler(); up = a.upstream / 'data/gsm8k'
    gt = [t['ground_truth'] for t in h.load_data(str(up / 'test.parquet'), split='test')]; n = len(gt)
    gts = [t['ground_truth'] for t in h.load_data(str(up / 'train.parquet'), split='train', max_samples=200)]
    ok = lambda v, g: bool(v) and bool(h.is_answer_correct(h.format_answer_for_check(v), g))

    def vote(ans, g):
        ans = [x for x in ans if x]
        return ok(Counter(ans).most_common(1)[0][0], g) if ans else False
    B = boot(n)
    for row, (ens, scf, basef) in GSM.items():
        sel = {p: 100 * float(np.mean([ok(x['a'], gts[i]) for i, x in enumerate(load(f'PS:{row}/sel_{p}_base.json'))])) for p in ('randopt', 'plain', 'boxed')}
        chosen = max(('randopt', 'plain', 'boxed'), key=lambda p: (sel[p], -('randopt', 'plain', 'boxed').index(p)))
        E = json.loads(Path(ens).read_text()); R = np.array([vote([E[m][i] for m in range(50)], gt[i]) for i in range(n)], float)
        sc = load(scf[chosen]); S = np.array([vote([x['a'] for x in sc[i][:50]], gt[i]) for i in range(n)], float)
        base = load(basef[chosen]); Bc = np.array([ok(x['a'], gt[i]) for i, x in enumerate(base)], float)
        d = R - S; c = ci(d, B); d2 = R - Bc
        res[row] = {'selection_acc': sel, 'chosen': chosen, 'randopt_K50': 100 * R.mean(), 'sc50_chosen': 100 * S.mean(), 'base_chosen': 100 * Bc.mean(),
                    'D': {'D': 100 * d.mean(), 'ci': c, 'outcome': outcome(c), 'equivalent_2pp': bool(c[0] >= -2 and c[1] <= 2)},
                    'randopt_minus_base_chosen': {'D': 100 * d2.mean(), 'ci': ci(d2, B)}}
    # GQA row (G3/G4 outputs)
    g = GQAHandler(); items = json.loads((A / 'g2-sameRun/pod/g2/items.json').read_text())
    gtq = [{'answer': r['answer'], 'full_answer': r['fullAnswer']} for r in items['test']]; nq = len(gtq)
    tf = tarfile.open(A / 'g3-termination/pod/g3_out_nontext.tgz')
    g3 = lambda name: json.load(io.TextIOWrapper(tf.extractfile(f'out/b256_{name}.json')))

    def vq(keys, i):
        keys = [x for x in keys if x]
        return bool(keys) and bool(g.is_voted_answer_correct(Counter(keys).most_common(1)[0][0], gtq[i]))
    sel = {p: 100 * float(np.mean([x['c'] for x in load(f'PS:gqa/{p}/sel_b256_base.json')])) for p in GQA_PROMPTS}
    chosen = max(GQA_PROMPTS, key=lambda p: (sel[p], -GQA_PROMPTS.index(p)))
    mem = [g3(f'rank{r}') for r in range(50)]; R = np.array([vq([mem[m][i]['key'] for m in range(50)], i) for i in range(nq)], float)
    scq = g3('sc') if chosen == 'cot' else json.loads((A / f'g4-direct-prompt/pod/g4/out/{chosen}/b256_sc.json').read_text())
    bq = g3('base') if chosen == 'cot' else json.loads((A / f'g4-direct-prompt/pod/g4/out/{chosen}/b256_base.json').read_text())
    S = np.array([vq([s['key'] for s in scq[i][:50]], i) for i in range(nq)], float); Bc = np.array([x['c'] for x in bq], float)
    Bq = boot(nq); d = R - S; c = ci(d, Bq); d2 = R - Bc
    res['gqa'] = {'selection_acc': sel, 'chosen': chosen, 'randopt_K50': 100 * R.mean(), 'sc50_chosen': 100 * S.mean(), 'base_chosen': 100 * Bc.mean(),
                  'D': {'D': 100 * d.mean(), 'ci': c, 'outcome': outcome(c), 'equivalent_2pp': bool(c[0] >= -2 and c[1] <= 2)},
                  'randopt_minus_base_chosen': {'D': 100 * d2.mean(), 'ci': ci(d2, Bq)}}
    res['selection_generations'] = {'prompt_selection': '3 prompts x 200 = 600 greedy', 'randopt': '5000 perturbations x 200 = 1,000,000 greedy'}
    res['summary'] = {r: res[r]['D']['outcome'] for r in ('q15', 'q3', 'olmo', 'gqa')}
    txt = json.dumps(res, indent=1, default=lambda o: o.item()); a.out.write_text(txt); print(txt)


if __name__ == '__main__':
    main()

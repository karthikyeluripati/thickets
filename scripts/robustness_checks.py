"""EXPLORATORY robustness checks of the Claim 1 results (not pre-registered). CPU. Needs RandOpt (--upstream).
1 alignment: GSM8K question order/ground truth across datasets, RandOpt's parquet and every saved run.
2 strict scoring: GSM8K answers count only inside \\boxed{} (boxed/plain prompts) or after "####" (RandOpt's prompt);
  GQA answers count only on exact normalized match with the ground-truth answer. Re-tests PS and G4 with strict SC
  (RandOpt's member texts were not saved, so its side keeps its own scorer, which can only favour RandOpt).
3 exact McNemar p-values for every primary D, Holm-corrected across the four PS rows.
4 GQA image-cluster bootstrap for PS/G4. 5 PS at K = 10."""
import argparse
from collections import Counter
import gzip
import io
import json
from math import comb
from pathlib import Path
import re
import sys
import tarfile

import numpy as np

A = Path('results/paper-analysis')


def mcnemar(b, c):  # exact two-sided binomial test on discordant pairs
    n = b + c; k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--upstream', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); sys.path.insert(0, str(a.upstream))
    import datasets
    from data_handlers.gqa import GQAHandler
    from data_handlers.gsm8k import GSM8KHandler
    h = GSM8KHandler(); test = h.load_data(str(a.upstream / 'data/gsm8k/test.parquet'), split='test'); gt = [t['ground_truth'] for t in test]; n = len(gt)
    res = {}
    # 1 alignment
    ds = datasets.load_dataset('openai/gsm8k', 'main', split='test')
    dgt = [r['answer'].split('####')[-1].strip().replace(',', '') for r in ds]
    pq_q = [str(t.get('prompt', t.get('question', ''))) for t in test]
    res['alignment_gsm8k'] = {'n': n, 'gt_equal_datasets_vs_parquet': sum(x == y for x, y in zip(dgt, gt)),
                              'question_text_in_parquet_prompt': sum(r['question'][:60] in q for r, q in zip(ds, pq_q))}
    ok = lambda v, i: bool(v) and bool(h.is_answer_correct(h.format_answer_for_check(v), gt[i]))

    def num(s):
        s = re.sub(r'[^0-9.\-]', '', s.replace(',', ''))
        try: return float(s)
        except ValueError: return None

    def strict(text, prompt):
        if prompt == 'randopt':
            v = h.reward_module.extract_solution(text, method='strict')
        else:
            m = re.findall(r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}', text); v = m[-1] if m else None
        return None if v is None else num(v)
    sok = lambda v, i: v is not None and abs(v - float(gt[i])) < 1e-6

    def svote(vals, i):
        vals = [v for v in vals if v is not None]
        return sok(Counter(vals).most_common(1)[0][0], i) if vals else False
    gz = lambda p: json.loads(gzip.decompress(Path(p).read_bytes()))
    # 2 strict GSM8K: base arms (all prompts) and SC arms that have texts
    src = {'q15': A / 'q2-qwen-prompt/pod/q2/out/q15', 'q3': A / 'q2-qwen-prompt/pod/q2/out/q3', 'olmo': A / 'o2-olmo-prompt/pod/o2/out'}
    ps = json.loads((A / 'ps-prompt-selection/ps_results.json').read_text())
    ens = {'q15': A / 'c-sameRun/pod6/c/out/ensemble_answers.json', 'q3': A / 'c3b-sameRun/pod/c3b/out/ensemble_answers.json',
           'olmo': A / 'o1-olmo-sameRun/pod/o1/out/ensemble_answers.json'}
    boot = np.random.default_rng(0).integers(0, n, (10000, n))
    ci = lambda x, b: [float(v) for v in np.percentile(x[b].mean(1) * 100, [2.5, 97.5])]
    res['strict_gsm8k'] = {}
    for row, d in src.items():
        r = {}
        for p in ('randopt', 'plain', 'boxed'):
            f = d / f'{p}_base.texts.json.gz'
            if f.exists():
                T = gz(f); recs = json.loads((d / f'{p}_base.json').read_text())
                r[f'base_{p}'] = {'lenient': 100 * float(np.mean([ok(x['a'], i) for i, x in enumerate(recs)])),
                                  'strict': 100 * float(np.mean([sok(strict(t, p), i) for i, t in enumerate(T)]))}
        chosen = ps[row]['chosen']
        scd = A / 'ps-prompt-selection/pod/ps/out' / row if (row in ('q15', 'q3') and chosen == 'boxed') else d
        T = gz(scd / f'{chosen}_sc.texts.json.gz')
        S_strict = np.array([svote([strict(t, chosen) for t in T[i][:50]], i) for i in range(n)], float)
        E = json.loads(ens[row].read_text())
        R = np.array([(lambda v: ok(Counter(v).most_common(1)[0][0], i) if v else False)([x for x in (E[m][i] for m in range(50)) if x]) for i in range(n)], float)
        dd = R - S_strict; c = ci(dd, boot); b_, c_ = int(((R == 1) & (S_strict == 0)).sum()), int(((R == 0) & (S_strict == 1)).sum())
        r['ps_chosen'] = chosen; r['sc50_chosen_strict'] = 100 * S_strict.mean(); r['sc50_chosen_lenient'] = ps[row]['sc50_chosen']
        r['D_randopt_minus_strict_sc'] = {'D': 100 * dd.mean(), 'ci': c, 'mcnemar_p': mcnemar(b_, c_)}
        r['D_lenient_ps'] = ps[row]['D']
        res['strict_gsm8k'][row] = r
    # 3 McNemar on the locked PS comparisons (lenient, as locked)
    g = GQAHandler(); items = json.loads((A / 'g2-sameRun/pod/g2/items.json').read_text())['test']; nq = len(items)
    gq = [{'answer': x['answer'], 'full_answer': x['fullAnswer']} for x in items]; img = [x['imageId'] for x in items]
    tf = tarfile.open(A / 'g3-termination/pod/g3_out_nontext.tgz'); g3 = lambda nm: json.load(io.TextIOWrapper(tf.extractfile(f'out/b256_{nm}.json')))

    def vq(k, i, strict_=False):
        k = [x for x in k if x]
        if not k: return False
        top = Counter(k).most_common(1)[0][0]
        if strict_:
            gk = h_key(gq[i]['answer']); return top == gk
        return bool(g.is_voted_answer_correct(top, gq[i]))

    def h_key(ans):
        nn = g._normalize_answer(ans); return ' '.join(g._canonicalize(g._singularize(w)) for w in nn.split()) if nn else ''
    M = [g3(f'rank{r}') for r in range(50)]
    scd = json.loads((A / 'g4-direct-prompt/pod/g4/out/direct/b256_sc.json').read_text())
    Rg = {s: np.array([vq([M[m][i]['key'] for m in range(K)], i, s) for i in range(nq)], float) for s in (False, True) for K in (50,)}
    Sg = {s: np.array([vq([x['key'] for x in scd[i][:50]], i, s) for i in range(nq)], float) for s in (False, True)}
    bq = np.random.default_rng(0).integers(0, nq, (10000, nq))
    uimg = sorted(set(img)); idx = {u: [i for i in range(nq) if img[i] == u] for u in uimg}; rng = np.random.default_rng(1)
    cl_draws = [np.concatenate([idx[uimg[p]] for p in rng.integers(0, len(uimg), len(uimg))]) for _ in range(2000)]
    out = {}
    for s in (False, True):
        dd = Rg[s] - Sg[s]; b_, c_ = int(((Rg[s] == 1) & (Sg[s] == 0)).sum()), int(((Rg[s] == 0) & (Sg[s] == 1)).sum())
        out['strict' if s else 'lenient'] = {'randopt': 100 * Rg[s].mean(), 'sc_direct': 100 * Sg[s].mean(), 'D': 100 * dd.mean(), 'ci': ci(dd, bq),
                                            'image_cluster_ci': [float(v) for v in np.percentile([dd[c].mean() * 100 for c in cl_draws], [2.5, 97.5])],
                                            'mcnemar_p': mcnemar(b_, c_)}
    res['gqa_ps_g4'] = out
    # PS McNemar + Holm (locked lenient comparisons)
    rows = {}
    for row in ('q15', 'q3', 'olmo'):
        E = json.loads(ens[row].read_text())
        R = np.array([(lambda v: ok(Counter(v).most_common(1)[0][0], i) if v else False)([x for x in (E[m][i] for m in range(50)) if x]) for i in range(n)], float)
        chosen = ps[row]['chosen']
        f = (A / 'ps-prompt-selection/pod/ps/out' / row / 'boxed_sc.json') if (row in ('q15', 'q3') and chosen == 'boxed') else (src[row] / f'{chosen}_sc.json')
        sc = json.loads(f.read_text())
        for K in (50, 10):
            RK = R if K == 50 else np.array([(lambda v: ok(Counter(v).most_common(1)[0][0], i) if v else False)([x for x in (E[m][i] for m in range(10)) if x]) for i in range(n)], float)
            S = np.array([(lambda v: ok(Counter(v).most_common(1)[0][0], i) if v else False)([x['a'] for x in sc[i][:K] if x['a']]) for i in range(n)], float)
            dd = RK - S; b_, c_ = int(((RK == 1) & (S == 0)).sum()), int(((RK == 0) & (S == 1)).sum())
            rows.setdefault(row, {})[f'K{K}'] = {'D': 100 * dd.mean(), 'ci': ci(dd, boot), 'mcnemar_p': mcnemar(b_, c_)}
    R10 = np.array([vq([M[m][i]['key'] for m in range(10)], i) for i in range(nq)], float)
    S10 = np.array([vq([x['key'] for x in scd[i][:10]], i) for i in range(nq)], float); dd = R10 - S10
    rows['gqa'] = {'K50': {'D': out['lenient']['D'], 'ci': out['lenient']['ci'], 'mcnemar_p': out['lenient']['mcnemar_p']},
                   'K10': {'D': 100 * dd.mean(), 'ci': ci(dd, bq), 'mcnemar_p': mcnemar(int(((R10 == 1) & (S10 == 0)).sum()), int(((R10 == 0) & (S10 == 1)).sum()))}}
    order = sorted(rows, key=lambda r: rows[r]['K50']['mcnemar_p']); m = len(order); running = 0
    for j, r in enumerate(order):
        running = max(running, min(1.0, (m - j) * rows[r]['K50']['mcnemar_p'])); rows[r]['K50']['holm_p'] = running
    res['ps_tests'] = rows
    txt = json.dumps(res, indent=1, default=lambda o: o.item()); a.out.write_text(txt); print(txt)


if __name__ == '__main__':
    main()

"""EXPLORATORY (not pre-registered): what do RandOpt's selected GQA models do differently from the base model?
Input: G3 outputs (results/paper-analysis/g3-termination/pod/g3_*.tgz, extracted to --d) and G2 items. CPU only.
A form (length, boxing, reasoning), B where the member gain comes from (by base outcome), C whether the answers
selection adds are already in the base model's sampling distribution, D answer form on open questions,
E within-question: are shorter member generations more often correct?"""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np

ITEMS = Path('results/paper-analysis/g2-sameRun/pod/g2/items.json')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--d', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--budget', type=int, default=256); a = ap.parse_args(); b = a.budget
    items = json.loads(ITEMS.read_text())['test']; n = len(items)
    yn = np.array([r['answer'].strip().lower() in ('yes', 'no') for r in items])
    J = lambda name: json.loads((a.d / f'out/b{b}_{name}.json').read_text())
    T = lambda name: json.loads(gzip.decompress((a.d / f'out/b{b}_{name}.texts.json.gz').read_bytes()))
    base = J('base'); sc = J('sc'); mem = [J(f'rank{r}') for r in range(50)]
    btxt = T('base'); mtxt = [T(f'rank{r}') for r in range(50)]
    arr = lambda rows, f: np.array([f(x) for x in rows], float)
    Bc = arr(base, lambda x: x['c']); Mc = np.array([arr(m, lambda x: x['c']) for m in mem])          # 50 x n
    Bt = arr(base, lambda x: x['ntok']); Mt = np.array([arr(m, lambda x: x['ntok']) for m in mem])
    term = lambda x: x['fin'] == 'stop' and x['boxed']
    Bterm = arr(base, term).astype(bool)
    steps = lambda t: ('step' in t.lower()) or ('1.' in t)
    res = {'budget': b, 'n': n, 'note': 'EXPLORATORY; not pre-registered'}

    # A. form
    def form(rows, texts):
        return {'mean_tokens': float(np.mean([x['ntok'] for x in rows])), 'median_tokens': float(np.median([x['ntok'] for x in rows])),
                'boxed': 100 * float(np.mean([x['boxed'] for x in rows])), 'hit_max_tokens': 100 * float(np.mean([x['fin'] == 'length' for x in rows])),
                'direct_le32_tokens': 100 * float(np.mean([x['ntok'] <= 32 for x in rows])),
                'stepwise_reasoning': 100 * float(np.mean([steps(t) for t in texts])) if texts else None}
    res['A_form'] = {'base': form(base, btxt), 'members': form([x for m in mem for x in m], [t for m in mtxt for t in m]),
                     'sc_samples': form([s for row in sc for s in row], None)}
    acc_m = Mc.mean(1) * 100; tok_m = Mt.mean(1)
    res['A_form']['members_acc_vs_mean_tokens_r'] = float(np.corrcoef(acc_m, tok_m)[0, 1])
    res['A_form']['members_mean_tokens_range'] = [float(tok_m.min()), float(tok_m.max())]
    res['A_form']['members_direct_share_range'] = [float(v) for v in np.percentile([100 * np.mean([x['ntok'] <= 32 for x in m]) for m in mem], [0, 50, 100])]

    # B. member gain by base outcome (per question averaged over the 50 members)
    groups = {'base_correct': Bc == 1, 'base_wrong_terminated': (Bc == 0) & Bterm, 'base_wrong_unterminated': (Bc == 0) & ~Bterm}
    gain = Mc.mean(0) - Bc
    res['B_gain_by_base_outcome'] = {k: {'n': int(g.sum()), 'base_acc': 100 * float(Bc[g].mean()), 'member_acc': 100 * float(Mc.mean(0)[g].mean()),
                                         'share_of_total_gain': float(gain[g].sum() / gain.sum())} for k, g in groups.items()}
    res['B_total_member_gain_pp'] = 100 * float(gain.mean())
    for typ, g in (('yes_no', yn), ('open', ~yn)):
        res['B_gain_by_base_outcome'][f'{typ}_gain_pp'] = 100 * float(gain[g].mean())

    # C. are the answers selection adds already in base's sampling distribution?
    keyc = lambda rows: {x['key']: x['c'] for x in rows if x['key']}
    def vote(keys):
        keys = [k for k in keys if k]; return Counter(keys).most_common(1)[0][0] if keys else ''
    Rk = [vote([mem[m][i]['key'] for m in range(50)]) for i in range(n)]
    Sk = [vote([s['key'] for s in sc[i]]) for i in range(n)]
    lookup = [{**keyc(sc[i]), **keyc([mem[m][i] for m in range(50)]), **keyc([base[i]])} for i in range(n)]
    Rc = np.array([bool(lookup[i].get(Rk[i], False)) for i in range(n)], float); Sc = np.array([bool(lookup[i].get(Sk[i], False)) for i in range(n)], float)
    res['C_vote_acc_approx'] = {'randopt': 100 * Rc.mean(), 'sc': 100 * Sc.mean(), 'D': 100 * (Rc - Sc).mean(),
                                'note': 'key-level correctness lookup; compare with locked G3 numbers'}
    disc = np.where((Rc == 1) & (Sc == 0))[0]
    share = np.array([np.mean([s['key'] == Rk[i] for s in sc[i]]) for i in disc])
    mshare = np.array([np.mean([mem[m][i]['key'] == Rk[i] for m in range(50)]) for i in disc])
    sc_top = np.array([np.mean([s['key'] == Sk[i] for s in sc[i]]) for i in disc])
    res['C_randopt_only_correct'] = {'n': int(len(disc)), 'correct_answer_in_sc_samples_pct': 100 * float((share > 0).mean()),
                                     'median_sc_share_of_correct_answer': float(np.median(share)), 'mean_sc_share_of_correct_answer': float(share.mean()),
                                     'median_member_share_of_correct_answer': float(np.median(mshare)), 'median_sc_share_of_sc_winner': float(np.median(sc_top)),
                                     'base_greedy_correct_pct': 100 * float(Bc[disc].mean()), 'open_question_pct': 100 * float((~yn[disc]).mean())}
    disc2 = np.where((Rc == 0) & (Sc == 1))[0]
    res['C_sc_only_correct'] = {'n': int(len(disc2))}

    # D. answer form on open questions (vote keys vs ground truth)
    gt = [r['answer'].strip().lower() for r in items]
    wc = lambda k: len(k.split()) if k else 0
    op = np.where(~yn)[0]
    res['D_open_answers'] = {'gt_words': float(np.mean([wc(gt[i]) for i in op])),
                             'base_words': float(np.mean([wc(base[i]['key']) for i in op])),
                             'member_words': float(np.mean([wc(mem[m][i]['key']) for m in range(50) for i in op])),
                             'sc_sample_words': float(np.mean([wc(s['key']) for i in op for s in sc[i]])),
                             'base_key_eq_gt': 100 * float(np.mean([base[i]['key'] == gt[i] for i in op])),
                             'member_key_eq_gt': 100 * float(np.mean([mem[m][i]['key'] == gt[i] for m in range(50) for i in op])),
                             'base_long_answer_gt3w': 100 * float(np.mean([wc(base[i]['key']) > 3 for i in op])),
                             'member_long_answer_gt3w': 100 * float(np.mean([wc(mem[m][i]['key']) > 3 for m in range(50) for i in op]))}

    # E. within question: shorter member generations vs longer ones (median split per question)
    diffs = []
    for i in range(n):
        t = Mt[:, i]; c = Mc[:, i]; med = np.median(t); s, l = t < med, t > med
        if s.any() and l.any(): diffs.append(c[s].mean() - c[l].mean())
    diffs = np.array(diffs); rng = np.random.default_rng(0)
    bs = [diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(5000)]
    res['E_within_question_short_minus_long_acc'] = {'n_questions': int(len(diffs)), 'pp': 100 * float(diffs.mean()),
                                                     'ci95': [100 * float(v) for v in np.percentile(bs, [2.5, 97.5])]}
    txt = json.dumps(res, indent=1); a.out.write_text(txt); print(txt)


if __name__ == '__main__':
    main()

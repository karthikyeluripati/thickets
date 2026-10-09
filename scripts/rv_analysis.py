"""Locked analysis for RV-1, RV-3, RV-4 and RV-5 (results/paper-analysis/rv-reviewer-round/plan_lock.md). CPU. RV-2a/2b use
scripts/gb_analysis.py (commands in the lock). --pod: the retrieved /workspace/rv folder; --ps: PS outputs (for the pooled
rule and the T = 0.7 comparators). --selftest replaces the new templates by PS's three candidates (and GB's OLMo run for
RV-3) and must reproduce PS's four rows and GB's own-prompt difference."""
import argparse
from collections import Counter
import io
import json
from pathlib import Path
import sys
import tarfile

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ps_analysis import A, GSM, GQA_PROMPTS  # noqa: E402
from rv_choose import GQA_NEW, GSM_NEW  # noqa: E402

PS_SEL_GQA = {'cot': 'cot', 'direct': 'direct', 'short': 'short'}
RO_ENS_OLMO = A / 'o1-olmo-sameRun/pod/o1/out/ensemble_answers.json'
SC_OLMO_OWN = A / 'o1-olmo-sameRun/pod/o1/sc/gsm8k_o1_T0.7.json.gz'
BASE_OLMO_OWN = A / 'o2-olmo-prompt/pod/o2/out/randopt_base.json'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--upstream', type=Path, required=True); ap.add_argument('--pod', type=Path, required=True)
    ap.add_argument('--ps', type=Path, required=True); ap.add_argument('--out', type=Path, required=True); ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args(); sys.path.insert(0, str(a.upstream))
    from data_handlers.gqa import GQAHandler
    from data_handlers.gsm8k import GSM8KHandler
    import gzip
    boot = lambda n: np.random.default_rng(0).integers(0, n, (10000, n))
    ci = lambda x, b: [float(v) for v in np.percentile(x[b].mean(1) * 100, [2.5, 97.5])]
    outcome = lambda c: 'RANDOPT AHEAD' if c[0] > 0 else ('SC AHEAD' if c[1] < 0 else 'NO DIFFERENCE DETECTED')
    stat = lambda d, b: {'D': 100 * d.mean(), 'ci': ci(d, b), 'outcome': outcome(ci(d, b)), 'equivalent_2pp': bool(ci(d, b)[0] >= -2 and ci(d, b)[1] <= 2)}

    def load(p):
        p = a.ps / p[3:] if isinstance(p, str) and p.startswith('PS:') else Path(p)
        if p.suffix == '.gz':  # p2_sc format {'res': {id: [{'a', 'c'}]}}
            r = json.loads(gzip.decompress(p.read_bytes()))['res']; return [r[str(i)] for i in range(len(r))]
        return json.loads(p.read_text())
    h = GSM8KHandler(); up = a.upstream / 'data/gsm8k'
    gt = [t['ground_truth'] for t in h.load_data(str(up / 'test.parquet'), split='test')]; n = len(gt); B = boot(n)
    gts = [t['ground_truth'] for t in h.load_data(str(up / 'train.parquet'), split='train', max_samples=200)]
    ok = lambda v, g: bool(v) and bool(h.is_answer_correct(h.format_answer_for_check(v), g))

    def vote(ans, g):
        ans = [x for x in ans if x]
        return ok(Counter(ans).most_common(1)[0][0], g) if ans else False
    sel_acc = lambda recs: 100 * float(np.mean([ok(x['a'], gts[i]) for i, x in enumerate(recs)]))
    res = {'selftest': a.selftest, 'rule': 'RV-1: argmax greedy BASE accuracy on the 200 selection questions among the NEW public templates; ties by listed order'}

    # ---- RV-1, GSM8K rows
    new = ('randopt', 'plain', 'boxed') if a.selftest else GSM_NEW
    for row, (ens, scf, basef) in GSM.items():
        E = json.loads(Path(ens).read_text()); R = np.array([vote([E[m][i] for m in range(50)], gt[i]) for i in range(n)], float)
        d = a.pod / 'rv1' / row
        if a.selftest:
            sel_new = {p: sel_acc(load(f'PS:{row}/sel_{p}_base.json')) for p in new}
            files = lambda p: (load(scf[p]), load(basef[p]))
        else:
            sel_new = {p: sel_acc(json.loads((d / f'sel_{p}_base.json').read_text())) for p in new}
            files = lambda p: (json.loads((d / f'{p}_sc.json').read_text()), json.loads((d / f'{p}_base.json').read_text()))
        chosen = max(new, key=lambda p: (sel_new[p], -new.index(p)))
        sc, base = files(chosen)
        S = np.array([vote([x['a'] for x in sc[i][:50]], gt[i]) for i in range(n)], float); Bc = np.array([ok(x['a'], gt[i]) for i, x in enumerate(base)], float)
        S10 = np.array([vote([x['a'] for x in sc[i][:10]], gt[i]) for i in range(n)], float)
        r = {'selection_acc_new': sel_new, 'chosen_new': chosen, 'randopt_K50': 100 * R.mean(), 'sc50_chosen_new': 100 * S.mean(), 'base_chosen_new': 100 * Bc.mean(),
             'RV1': stat(R - S, B), 'K10': stat(R - S10, B), 'randopt_minus_one_generation': stat(R - Bc, B)}
        if not a.selftest:  # pooled rule over the six candidates (secondary); PS's selection accuracies for the three old ones
            sel_old = {p: sel_acc(load(f'PS:{row}/sel_{p}_base.json')) for p in ('randopt', 'plain', 'boxed')}
            pool = {**sel_old, **sel_new}; order = ('randopt', 'plain', 'boxed') + GSM_NEW
            pooled = max(order, key=lambda p: (pool[p], -order.index(p)))
            if pooled in sel_new: Sp = S if pooled == chosen else None
            else:
                scp = load(scf[pooled]); Sp = np.array([vote([x['a'] for x in scp[i][:50]], gt[i]) for i in range(n)], float)
            r['pooled'] = {'selection_acc_all': pool, 'chosen_pooled': pooled,
                           'RV1_pooled': stat(R - Sp, B) if Sp is not None else 'the pooled choice is a new template that was not the new-only choice: not run'}
        res[row] = r
    # ---- RV-1, GQA row
    g = GQAHandler(); items = json.loads((A / 'g2-sameRun/pod/g2/items.json').read_text())
    gtq = [{'answer': r['answer'], 'full_answer': r['fullAnswer']} for r in items['test']]; nq = len(gtq); Bq = boot(nq)
    tf = tarfile.open(A / 'g3-termination/pod/g3_out_nontext.tgz'); g3 = lambda name: json.load(io.TextIOWrapper(tf.extractfile(f'out/b256_{name}.json')))

    def vq(keys, i):
        keys = [x for x in keys if x]
        return bool(keys) and bool(g.is_voted_answer_correct(Counter(keys).most_common(1)[0][0], gtq[i]))
    mem = [g3(f'rank{r}') for r in range(50)]; Rq = np.array([vq([mem[m][i]['key'] for m in range(50)], i) for i in range(nq)], float)
    old_test = lambda p: ((g3('sc'), g3('base')) if p == 'cot' else tuple(json.loads((A / f'g4-direct-prompt/pod/g4/out/{p}/b256_{x}.json').read_text()) for x in ('sc', 'base')))
    newq = GQA_PROMPTS if a.selftest else GQA_NEW
    if a.selftest:
        selq = {p: 100 * float(np.mean([x['c'] for x in load(f'PS:gqa/{p}/sel_b256_base.json')])) for p in newq}; filesq = old_test
    else:
        dq = a.pod / 'rv1' / 'gqa'
        selq = {p: 100 * float(np.mean([x['c'] for x in json.loads((dq / p / 'sel_b256_base.json').read_text())])) for p in newq}
        filesq = lambda p: tuple(json.loads((dq / p / f'b256_{x}.json').read_text()) for x in ('sc', 'base'))
    chosenq = max(newq, key=lambda p: (selq[p], -newq.index(p))); scq, bq = filesq(chosenq)
    Sq = np.array([vq([s['key'] for s in scq[i][:50]], i) for i in range(nq)], float); Sq10 = np.array([vq([s['key'] for s in scq[i][:10]], i) for i in range(nq)], float)
    Bcq = np.array([x['c'] for x in bq], float)
    r = {'selection_acc_new': selq, 'chosen_new': chosenq, 'randopt_K50': 100 * Rq.mean(), 'sc50_chosen_new': 100 * Sq.mean(), 'base_chosen_new': 100 * Bcq.mean(),
         'RV1': stat(Rq - Sq, Bq), 'K10': stat(Rq - Sq10, Bq), 'randopt_minus_one_generation': stat(Rq - Bcq, Bq)}
    if not a.selftest:
        sel_old = {p: 100 * float(np.mean([x['c'] for x in load(f'PS:gqa/{p}/sel_b256_base.json')])) for p in GQA_PROMPTS}
        pool = {**sel_old, **selq}; order = GQA_PROMPTS + GQA_NEW; pooled = max(order, key=lambda p: (pool[p], -order.index(p)))
        if pooled in selq: Sp = Sq if pooled == chosenq else None
        else:
            scp, _ = old_test(pooled); Sp = np.array([vq([s['key'] for s in scp[i][:50]], i) for i in range(nq)], float)
        r['pooled'] = {'selection_acc_all': pool, 'chosen_pooled': pooled,
                       'RV1_pooled': stat(Rq - Sp, Bq) if Sp is not None else 'the pooled choice is a new template that was not the new-only choice: not run'}
    res['gqa'] = r
    res['RV1_summary'] = {k: res[k]['RV1']['outcome'] + (' (EQUIVALENT)' if res[k]['RV1']['equivalent_2pp'] else '') for k in ('q15', 'q3', 'olmo', 'gqa')}

    # ---- RV-5 temperature (Qwen-3B boxed; GQA direct), T = 0.7 from PS / G4 for reference
    if not a.selftest:
        E = json.loads(Path(GSM['q3'][0]).read_text()); R3 = np.array([vote([E[m][i] for m in range(50)], gt[i]) for i in range(n)], float)
        rv5 = {}
        for T, name in (('0.7', 'PS:q3/boxed_sc.json'), ('0.5', a.pod / 'rv5/q3/boxed_sc_T0.5.json'), ('1.0', a.pod / 'rv5/q3/boxed_sc_T1.json')):
            sc = load(name) if isinstance(name, str) else json.loads(name.read_text())
            S = np.array([vote([x['a'] for x in sc[i][:50]], gt[i]) for i in range(n)], float)
            rv5[f'q3_boxed_T{T}'] = {'sc50': 100 * S.mean(), **stat(R3 - S, B)}
        for T, name in (('0.7', A / 'g4-direct-prompt/pod/g4/out/direct/b256_sc.json'), ('0.5', a.pod / 'rv5/gqa/direct/b256_sc_T0.5.json'), ('1.0', a.pod / 'rv5/gqa/direct/b256_sc_T1.json')):
            sc = json.loads(name.read_text()); S = np.array([vq([s['key'] for s in sc[i][:50]], i) for i in range(nq)], float)
            rv5[f'gqa_direct_T{T}'] = {'sc50': 100 * S.mean(), **stat(Rq - S, Bq)}
        res['RV5'] = rv5

    # ---- RV-3: second OLMo search (seed 43) under RandOpt's prompt vs SC@50 (O1's) and vs O1's search (seed 42)
    d3 = (A / 'gb-gsm8k-boxed-search/pod/gb/olmo') if a.selftest else (a.pod / 'olmo2')
    if (d3 / 'out' / 'test_rank49.json').exists():
        E2 = [json.loads((d3 / 'out' / f'test_rank{r}.json').read_text())['ans'] for r in range(50)]
        R2 = {K: np.array([vote([E2[m][i] for m in range(K)], gt[i]) for i in range(n)], float) for K in (50, 10)}
        M2 = np.array([[ok(E2[m][i], gt[i]) for i in range(n)] for m in range(50)], float)
        sc = load(SC_OLMO_OWN); S = {K: np.array([vote([x['a'] for x in sc[i][:K]], gt[i]) for i in range(n)], float) for K in (50, 10)}
        Eo = json.loads(RO_ENS_OLMO.read_text()); Ro = np.array([vote([Eo[m][i] for m in range(50)], gt[i]) for i in range(n)], float)
        Bo = np.array([ok(x['a'], gt[i]) for i, x in enumerate(json.loads(BASE_OLMO_OWN.read_text()))], float)
        Bt = np.array([ok(v, gt[i]) for i, v in enumerate(json.loads((d3 / 'out' / 'test_base.json').read_text())['ans'])], float)
        top = json.loads((d3 / 'out' / 'topk.json').read_text()); sel_base = 100 * json.loads((d3 / 'out' / 'base_select.json').read_text())['reward']
        res['RV3'] = {'randopt_seed43_K50': 100 * R2[50].mean(), 'sc50_own_prompt': 100 * S[50].mean(), 'RV3': stat(R2[50] - S[50], B), 'K10': stat(R2[10] - S[10], B),
                      'seed_to_seed_minus_O1': stat(R2[50] - Ro, B), 'O1_K50': 100 * Ro.mean(), 'members_mean': 100 * M2.mean(),
                      'members_minus_base': stat(M2.mean(0) - Bt, B), 'base_this_run': 100 * Bt.mean(), 'base_O2': 100 * Bo.mean(),
                      'selection_base': sel_base, 'selection_top_reward': [100 * top[0]['reward'], 100 * top[-1]['reward']],
                      'vote_gain_over_members': 100 * (R2[50].mean() - M2.mean())}
    # ---- RV-4
    if not a.selftest:
        for f in (a.pod / 'rv4').glob('randopt_*.log'):
            import re
            m = re.search(r'Train reward: ([\d.]+)%', f.read_text(errors='ignore'))
            res.setdefault('RV4', {})[f.stem] = float(m.group(1)) if m else None
        bs = a.pod / 'q15/out_fid/base_select.json'
        if bs.exists(): res.setdefault('RV4', {})['ours_fast_runner'] = 100 * json.loads(bs.read_text())['reward']
        res.setdefault('RV4', {}).update({'C_printed': 73.0, 'PS': 68.5, 'GB2': 68.0})
    txt = json.dumps(res, indent=1, default=lambda o: o.item()); a.out.write_text(txt); print(txt)


if __name__ == '__main__':
    main()

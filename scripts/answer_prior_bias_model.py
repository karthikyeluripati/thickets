"""Post-hoc, CPU-only: how much of a candidate's behavior does a PURE answer-content bias reproduce?

Model: z_cand(j, k) ~ z_base(j, k) + sum_f b_f * 1[option k contains f] (+ per-item constant), f in FEATS.
b is fitted by least squares on centered score changes. Predicted answers = argmax(z_base + X b).
Fits: RERANK (2-fold split by uid hash for out-of-sample), then the RERANK-fitted b applied to TEST base scores.
Winner and the 12 controls (vLLM letter log-probs).
"""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from answer_prior_analysis import FEATS, LET, OUT, PA, feats


def main():
    ex = {}
    for s in ('validation', 'test'):
        for l in Path(f'examples/omnispatial-perspective-taking/{s}.jsonl').read_text(encoding='utf-8').splitlines():
            r = json.loads(l); ex[r['uid']] = r
    ld = lambda f: json.loads(gzip.decompress(Path(f).read_bytes()))
    vec = lambda q: np.array([q['letter_logprobs'][a] for a in LET], float)
    base = {q['uid']: vec(q) for q in ld(PA / 'causal-diagnostic/session1/base_full.json.gz')}
    cands = {'9504111': {q['uid']: vec(q) for q in ld(PA / 'causal-diagnostic/session1/candidate_full.json.gz')}}
    for c in json.loads((PA / 'random-control-transfer/frozen_controls.json').read_text())['candidates']:
        rr = ld(PA / f"random-control-transfer/gpu/control_{c['seed']}.json.gz")
        cands[str(c['seed'])] = {q['uid']: vec(q) for ph in ('RERANK', 'TEST') for q in rr[ph]}
    X = {u: np.array([[float(feats(o)[f]) for f in FEATS] for o in ex[u]['options']]) for u in ex}
    Xc = {u: X[u] - X[u].mean(0) for u in X}
    fold = {u: int(hashlib.sha256(u.encode()).hexdigest(), 16) % 2 for u in ex}

    def fit(Z, uids):
        A = np.vstack([Xc[u] for u in uids]); y = np.concatenate([(Z[u] - Z[u].mean()) - (base[u] - base[u].mean()) for u in uids])
        b, *_ = np.linalg.lstsq(A, y, rcond=None)
        r2 = 1 - ((y - A @ b) ** 2).sum() / (y ** 2).sum()
        return b, float(r2)

    def evaluate(Z, b, uids):
        gold = np.array([ex[u]['answer'] for u in uids])
        pb = np.array([int(np.argmax(base[u])) for u in uids]); pc = np.array([int(np.argmax(Z[u])) for u in uids])
        pm = np.array([int(np.argmax(base[u] + X[u] @ b)) for u in uids])
        ch = pc != pb
        return {'n': len(uids), 'actual_gain_pp': float(100 * ((pc == gold).mean() - (pb == gold).mean())),
                'bias_model_gain_pp': float(100 * ((pm == gold).mean() - (pb == gold).mean())),
                'actual_changed': int(ch.sum()), 'model_changed': int((pm != pb).sum()),
                'model_reproduces_changed_answer': int((ch & (pm == pc)).sum()), 'model_false_changes': int(((pm != pb) & ~ch).sum()),
                'actual_repairs': int(((pc == gold) & (pb != gold)).sum()), 'actual_regressions': int(((pc != gold) & (pb == gold)).sum()),
                'model_repairs': int(((pm == gold) & (pb != gold)).sum()), 'model_regressions': int(((pm != gold) & (pb == gold)).sum())}
    R = {'post_hoc': True, 'features': list(FEATS), 'per_candidate': {}}
    for name, Z in cands.items():
        RU = [u for u in Z if not u.startswith('test:')]; TU = [u for u in Z if u.startswith('test:')]
        bR, r2R = fit(Z, RU); bT, r2T = fit(Z, TU)
        oos = {}
        for k in (0, 1):
            tr = [u for u in RU if fold[u] != k]; te = [u for u in RU if fold[u] == k]
            bk, _ = fit(Z, tr); e = evaluate(Z, bk, te)
            for kk, v in e.items(): oos[kk] = oos.get(kk, 0) + (v if kk not in ('actual_gain_pp', 'bias_model_gain_pp') else v * len(te) / len(RU))
        R['per_candidate'][name] = {'b_RERANK': dict(zip(FEATS, map(float, bR))), 'score_r2_RERANK': r2R,
                                    'b_TEST': dict(zip(FEATS, map(float, bT))), 'score_r2_TEST': r2T,
                                    'RERANK_in_sample': evaluate(Z, bR, RU), 'RERANK_2fold_out_of_sample': oos,
                                    'TEST_with_RERANK_bias': evaluate(Z, bR, TU), 'TEST_with_TEST_bias': evaluate(Z, bT, TU)}
    (OUT / 'answer_prior_bias_model.json').write_text(json.dumps(R, indent=1, default=float))
    return R


if __name__ == '__main__':
    R = main()
    for n, v in R['per_candidate'].items():
        o, t, tt = v['RERANK_2fold_out_of_sample'], v['TEST_with_RERANK_bias'], v['TEST_with_TEST_bias']
        print(f"{n}: bR={ {k: round(x, 2) for k, x in v['b_RERANK'].items()} } r2R={v['score_r2_RERANK']:.2f} r2T={v['score_r2_TEST']:.2f}")
        print(f"   RERANK oos: actual {o['actual_gain_pp']:+.1f} model {o['bias_model_gain_pp']:+.1f} | changed {o['actual_changed']} reproduced {o['model_reproduces_changed_answer']} false {o['model_false_changes']}")
        print(f"   TEST (RERANK b): actual {t['actual_gain_pp']:+.2f} model {t['bias_model_gain_pp']:+.2f} | TEST (own b): model {tt['bias_model_gain_pp']:+.2f} reproduced {tt['model_reproduces_changed_answer']}/{tt['actual_changed']}")

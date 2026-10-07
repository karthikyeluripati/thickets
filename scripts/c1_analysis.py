"""Locked analysis for C1 / item 3 (results/paper-analysis/c1/plan_lock.md): cross-fitted answer-content prior calibration
on the M1 hold-out."""
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from answer_prior_analysis import feats  # noqa: E402

PA = Path('results/paper-analysis')
M1 = PA / 'm1'
D = PA / 'c1'
WIN = '25a60f0b59f103ebf782a31f94bb569d897a652a0bc1bd95c5cdc5521418b843'
LET = 'ABCD'
FE = ('front', 'back', 'left', 'right')


def ld(f):
    return json.loads(gzip.decompress(Path(f).read_bytes()))


def zvec(x):
    s = x['letter_logprobs']; fl = x['top20_floor']
    return np.array([s[L] if s[L] is not None else fl for L in LET], float)


def main():
    rows = [json.loads(l) for l in (M1 / 'pod/holdout.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    gold = np.array([r['answer'] for r in rows]); img = np.array([r['image_sha256'] for r in rows])
    half = np.array([int(hashlib.sha256(('c1-half-v1:' + h).encode()).hexdigest()[0], 16) % 2 == 0 for h in img])
    X = np.array([[[float(feats(o)[f]) for f in FE] for o in r['options']] for r in rows])  # items x 4 options x 4 feats
    Xc = X - X.mean(1, keepdims=True)
    base = ld(M1 / 'pod/base.json.gz'); base = base['HOLDOUT'] if isinstance(base, dict) else base; zb = np.array([zvec(x) for x in base]); zbc = zb - zb.mean(1, keepdims=True)
    cb = zb.argmax(1) == gold
    cands = json.loads((M1 / 'frozen_candidates.json').read_text())['candidates']
    res, keep = {}, {}
    for c in cands:
        f = M1 / f"pod/cand_{c['seed']}.json.gz"
        if not f.exists(): continue
        out = ld(f)['HOLDOUT']; z = np.array([zvec(x) for x in out]); zc = z - z.mean(1, keepdims=True)
        dz = zc - zbc; zcal = z.copy(); r2 = []
        for fit_mask in (half, ~half):
            A = Xc[fit_mask].reshape(-1, len(FE)); y = dz[fit_mask].reshape(-1)
            b, *_ = np.linalg.lstsq(A, y, rcond=None); r2.append(1 - ((y - A @ b) ** 2).sum() / (y ** 2).sum())
            zcal[~fit_mask] = z[~fit_mask] - Xc[~fit_mask] @ b
        craw = z.argmax(1) == gold; ccal = zcal.argmax(1) == gold
        parsed_agree = float(np.mean([x['parsed'] == LET[i] for x, i in zip(out, z.argmax(1))]))
        res[c['candidate_id']] = {'role': c['role'], 'g': float(100 * (craw.mean() - cb.mean())), 'g_cal': float(100 * (ccal.mean() - cb.mean())),
                                  'shift_r2_mean': float(np.mean(r2)), 'argmax_parsed_agreement': parsed_agree}
        keep[c['candidate_id']] = (craw, ccal)
    R = {'n_items': len(rows), 'n_candidates': len(res), 'base_argmax_acc': float(100 * cb.mean()),
         'base_argmax_parsed_agreement': float(np.mean([x['parsed'] == LET[i] for x, i in zip(base, zb.argmax(1))]))}
    ui = np.unique(img); groups = [np.flatnonzero(img == u) for u in ui]; rng = np.random.default_rng(20261006)
    if WIN in res:
        g, gc = res[WIN]['g'], res[WIN]['g_cal']; craw, ccal = keep[WIN]
        bs = []
        for _ in range(2000):
            ix = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))]); bs.append(100 * (craw[ix].mean() - ccal[ix].mean()))
        ci = [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]
        verdict = ('NOT APPLICABLE (no raw gain)' if g <= 0 else 'PRIOR-SHIFT EXPLAINS' if gc <= 0.5 * g and ci[0] > 0
                   else 'NOT EXPLAINED' if gc >= 0.8 * g else 'PARTIAL')
        R['C1_1'] = {'raw_g': g, 'g_cal': gc, 'g_minus_gcal_ci95': ci, 'verdict': verdict, 'shift_r2': res[WIN]['shift_r2_mean']}
    pos = {k: v for k, v in res.items() if v['g'] > 0}
    if pos:
        gs = np.array([v['g'] for v in pos.values()]); gcs = np.array([v['g_cal'] for v in pos.values()])
        bs = [(lambda ix: gcs[ix].sum() / gs[ix].sum())(rng.integers(0, len(gs), len(gs))) for _ in range(2000)]
        R['C1_2'] = {'n_positive': len(pos), 'surviving_share': float(gcs.sum() / gs.sum()), 'ci95': [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                     'supports_mostly_prior_shift': gcs.sum() / gs.sum() <= 0.5}
    allg = np.array([v['g'] for v in res.values()]); allc = np.array([v['g_cal'] for v in res.values()])
    R['C1_3'] = {'r_g_gcal': float(np.corrcoef(allg, allc)[0, 1]),
                 'top50_mean_g': float(np.mean([v['g'] for v in res.values() if v['role'] in ('top50', 'winner')])),
                 'top50_mean_gcal': float(np.mean([v['g_cal'] for v in res.values() if v['role'] in ('top50', 'winner')])),
                 'controls_mean_g': float(np.mean([v['g'] for v in res.values() if v['role'] == 'control'])),
                 'controls_mean_gcal': float(np.mean([v['g_cal'] for v in res.values() if v['role'] == 'control']))}
    R['C1_4'] = {'median_shift_r2': float(np.median([v['shift_r2_mean'] for v in res.values()]))}
    D.mkdir(parents=True, exist_ok=True)
    (D / 'c1_results.json').write_text(json.dumps(R, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
    (D / 'c1_per_candidate.json').write_text(json.dumps(res, indent=1))
    return R


if __name__ == '__main__':
    print(json.dumps(main(), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))

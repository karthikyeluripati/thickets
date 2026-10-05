"""Parts N/O: POST_HOC_PROMPT_ROBUSTNESS_DIAGNOSTIC table and fig9 from the Part N GPU outputs.

Official-style = OmniSpatial manual-CoT prompt + official `re` parser (last 'Answer: X', fallback 'A'),
greedy. The strict parser (no fallback) is reported alongside. Paired bootstrap over questions.
"""
import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import ACCENT, BASE_C, CAND, INK2, MUTED, dump, load, master, save, write_tex, CANDIDATE

PR = Path('results/paper-analysis/prompt-robustness')
LETTERS = 'ABCD'


def gold(split):
    name = {'RERANK': 'validation', 'TEST': 'test'}[split]
    rows = [json.loads(l) for l in Path(f'examples/omnispatial-perspective-taking/{name}.jsonl').read_text(encoding='utf-8').splitlines()]
    return [r['uid'] for r in rows], np.array([LETTERS[r['answer']] for r in rows])


def boot(b, c, n=10000, seed=20261005):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(b), (n, len(b)))
    d = (c[idx].mean(1) - b[idx].mean(1)) * 100
    return [float(np.quantile(d, .025)), float(np.quantile(d, .975))]


def main():
    df = master()
    rows, out = [], {}
    for split in ('RERANK', 'TEST'):
        uids, g = gold(split)
        sub = df[(df.phase == split)]
        bd = sub[sub.candidate_id == 'BASE'].set_index('example_id').loc[uids]
        cd = sub[sub.candidate_id == CANDIDATE].set_index('example_id').loc[uids]
        b, c = bd.base_correct.to_numpy(bool), cd.candidate_correct.to_numpy(bool)
        out[('direct', split)] = dict(base=int(b.sum()), cand=int(c.sum()), n=len(g), gain=100 * (c.mean() - b.mean()), ci=boot(b, c))
        sfx = split.lower()
        fb, fc = PR / f'p{2 if split == "RERANK" else 1}_base_{sfx}_manualcot_greedy.json.gz', PR / f'p2_cand9504111_{sfx}_manualcot_greedy.json.gz'
        if fb.exists() and fc.exists():
            B, C = load(fb), load(fc)
            assert B['uids'] == uids and C['uids'] == uids
            for parser in ('official', 'strict'):
                b2 = np.array(B[f'pred_{parser}']) == g; c2 = np.array(C[f'pred_{parser}']) == g
                out[(f'cot_{parser}', split)] = dict(base=int(b2.sum()), cand=int(c2.sum()), n=len(g), gain=100 * (c2.mean() - b2.mean()), ci=boot(b2, c2),
                                                     base_no_answer=B['no_answer_match'], cand_no_answer=C['no_answer_match'],
                                                     base_truncated=B['truncated'], cand_truncated=C['truncated'])
    label = {'direct': 'Direct letter (original)', 'cot_official': 'Official manual-CoT, \\texttt{re} parser',
             'cot_strict': 'Official manual-CoT, strict parser'}
    for k in ('direct', 'cot_official', 'cot_strict'):
        for split in ('RERANK', 'TEST'):
            if (k, split) not in out: continue
            o = out[(k, split)]
            rows.append([label[k] if split == 'RERANK' else '', split, f"{o['base']}/{o['n']} ({100 * o['base'] / o['n']:.2f})",
                         f"{o['cand']}/{o['n']} ({100 * o['cand'] / o['n']:.2f})", f"{o['gain']:+.2f} [{o['ci'][0]:+.1f}, {o['ci'][1]:+.1f}]"])
        rows.append('MIDRULE')
    rows = rows[:-1]
    dump('table_prompt_robustness', [{'prompt': k, 'split': s, **v} for (k, s), v in out.items()])
    write_tex('table_prompt_robustness', ['Prompt / parser', 'Split', 'Base', 'Seed 9504111', 'Gain (pp) [95\\% CI]'], rows,
              'POST\\_HOC\\_PROMPT\\_ROBUSTNESS\\_DIAGNOSTIC: seed 9504111 (selected under the direct-letter prompt) re-evaluated '
              'with the official OmniSpatial manual-CoT prompt (greedy). Paired bootstrap over questions. One prompt pair; not a general result.',
              'tab:prompt', colspec='llrrr')
    if ('cot_official', 'TEST') in out:
        fig, ax = plt.subplots(figsize=(3.4, 2.5))
        for k, col, mk, name in (('direct', CAND, 'o', 'Direct letter (original)'), ('cot_official', ACCENT, 's', 'Official manual-CoT')):
            y = [out[(k, s)]['gain'] for s in ('RERANK', 'TEST')]
            lo = [y[i] - out[(k, s)]['ci'][0] for i, s in enumerate(('RERANK', 'TEST'))]
            hi = [out[(k, s)]['ci'][1] - y[i] for i, s in enumerate(('RERANK', 'TEST'))]
            dx = -.04 if k == 'direct' else .04
            ax.errorbar(np.array([0, 1]) + dx, y, yerr=[lo, hi], color=col, marker=mk, ms=6, capsize=2, lw=1.6, label=name)
        ax.axhline(0, color=MUTED, lw=.8)
        ax.set_xticks([0, 1], ['RERANK (n=200)', 'TEST (n=561)']); ax.set_xlim(-.4, 1.4)
        ax.set_ylabel('Seed 9504111 gain over base (pp)')
        ax.legend(frameon=False, fontsize=6.5, loc='upper center', bbox_to_anchor=(.5, -.18), ncol=1)
        save(fig, 'fig9_prompt_robustness')
    print(json.dumps({f'{k}|{s}': v for (k, s), v in out.items()}, indent=1))


if __name__ == '__main__':
    main()

"""CPU analysis of causal-diagnostic session outputs (LOCALIZATION only for session 1).

Implements the predeclared rules in paper/CAUSAL_DIAGNOSTIC_9504111.md section 6:
R_G = share of candidate answer changes that revert to the base answer under REMOVAL(G);
I_G = share of candidate answer changes reproduced under INSERTION(G); S_G = (R_G + I_G)/2,
computed separately for repairs and regressions (both phases pooled). Answer-score movement uses the fixed contrast
c = score(original candidate answer) - score(original base answer) at the first answer token:
m = (c_hybrid - c_base) / (c_candidate - c_base)   (0 = base-like, 1 = candidate-like).
"""
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from causal_diag_9504111 import GROUPS, LETTERS, contrast, margin  # noqa: E402

OUT = Path('results/paper-analysis/causal-diagnostic')
CELLS = ('repair', 'regression', 'both_wrong_diff', 'both_correct', 'both_wrong_same')


def load(d, name):
    return {x['uid']: x for x in json.loads(gzip.decompress((d / f'{name}.json.gz').read_bytes()))}


def per_example(d, labels, uids, hybrids):
    B, C = load(d, 'base_full'), load(d, 'candidate_full')
    rows = []
    for h in hybrids:
        H = load(d, h)
        for u in uids:
            lab = labels[u]; b, c = lab['base'], lab['cand']
            cb, cc, ch = contrast(B[u]['letter_logprobs'], c, b), contrast(C[u]['letter_logprobs'], c, b), contrast(H[u]['letter_logprobs'], c, b)
            m = (ch - cb) / (cc - cb) if None not in (cb, cc, ch) and abs(cc - cb) > 1e-6 else None
            rows.append({'hybrid': h, 'uid': u, 'phase': lab['phase'], 'subtask': lab['subtask'], 'transition': lab['transition'], 'gold': lab['gold'],
                         'base_ans': b, 'cand_ans': c, 'hyb_ans': H[u]['parsed'], 'eq_base': H[u]['parsed'] == b, 'eq_cand': H[u]['parsed'] == c,
                         'invalid': H[u]['parsed'] == '', 'correct': H[u]['parsed'] == lab['gold'], 'c_base': cb, 'c_cand': cc, 'c_hyb': ch, 'm': m,
                         'margin_hyb': margin(H[u]['letter_logprobs'], lab['gold'])})
    return pd.DataFrame(rows)


def summarize(df):
    out = {}
    for h, g in df.groupby('hybrid'):
        s = {}
        for (ph, t), x in g.groupby(['phase', 'transition']):
            s[f'{ph}|{t}'] = {'n': len(x), 'eq_cand': int(x.eq_cand.sum()), 'eq_base': int(x.eq_base.sum()), 'other': int((~x.eq_cand & ~x.eq_base).sum()),
                              'invalid': int(x.invalid.sum()), 'correct': int(x.correct.sum()),
                              'mean_m': float(x.m.dropna().mean()) if x.m.notna().any() else None}
        out[h] = s
    return out


def selection(df):
    """Predeclared S_G for repairs and regressions (both phases pooled)."""
    res = {}
    for g in GROUPS:
        rem, ins = df[df.hybrid == f'removal_{g}_loc'], df[df.hybrid == f'insertion_{g}_loc']
        r = {}
        for cls in ('repair', 'regression', 'both_wrong_diff'):
            a, b = rem[rem.transition == cls], ins[ins.transition == cls]
            R, I = float(a.eq_base.mean()), float(b.eq_cand.mean())
            r[cls] = {'n': len(a), 'R_removal_reverts': R, 'I_insertion_reproduces': I, 'S': (R + I) / 2,
                      'removal_mean_m': float(a.m.dropna().mean()), 'insertion_mean_m': float(b.m.dropna().mean())}
        ctrl_r = rem[rem.transition.isin(['both_correct', 'both_wrong_same'])]; ctrl_i = ins[ins.transition.isin(['both_correct', 'both_wrong_same'])]
        r['controls_changed_answer'] = {'removal': int((~ctrl_r.eq_base).sum()), 'insertion': int((~ctrl_i.eq_base).sum()), 'n': len(ctrl_r)}
        res[g] = r
    pick = {}
    for cls in ('repair', 'regression'):
        best = max(GROUPS, key=lambda g: (round(res[g][cls]['S'], 9), -GROUP_SIZE.get(g, 0)))
        pick[cls] = {'group': best, 'S': res[best][cls]['S'], 'meets_0.5': res[best][cls]['S'] >= .5}
    return res, pick


GROUP_SIZE = {}


def letter_offsets(d, labels, uids):
    """Mean centered candidate-minus-base A-D log-prob offsets on the given examples (all four letters present)."""
    B, C = load(d, 'base_full'), load(d, 'candidate_full')
    diffs = []
    for u in uids:
        b, c = B[u]['letter_logprobs'], C[u]['letter_logprobs']
        if all(b[L] is not None and c[L] is not None for L in LETTERS):
            v = np.array([c[L] - b[L] for L in LETTERS]); diffs.append(v - v.mean())
    D = np.array(diffs)
    return {'n': len(D), 'mean_offset': dict(zip(LETTERS, D.mean(0).round(4).tolist())), 'sd': dict(zip(LETTERS, D.std(0, ddof=1).round(4).tolist()))}


def offset_model_accuracy(d, labels, uids, offsets):
    """Predict the candidate answer as argmax(base letter scores + frozen offsets); share of answer CHANGES predicted."""
    B = load(d, 'base_full'); hit = n = 0
    for u in uids:
        lab = labels[u]
        if lab['base'] == lab['cand']:
            continue
        b = B[u]['letter_logprobs']
        if any(b[L] is None for L in LETTERS):
            continue
        pred = max(LETTERS, key=lambda L: b[L] + offsets[L])
        n += 1; hit += pred == lab['cand']
    return {'changed_examples': n, 'predicted_candidate_answer': hit, 'share': hit / n if n else None}


def base_closeness(d, labels):
    """Were changed answers close contests under the base? Base top-2 gap for changed vs unchanged (all 761)."""
    B = load(d, 'base_full'); rows = []
    for u, x in B.items():
        s = [v for v in x['letter_logprobs'].values() if v is not None]
        gap = sorted(s)[-1] - sorted(s)[-2] if len(s) >= 2 else None
        rows.append({'uid': u, 'phase': labels[u]['phase'], 'transition': labels[u]['transition'], 'gap': gap})
    r = pd.DataFrame(rows)
    return {f'{ph}|{t}': {'n': len(g), 'median_base_top2_gap': float(g.gap.median()), 'q25': float(g.gap.quantile(.25)), 'q75': float(g.gap.quantile(.75))}
            for (ph, t), g in r.groupby(['phase', 'transition'])}


def main(session_dir):
    d = Path(session_dir)
    man = json.loads((OUT / 'example_manifest.json').read_text())
    labels, loc = man['labels'], man['localization']
    norms = json.loads((d / 'group_norms.json').read_text())
    GROUP_SIZE.update({g: v['params'] for g, v in norms.items()})
    hybrids = ['control_noop_candidate_copy_loc', 'control_reset_base_loc'] + [f'{k}_{g}_loc' for g in GROUPS for k in ('removal', 'insertion')]
    df = per_example(d, labels, loc, hybrids)
    df.to_csv(OUT / 'session1_localization_per_example.csv', index=False)
    sel, pick = selection(df)
    offs = letter_offsets(d, labels, loc)
    R = {'summary_by_cell': summarize(df), 'selection': sel, 'predeclared_pick': pick, 'group_norms': norms,
         'letter_offsets_localization': offs, 'offset_model_localization': offset_model_accuracy(d, labels, loc, offs['mean_offset']),
         'base_closeness_all761': base_closeness(d, labels)}
    (OUT / 'session1_analysis.json').write_text(json.dumps(R, indent=1))
    return R


if __name__ == '__main__':
    R = main(sys.argv[1] if len(sys.argv) > 1 else OUT / 'session1')
    for g, v in R['selection'].items():
        print(f"{g:16s} " + ' | '.join(f"{c}: R {v[c]['R_removal_reverts']:.2f} I {v[c]['I_insertion_reproduces']:.2f} S {v[c]['S']:.2f} (n={v[c]['n']})"
                                       for c in ('repair', 'regression', 'both_wrong_diff')) + f" | ctrl changed {v['controls_changed_answer']}")
    print('PICK', R['predeclared_pick']); print('OFFSETS', R['letter_offsets_localization']); print('OFFSET MODEL', R['offset_model_localization'])


def figures():
    """Fig 1: per-group insertion/removal effects on the fixed contrast, by phase x transition.
    Fig 2: mechanically selected traces: in each changed cell, the LOCALIZATION example with the median base top-2 gap
    (ties: lowest uid); bars show the fixed contrast under base, candidate and each single-group insertion."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sys.path.insert(0, 'paper/figures/scripts')
    from common import ACCENT, BASE_C, CAND, GOOD, MUTED  # noqa: F401  (applies the paper rcParams)
    figd = Path('paper/figures/causal-diagnostic'); figd.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(OUT / 'session1_localization_per_example.csv')
    man = json.loads((OUT / 'example_manifest.json').read_text())
    A = json.loads((OUT / 'session1_analysis.json').read_text())
    cells = [('RERANK', 'repair'), ('RERANK', 'regression'), ('TEST', 'repair'), ('TEST', 'regression')]
    cols = ['#5a9be0', '#9cc2ee', CAND, ACCENT]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3), sharey=True)
    x = np.arange(len(GROUPS)); w = .2
    for ax, kind, title in ((axes[0], 'insertion', 'A  INSERTION (base + candidate values in G)\n   share of candidate contrast shift reproduced'),
                            (axes[1], 'removal', 'B  REMOVAL (candidate + base values in G)\n   share of candidate contrast shift removed')):
        for i, ((ph, t), col) in enumerate(zip(cells, cols)):
            vals = []
            for g in GROUPS:
                s = df[(df.hybrid == f'{kind}_{g}_loc') & (df.phase == ph) & (df.transition == t)].m
                vals.append(s.mean() if kind == 'insertion' else (1 - s).mean())
            n = int(((df.hybrid == f'{kind}_{GROUPS[0]}_loc') & (df.phase == ph) & (df.transition == t)).sum())
            ax.bar(x + (i - 1.5) * w, vals, w, color=col, label=f'{ph} {t}s (n={n})', zorder=3)
        ax.axhline(0, color=MUTED, lw=.8)
        ax.set_xticks(x, [f"{g.replace('final_norm_head', 'norm+head')}\n{A['group_norms'][g]['params'] / 1e9:.2f}B" for g in GROUPS], fontsize=6.3)
        ax.set_title(title, loc='left', fontsize=7)
    axes[0].set_ylabel('Mean share of the fixed contrast shift'); axes[0].legend(frameon=False, fontsize=6, loc='lower left')
    fig.tight_layout(); fig.savefig(figd / 'fig_group_effects.pdf', metadata={'CreationDate': None}); fig.savefig(figd / 'fig_group_effects.png', dpi=300); plt.close(fig)

    d = Path(OUT / 'session1'); B, C = load(d, 'base_full'), load(d, 'candidate_full')
    picks = []
    for ph, t in cells:
        us = sorted(u for u in man['localization'] if man['labels'][u]['phase'] == ph and man['labels'][u]['transition'] == t)
        gaps = []
        for u in us:
            s = sorted(v for v in B[u]['letter_logprobs'].values() if v is not None)
            gaps.append((s[-1] - s[-2], u))
        gaps.sort(); picks.append((ph, t, gaps[(len(gaps) - 1) // 2][1]))
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.6), sharey=False)
    for ax, (ph, t, u) in zip(axes, picks):
        lab = man['labels'][u]; sub = df[df.uid == u].set_index('hybrid')
        names = ['base', 'candidate'] + [f'+{g}' for g in GROUPS]
        vals = [sub.c_base.iloc[0], sub.c_cand.iloc[0]] + [sub.loc[f'insertion_{g}_loc', 'c_hyb'] for g in GROUPS]
        answers = [lab['base'], lab['cand']] + [sub.loc[f'insertion_{g}_loc', 'hyb_ans'] for g in GROUPS]
        colors = [BASE_C, ACCENT] + [CAND] * len(GROUPS)
        ax.barh(range(len(vals))[::-1], vals, color=colors, zorder=3)
        for k, (v, a_) in enumerate(zip(vals, answers)):
            ax.text(v, len(vals) - 1 - k, f' {a_}', va='center', ha='left' if v >= 0 else 'right', fontsize=5.6)
        ax.axvline(0, color=MUTED, lw=.8)
        ax.set_yticks(range(len(vals))[::-1], [n.replace('final_norm_head', 'norm+head') for n in names], fontsize=5.8)
        ax.set_title(f"{ph} {t}\n{u} ({lab['subtask'][:4]}), gold {lab['gold']}\nbase {lab['base']} → cand {lab['cand']}", loc='left', fontsize=6)
        ax.set_xlabel(f"log p({lab['cand']}) − log p({lab['base']})", fontsize=6)
    fig.tight_layout(); fig.savefig(figd / 'fig_example_traces.pdf', metadata={'CreationDate': None}); fig.savefig(figd / 'fig_example_traces.png', dpi=300); plt.close(fig)
    return picks

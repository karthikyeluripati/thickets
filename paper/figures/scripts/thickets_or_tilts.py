"""Figures and tables for "Thickets or Tilts?" (paper/WRITING_PROMPT.md). CPU only; every number comes from committed
result files (paper/RESULTS_MASTER.md). Output: paper/figures/thickets-or-tilts/{fig*.pdf,png,svg, table*.md, table*.tex}.
Run from the repo root:  python paper/figures/scripts/thickets_or_tilts.py
"""
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PA = Path('results/paper-analysis')
OUT = Path('paper/figures/thickets-or-tilts'); OUT.mkdir(parents=True, exist_ok=True)
RANDOPT_SRC = Path(os.environ.get('RANDOPT_DIR', 'third_party/RandOpt'))  # RandOpt @ 4000d34 (see README)

# Okabe-Ito (colour-blind safe); one colour per method, fixed across all figures
C = {'base': '#7F7F7F', 'randopt': '#E69F00', 'sc': '#0072B2', 'control': '#BDBDBD', 'ttmv': '#CC79A7',
     'pert': '#D55E00', 'green': '#009E73', 'sky': '#56B4E9', 'ink': '#222222', 'muted': '#666666'}
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'legend.fontsize': 8, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': '#E6E6E6', 'grid.linewidth': 0.6, 'axes.axisbelow': True, 'axes.edgecolor': '#888888',
                     'savefig.dpi': 300, 'savefig.bbox': 'tight', 'font.family': 'DejaVu Sans'})


def ld(f):
    return json.loads(gzip.decompress(Path(f).read_bytes()))


def save(fig, name):
    for ext in ('pdf', 'png', 'svg'):
        fig.savefig(OUT / f'{name}.{ext}')
    plt.close(fig)


def label(ax, s):
    ax.text(-0.13, 1.06, s, transform=ax.transAxes, fontsize=11, fontweight='bold', va='bottom')


def rline(ax, x, y, color, text_xy=(0.04, 0.93)):
    x, y = np.asarray(x, float), np.asarray(y, float); r = np.corrcoef(x, y)[0, 1]; b = np.polyfit(x, y, 1)
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max()); pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color='#AAAAAA', lw=0.8, ls='--', zorder=1)
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
    ax.text(*text_xy, f'r = {r:.3f}\nslope = {b[0]:.2f}', transform=ax.transAxes, va='top', fontsize=8, color=C['ink'],
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#DDDDDD'))
    return r


P2 = json.loads((PA / 'p2/p2_results.json').read_text()); P3 = json.loads((PA / 'p3/p3_results.json').read_text())
ROWS = {'GSM8K\nQwen2.5-1.5B': P2['gsm8k_q15'], 'GSM8K\nQwen2.5-3B': P3['gsm8k_q3'], 'GQA\nQwen2.5-VL-3B': P2['gqa_vl3']}


# ---------------------------------------------------------------- Figure 1b: SC vs RandOpt (teaser bars)
def fig1():
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    names = list(ROWS); x = np.arange(len(names)); w = 0.26
    for k, (name, v) in enumerate(ROWS.items()):
        comparable = v['gate_pass']; hatch = None if comparable else '///'
        sc = v['SC_T0.7']['curve']['50']; ci = v['SC_T0.7']['sc50_ci95']
        bars = [(v['paper']['base'], C['base'], 'Base (paper)'), (v['paper']['randopt'], C['randopt'], 'RandOpt, K=50 (paper)'),
                (sc, C['sc'], 'Self-consistency@50 (ours, no weight search)')]
        for j, (val, col, lab) in enumerate(bars):
            ax.bar(x[k] + (j - 1) * w, val, w * 0.92, color=col, hatch=hatch, edgecolor='white' if comparable else '#555555',
                   linewidth=0.6, label=lab if k == 0 else None)
            top = ci[1] if j == 2 else val
            ax.text(x[k] + (j - 1) * w, top + 0.8, f'{val:.1f}', ha='center', va='bottom', fontsize=7, color=C['ink'])
        ax.errorbar(x[k] + w, sc, yerr=[[sc - ci[0]], [ci[1] - sc]], fmt='none', ecolor=C['ink'], elinewidth=0.9, capsize=2)
    ax.set_xticks(x); ax.set_xticklabels([n + ('' if ROWS[n]['gate_pass'] else '\n(not comparable*)') for n in names])
    ax.set_ylabel('Accuracy (%)'); ax.set_ylim(40, 95)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01), frameon=False, ncol=3, fontsize=7)
    ax.text(1.0, -0.36, '*our greedy base differs from the paper\'s by more than the pre-registered gate (GQA: 54.0 vs 56.6)',
            transform=ax.transAxes, ha='right', fontsize=6.5, color=C['muted'])
    save(fig, 'fig1b_sc_vs_randopt')


# ---------------------------------------------------------------- Figure 2: SC@K curves with paper reference lines
def fig2():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=False)
    for ax, (name, v), lab in zip(axes, ROWS.items(), 'abc'):
        Ks = [1, 5, 10, 20, 50]; y = [v['SC_T0.7']['curve'][str(K)] for K in Ks]
        ax.plot(Ks, y, '-o', color=C['sc'], ms=3.5, lw=2, label='Self-consistency (ours)', zorder=3)
        ax.axhline(v['paper']['randopt'], color=C['randopt'], lw=2, label='RandOpt K=50 (paper)')
        ax.axhline(v['paper']['base'], color=C['base'], lw=1.5, ls=':', label='Base (paper)')
        if v['paper'].get('ttmv') is not None:
            ax.axhline(v['paper']['ttmv'], color=C['ttmv'], lw=1.5, ls='--', label='TT-MV (paper)')
        ax.set_xscale('log'); ax.set_xticks(Ks); ax.set_xticklabels(Ks); ax.set_xlabel('K (samples voted)')
        ax.set_title(name.replace('\n', ', ').replace('Qwen2.5-', '') + ('' if v['gate_pass'] else '*'), fontsize=8.5, pad=8)
        label(ax, lab)
    axes[0].set_ylabel('Accuracy (%)')
    h_, l_ = axes[0].get_legend_handles_labels(); fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.legend(h_, l_, loc='lower center', bbox_to_anchor=(0.5, 0.04), ncol=4, frameon=False, fontsize=7)
    fig.text(0.5, 0.0, '*not comparable: our greedy base differs from the paper\'s by more than the pre-registered gate', ha='center', fontsize=6, color=C['muted'])
    save(fig, 'fig2_sc_at_k')


# ---------------------------------------------------------------- Figure 3: the first-order tilt law
def fig3():
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0)); axes = axes.ravel()
    s12 = json.loads((PA / 'stage12/stage12_per_perturbation.json').read_text())
    r1 = json.loads((PA / 'r1/r1_per_perturbation.json').read_text())
    r2b = json.loads((PA / 'r2b/r2b_per_perturbation.json').read_text())
    for ax, rows, title, lab in ((axes[0], s12, 'Qwen3-VL-8B, "front"\n(Perspective; Stage 2)', 'a'),
                                 (axes[1], r1, 'Qwen3-VL-8B, "left"\n(Complex Logic; R1)', 'b'),
                                 (axes[2], r2b, 'Qwen2.5-VL-7B, "front"\n(Perspective; R2b)', 'c')):
        x = [z['pred_NOV'] for z in rows]; y = [z['T_NOV'] for z in rows]
        ax.scatter(x, y, s=12, color=C['pert'], alpha=0.8, edgecolor='white', linewidth=0.4, zorder=3)
        rline(ax, x, y, C['pert']); ax.set_title(title, fontsize=8); ax.set_xlabel('predicted tilt (nats)'); label(ax, lab)
    axes[0].set_ylabel('measured tilt (nats)')
    # inset: per-item r from I5
    pred_i = np.load(PA / 'i5/pod/pred_items.npy'); meta = json.loads((PA / 'i5/pod/meta.json').read_text())
    m = json.loads((PA / 'stage12/pod/measurements.json').read_text()); base = m['base']; order = {c: k for k, c in enumerate(meta['candidate_ids'])}
    meas = np.zeros_like(pred_i)
    for rec in m['perturbations']:
        meas[:, order[rec['candidate_id']]] = [x_['B'] - b['B'] for x_, b in zip(rec['NOV'], base)]
    per = np.array([np.corrcoef(pred_i[j], meas[j])[0, 1] for j in range(len(pred_i))])
    ins = axes[0].inset_axes([0.66, 0.17, 0.31, 0.2]); ins.hist(per, bins=np.linspace(0.4, 1, 13), color=C['sky'], edgecolor='white')
    ins.set_xlabel(f'per-item r (median {np.median(per):.2f})', fontsize=5, labelpad=1); ins.set_yticks([]); ins.tick_params(labelsize=5); ins.grid(False)
    # P0: GQA per-question contrast, coloured by sigma
    # P0 (GQA) and the non-Qwen OLMo panels (S1-A, S1-7B): per-question contrast, coloured by sigma
    cols = {0.001: C['green'], 0.002: C['sc'], 0.005: C['control']}
    s1sig = np.array([c['sigma'] for c in json.loads((PA / 's1/frozen_A.json').read_text())['candidates']])
    panels = [(axes[3], np.load(PA / 'p0/pod/pred.npy'), np.load(PA / 'p0/pod/meas.npy'),
               np.array([c['sigma'] for c in json.loads((PA / 'p0/candidates.json').read_text())['candidates']]),
               'Qwen2.5-VL-3B, GQA (P0)\nLM-only, answer contrast', 'd', 6.0),
              (axes[4], np.load(PA / 's1/pod/out/A/A_pred.npy'), np.load(PA / 's1/pod/out/A/A_meas.npy'), s1sig,
               'OLMo-2-1B, ARC (S1-A)\nnon-Qwen, text-only', 'e', None),
              (axes[5], np.load(PA / 's1-7b/pod/s17b/A/A_pred.npy'), np.load(PA / 's1-7b/pod/s17b/A/A_meas.npy'), s1sig,
               'OLMo-2-7B, ARC (S1-7B)\nnon-Qwen, text-only', 'f', None)]
    for ax, pred, mm, sig, title, lab, lim in panels:
        sig = sig[: pred.shape[1]]
        for s in (0.005, 0.002, 0.001):
            k = sig == s; r = np.corrcoef(pred[:, k].ravel(), mm[:, k].ravel())[0, 1]
            ax.scatter(pred[:, k].ravel(), mm[:, k].ravel(), s=3, alpha=0.35 if s != 0.005 else 0.25, color=cols[s],
                       label=f'σ={s}: r={r:.2f}', zorder=3 if s != 0.005 else 2)
        if lim is None:  # axes from the sigma <= 0.002 bulk; sigma = 0.005 points beyond it are clipped
            v = np.abs(np.concatenate([pred[:, sig <= 0.002].ravel(), mm[:, sig <= 0.002].ravel()])); lim = float(np.percentile(v, 99.5)) * 1.6
        ax.plot([-lim, lim], [-lim, lim], color='#AAAAAA', lw=0.8, ls='--'); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_title(title, fontsize=8); ax.set_xlabel('predicted Δ (nats)')
        ax.legend(loc='upper left', frameon=False, fontsize=6, markerscale=3); label(ax, lab)
    axes[3].set_ylabel('measured Δ (nats)')
    fig.tight_layout(); save(fig, 'fig3_tilt_law')
    return per


# ---------------------------------------------------------------- Figure 4: localization
def fig4():
    S3 = json.loads((PA / 'stage3/stage3_results.json').read_text())
    groups = ['embed', 'L00-05', 'L06-11', 'L12-17', 'L18-23', 'L24-29', 'L30-35', 'final_norm_head']
    labels = ['embed', '0–5', '6–11', '12–17', '18–23', '24–29', '30–35', 'norm+\nhead']
    pr = [S3['tauP96_group_var_share'][g] for g in groups]; me = [S3['3B3_measured_var_share'][g] for g in groups]
    fig, (ax, ax3, ax2) = plt.subplots(1, 3, figsize=(9.6, 2.7), gridspec_kw={'width_ratios': [2.0, 1.05, 1.0]})
    x = np.arange(len(groups)); w = 0.38
    ax.bar(x - w / 2, pr, w * 0.92, color=C['sc'], label='predicted from noise')
    ax.bar(x + w / 2, me, w * 0.92, color=C['pert'], label='measured (block insertions)')
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_xlabel('Qwen3-VL-8B language layers (block)'); ax.set_ylabel('share of tilt variance')
    ax.set_ylim(0, 0.5); ax.legend(frameon=False, loc='upper right', fontsize=7, bbox_to_anchor=(1.0, 1.0)); label(ax, 'a')
    # inset: block partial vs measured insertion
    meta = json.loads((PA / 'stage3/pod/meta.json').read_text()); G = meta['groups']; idx = {c[0]: k for k, c in enumerate(meta['candidates'])}
    P = np.load(PA / 'stage3/pod/proj_P96.npy'); ins3 = json.loads((PA / 'stage3/pod/insertions_3B.json').read_text())
    xs, ys = [], []
    for rec in ins3['records']:
        for g, t in rec['insertion_T'].items(): xs.append(P[idx[rec['candidate_id']], G.index(g)]); ys.append(t)
    ax3.scatter(xs, ys, s=8, color=C['pert'], alpha=0.7, edgecolor='white', linewidth=0.3, zorder=3)
    rline(ax3, xs, ys, C['pert']); ax3.set_xlabel('predicted block tilt (nats)'); ax3.set_ylabel('measured insertion tilt (nats)')
    ax3.set_title('20 perturbations x 8 blocks', fontsize=8); label(ax3, 'b')
    # relative-depth replication
    R1 = json.loads((PA / 'r1/r1_results.json').read_text()); R2b = json.loads((PA / 'r2b/r2b_results.json').read_text())
    vals = [('8B\n"front"\n(measured)', S3['3B3_measured_var_share']['L06-11'] + S3['3B3_measured_var_share']['L12-17'] + S3['3B3_measured_var_share']['L18-23'], 'measured'),
            ('8B\n"left"\n(R1)', R1['L1_mid_layers_6_23_share'], 'predicted'),
            ('7B\n"front"\n(R2b)', R2b['reported']['mid_blocks_share_80'], 'predicted')]
    ax2.bar(range(3), [v[1] for v in vals], color=[C['pert'], C['sc'], C['sc']], width=0.6)
    for i, v in enumerate(vals): ax2.text(i, v[1] + 0.02, f'{v[1]:.2f}', ha='center', fontsize=7)
    ax2.set_xticks(range(3)); ax2.set_xticklabels([v[0] for v in vals], fontsize=6.5); ax2.set_ylim(0, 1)
    ax2.set_ylabel('share in middle layers\n(rel. depth ≈ 0.15–0.7)'); label(ax2, 'c')
    fig.tight_layout(); save(fig, 'fig4_localization')


# ---------------------------------------------------------------- Figure 5: selection and transfer (OmniSpatial)
def fig5():
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.7), gridspec_kw={'width_ratios': [1.0, 1.0, 1.25, 1.2]})
    ap = json.loads((PA / 'answer-prior/answer_prior_results.json').read_text())['composition']
    ax = axes[0]
    comp = [('SEARCH', ap['SEARCH']['format'].get('compass', 0), 200), ('RERANK', ap['RERANK']['format'].get('compass', 0), 200),
            ('official\nTEST', ap['TEST']['format'].get('compass', 0), 561)]
    ax.bar(range(3), [100 * c / n for _, c, n in comp], color=[C['sc'], C['sc'], C['pert']], width=0.6)
    for i, (_, c, n) in enumerate(comp): ax.text(i, 100 * c / n + 2, f'{c}/{n}', ha='center', fontsize=7)
    ax.set_xticks(range(3)); ax.set_xticklabels([c[0] for c in comp]); ax.set_ylabel('% compass-format items'); ax.set_ylim(0, 110)
    ax.set_title('selection and test sets\nuse different formats', fontsize=8); label(ax, 'a')
    ax = axes[1]
    vals = [('RERANK\n(selected)', 8.0, None), ('official\nTEST', -2.67, None), ('fresh,\nsame format', 2.67, (0.17, 4.93))]
    for i, (n, v, ci) in enumerate(vals):
        ax.bar(i, v, color=[C['randopt'], C['control'], C['green']][i], width=0.6)
        if ci: ax.errorbar(i, v, yerr=[[v - ci[0]], [ci[1] - v]], fmt='none', ecolor=C['ink'], capsize=2, elinewidth=0.9)
        ax.text(i + (0.33 if ci else 0), v + (0.3 if v >= 0 else -0.9), f'{v:+.1f}', ha='left' if ci else 'center', fontsize=7)
    ax.axhline(0, color='#888888', lw=0.8); ax.set_xticks(range(3)); ax.set_xticklabels([v[0] for v in vals], fontsize=6.5); ax.set_ylim(-4, 9.5)
    ax.set_ylabel('winner gain (pp)'); ax.set_title('selected winner 9504111', fontsize=8); label(ax, 'b')
    ax = axes[2]
    ens = [('winner', 2.7, 0.3, 4.9), ('top-10\nvote', 2.8, 1.0, 4.7), ('47-model\nvote', 0.2, -1.0, 1.3), ('controls\nvote', -0.3, -2.2, 1.5)]
    cols = [C['green'], C['sc'], C['randopt'], C['control']]
    for i, (n, v, lo, hi) in enumerate(ens):
        ax.bar(i, v, color=cols[i], width=0.6); ax.errorbar(i, v, yerr=[[v - lo], [hi - v]], fmt='none', ecolor=C['ink'], capsize=2, elinewidth=0.9)
    ax.axhline(0, color='#888888', lw=0.8); ax.set_xticks(range(4)); ax.set_xticklabels([e[0] for e in ens], fontsize=6.5)
    ax.set_ylabel('gain on fresh matched items (pp)'); ax.set_title('little ensemble gain at K≈50\n(exploratory)', fontsize=8); label(ax, 'c')
    ax = axes[3]
    S = pd.read_csv(PA / 'answer-prior/candidate_phase_answer_prior.csv'); S = S[S.phase == 'SEARCH']
    hb = ax.hexbin(S.shift_front, S.gain, gridsize=22, cmap='Blues', mincnt=1, linewidths=0.2)
    w = S[S.is_winner]; ax.scatter(w.shift_front, w.gain, s=55, marker='*', color=C['randopt'], edgecolor=C['ink'], linewidth=0.5, zorder=4)
    ax.annotate('winner', (float(w.shift_front.iloc[0]), float(w.gain.iloc[0])), xytext=(5.5, -7.0), textcoords='data', fontsize=7,
                arrowprops=dict(arrowstyle='-', color=C['ink'], lw=0.6))
    r = np.corrcoef(S.shift_front, S.gain)[0, 1]
    ax.text(0.04, 0.95, f'5000 candidates\nr = {r:.2f}', transform=ax.transAxes, va='top', fontsize=7, bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#DDDDDD'))
    ax.set_xlabel('shift in "front" answers (pp)'); ax.set_ylabel('SEARCH accuracy gain (pp)')
    ax.set_title('gains co-vary with a label-prior tilt\n(exploratory)', fontsize=8); label(ax, 'd')
    fig.tight_layout(); save(fig, 'fig5_selection_transfer')


# ---------------------------------------------------------------- Figure 6: CoT regime (GQA)
def fig6():
    sys.path.insert(0, str(RANDOPT_SRC)); from data_handlers.gqa import GQAHandler  # noqa
    h = GQAHandler(); norm = lambda s: h._normalize_answer(str(s))
    D = PA / 'p1'; it = json.loads((D / 'items.json').read_text(encoding='utf-8')); ALL = it['selection'] + it['heldout']
    base = ld(D / 'pod/eval/base_p1.json.gz'); bans = {r['id']: norm(base[r['id']]['pred']) for r in ALL}
    P = [ld(f)['res'] for f in sorted((D / 'pod/eval/cands').glob('cand_*.json.gz'))]
    S = ld(D / 'pod/samples_T0.3.json.gz')
    fP = np.array([np.mean([norm(p[r['id']]['pred']) != bans[r['id']] for p in P]) for r in ALL])
    fT = np.array([np.mean([norm(x['pred']) != bans[r['id']] for x in S[r['id']]]) for r in ALL])
    rho = pd.Series(fP).corr(pd.Series(fT), method='spearman')
    R = json.loads((D / 'p1_results.json').read_text())
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.4, 2.8), gridspec_kw={'width_ratios': [1, 1.3]})
    ax.scatter(fT, fP, s=8, color=C['pert'], alpha=0.55, edgecolor='none')
    ax.plot([0, 1], [0, 1], color='#AAAAAA', lw=0.8, ls='--'); ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel('per-question flip rate, temperature sampling (T=0.3)'); ax.set_ylabel('per-question flip rate,\n64 weight perturbations')
    ax.set_title('similarly susceptible questions', fontsize=8)
    ax.text(0.04, 0.95, f'400 GQA questions\nSpearman ρ = {rho:.2f}', transform=ax.transAxes, va='top', fontsize=7, bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#DDDDDD'))
    label(ax, 'a')
    k8 = R['P1_C']['K8']; bars = [('base\ngreedy', R['base_acc']['H'], C['base']), ('top-8\nselected', k8['randopt_topK_vote'], C['randopt']),
                                  ('8 random\nperturbed', k8['random_K_perturbation_vote'], C['control']),
                                  ('SC@8\nT=0.3', k8['sc_vote_Tstar'], C['sky']), ('SC@8\nT=0.7', k8['sc_vote_T0.7'], C['sc'])]
    for i, (n, v, c) in enumerate(bars):
        ax2.bar(i, v, color=c, width=0.62); ax2.text(i, v + 0.4, f'{v:.1f}', ha='center', fontsize=7)
    ax2.set_xticks(range(len(bars))); ax2.set_xticklabels([b[0] for b in bars], fontsize=6.5); ax2.set_ylim(50, 66)
    ax2.set_ylabel('held-out accuracy (%)'); ax2.set_title('vote of perturbed models vs vote of samples\n(200 held-out questions; underpowered)', fontsize=8); label(ax2, 'b')
    fig.tight_layout(); save(fig, 'fig6_cot_regime')
    return rho


# ---------------------------------------------------------------- Tables
def tables():
    rows = [('GSM8K', 'Qwen2.5-1.5B', P2['gsm8k_q15']), ('GSM8K', 'Qwen2.5-3B', P3['gsm8k_q3']), ('GSM8K', 'Qwen2.5-0.5B', P2['gsm8k_q05']),
            ('MATH-500', 'Qwen2.5-1.5B', P3['math500_q15']), ('MATH-500', 'Qwen2.5-3B', P3['math500_q3']), ('GQA (2000)', 'Qwen2.5-VL-3B', P2['gqa_vl3'])]
    md = ['| Task | Model | Base ours / paper | Gate | SC@50 (T=0.7) [95% CI] | TT-MV (paper) | RandOpt (paper) | Verdict |', '|---|---|---|---|---|---|---|---|']
    tex = [r'\begin{tabular}{llcccccc}', r'\toprule', r'Task & Model & Base (ours / paper) & Gate & SC@50 & TT-MV$^\dagger$ & RandOpt$^\dagger$ & Verdict \\', r'\midrule']
    for t, mname, v in rows:
        sc = v['SC_T0.7']['curve']['50']; ci = v['SC_T0.7']['sc50_ci95']; tt = v['paper']['ttmv']; tts = f'{tt:.1f}' if tt is not None else '--'
        verdict = v['decision_vs_randopt'].replace(' (gate)', '').lower() if not v['gate_pass'] else v['decision_vs_randopt'].lower()
        md.append(f"| {t} | {mname} | {v['greedy']:.1f} / {v['paper']['base']:.1f} | {'pass' if v['gate_pass'] else 'fail'} | {sc:.1f} [{ci[0]:.1f}, {ci[1]:.1f}] | {tts} | {v['paper']['randopt']:.1f} | {verdict} |")
        tex.append(f"{t} & {mname} & {v['greedy']:.1f} / {v['paper']['base']:.1f} & {'pass' if v['gate_pass'] else 'fail'} & {sc:.1f} [{ci[0]:.1f}, {ci[1]:.1f}] & {tts} & {v['paper']['randopt']:.1f} & {verdict} \\\\")
    tex += [r'\bottomrule', r'\end{tabular}']
    (OUT / 'table1_sc_vs_randopt.md').write_text('\n'.join(md) + '\n\n$\\dagger$ reported by Gan & Isola (2026), Table 1 and Table 4.\n', encoding='utf-8')
    (OUT / 'table1_sc_vs_randopt.tex').write_text('\n'.join(tex) + '\n', encoding='utf-8')
    ledger = [
        ('GPU-A H*, F1 (whole model incl. vision)', '79252d4', 'pooled r(pred, measured contrast)', '≥0.6 support / <0.3 falsify', '0.137', 'falsified'),
        ('GPU-A H*, S2d (group partials)', '79252d4', 'pooled r', '<0.5 falsify', '0.239', 'falsified'),
        ('Stage 2: tilt law', '09c1f1f', 'r(pred, measured), 80 pert.', '≥0.6 pass', '0.938', 'pass'),
        ('Stage 3A-1: τ vs SEARCH gain', 'ce60c5d', 'r over 5000', '≥0.25 pass / <0.10 falsify', '0.240', 'inconclusive'),
        ('Stage 3A-2: τ vs RERANK gain', 'ce60c5d', 'r over 543', '≥0.25 pass', '0.402', 'pass'),
        ('Stage 3B: block localization', 'ce60c5d', 'pooled r, 160 pairs', '≥0.6 pass', '0.978', 'pass'),
        ('R1: new task / content word', '00155cc', 'r(pred, measured)', '≥0.6 pass', '0.915', 'replicated'),
        ('R2: second model (96 items)', '47d2102', 'r; reliability gate', 'reliability ≥0.7', '0.920; rel. 0.52', 'inconclusive'),
        ('R2b: second model (363 items)', '1cb3252', 'r; reliability gate', 'same', '0.925; rel. 0.95', 'replicated'),
        ('I5: per-item law', 'a1c2a7a', 'pooled r, 7680 pairs', '≥0.6 pass', '0.771', 'pass'),
        ('M1-1: winner on fresh matched items', '5eb864b', 'gain, cluster CI', 'CI > 0', '+2.67 [0.17, 4.93]', 'transfers (not full)'),
        ('M1-2: front concentration', '5eb864b', 'front − non-front gain', 'CI > 0', '+1.95 [−2.57, 6.47]', 'not supported'),
        ('M1-3: τ vs fresh gain', '5eb864b', 'r over 59', '≥0.25', '0.61', 'pass'),
        ('M1-4: top-50 vs controls', '5eb864b', 'mean gain difference', 'reported', '+1.16 [0.31, 2.03]', '–'),
        ('C1-1: prior calibration (winner)', '5aa4c06', 'calibrated vs raw gain', '≤0.5× explains', '2.33 vs 2.67', 'not explained'),
        ('P0: per-question first-order (LM-only GQA)', 'd5152f7', 'pooled r, σ≤0.002', '≥0.5 GO', '0.930', 'GO'),
        ('P1-A: flips match sampling', '464aed0', 'Spearman over 400 items', '≥0.6', '0.946', 'supports'),
        ('P1-B: persistence', '464aed0', 'r(sel, held-out) over 64', 'persistent if ≥0.3', '0.74', 'persistent'),
        ('P1-C: top-8 vote vs SC', '464aed0', 'diff vs SC@T*, CI', 'advantage if CI > 0', '+2.5 [−3.5, 8.5]', 'no advantage'),
        ('P2: GSM8K-1.5B SC vs RandOpt', '01efd9d', 'SC@50 − RandOpt', 'match ≥ −1.0', '+3.4', 'matches'),
        ('P3: GSM8K-3B SC vs RandOpt', 'c60b600', 'SC@50 − RandOpt', 'match ≥ −1.0', '+1.1', 'matches'),
        ('P2/P3: other 4 rows', '01efd9d / c60b600', 'base-reproduction gate', '±2.0 pp (GQA ±2.5)', 'failed', 'not comparable')]
    master = Path('paper/RESULTS_MASTER.md').read_text(encoding='utf-8')
    sec = master.split('## R9.')[1].split('## R10.')[0]
    rows9 = [l for l in sec.splitlines() if l.startswith('|')]
    (OUT / 'table2_preregistration_ledger.md').write_text('\n'.join(rows9) + '\n', encoding='utf-8')
    md3 = ['| Test | Model | Content word | σ | r(pred, measured) |', '|---|---|---|---|---|',
           '| Stage 2 | Qwen3-VL-8B | front | 0.002 | 0.938 |', '| R1 | Qwen3-VL-8B | left | 0.002 | 0.915 |',
           '| R2b | Qwen2.5-VL-7B | front | 0.002 | 0.925 |', '| I5 (per item, pooled) | Qwen3-VL-8B | front | 0.002 | 0.771 |',
           '| P0 | Qwen2.5-VL-3B (LM-only) | per-question answer contrast | 0.001 | 0.977 |',
           '| P0 | Qwen2.5-VL-3B (LM-only) | per-question answer contrast | 0.002 | 0.915 |',
           '| P0 | Qwen2.5-VL-3B (LM-only) | per-question answer contrast | 0.005 | 0.306 |']
    (OUT / 'table3_tilt_law_by_model_sigma.md').write_text('\n'.join(md3) + '\n', encoding='utf-8')


# ---------------------------------------------------------------- Same-run rows, prompt controls, prompt selection
def j(f):
    return json.loads((PA / f).read_text())


def same_run_rows():
    """Four same-run rows: RandOpt K = 50 − SC@50 under RandOpt's prompt, and under the prompt chosen on the selection set."""
    C_, C3, G2 = j('c-sameRun/c_results.json'), j('c3b-sameRun/c3b_results.json'), j('g2-sameRun/g2_results.json')
    O1, O2, Q2 = j('o1-olmo-sameRun/pod/o1/o1_results.json'), j('o2-olmo-prompt/pod/o2/o2_results.json'), j('q2-qwen-prompt/pod/q2/q2_results.json')
    G4, PS = j('g4-direct-prompt/pod/g4/g4_results.json'), j('ps-prompt-selection/ps_results.json')
    rows = [  # name, D under RandOpt's prompt, PS entry, base under RandOpt's prompt, plain/direct base, chosen-prompt base
        ('GSM8K\nQwen2.5-1.5B', C_['K50'], PS['q15'], Q2['q15']['randopt_prompt_base'], Q2['q15']['plain_base']),
        ('GSM8K\nQwen2.5-3B', C3['K50'], PS['q3'], Q2['q3']['randopt_prompt_base'], Q2['q3']['plain_base']),
        ('GQA\nQwen2.5-VL-3B', G2['K50'], PS['gqa'], G4['reference_cot']['base'], G4['direct']['base']),
        ('GSM8K\nOLMo-2-1B', O1['s42']['K50'], PS['olmo'], O2['randopt_prompt_base'], O2['plain']['base'])]
    held = {'GQA\nQwen2.5-VL-3B': j('gd-gqa-direct-search/pod2/gd/gd_results.json')['K50'],
            'GSM8K\nOLMo-2-1B': j('gb-gsm8k-boxed-search/pod/gb/olmo/gb_results.json')['K50']}  # RandOpt searched under the chosen prompt
    for n, f in (('GSM8K\nQwen2.5-3B', 'rv-reviewer-round/rv2a_results.json'), ('GSM8K\nQwen2.5-1.5B', 'rv-reviewer-round/rv2b_results.json')):
        if (PA / f).exists(): held[n] = j(f)['K50']
    RV = j('rv-reviewer-round/rv_results.json')  # RV-1: SC@50 under the public template chosen on the selection set
    rvk = {'GSM8K\nQwen2.5-1.5B': 'q15', 'GSM8K\nQwen2.5-3B': 'q3', 'GQA\nQwen2.5-VL-3B': 'gqa', 'GSM8K\nOLMo-2-1B': 'olmo'}
    return [dict(name=n, d0=k['D'], ci0=k['D_ci95'], d1=p['D']['D'], ci1=p['D']['ci'], chosen=p['chosen'], randopt=p['randopt_K50'],
                 sc_chosen=p['sc50_chosen'], base_ro=b0, dmg_plain=bp - b0, dmg_chosen=p['base_chosen'] - b0,
                 d2=held[n]['D'] if n in held else None, ci2=held[n]['ci'] if n in held else None,
                 randopt_held=(held[n].get('randopt_direct', held[n].get('randopt_boxed')) if n in held else None),
                 d3=RV[rvk[n]]['RV1']['D'], ci3=RV[rvk[n]]['RV1']['ci'], chosen_public=RV[rvk[n]]['chosen_new'],
                 sc_public=RV[rvk[n]]['sc50_chosen_new'])
            for n, k, p, b0, bp in rows]


def fig7():
    R = same_run_rows(); fig, ax = plt.subplots(figsize=(5.4, 4.0)); y = np.arange(len(R))[::-1]
    for k, r in enumerate(R):
        marks = [(r['d0'], r['ci0'], C['sc'], 'o', 0.27), (r['d1'], r['ci1'], C['green'], 's', 0.09), (r['d3'], r['ci3'], C['sky'], 'D', -0.09)]
        if r['d2'] is not None: marks.append((r['d2'], r['ci2'], C['pert'], '^', -0.27))
        for d, ci, col, mk, off in marks:
            ax.errorbar(d, y[k] + off, xerr=[[d - ci[0]], [ci[1] - d]], fmt=mk, ms=6, color=col, mec='white', mew=0.8, elinewidth=1.4, capsize=0)
            ax.text(ci[1] + 0.6, y[k] + off, f'{d:+.1f}', va='center', fontsize=7, color=C['ink'])
    ax.axvline(0, color=C['muted'], lw=0.9)
    ax.set_yticks(y); ax.set_yticklabels([r['name'] + f"\n(chosen: {r['chosen']})" for r in R], fontsize=7.5)
    ax.set_xlabel('RandOpt K=50 − self-consistency@50 (pp, 95% CI)\n← self-consistency better      RandOpt better →')
    ax.errorbar([], [], xerr=[], fmt='o', color=C['sc'], label="RandOpt vs SC, both under RandOpt's prompt")
    ax.errorbar([], [], xerr=[], fmt='s', color=C['green'], label="RandOpt (its prompt) vs SC under the prompt chosen on the selection set")
    ax.errorbar([], [], xerr=[], fmt='D', color=C['sky'], label='RandOpt (its prompt) vs SC under the public template chosen on the selection set')
    ax.errorbar([], [], xerr=[], fmt='^', color=C['pert'], label='RandOpt searched under the chosen prompt vs SC under it')
    ax.legend(loc='lower center', bbox_to_anchor=(0.45, 1.01), frameon=False, ncol=1, fontsize=7); ax.grid(axis='y', visible=False)
    ax.set_xlim(-27, 14); save(fig, 'fig7_same_run_prompt_selection')


def fig10():
    """Selection-set gain vs test gain of the selected models, every N = 5000 search (exploratory)."""
    T = json.loads((PA / 'transfer/transfer_results.json').read_text())
    held = {'GQA Qwen2.5-VL-3B (GD)', 'GSM8K OLMo-2-1B (GB)', 'GSM8K Qwen2.5-3B (RV-2a)', 'GSM8K Qwen2.5-1.5B (RV-2b)'}
    short = {'GSM8K Qwen2.5-1.5B (C)': 'Qwen-1.5B', 'GSM8K Qwen2.5-3B (C3B)': 'Qwen-3B', 'GSM8K OLMo-2-1B (O1)': 'OLMo-1B',
             'GQA Qwen2.5-VL-3B (G2)': 'GQA (seed 42)', 'GQA Qwen2.5-VL-3B (G2R)': 'GQA (seed 43)', 'GQA Qwen2.5-VL-3B (GD)': 'GQA, direct',
             'GSM8K OLMo-2-1B (GB)': 'OLMo-1B, boxed', 'GSM8K Qwen2.5-3B (RV-2a)': 'Qwen-3B, boxed', 'GSM8K Qwen2.5-1.5B (RV-2b)': 'Qwen-1.5B, boxed'}
    off = {'GQA (seed 42)': (8, 6), 'GQA (seed 43)': (-70, 10), 'Qwen-1.5B': (16, -14), 'Qwen-3B': (6, 4), 'OLMo-1B': (-36, -16),
           'GQA, direct': (6, 4), 'OLMo-1B, boxed': (6, -10), 'Qwen-3B, boxed': (-10, -16), 'Qwen-1.5B, boxed': (8, 8)}
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    lim = [-2, 14]; ax.plot(lim, lim, color='#BBBBBB', lw=0.8, ls='--', zorder=1); ax.axhline(0, color=C['muted'], lw=0.8)
    ax.text(2.75, 2.2, 'full transfer\n(y = x)', fontsize=6.5, color=C['muted'])
    for r in T:
        h = r['search'] in held; x, yv = r['selection_gain'], r['test_gain']
        ax.scatter(x, yv, s=36, marker='^' if h else 'o', color=C['pert'] if h else C['randopt'], ec='white', lw=0.6, zorder=3)
        nm = short[r['search']]
        ax.annotate(nm, (x, yv), xytext=off[nm], textcoords='offset points', fontsize=6.5, color=C['ink'],
                    arrowprops=dict(arrowstyle='-', color='#999999', lw=0.5, shrinkA=0, shrinkB=3))
    ax.scatter([], [], marker='o', color=C['randopt'], label="searched under RandOpt's prompt")
    ax.scatter([], [], marker='^', color=C['pert'], label='searched under the chosen prompt')
    ax.legend(loc='upper right', frameon=False, fontsize=6.5)
    ax.set_xlim(0, 14); ax.set_ylim(-2, 7)
    ax.set_xlabel('selection gain: top-50 mean − base\non the 200 selection questions (pp)'); ax.set_ylabel('test gain: selected models\n(mean) − base (pp)')
    save(fig, 'fig10_transfer')


def table_same_run_tex():
    R = same_run_rows()
    fmt = lambda d, ci: rf"${d:+.2f}$ {{\scriptsize[${ci[0]:+.2f}$, ${ci[1]:+.2f}$]}}"
    L = [r'\begin{tabular}{lcccccccc}', r'\toprule',
         r" & \multicolumn{2}{c}{RandOpt's prompt} & \multicolumn{2}{c}{Prompt chosen on selection set} & \multicolumn{2}{c}{Public template chosen} & \multicolumn{2}{c}{Search under chosen prompt} \\",
         r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}',
         r'Row & RandOpt & RandOpt $-$ SC & SC@50 & RandOpt $-$ SC & SC@50 & RandOpt $-$ SC & RandOpt & RandOpt $-$ SC \\', r'\midrule']
    for r in R:
        nm = r['name'].replace('\n', ' / ') + f" ({r['chosen']})"
        held = (f"{r['randopt_held']:.1f} & " + fmt(r['d2'], r['ci2'])) if r['d2'] is not None else r'-- & --'
        L.append(f"{nm} & {r['randopt']:.1f} & {fmt(r['d0'], r['ci0'])} & {r['sc_chosen']:.1f} & {fmt(r['d1'], r['ci1'])} & "
                 f"{r['sc_public']:.1f} & {fmt(r['d3'], r['ci3'])} & {held} \\\\")
    L += [r'\bottomrule', r'\end{tabular}']
    (OUT / 'table1_same_run.tex').write_text('\n'.join(L) + '\n', encoding='utf-8')


def fig8():
    R = same_run_rows(); fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for r in R:
        ax.errorbar(r['dmg_plain'], r['d0'], yerr=[[r['d0'] - r['ci0'][0]], [r['ci0'][1] - r['d0']]], fmt='o', ms=6, color=C['sc'], mec='white',
                    elinewidth=1.2, zorder=3)
        ax.plot(r['dmg_chosen'], r['d0'], marker='o', ms=6, mfc='white', mec=C['sc'], mew=1.2, ls='none', zorder=2)
        ax.plot([r['dmg_plain'], r['dmg_chosen']], [r['d0'], r['d0']], color='#BBBBBB', lw=0.8, zorder=1)
        below = '3B' in r['name'] and 'VL' not in r['name']  # keeps the label clear of the 1.5B whisker
        ax.annotate(r['name'].replace('\n', ' '), (min(r['dmg_plain'], r['dmg_chosen']) if below else max(r['dmg_plain'], r['dmg_chosen']), r['d0']),
                    xytext=(4, 11) if below else (6, 4), textcoords='offset points', fontsize=6.5, color=C['ink'])
    ax.axhline(0, color=C['muted'], lw=0.9); ax.axvline(0, color='#DDDDDD', lw=0.8)
    ax.set_xlabel("Damage of RandOpt's prompt to the base model (pp)\n= base accuracy with another prompt − with RandOpt's prompt")
    ax.set_ylabel('RandOpt − SC@50, both under\nRandOpt\'s prompt (pp)')
    ax.plot([], [], 'o', color=C['sc'], label='vs plain / direct prompt (pre-registered, Q2)')
    ax.plot([], [], 'o', mfc='white', mec=C['sc'], label='vs prompt chosen on the selection set (PS)')
    ax.legend(loc='upper left', frameon=False, fontsize=6.5); ax.set_xlim(-10, 45); save(fig, 'fig8_prompt_damage')


def table4():
    R = same_run_rows()
    md = ['| Row | Base (RandOpt prompt) | RandOpt K=50 | RandOpt − SC@50, RandOpt prompt | Prompt chosen on selection set | SC@50, chosen prompt | RandOpt − SC@50, chosen prompt |',
          '|---|---|---|---|---|---|---|']
    for r in R:
        md.append(f"| {r['name'].replace(chr(10), ' / ')} | {r['base_ro']:.1f} | {r['randopt']:.1f} | {r['d0']:+.2f} [{r['ci0'][0]:+.2f}, {r['ci0'][1]:+.2f}] | "
                  f"{r['chosen']} | {r['sc_chosen']:.1f} | {r['d1']:+.2f} [{r['ci1'][0]:+.2f}, {r['ci1'][1]:+.2f}] |")
    (OUT / 'table4_same_run_prompt_selection.md').write_text('\n'.join(md) + '\n\nSources: C, C3B, G2, O1 (same-run RandOpt vs SC); PS (prompt selection). '
                                                             'All locked; see results/README.md.\n', encoding='utf-8')


def fig9():
    """Same 5000 perturbations, selection reward under RandOpt's prompt (x) vs under the better prompt (y)."""
    import re as _re

    def log_rewards(p):
        t = Path(p).read_text(encoding='utf-8', errors='replace')
        return np.array([float(x.strip().strip("'")) for m in _re.finditer(r'Batch \d+ \| \d+/5000 \| \[(.*?)\]', t) for x in m.group(1).split(',')])

    def jsonl(d):
        r = {}
        for f in Path(d).glob('select_*.jsonl'):
            for l in f.read_text().splitlines():
                if l.strip():
                    x = json.loads(l); r[x['k']] = x['reward']
        return np.array([r[k] for k in range(5000)])
    rows = [('GSM8K / OLMo-2-1B', "RandOpt's prompt", 'boxed prompt', log_rewards(PA / 'o1-olmo-sameRun/pod/o1/out/randopt.log'),
             jsonl(PA / 'gb-gsm8k-boxed-search/pod/gb/olmo/out'), 0.415, 0.795),
            ('GQA / Qwen2.5-VL-3B', 'CoT prompt', 'direct prompt', jsonl(PA / 'g2-sameRun/pod/g2/out'), jsonl(PA / 'gd-gqa-direct-search/pod/gd/out'),
             json.loads((PA / 'g2-sameRun/pod/g2/out/base_select.json').read_text())['reward'], json.loads((PA / 'gd-gqa-direct-search/pod/gd/out/base_select.json').read_text())['reward'])]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0)); rng = np.random.default_rng(0)
    for k, (ax, (title, xa, ya, a, b, ba, bb)) in enumerate(zip(axes, rows)):
        jx, jy = rng.uniform(-.0015, .0015, 5000), rng.uniform(-.0015, .0015, 5000)  # rewards move in steps of 1/200
        ta, tb = np.argsort(-a, kind='stable')[:50], np.argsort(-b, kind='stable')[:50]
        ax.scatter(100 * (a + jx), 100 * (b + jy), s=3, color='#BDBDBD', alpha=.5, lw=0, rasterized=True, label='all 5000 perturbations')
        ax.scatter(100 * (a[ta] + jx[ta]), 100 * (b[ta] + jy[ta]), s=16, color=C['randopt'], ec='white', lw=.4, label=f'top 50 under {xa}')
        ax.scatter(100 * (a[tb] + jx[tb]), 100 * (b[tb] + jy[tb]), s=16, marker='s', color=C['green'], ec='white', lw=.4, label=f'top 50 under {ya}')
        ax.axvline(100 * ba, color=C['muted'], lw=.8, ls='--'); ax.axhline(100 * bb, color=C['muted'], lw=.8, ls='--')
        from scipy.stats import spearmanr
        ax.text(.03, .03, f'Spearman {spearmanr(a, b).correlation:.3f}\ntop-50 overlap {len(set(ta) & set(tb))}', transform=ax.transAxes, va='bottom',
                fontsize=7, color=C['ink'], bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#DDDDDD'))
        lx, hx = np.percentile(100 * a, [0.5, 100]); ly, hy = np.percentile(100 * b, [0.5, 100])
        lx, ly = min(lx, 100 * ba) - 2, min(ly, 100 * bb) - 2; hx, hy = hx + 1.5, hy + 1.5
        off = int(((100 * a < lx) | (100 * b < ly)).sum())
        ax.set_xlim(lx, hx); ax.set_ylim(ly, hy)
        if off: ax.text(.97, .03, f'{off} perturbations off-axis (lower)', transform=ax.transAxes, ha='right', va='bottom', fontsize=6.5, color=C['muted'])
        ax.set_title(f"{'ab'[k]}   {title}", fontsize=9, loc='left', fontweight='bold')
        ax.set_xlabel(f'selection accuracy, {xa} (%)'); ax.set_ylabel(f'selection accuracy, {ya} (%)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, ['all 5000 perturbations', "top 50 under RandOpt's prompt", 'top 50 under the better prompt'], loc='lower center',
               bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=7, markerscale=1.6)
    fig.tight_layout(); save(fig, 'fig9_prompt_specific_experts')


if __name__ == '__main__' and sys.argv[1:] == ['law']:
    fig3(); sys.exit()

if __name__ == '__main__' and sys.argv[1:] == ['experts']:
    fig9(); sys.exit()

if __name__ == '__main__' and sys.argv[1:] == ['same-run']:
    fig7(); fig8(); table4(); fig10(); table_same_run_tex(); print(sorted(p.name for p in OUT.iterdir() if p.stem.startswith(('fig7', 'fig8', 'table4')))); sys.exit()

if __name__ == '__main__':
    fig1(); fig2(); per = fig3(); fig4(); fig5(); rho = fig6(); tables(); fig7(); fig8(); table4()
    print('done; I5 per-item median r = %.3f; P1 rho = %.3f' % (np.median(per), rho))
    print(sorted(p.name for p in OUT.iterdir()))

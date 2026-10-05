"""Shared helpers: data loading, palette, figure export (PDF + SVG + 300-dpi PNG) and data dumps.

Stage names used in the paper: SEARCH (SEARCH200), RERANK (historical VALIDATION200,
the second-stage selection set) and TEST (official Perspective-Taking test, 561).
"""
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path('results/perspective-taking-n5000-20261004')
MASTER = Path('results/paper-analysis/paper_master_predictions.parquet')
FIG = Path('paper/figures')
DATA = FIG / 'data'
TABLES = Path('paper/tables')
CANDIDATE = '25a60f0b59f103ebf782a31f94bb569d897a652a0bc1bd95c5cdc5521418b843'  # seed 9504111, sigma 0.002
SIGMAS = (0.00025, 0.0005, 0.001, 0.002)

# Colorblind-safe reference palette (validated categorical slots 1-3) + neutral inks.
INK, INK2, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df', '#ffffff'
CAND, BASE_C, ACCENT, GOOD = '#2a78d6', '#52514e', '#eb6834', '#1baf7a'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5, 'legend.fontsize': 7.5, 'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .5,
    'lines.linewidth': 1.6, 'savefig.bbox': 'tight', 'savefig.pad_inches': .02, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})


def load(p):
    raw = Path(p).read_bytes()
    return json.loads(gzip.decompress(raw) if str(p).endswith('.gz') else raw)


def master():
    return pd.read_parquet(MASTER)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    # No embedded dates, and fixed SVG ids, so reruns are byte-identical.
    plt.rcParams['svg.hashsalt'] = name
    for ext, kw in (('pdf', {'metadata': {'CreationDate': None}}), ('svg', {'metadata': {'Date': None}}), ('png', {'dpi': 300})):
        fig.savefig(FIG / f'{name}.{ext}', **kw)
    plt.close(fig)


def dump(name, obj):
    DATA.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(DATA / f'{name}.csv', index=False)
    else:
        (DATA / f'{name}.json').write_text(json.dumps(obj, indent=1, default=lambda x: x.item() if hasattr(x, 'item') else str(x)))


def correct_matrix(df, phase, ids=None):
    """(candidates x questions) boolean matrix in fixed question order, plus base vector and question metadata."""
    d = df[df.phase == phase]
    q = d[d.candidate_id == 'BASE'][['example_id', 'subtask', 'question_form', 'question_text', 'true_option', 'base_prediction', 'base_correct']].reset_index(drop=True)
    order = {e: i for i, e in enumerate(q.example_id)}
    cands = [c for c in d.candidate_id.unique() if c != 'BASE'] if ids is None else ids
    sub = d[d.candidate_id.isin(cands)]
    m = np.zeros((len(cands), len(q)), dtype=bool)
    p = np.empty((len(cands), len(q)), dtype=object)
    ci = {c: i for i, c in enumerate(cands)}
    for c, e, ok, pr in zip(sub.candidate_id, sub.example_id, sub.candidate_correct, sub.candidate_prediction):
        m[ci[c], order[e]] = ok; p[ci[c], order[e]] = pr
    return list(cands), m, p, q


def tex_escape(s):
    return str(s).replace('_', '\\_').replace('%', '\\%').replace('&', '\\&')


def write_tex(name, header, rows, caption, label, colspec=None):
    TABLES.mkdir(parents=True, exist_ok=True)
    colspec = colspec or 'l' + 'r' * (len(header) - 1)
    # Requires \usepackage{booktabs,adjustbox}; adjustbox only shrinks tables wider than the column.
    lines = ['\\begin{table}[t]', '\\centering', '\\small', '\\begin{adjustbox}{max width=\\linewidth}',
             f'\\begin{{tabular}}{{{colspec}}}', '\\toprule',
             ' & '.join(header) + ' \\\\', '\\midrule']
    for r in rows:
        lines.append('\\midrule' if r == 'MIDRULE' else ' & '.join(str(x) for x in r) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{adjustbox}', f'\\caption{{{caption}}}', f'\\label{{{label}}}', '\\end{table}']
    (TABLES / f'{name}.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')

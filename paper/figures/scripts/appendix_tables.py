"""Appendix tables generated from paper/RESULTS_MASTER.md so they cannot drift from it: the R9 confirmatory-test ledger
(longtable) and the tilt-law-by-model table. Run from the repo root: python paper/figures/scripts/appendix_tables.py"""
from pathlib import Path
import re

OUT = Path('paper/figures/thickets-or-tilts')
SUB = [('\\', r'\textbackslash{}'), ('&', r'\&'), ('%', r'\%'), ('_', r'\_'), ('#', r'\#'), ('$', r'\$'),
       ('≤', r'$\le$'), ('≥', r'$\ge$'), ('−', r'$-$'), ('×', r'$\times$'), ('∋', r'$\ni$'), ('→', r'$\rightarrow$'),
       ('Δ', r'$\Delta$'), ('σ', r'$\sigma$'), ('ρ', r'$\rho$'), ('τ', r'$\tau$'), ('Φ', r'$\Phi$'), ('‖', r'$\|$'),
       ('±', r'$\pm$'), ('≈', r'$\approx$'), ('·', r'$\cdot$'), ('†', r'$\dagger$'), ('’', "'"), ('“', '``'), ('”', "''"),
       ('Σ', r'$\Sigma$'), ('₁', r'$_1$'), ('₂', r'$_2$'), ('²', r'$^2$'), ('ε', r'$\epsilon$'),
       ('θ', r'$\theta$'), ('μ', r'$\mu$'), ('<', r'$<$'), ('>', r'$>$'), ('–', '--'), ('—', '---'), ('\u00a0', ' ')]


def tex(s):
    s = s.strip()
    for a, b in SUB:
        s = s.replace(a, b)
    s = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', s)
    s = re.sub(r'`(.+?)`', r'\\texttt{\1}', s)
    s = s.replace('\\|', '|')
    return s


def ledger():
    m = Path('paper/RESULTS_MASTER.md').read_text(encoding='utf-8')
    sec = m.split('## R9.')[1].split('## R10.')[0]
    rows = [l for l in sec.splitlines() if l.startswith('|')][2:]  # drop header and rule
    L = [r'\begin{longtable}{p{1.2cm}p{1.3cm}p{3.8cm}p{2.1cm}p{2.1cm}p{2.4cm}p{2.7cm}}', r'\toprule',
         r'Study & Lock & Test & Statistic & Rule (fixed in the lock) & Result & Outcome \\', r'\midrule', r'\endfirsthead',
         r'\toprule', r'Study & Lock & Test & Statistic & Rule & Result & Outcome \\', r'\midrule', r'\endhead']
    for r in rows:
        cells = [tex(c) for c in r.strip().strip('|').split(' | ')]
        if len(cells) != 7:
            cells = [tex(c) for c in re.split(r'(?<!\\)\|', r.strip().strip('|'))]
        L.append(' & '.join(cells[:7]) + r' \\')
    L += [r'\bottomrule', r'\end{longtable}']
    (OUT / 'table_ledger.tex').write_text('\n'.join(L) + '\n', encoding='utf-8')
    return len(rows)


def tilt_law():
    md = (OUT / 'table3_tilt_law_by_model_sigma.md').read_text(encoding='utf-8').splitlines()
    rows = [l for l in md if l.startswith('|')][2:]
    L = [r'\begin{tabular}{llllc}', r'\toprule', r'Test & Model & Contrast & $\sigma$ & $r$(predicted, measured) \\', r'\midrule']
    for r in rows:
        L.append(' & '.join(tex(c) for c in r.strip().strip('|').split('|')) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}']
    (OUT / 'table3_tilt_law.tex').write_text('\n'.join(L) + '\n', encoding='utf-8')
    return len(rows)


if __name__ == '__main__':
    print('ledger rows', ledger(), 'tilt rows', tilt_law())

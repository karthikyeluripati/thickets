"""Part L: precommitted vs post-hoc provenance. Commit hashes and timestamps are read from git."""
import subprocess

from common import dump, write_tex

ROWS = [  # (analysis, status, commit, what it fixed or found)
    ('Data splits (SEARCH/RERANK/TEST)', 'precommitted', 'faf8f54', 'Splits frozen before any inference'),
    ('Protocol, K, sigma grid, density audit', 'precommitted', 'e5bb2ce', 'GO rules, 500 audit candidates fixed'),
    ('Base outputs', 'precommitted', '833e445', 'Base answers frozen before search'),
    ('SEARCH ranking and top 50', 'precommitted', '220ba16', 'Locked before RERANK inference'),
    ('RERANK selection (rank 1, top 5/10)', 'precommitted', 'f0c3cf6', 'Locked before TEST inference'),
    ('Standalone TEST result', 'precommitted', 'a193a87', 'NO-GO: winner -2.67 pp'),
    ('RandOpt top-50 committee', 'post hoc (addendum)', 'b43e47f', 'Committee frozen before its TEST run; = base'),
    ('Forensic audit and GPU reproduction', 'post hoc', '0099a29', 'Result reproduced; split-specific effect'),
    ('Allocentric specialist search', 'post hoc', '4ec4dd4', 'No signal beyond structured null'),
    ('Paper analyses (decomposition, overlap, nulls, oracle)', 'post hoc', 'this branch', 'Descriptive; no new candidates'),
    ('Official-prompt robustness (1 GPU, \\$2.50)', 'post hoc', 'this branch', 'RERANK +8.0 becomes -1.5; TEST -4.3'),
]


def when(c):
    if c == 'this branch':
        return '2026-10-05'
    return subprocess.run(['git', 'log', '-1', '--date=iso-strict-local', '--format=%cd', c], capture_output=True, text=True, check=True, env={**__import__('os').environ, 'TZ': 'UTC'}).stdout.strip()


def main():
    out = [{'analysis': a, 'status': s, 'commit': c, 'committed_at': when(c), 'note': n} for a, s, c, n in ROWS]
    dump('table_analysis_provenance', out)
    rows = [[o['analysis'], o['status'], f"\\texttt{{{o['commit']}}}", o['committed_at'][:16].replace('T', ' '), o['note']] for o in out]
    rows.insert(6, 'MIDRULE')
    write_tex('table_analysis_provenance', ['Analysis', 'Status', 'Commit', 'Committed (UTC)', 'Outcome'], rows,
              'Provenance of every analysis. Rows above the rule were frozen in git before the data they '
              'evaluate were generated; rows below are post hoc and are reported as exploratory.',
              'tab:provenance', colspec='p{4.2cm}llll')
    for o in out:
        print(o)


if __name__ == '__main__':
    main()

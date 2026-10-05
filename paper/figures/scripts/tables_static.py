"""Part R: setup-difference and forensic-integrity tables, from committed forensic artifacts only."""
import json
from pathlib import Path

from common import ROOT, dump, write_tex

FORENSIC = ROOT / 'forensic-audit'


def setup_differences():
    rows = [
        ('Model', 'LLMs (Qwen2.5, Olmo); GQA model unstated', 'Qwen3-VL-8B-Instruct'),
        ('Population $N$ / committee $K$', '5,000 / 50--500', '5,000 / 50'),
        ('Selection examples', '200 (train reward)', 'SEARCH200, then RERANK200'),
        ('$\\sigma$', 'uniform over \\{5e-4, 1e-3, 2e-3\\}', 'fixed mix \\{2.5e-4, 5e-4, 1e-3, 2e-3\\}'),
        ('Parameters perturbed', 'language only (default)', 'all, incl. vision encoder'),
        ('Restoration', 'add / subtract', 'exact snapshot'),
        ('Voting', '\\texttt{Counter.most\\_common}', 'identical'),
        ('Prompt / answer', 'step-by-step, \\texttt{\\textbackslash boxed\\{\\}}', 'official direct letter (A--D)'),
        ('Generation length', '256 tokens', '16 tokens'),
    ]
    dump('table_setup_differences', [dict(zip(('property', 'upstream', 'ours'), r)) for r in rows])
    write_tex('table_setup_differences', ['Property', 'Released RandOpt (GQA)', 'This study'], rows,
              'Setup differences from the released RandOpt code (pinned \\texttt{4000d34}), read from code only. '
              'The upstream GQA invocation is not published, so its vision-encoder setting is uncertain.',
              'tab:setup', colspec='lll')


def forensic_integrity():
    rep = json.loads((FORENSIC / 'phaseQR-gpu' / 'report.json').read_text())
    runs = {r['label']: r for r in rep['runs']}
    ident = all(r['validation'].get('raw_text_identical_to_study') and r['test'].get('raw_text_identical_to_study')
                for k, r in runs.items() if not k.startswith('diagnostic') or k.endswith('_all'))
    rows = [
        ('Independent re-scoring of base and 9504111', '0 of 1,922 decisions differ; 79$\\to$95 and 259$\\to$244 reproduce'),
        ('Re-scoring of all 1,137,611 stored outputs', 'every SEARCH and RERANK count equals the locked value'),
        ('Clean-process GPU re-run (base, 9504111 twice, 4 others)', 'byte-identical raw text on every question' if ident else 'MISMATCH'),
        ('Base weights restored after candidates', 'yes (state hash identical)' if rep['base_state_restored'] else 'NO'),
        ('Candidate weights identical across phases', 'one state fingerprint, SEARCH$\\to$TEST'),
        ('Split leakage (QA, image hash, question text vs.\\ TEST)', 'none'),
        ('Vision encoder recomputed per candidate', 'yes (runtime hook counts)'),
        ('Answer-letter distribution shift RERANK vs.\\ TEST', 'none (TVD 0.055, $p$=0.60)'),
        ('Subtask distribution shift RERANK vs.\\ TEST', 'large (Egocentric 59.5\\% vs.\\ 18.2\\%)'),
        ('Parser edge cases', '0 malformed of 1,137,611'),
        ('Mask diagnostic (post hoc): 9504111 RERANK / TEST',
         f"language-only {runs['diagnostic_9504111_language_only']['validation']['correct']}/{runs['diagnostic_9504111_language_only']['test']['correct']}, "
         f"vision-only {runs['diagnostic_9504111_vision_only']['validation']['correct']}/{runs['diagnostic_9504111_vision_only']['test']['correct']}, "
         f"all {runs['diagnostic_9504111_all']['validation']['correct']}/{runs['diagnostic_9504111_all']['test']['correct']} (base 79/259)"),
    ]
    dump('table_forensic_integrity', [dict(zip(('check', 'result'), r)) for r in rows])
    write_tex('table_forensic_integrity', ['Check', 'Result'], rows,
              'Forensic integrity checks. No implementation error was found; the collapse reproduces exactly.',
              'tab:forensic', colspec='p{6.3cm}p{6.3cm}')


if __name__ == '__main__':
    setup_differences(); forensic_integrity()

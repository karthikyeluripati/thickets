"""Read only completed selection-phase records; commit the full ranking next."""
import argparse
from pathlib import Path
import sys

from thicket_runtime.cli import revision_info
from thicket_runtime.committee_validation import committed
from thicket_runtime.line_tracing import sha
from thicket_runtime.visual_runtime import read, write
from thicket_runtime.visual_final import rank_selection, verify_inputs
from thicket_runtime.visual_final_audit import phase

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--run',type=Path,required=True)
p.add_argument('--protocol',type=Path,default=Path('experiments/visual_line_tracing_final_protocol.json'))
p.add_argument('--out',type=Path,required=True)
a = p.parse_args()
committed(a.protocol)
protocol = read(a.protocol)
verify_inputs(protocol)
if read(a.run/'run-manifest.json')['phase'] != 'search':
    raise ValueError('ranking cannot accept held-out candidate results')
audit = phase(a.run,protocol,a.protocol)
ranking = rank_selection(audit['records'],protocol)
write(a.out,{'protocol_sha256':sha(a.protocol),'source':revision_info(),'command':[sys.executable,*sys.argv],
            'search_manifest_sha256':sha(a.run/'run-manifest.json'),'search_checksums_sha256':sha(a.run/'sha256.json'),
            'ranked':ranking,'best':ranking[0],'top5':ranking[:5],'top10':ranking[:10],
            'ranking_inputs':'search-phase selection150 records only; diagnostic and heldout candidate outputs never read',
            'controls_pass':True,'candidate_count':300})
print({'frozen_candidates':300,'rank1_selection_correct':ranking[0]['correct_count'],'rank1_id':ranking[0]['candidate']['candidate_id']})

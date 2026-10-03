"""Create immutable phase locks from complete audited data, without GPU generation."""
import argparse
from pathlib import Path
import sys
from thicket_runtime.cli import revision_info
from thicket_runtime.line_tracing import load_split, sha
from thicket_runtime.visual_runtime import read,write
from thicket_runtime.visual_decisions import calibrate,rank_selection,summarize

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--phase',choices=['baseline','calibration','search'],required=True)
p.add_argument('--protocol',type=Path,default=Path('experiments/visual_line_tracing_protocol.json'))
p.add_argument('--run',type=Path,required=True)
p.add_argument('--baseline-lock',type=Path)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
protocol=read(a.protocol)
manifest,controls=read(a.run/'run-manifest.json'),read(a.run/'controls.json')
if manifest['status']!='complete' or manifest['phase']!=a.phase or not all(controls.values()):
    raise ValueError('complete valid phase required')
for name,expected in read(a.run/'sha256.json').items():
    if sha(a.run/name)!=expected: raise ValueError('run artifact changed: '+name)
if manifest['protocol_sha256']!=sha(a.protocol): raise ValueError('protocol changed')
dataset=Path(protocol['dataset'])
lock={'phase':a.phase,'protocol_sha256':sha(a.protocol),'dataset_manifest_sha256':sha(dataset/'manifest.json'),
      'run_manifest_sha256':sha(a.run/'run-manifest.json'),'run_checksum_sha256':sha(a.run/'sha256.json'),
      'source':revision_info(),'command':[sys.executable,*sys.argv],'base_id':manifest['base_id'],
      'controls_pass':True}
if a.phase=='baseline':
    bases=read(a.run/'base.json.gz')
    summary={s:summarize(v['outputs'],load_split(dataset,s)) for s,v in bases.items()}
    lo,hi=protocol['baseline']['mandatory_rebuild_if_either_split_outside']
    lock.update(accepted=all(lo<=v['accuracy']<=hi for v in summary.values()),summary=summary,
                output_identities={s:v['output_identity'] for s,v in bases.items()},
                baseline_only_heldout_exception=True)
elif a.phase=='calibration':
    base=read(a.baseline_lock)
    lock.update(calibrate(read(a.run/'records.json'),base['summary']['selection']['accuracy'],protocol['calibration']))
    lock['heldout_candidate_outcomes_used']=False
else:
    ranked=rank_selection(read(a.run/'records.json'),protocol['search'])
    lock.update(best=ranked[0],top5=ranked[:5],top10=ranked[:10],ranked_candidates=ranked,
                heldout_candidate_outcomes_used=False)
write(a.out,lock)
print({k:lock[k] for k in ('accepted','summary','scale_valid','sigma','table') if k in lock} if a.phase!='search' else {'frozen_top10':10,'best_selection_correct':lock['best']['correct_count']})

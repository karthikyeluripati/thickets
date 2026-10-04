"""Create immutable, audited v2.1 locks; never use easy for candidate selection."""
import argparse
from pathlib import Path
import sys

from thicket_runtime.cli import revision_info
from thicket_runtime.committee_validation import committed
from thicket_runtime.line_tracing import sha
from thicket_runtime.visual_runtime import read,write
from thicket_runtime.visual_v21 import baseline_summary,calibrate,capability_gate,rank_selection,verify_dataset
from thicket_runtime.visual_v21_audit import phase

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--phase',choices=['baseline','calibration','search'],required=True)
p.add_argument('--protocol',type=Path,default=Path('experiments/visual_line_tracing_v21_protocol.json'))
p.add_argument('--run',type=Path,required=True)
p.add_argument('--baseline-lock',type=Path)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
protocol=read(a.protocol)
verify_dataset(protocol)
base=read(a.baseline_lock) if a.baseline_lock else None
if a.phase!='baseline' and not (base and base['accepted']): raise ValueError('accepted baseline lock required')
if a.baseline_lock:
    committed(a.baseline_lock)
    if a.baseline_lock.read_bytes()!=(a.run/a.baseline_lock.name).read_bytes():
        raise ValueError('phase used a different baseline lock')
audited=phase(a.run,protocol,a.protocol,base)
if audited['manifest']['phase']!=a.phase: raise ValueError('wrong phase')
lock={'phase':a.phase,'protocol_sha256':sha(a.protocol),'dataset_manifest_sha256':protocol['dataset_pin']['manifest_sha256'],
      'run_manifest_sha256':sha(a.run/'run-manifest.json'),'run_checksum_sha256':sha(a.run/'sha256.json'),
      'source':revision_info(),'command':[sys.executable,*sys.argv],'base_id':audited['manifest']['base_id'],'controls_pass':True}
if a.phase=='baseline':
    summary=baseline_summary(audited['bases'],audited['datasets'])
    lock.update(capability_gate(summary,protocol['baseline']))
    lock.update(summary=summary,output_identities={s:r['output_identity'] for s,r in audited['bases'].items()},
                baseline_only_heldout_exception=True)
elif a.phase=='calibration':
    lock.update(calibrate(audited['records'],base['summary']['selection']['primary']['correct_count'],protocol))
else:
    ranked=rank_selection(audited['records'],protocol)
    lock.update(best=ranked[0],top5=ranked[:5],top10=ranked[:10],ranked_candidates=ranked,easy_used=False,heldout_used=False)
write(a.out,lock)
print({k:lock[k] for k in ('decision','accepted','primary_heldout_accuracy','continue_search','sigma','table') if k in lock}
      if a.phase!='search' else {'frozen_top10':10,'best_primary_correct':lock['best']['correct_count']})

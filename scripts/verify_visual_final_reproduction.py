"""Record independent local reproduction of the frozen remote paired analysis."""
import argparse
import math
from pathlib import Path
import sys

from thicket_runtime.cli import revision_info
from thicket_runtime.line_tracing import sha
from thicket_runtime.visual_runtime import read, write

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True)
p.add_argument('--reproduced',type=Path,required=True)
a = p.parse_args()
expected = a.root/'analysis'
differences = []


def compare(left,right,path):
    if isinstance(left,float) and isinstance(right,(int,float)):
        if not math.isclose(left,right,rel_tol=0,abs_tol=1e-12): differences.append(path)
    elif isinstance(left,dict) and isinstance(right,dict):
        if set(left) != set(right): differences.append(path+':keys')
        else:
            for key in left: compare(left[key],right[key],path+'/'+key)
    elif isinstance(left,list) and isinstance(right,list):
        if len(left) != len(right): differences.append(path+':length')
        else:
            for i,(x,y) in enumerate(zip(left,right)): compare(x,y,path+'/'+str(i))
    elif left != right: differences.append(path)


files = [x for x in sorted(a.reproduced.iterdir()) if x.name != 'manifest.json']
for path in files:
    original = expected/path.name
    if path.suffix in ('.json','.gz'): compare(read(original),read(path),path.name)
    elif original.read_bytes() != path.read_bytes(): differences.append(path.name)
remote,local = read(expected/'manifest.json'),read(a.reproduced/'manifest.json')
for field in ('bootstrap_indices_sha256','phase_manifest_sha256','protocol_sha256','dataset_proof','selection_lock_commit'):
    compare(remote[field],local[field],field)
if differences: raise ValueError('independent reproduction differs: '+str(differences[:10]))
records = read(a.root/'search-01/records.json')
unique_states = len({r['candidate_state_id'] for r in records})
if unique_states != 300: raise ValueError('search did not produce 300 distinct states')
requests = {}
for phase in ('diagnostic','search','heldout'):
    root = a.root/(phase+'-01')
    paths = [*sorted((root/'candidates').glob('*.json.gz'))]
    paths += [p for p in (root/'zero.json.gz',root/'repeat.json.gz') if p.exists()]
    requests[phase] = sum(len(v['outputs']) for path in paths for v in read(path)['splits'].values())
if requests != {'diagnostic':500,'search':45300,'heldout':6000}:
    raise ValueError('image request budget differs from protocol')
write(a.root/'independent-reproduction.json',{'source':revision_info(),'command':[sys.executable,*sys.argv],
      'remote_analysis_manifest_sha256':sha(expected/'manifest.json'),'local_reproduction_manifest':local,
      'files_compared':[p.name for p in files],'float_absolute_tolerance':1e-12,
      'bootstrap_indices_identical':True,'all_outcomes_and_gate_reproduced':True,
      'unique_search_state_fingerprints':unique_states,'actual_image_requests':requests,
      'total_image_requests':sum(requests.values()),'mismatches':differences})
print({'independent_reproduction':'pass','files_compared':len(files),'image_requests':sum(requests.values())})

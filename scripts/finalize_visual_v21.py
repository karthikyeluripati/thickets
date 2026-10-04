"""Verify v2.1 evidence, frozen datasets and preservation before the final commit."""
import argparse
from pathlib import Path
import subprocess
import sys

from thicket_runtime.cli import revision_info
from thicket_runtime.line_tracing import sha
from thicket_runtime.visual_runtime import read,write
from thicket_runtime.visual_v21 import verify_dataset

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True)
p.add_argument('--protocol',type=Path,default=Path('experiments/visual_line_tracing_v21_protocol.json'))
a=p.parse_args()
if (a.root/'integrity.json').exists() or (a.root/'sha256.json').exists():
    raise FileExistsError('final evidence is immutable')
protocol=read(a.protocol)
dataset=verify_dataset(protocol)
verified={}
for path in sorted(Path('results').glob('*/sha256.json')):
    if path.parent.resolve()==a.root.resolve(): continue
    index=read(path)
    for name,expected in index.items():
        if sha(path.parent/name)!=expected: raise ValueError('prior artifact changed: '+str(path.parent/name))
    verified[path.parent.as_posix()]=len(index)
parent=protocol['dataset_pin']['commit']
old=set(subprocess.check_output(['git','ls-tree','-r','--name-only',parent],text=True).splitlines())
changed=set(subprocess.check_output(['git','diff','--name-only',parent],text=True).splitlines())
protected=(old & changed)-{'README.md'}
if protected: raise ValueError('old tracked files changed: '+str(sorted(protected)))
main=subprocess.check_output(['git','rev-parse','main'],text=True).strip()
if main!='deb414253139cc2559d19cdfe7e6b4786e7c40db': raise ValueError('main changed')
phases={}
for path in sorted(a.root.glob('*/run-manifest.json')):
    root=path.parent; manifest=read(path)
    for name,expected in read(root/'sha256.json').items():
        if sha(root/name)!=expected: raise ValueError('phase artifact changed: '+name)
    source=manifest['source']['commit']; count=0
    for f in (root/'measured-source').rglob('*'):
        if f.is_file():
            name=f.relative_to(root/'measured-source').as_posix()
            if subprocess.check_output(['git','show',source+':'+name])!=f.read_bytes():
                raise ValueError('measured source differs from execution commit: '+name)
            count+=1
    phases[root.name]={'status':manifest['status'],'source':source,'verified_source_files':count}
if not phases or not (a.root/'analysis/gate.json').is_file(): raise ValueError('audited terminal result required')
write(a.root/'integrity.json',{'source':revision_info(),'command':[sys.executable,*sys.argv],
      'prior_indexed_artifacts_verified':verified,'protected_old_file_changes':[],'main_ref':main,
      'dataset_pin_verified':dataset,'phases':phases,'mismatches':0})
write(a.root/'sha256.json',{f.relative_to(a.root).as_posix():sha(f) for f in sorted(a.root.rglob('*')) if f.is_file()})
print({'prior_artifacts_verified':sum(verified.values()),'phases':len(phases),'integrity_pass':True})

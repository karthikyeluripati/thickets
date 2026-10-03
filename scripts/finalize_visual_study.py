"""Verify immutable evidence and historical preservation before the final commit."""
import argparse
from pathlib import Path
import subprocess
import sys
from thicket_runtime.cli import revision_info
from thicket_runtime.line_tracing import sha
from thicket_runtime.visual_runtime import read,write

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True)
a=p.parse_args()
if (a.root/'sha256.json').exists() or (a.root/'integrity.json').exists():
    raise FileExistsError('final study evidence is immutable')
old=['llm-bounded-20261003','work1-closure-20261003','shared-speculative-20261003','adaptive-evaluation-20261003','complementarity-selection-20261003']
verified={}
for path in [Path('results')/n for n in old]:
    index=read(path/'sha256.json')
    for name,expected in index.items():
        if sha(path/name)!=expected: raise ValueError('historical file changed: '+str(path/name))
    verified[str(path)]=len(index)
changed=subprocess.check_output(['git','diff','--name-only','a7e435a5ddd3b843c8b30b2ea83b81d475834895','--',*[str(Path('results')/n) for n in old]])
if changed.strip(): raise ValueError('prior research artifacts changed')
main=subprocess.check_output(['git','rev-parse','main'],text=True).strip()
if main!='deb414253139cc2559d19cdfe7e6b4786e7c40db': raise ValueError('main changed')
phases={}
for path in sorted(a.root.rglob('run-manifest.json')):
    root=path.parent
    manifest=read(path)
    for name,expected in read(root/'sha256.json').items():
        if sha(root/name)!=expected: raise ValueError('phase artifact changed: '+name)
    source=manifest['source']['commit']
    count=0
    for f in (root/'measured-source').rglob('*'):
        if f.is_file():
            name=f.relative_to(root/'measured-source').as_posix()
            if subprocess.check_output(['git','show',source+':'+name])!=f.read_bytes():
                raise ValueError('measured source differs: '+name)
            count+=1
    phases[root.relative_to(a.root).as_posix()]={'status':manifest['status'],'source':source,'verified_source_files':count}
if not phases: raise ValueError('no GPU phase exists; preparation is not a completed study')
write(a.root/'integrity.json',{'source':revision_info(),'command':[sys.executable,*sys.argv],
    'old_indexed_files_verified':verified,'historical_result_changes':[],'main_ref':main,'phases':phases,'mismatches':0})
write(a.root/'sha256.json',{f.relative_to(a.root).as_posix():sha(f) for f in sorted(a.root.rglob('*')) if f.is_file()})
print({'old_verified':sum(verified.values()),'phases':len(phases),'integrity_pass':True})

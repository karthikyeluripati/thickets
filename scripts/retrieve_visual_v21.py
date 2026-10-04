"""Retrieve completed v2.1 visual-study evidence and lock commits without overwrites.

Run from the local repository root. SSH stdout is captured privately in ignored
runs/; only decoded, SHA256-verified research artifacts enter results/.
"""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--ssh-target',required=True)
p.add_argument('--remote-root',default='/workspace/thickets-visual-v21')
p.add_argument('--study',default='visual-line-tracing-v21-20261003')
p.add_argument('--cached',action='store_true')
a=p.parse_args()
root=Path(__file__).resolve().parents[1]
destination=(root/'results'/a.study).resolve()
if destination.parent!=root/'results' or not re.fullmatch(r'[a-zA-Z0-9/_-]+',a.remote_root):
    raise ValueError('invalid study or remote path')
remote=r"""stty -echo
export TERM=dumb
python - <<'PYREMOTE'
import base64,hashlib,json,pathlib,subprocess,tarfile
repo=pathlib.Path(REMOTE_ROOT)
root=repo/'results'/STUDY
phases=list(root.glob('*/run-manifest.json'))
assert phases and all(json.loads(p.read_text())['status']=='complete' for p in phases)
assert (root/'locks/baseline.json').is_file()
bundle=pathlib.Path('/workspace/visual-v21-locks.bundle')
subprocess.run(['git','bundle','create',str(bundle),'HEAD','^7ca95bb633bceab6bff6f605225bffb36b5b91b9'],cwd=repo,check=True,stdout=subprocess.DEVNULL)
archive=pathlib.Path('/workspace/visual-v21-results.tar.xz')
with tarfile.open(archive,'w:xz',preset=3) as t:
 for path in sorted(root.iterdir()):
  if path.name not in ('locks','cpu-tests.txt','cpu-v21-tests.txt','.gitattributes'):
   t.add(path,arcname='results/'+STUDY+'/'+path.name)
 t.add(bundle,arcname='visual-v21-locks.bundle')
data=archive.read_bytes()
encoded=base64.b64encode(data).decode()
print('ARTIFACT_BEGIN')
print(hashlib.sha256(data).hexdigest())
for i in range(0,len(encoded),60): print(encoded[i:i+60])
print('ARTIFACT_END')
PYREMOTE
exit
""".replace('REMOTE_ROOT',repr(a.remote_root)).replace('STUDY',repr(a.study))
capture=root/'runs/visual-v21-results-transfer.stdout'
if not a.cached:
    result=subprocess.run(['ssh','-tt','-o','BatchMode=yes','-o','ConnectTimeout=20',
        '-o','StrictHostKeyChecking=yes',a.ssh_target],input=remote.encode(),
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=300)
    capture.write_bytes(result.stdout)
    (root/'runs/visual-v21-results-transfer.stderr').write_bytes(result.stderr)
    if result.returncode: raise RuntimeError('SSH retrieval failed; transport log retained')
text=capture.read_bytes().decode().replace('\r','')
match=re.search(r'ARTIFACT_BEGIN\n([a-f0-9]{64})\n([A-Za-z0-9+/=\n]+)\nARTIFACT_END',text)
if not match: raise RuntimeError('incomplete artifact framing; transport log retained')
payload=base64.b64decode(match[2])
if hashlib.sha256(payload).hexdigest()!=match[1]: raise ValueError('transfer checksum mismatch')
(root/'runs/visual-v21-results.tar.xz').write_bytes(payload)
with tarfile.open(fileobj=io.BytesIO(payload),mode='r:xz') as archive:
    for item in archive.getmembers():
        target=(root/item.name).resolve()
        if item.name=='visual-v21-locks.bundle': target=root/'runs/visual-v21-locks.bundle'
        elif not target.is_relative_to(destination): raise ValueError('unsafe archive path: '+item.name)
        if item.issym() or item.islnk() or not (item.isfile() or item.isdir()):
            raise ValueError('unsafe archive item: '+item.name)
        if target.exists() and not item.isdir(): raise FileExistsError(target)
    for item in archive.getmembers():
        target=root/'runs/visual-v21-locks.bundle' if item.name=='visual-v21-locks.bundle' else root/item.name
        if item.isdir(): target.mkdir(parents=True,exist_ok=True)
        else:
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(archive.extractfile(item).read())
receipt=destination/'retrieval.json'
if receipt.exists(): raise FileExistsError(receipt)
receipt.write_bytes((json.dumps({'command':[sys.executable,*sys.argv],
    'archive_bytes':len(payload),'archive_sha256':match[1],
    'lock_bundle_sha256':hashlib.sha256((root/'runs/visual-v21-locks.bundle').read_bytes()).hexdigest(),
    'verified_transport_sha256':True,'existing_files_overwritten':False},indent=2)+'\n').encode())
print(f'Retrieved {len(payload)} bytes; SHA256 {match[1]}')

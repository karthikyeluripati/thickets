"""On the GPU host: fetch the pinned official zips, verify them, extract Complex_Logic images.

Every zip must match its pinned LFS SHA256 and every extracted image must match
the SHA256 frozen in the committed split files. Images are never committed.
"""
import argparse
import hashlib
from pathlib import Path
import zipfile

from huggingface_hub import hf_hub_download

from thicket_runtime.omnispatial_data import DATASET, DATASET_REVISION, TASK, ZIPS, load

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--dataset', default='examples/omnispatial-complex-logic')
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
need = {}
for split in ('search', 'validation', 'test'):
    for r in load(a.dataset, split):
        need[r['image_member']] = (r['source_split'], r['image_sha256'])
done = 0
for source, (name, expected) in ZIPS.items():
    path = hf_hub_download(DATASET, name, repo_type='dataset', revision=DATASET_REVISION)
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''): h.update(chunk)
    if h.hexdigest() != expected: raise ValueError('official zip differs from pinned LFS sha256: ' + name)
    with zipfile.ZipFile(path) as z:
        for member in z.namelist():
            if member in need and need[member][0] == source:
                data = z.read(member)
                if hashlib.sha256(data).hexdigest() != need[member][1]: raise ValueError('image differs: ' + member)
                target = a.out / source / member.rsplit('/', 1)[1]
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and target.read_bytes() != data: raise FileExistsError(target)
                target.write_bytes(data)
                done += 1
if done != len(need): raise ValueError(f'extracted {done} of {len(need)} frozen images')
print({'images_verified': done, 'zips_verified': list(ZIPS), 'task': TASK})

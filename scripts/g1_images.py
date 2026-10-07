"""G1: fetch the pinned GQA testdev_balanced_images parquet and write the frozen items' images as {imageId}.jpg."""
import hashlib
import io
import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
import pandas as pd

REV = 'a6e72d6e1b912da88af8b2f9eba05d5ea8ec2dd8'
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
items = json.loads(Path('results/paper-analysis/g1/frozen_items.json').read_text(encoding='utf-8'))
need = {r['imageId'] for k in ('selection', 'heldout') for r in items[k]}
f = hf_hub_download('lmms-lab-encoder/GQA', 'testdev_balanced_images/testdev-00000-of-00001.parquet', repo_type='dataset', revision=REV)
d = pd.read_parquet(f)
idcol = 'id' if 'id' in d.columns else [c for c in d.columns if 'id' in c.lower()][0]
imcol = [c for c in d.columns if 'image' in c.lower() and c != idcol][0]
from PIL import Image
got, shas = 0, {}
for _, row in d.iterrows():
    iid = str(row[idcol])
    if iid not in need: continue
    v = row[imcol]; b = v['bytes'] if isinstance(v, dict) else v
    img = Image.open(io.BytesIO(b)).convert('RGB'); p = out / f'{iid}.jpg'
    if b[:3] == b'\xff\xd8\xff': p.write_bytes(b)
    else: img.save(p, format='JPEG', quality=95)
    shas[iid] = hashlib.sha256(p.read_bytes()).hexdigest(); got += 1
missing = need - set(shas)
Path('results/paper-analysis/g1/gpu').mkdir(parents=True, exist_ok=True)
Path('results/paper-analysis/g1/gpu/image_sha256.json').write_text(json.dumps(shas, indent=0))
print('G1_IMAGES', got, 'needed', len(need), 'missing', len(missing), 'columns', list(d.columns))
assert not missing

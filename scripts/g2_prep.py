"""G2 (results/paper-analysis/g2-sameRun/plan_lock.md): selection = frozen G1 selection 200; test = P2's 2000-question hash
sample minus questions whose image is used by a selection question. Writes /workspace/g2/items.json and images."""
import hashlib
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

REV = 'a6e72d6e1b912da88af8b2f9eba05d5ea8ec2dd8'
OUT = Path('/workspace/g2'); (OUT / 'images').mkdir(parents=True, exist_ok=True)
q = pd.read_parquet(hf_hub_download('lmms-lab-encoder/GQA', 'testdev_balanced_instructions/testdev-00000-of-00001.parquet', repo_type='dataset', revision=REV))
q['h'] = [hashlib.sha256(('p2-gqa-v1:' + str(i)).encode()).hexdigest() for i in q.id]
p2 = q.sort_values('h').head(2000)
sel = json.loads(Path('results/paper-analysis/g1/frozen_items.json').read_text(encoding='utf-8'))['selection']
sel_imgs = {r['imageId'] for r in sel}
test = [{'id': str(r.id), 'question': r.question, 'answer': str(r.answer).strip().lower(), 'fullAnswer': str(r.fullAnswer), 'imageId': str(r.imageId)}
        for r in p2.itertuples() if str(r.imageId) not in sel_imgs]
selection = [{'id': str(r['id']), 'question': r['question'], 'answer': str(r['answer']).strip().lower(), 'fullAnswer': str(r.get('fullAnswer', '')),
              'imageId': str(r['imageId'])} for r in sel]
need = {r['imageId'] for r in test + selection}
imgs = pd.read_parquet(hf_hub_download('lmms-lab-encoder/GQA', 'testdev_balanced_images/testdev-00000-of-00001.parquet', repo_type='dataset', revision=REV))
n = 0
for _, r in imgs.iterrows():
    if str(r['id']) in need:
        v = r['image']; b = v['bytes'] if isinstance(v, dict) else v
        (OUT / 'images' / f"{r['id']}.jpg").write_bytes(b); n += 1
assert n == len(need), (n, len(need))
(OUT / 'items.json').write_text(json.dumps({'selection': selection, 'test': test, 'n_p2_removed': 2000 - len(test)}))
print('G2_PREP', len(selection), len(test), 'removed', 2000 - len(test), 'images', n, flush=True)

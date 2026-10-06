"""R1 replication helpers (results/paper-analysis/r1/plan_lock.md): OmniSpatial Complex_Logic, content class 'left'."""
import json
from pathlib import Path
import re

DATA = Path('examples/omnispatial-complex-logic')
INPUTS = Path('results/paper-analysis/r1/frozen_inputs.json')


def has_left(text):
    return 'left' in re.findall(r'[a-z]+', text.lower())


def tilt_weights(options):
    F = [has_left(o) for o in options]; nf, nn = sum(F), len(F) - sum(F)
    return [(1.0 / nf if f else -1.0 / nn) for f in F]


def examples():
    ex = {}
    for s in ('search', 'validation'):
        for l in (DATA / f'{s}.jsonl').read_text(encoding='utf-8').splitlines():
            r = json.loads(l); ex[r['uid']] = r
    return ex


def image_path(images, r):
    return images / r['source_split'] / r['image_member'].rsplit('/', 1)[1]

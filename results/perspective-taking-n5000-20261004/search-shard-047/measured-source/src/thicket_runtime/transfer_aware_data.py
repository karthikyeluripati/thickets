"""Fresh v1 line-tracing splits and search folds for the transfer-aware first bridge.

Geometry, rendering, prompt, parser and labels are the unchanged v1 functions in
line_tracing.py. Only new seeds and a balanced (difficulty, label, start) slot
assignment are defined here. No model output enters generation.
"""
from collections import Counter
import json
from pathlib import Path
import random

from .line_tracing import LEVELS, PROMPT, draw_example, sha, topology

SPLITS = (('search', 300, 8100000), ('validation', 300, 8200000), ('test', 1000, 8300000))
FOLDS = ('A', 'B', 'C')
LEVEL_NAMES = ('easy', 'medium', 'hard')
PREVIOUS_DATASETS = ('examples/visual-line-tracing-v1', 'examples/visual-line-tracing-v2')


def slots(n, seed_base):
    """Equal difficulty counts; labels balanced overall; starts balanced within each cell."""
    rng = random.Random(seed_base + 999000)
    counts = [n // 3 + (i < n % 3) for i in range(3)]
    result, pointer = [], 0
    for level, m in zip(LEVEL_NAMES, counts):
        per = [m // 4] * 4
        for _ in range(m % 4):
            per[pointer % 4] += 1
            pointer += 1
        for end in range(4):
            starts = [s for _ in range(per[end] // 4 + 1) for s in rng.sample(range(4), 4)][:per[end]]
            result += [(level, start, end) for start in starts]
    rng.shuffle(result)
    return result


def assign_folds(rows, seed):
    """Stratify by (difficulty, label); the 25th-example remainder rotates across folds."""
    rng, folds = random.Random(seed), {f: [] for f in FOLDS}
    for c, (level, label) in enumerate((l, a) for l in LEVEL_NAMES for a in '1234'):
        cell = [r['id'] for r in rows if r['difficulty'] == level and r['answer'] == label]
        rng.shuffle(cell)
        for k, rid in enumerate(cell):
            folds[FOLDS[(k + c) % 3]].append(rid)
    order = {r['id']: i for i, r in enumerate(rows)}
    return {f: sorted(ids, key=order.get) for f, ids in folds.items()}


def previous_hashes_and_seeds(repo):
    hashes, seeds = set(), set()
    for root in PREVIOUS_DATASETS:
        for path in sorted((Path(repo) / root).glob('*.jsonl')):
            for line in path.read_text().splitlines():
                row = json.loads(line)
                hashes.add(row['image_sha256'])
                seeds.add(row['seed'])
    return hashes, seeds


def freeze_dataset(root, repo='.'):
    import PIL
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    (root / 'images').mkdir()
    old_hashes, old_seeds = previous_hashes_and_seeds(repo)
    records = {}
    for split, count, seed_base in SPLITS:
        rows = []
        for i, (level, start, end) in enumerate(slots(count, seed_base)):
            seed = seed_base + i
            row = {'id': f'{split}-{i:04d}', 'split': split, 'seed': seed, 'difficulty': level,
                   'answer': str(end + 1), 'prompt': PROMPT, 'image': f'images/{split}-{i:04d}.png',
                   'width': 448, 'height': 448, 'paths': 4, 'style': dict(LEVELS[level]),
                   'topology': topology(seed, level, start, end)}
            draw_example(row).save(root / row['image'], optimize=False)
            row['image_sha256'] = sha(root / row['image'])
            rows.append(row)
        records[split] = rows
        (root / (split + '.jsonl')).write_bytes(('\n'.join(json.dumps(r, sort_keys=True) for r in rows) + '\n').encode())
    every = [r for rows in records.values() for r in rows]
    if len({r['seed'] for r in every}) != 1600 or {r['seed'] for r in every} & old_seeds:
        raise ValueError('generation seeds are not fresh and disjoint')
    if len({r['image_sha256'] for r in every}) != 1600 or {r['image_sha256'] for r in every} & old_hashes:
        raise ValueError('image hashes are not fresh and disjoint')
    folds = assign_folds(records['search'], 8100000 + 777000)
    (root / 'search_folds.json').write_bytes((json.dumps(folds, indent=2) + '\n').encode())
    fold_of = {rid: f for f, ids in folds.items() for rid in ids}

    def stats(rows):
        return {'n': len(rows), 'difficulty': dict(Counter(r['difficulty'] for r in rows)),
                'labels': dict(sorted(Counter(r['answer'] for r in rows).items())),
                'difficulty_by_label': {l: dict(sorted(Counter(r['answer'] for r in rows if r['difficulty'] == l).items())) for l in LEVEL_NAMES},
                'start_end_cells': dict(sorted(Counter(f"{r['topology']['target_path'] + 1}->{r['answer']}" for r in rows).items())),
                'straight_endpoint_heuristic_accuracy': sum(int(r['answer']) == r['topology']['target_path'] + 1 for r in rows) / len(rows)}
    manifest = {'schema': 'visual-line-tracing-v1-transfer-aware', 'pillow': PIL.__version__, 'size': [448, 448],
                'renderer': 'unchanged v1 draw_example: 3x supersampling; Pillow bundled default font; deterministic frozen PNGs',
                'v1_generator_sha256': sha(Path(__file__).with_name('line_tracing.py')),
                'split_builder_sha256': sha(__file__), 'level_specifications': LEVELS, 'prompt': PROMPT,
                'split_jsonl_sha256': {s: sha(root / (s + '.jsonl')) for s in records},
                'search_folds_sha256': sha(root / 'search_folds.json'),
                'seed_ranges': {s: [b, b + n - 1] for s, n, b in SPLITS},
                'seed_and_image_disjoint_across_splits': True,
                'seed_and_image_disjoint_from_previous_datasets': list(PREVIOUS_DATASETS),
                'splits': {s: stats(v) for s, v in records.items()},
                'folds': {f: stats([r for r in records['search'] if fold_of[r['id']] == f]) for f in FOLDS},
                'source_outcomes_used': False}
    (root / 'manifest.json').write_bytes((json.dumps(manifest, indent=2) + '\n').encode())
    (root / '.gitattributes').write_bytes(b'* -text\n')
    return manifest


def load(root, split):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_bytes())
    path = root / (split + '.jsonl')
    if sha(path) != manifest['split_jsonl_sha256'][split]:
        raise ValueError('dataset split changed')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        if sha(root / row['image']) != row['image_sha256']:
            raise ValueError('image changed')
    return rows


def load_folds(root):
    root = Path(root)
    path = root / 'search_folds.json'
    if sha(path) != json.loads((root / 'manifest.json').read_bytes())['search_folds_sha256']:
        raise ValueError('search folds changed')
    return json.loads(path.read_bytes())

"""Easier, separately frozen line tracing; preserve the original v1 generator.

The renderer is retained from v1 with row-local style lookup. Its equivalence on
v1 metadata is tested, so the visual change is the reduced crossing geometry.
"""
from collections import Counter
from functools import lru_cache
from itertools import product
import json
import math
from pathlib import Path
import random

from .line_tracing import LEVELS as V1_LEVELS, PROMPT, sha

LEVELS = {
    level: {**V1_LEVELS[level], 'stages': stages, 'minimum_target_crossings': minimum}
    for level, stages, minimum in [('easy', [1], 1), ('medium', [2, 3], 2), ('hard', [4, 5], 3)]
}


def style(level, stages):
    if stages not in LEVELS[level]['stages']:
        raise ValueError('swap count outside the frozen difficulty range')
    return {**LEVELS[level], 'stages': stages}


@lru_cache(maxsize=None)
def feasible_cells(level, stages):
    """Enumerate reachable start/end pairs; one swap cannot cover all 16 pairs."""
    spec = style(level, stages)
    cells = set()
    for choices in product(range(3), repeat=stages):
        lanes, visits = list(range(4)), [0] * 4
        for lane in choices:
            a, b = lanes[lane:lane + 2]
            visits[a] += 1
            visits[b] += 1
            lanes[lane:lane + 2] = [b, a]
        cells.update((start, lanes.index(start)) for start in range(4)
                     if visits[start] >= spec['minimum_target_crossings'])
    return tuple(sorted(cells))


def topology(seed, level, start_lane, end_lane, stages):
    spec = style(level, stages)
    if (start_lane, end_lane) not in feasible_cells(level, stages):
        raise ValueError('unreachable start/end pair at this difficulty')
    rng = random.Random(seed)
    for attempt in range(100000):
        lanes, swaps, visits = list(range(4)), [], [0] * 4
        for _ in range(stages):
            lane = rng.randrange(3)
            a, b = lanes[lane:lane + 2]
            visits[a] += 1
            visits[b] += 1
            swaps.append({'lane': lane, 'crossing_paths': [a, b], 'over_path': rng.choice([a, b])})
            lanes[lane], lanes[lane + 1] = b, a
        if lanes[end_lane] == start_lane and visits[start_lane] >= spec['minimum_target_crossings']:
            break
    else:
        raise RuntimeError('could not construct feasible topology')
    return {'stages': swaps, 'final_lane_to_path': lanes, 'target_path': start_lane,
            'target_end_lane': end_lane, 'target_crossings': visits[start_lane],
            'path_crossings': visits, 'rejection_attempts': attempt,
            'boundary_offsets': [rng.uniform(-5, 5) for _ in range(stages + 1)],
            'stage_wiggles': [rng.uniform(-spec['wiggle'], spec['wiggle']) for _ in swaps]}


def draw_example(meta):
    from PIL import Image, ImageDraw, ImageFont
    spec, graph = meta['style'], meta['topology']
    scale, width = 3, 3.2
    image = Image.new('RGB', (448 * scale, 448 * scale), 'white')
    draw = ImageDraw.Draw(image)
    x0, x1 = 32, 396
    ys = [224 + (i - 1.5) * spec['spacing'] for i in range(4)]
    lanes = list(range(4))
    bridges, polylines = [], [[] for _ in range(4)]
    for j, stage in enumerate(graph['stages']):
        updated = lanes.copy()
        lane = stage['lane']
        updated[lane], updated[lane + 1] = updated[lane + 1], updated[lane]
        curves = {}
        for path in range(4):
            y0, y1 = ys[lanes.index(path)], ys[updated.index(path)]
            points = []
            for k in range(81):
                t = k / 80
                smooth = 3*t*t - 2*t*t*t
                y = y0 + (y1-y0)*smooth + graph['boundary_offsets'][j]*(1-smooth) + graph['boundary_offsets'][j+1]*smooth
                y += graph['stage_wiggles'][j] * math.sin(math.pi*t)**2
                x = x0 + (x1-x0)*(j+t)/spec['stages']
                points.append((x*scale, y*scale))
            curves[path] = points
            polylines[path].extend(points)
        bridges.append(curves[stage['over_path']][34:47])
        lanes = updated
    for points in polylines:
        draw.line(points, fill='#202020', width=round(width*scale), joint='curve')
    for points in bridges:
        draw.line(points, fill='white', width=round((width+5)*scale), joint='curve')
        draw.line(points, fill='#202020', width=round(width*scale), joint='curve')
    for lane, y in enumerate(ys):
        sy, ey = y + graph['boundary_offsets'][0], y + graph['boundary_offsets'][-1]
        radius = 7 if lane == graph['target_path'] else 4
        color = '#dc2020' if lane == graph['target_path'] else '#202020'
        draw.ellipse(((x0-radius)*scale,(sy-radius)*scale,(x0+radius)*scale,(sy+radius)*scale), fill=color)
        draw.ellipse(((x1-4)*scale,(ey-4)*scale,(x1+4)*scale,(ey+4)*scale), fill='#202020')
        draw.text((420*scale,ey*scale),str(lane+1),font=ImageFont.load_default(size=26*scale),anchor='mm',fill='black')
    return image.resize((448,448), Image.Resampling.LANCZOS)


def shortcut_prediction(row, use_start=False):
    """Topology-prior prediction fixed without fitting either split's answers."""
    graph = row['topology']
    cells = feasible_cells(row['difficulty'], row['style']['stages'])
    counts = Counter(end for start, end in cells if not use_start or start == graph['target_path'])
    return str(min(counts, key=lambda end: (-counts[end], end)) + 1)


def freeze_dataset(root):
    import PIL
    if PIL.__version__ != '12.3.0':
        raise ValueError('use the v1 renderer version, Pillow 12.3.0')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    (root / 'images').mkdir()
    records = {}
    for split, count, seed_base in (('selection', 150, 1100000), ('heldout', 500, 2200000)):
        rows = []
        for i in range(count):
            level = ('easy', 'medium', 'hard')[i % 3]
            j = i // 3
            choices = LEVELS[level]['stages'][:]
            random.Random(seed_base + 998000 + i % 3).shuffle(choices)
            stages = choices[j % len(choices)]
            # Cycle balanced feasible cells independently for each swap count.
            cells = list(feasible_cells(level, stages))
            random.Random(seed_base + 999000 + i % 3 + 100*stages).shuffle(cells)
            start, end = cells[(j // len(choices)) % len(cells)]
            seed = seed_base + i
            row = {'id': f'{split}-{i:04d}', 'split': split, 'seed': seed, 'difficulty': level,
                   'answer': str(end+1), 'prompt': PROMPT, 'image': f'images/{split}-{i:04d}.png',
                   'width': 448, 'height': 448, 'paths': 4, 'style': style(level, stages),
                   'topology': topology(seed, level, start, end, stages)}
            draw_example(row).save(root / row['image'], optimize=False)
            row['image_sha256'] = sha(root / row['image'])
            rows.append(row)
        records[split] = rows
        (root / (split+'.jsonl')).write_bytes(('\n'.join(json.dumps(r, sort_keys=True) for r in rows)+'\n').encode())
    a, b = records.values()
    assert not {r['seed'] for r in a} & {r['seed'] for r in b}
    assert len({r['image_sha256'] for r in a+b}) == 650
    manifest = {
        'schema': 'visual-line-tracing-v2', 'pillow': PIL.__version__, 'size': [448, 448],
        'renderer': 'Unchanged v1 drawing algorithm; row-local swap count; 3x supersampling and Pillow default font',
        'generator_sha256': sha(__file__), 'shared_v1_module_sha256': sha(Path(__file__).with_name('line_tracing.py')),
        'level_specifications': LEVELS,
        'split_jsonl_sha256': {s: sha(root/(s+'.jsonl')) for s in records},
        'seed_ranges': {'selection': [1100000,1100149], 'heldout': [2200000,2200499]},
        'v1_seed_pairing': 'Same IDs, seeds, split sizes and difficulty assignment as v1; new feasible pairs, geometry and labels',
        'seed_and_image_disjoint': True, 'counts': {s: len(v) for s, v in records.items()},
        'cell_balance': 'Cycle all feasible start/end pairs separately per split, difficulty and swap count; counts differ by at most one',
        'feasible_cells': {l: {str(n): list(feasible_cells(l,n)) for n in conf['stages']} for l, conf in LEVELS.items()},
        'label_counts': {s: dict(Counter(r['answer'] for r in v)) for s, v in records.items()},
        'swap_counts_by_difficulty': {s: {l: dict(Counter(r['style']['stages'] for r in v if r['difficulty']==l)) for l in LEVELS} for s, v in records.items()},
        'start_end_cells_by_difficulty': {s: {l: dict(Counter(f"{r['topology']['target_path']}->{r['answer']}" for r in v if r['difficulty']==l)) for l in LEVELS} for s, v in records.items()},
        'shortcut_accuracy_by_difficulty': {s: {l: {
            'straight_endpoint': sum(r['answer']==str(r['topology']['target_path']+1) for r in v if r['difficulty']==l)/sum(r['difficulty']==l for r in v),
            'feasible_cell_prior': sum(r['answer']==shortcut_prediction(r) for r in v if r['difficulty']==l)/sum(r['difficulty']==l for r in v),
            'feasible_cell_prior_given_start': sum(r['answer']==shortcut_prediction(r,True) for r in v if r['difficulty']==l)/sum(r['difficulty']==l for r in v)
        } for l in LEVELS} for s, v in records.items()},
        'revision_reason': 'User-requested difficulty-only v2 after the completed v1 scale-failure study',
        'per_example_model_outcomes_used': False, 'v2_model_outcomes_used': False
    }
    (root/'manifest.json').write_bytes((json.dumps(manifest, indent=2)+'\n').encode())
    (root/'.gitattributes').write_bytes(b'* -text\n')
    return manifest

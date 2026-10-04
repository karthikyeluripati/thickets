"""Deterministic, topology-labelled cable tracing; no model outputs enter generation."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re

PROMPT = ('Follow the black cable from the red starting dot on the left to its numbered '
          'endpoint on the right. At a crossing, cables pass over or under each other; '
          'they never join. Continue along the same cable. Which endpoint does it reach? '
          'Answer with only one digit: 1, 2, 3, or 4.')
LEVELS = {'easy': {'stages': 4, 'spacing': 92, 'wiggle': 3, 'minimum_target_crossings': 1},
          'medium': {'stages': 7, 'spacing': 78, 'wiggle': 7, 'minimum_target_crossings': 3},
          'hard': {'stages': 10, 'spacing': 66, 'wiggle': 10, 'minimum_target_crossings': 5}}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_answer(text):
    match = re.fullmatch(r'\s*([1-4])[.!]?\s*', text)
    return match.group(1) if match else ''


def vote(answers):
    valid = Counter(a for a in answers if a in ('1', '2', '3', '4'))
    ranked = valid.most_common()
    first = ranked[0] if ranked else ('', 0)
    runner = ranked[1][1] if len(ranked) > 1 else 0
    return {'answer': first[0], 'counts': [[a, n] for a, n in ranked],
            'margin': first[1] - runner, 'valid_votes': sum(valid.values()),
            'tied_winners': sum(n == first[1] for a, n in ranked)}


def topology(seed, level, start_lane, end_lane):
    """Condition only on balanced start/end cells and declared geometric difficulty."""
    rng, spec = random.Random(seed), LEVELS[level]
    for attempt in range(100000):
        lanes, stages, visits = list(range(4)), [], [0] * 4
        for _ in range(spec['stages']):
            lane = rng.randrange(3)
            a, b = lanes[lane:lane + 2]
            visits[a] += 1
            visits[b] += 1
            stages.append({'lane': lane, 'crossing_paths': [a, b], 'over_path': rng.choice([a, b])})
            lanes[lane], lanes[lane + 1] = b, a
        if lanes[end_lane] == start_lane and visits[start_lane] >= spec['minimum_target_crossings']:
            break
    else:
        raise RuntimeError('could not construct balanced topology')
    return {'stages': stages, 'final_lane_to_path': lanes, 'target_path': start_lane,
            'target_end_lane': end_lane, 'target_crossings': visits[start_lane],
            'path_crossings': visits, 'rejection_attempts': attempt,
            'boundary_offsets': [rng.uniform(-5, 5) for _ in range(spec['stages'] + 1)],
            'stage_wiggles': [rng.uniform(-spec['wiggle'], spec['wiggle']) for _ in stages]}


def draw_example(meta):
    from PIL import Image, ImageDraw, ImageFont
    spec, graph = LEVELS[meta['difficulty']], meta['topology']
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


def freeze_dataset(root):
    import PIL
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    (root / 'images').mkdir()
    records = {}
    for split, count, seed_base in (('selection',150,1100000),('heldout',500,2200000)):
        rows = []
        for i in range(count):
            level = ('easy','medium','hard')[i % 3]
            j = i // 3
            # Balance start/end joint cells within each level, with a seed-based
            # permutation of cell order so no metadata ordering predicts labels.
            cells = list(range(16))
            random.Random(seed_base + 999000 + i % 3).shuffle(cells)
            cell = cells[j % 16]
            start, end = divmod(cell, 4)
            seed = seed_base + i
            row = {'id': f'{split}-{i:04d}', 'split': split, 'seed': seed, 'difficulty': level,
                   'answer': str(end+1), 'prompt': PROMPT, 'image': f'images/{split}-{i:04d}.png',
                   'width':448, 'height':448, 'paths':4, 'style':dict(LEVELS[level]),
                   'topology':topology(seed,level,start,end)}
            draw_example(row).save(root / row['image'], optimize=False)
            row['image_sha256'] = sha(root / row['image'])
            rows.append(row)
        records[split] = rows
        (root / (split+'.jsonl')).write_bytes(('\n'.join(json.dumps(r,sort_keys=True) for r in rows)+'\n').encode())
    a, b = records.values()
    assert not {r['seed'] for r in a} & {r['seed'] for r in b}
    assert not {r['image_sha256'] for r in a} & {r['image_sha256'] for r in b}
    assert len({r['image_sha256'] for r in a+b}) == 650
    manifest = {'schema':'visual-line-tracing-v1', 'pillow':PIL.__version__, 'size':[448,448],
                'renderer':'3x supersampling; Pillow bundled default font; deterministic frozen PNGs',
                'generator_sha256':sha(__file__), 'level_specifications':LEVELS,
                'split_jsonl_sha256':{s:sha(root/(s+'.jsonl')) for s in records},
                'seed_ranges':{'selection':[1100000,1100149],'heldout':[2200000,2200499]},
                'seed_and_image_disjoint':True, 'counts':{s:len(v) for s,v in records.items()},
                'label_counts':{s:dict(Counter(r['answer'] for r in v)) for s,v in records.items()},
                'start_end_cells_by_difficulty':{s:{l:dict(Counter(f"{r['topology']['target_path']}->{r['answer']}" for r in v if r['difficulty']==l)) for l in LEVELS} for s,v in records.items()},
                'straight_endpoint_heuristic_accuracy':{s:sum(int(r['answer'])==r['topology']['target_path']+1 for r in v)/len(v) for s,v in records.items()},
                'source_outcomes_used':False}
    (root/'manifest.json').write_bytes((json.dumps(manifest,indent=2)+'\n').encode())
    (root/'.gitattributes').write_bytes(b'* -text\n')
    return manifest


def load_split(root, split):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_bytes())
    path = root/(split+'.jsonl')
    if sha(path) != manifest['split_jsonl_sha256'][split]:
        raise ValueError('dataset split changed')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        if sha(root/row['image']) != row['image_sha256']:
            raise ValueError('image changed')
    return rows

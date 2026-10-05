"""OmniSpatial Perspective-Taking SEARCH200 / VALIDATION200 / official TEST.

Uses the pinned OmniSpatial sources, official QA/options/answers and the
official direct prompt/scorer from omnispatial_data. Only the split rule is new.
"""
from collections import Counter
import json
from pathlib import Path
import random

from .omnispatial_data import (DATASET, DATASET_REVISION, OFFICIAL_BLOBS, REPO, REPO_COMMIT, ZIPS, image_member, sha256)

TASK = 'Perspective_Taking'
SPLIT_SEED = 20261005
SEARCH_N = VALIDATION_N = 200


def largest_remainder(weights, total):
    keys = sorted(weights)
    s = sum(weights.values())
    raw = {k: weights[k] * total / s for k in keys}
    out = {k: int(raw[k]) for k in keys}
    for k in sorted(keys, key=lambda k: (-(raw[k] - out[k]), k))[:total - sum(out.values())]:
        out[k] += 1
    return out


def targets(train):
    """Per (sub_task_type, answer) counts for a 200-item split: sub-task shares follow
    the official train distribution; answer positions follow it within each sub-task."""
    subs = largest_remainder(Counter(r['sub_task_type'] for r in train), SEARCH_N)
    cells = {}
    for s, n in subs.items():
        for a, k in largest_remainder(Counter(r['answer'] for r in train if r['sub_task_type'] == s), n).items():
            cells[(s, a)] = k
    return cells


def select(train, test_hashes):
    """One QA pair per image; no image (by sha256) shared between SEARCH, VALIDATION
    or the official TEST. Within each (sub_task, answer) cell, QA pairs are visited in
    a seeded random order; the first eligible pairs fill SEARCH, the next VALIDATION.
    Remaining train records are left unused."""
    rng = random.Random(SPLIT_SEED)
    order = sorted(train, key=lambda r: r['uid'])
    rng.shuffle(order)
    used, chosen = set(test_hashes), {'search': [], 'validation': []}
    cells = targets(train)
    for cell in sorted(cells):
        pool = [r for r in order if (r['sub_task_type'], r['answer']) == cell]
        for split in ('search', 'validation'):
            need = cells[cell]
            for r in pool:
                if need == 0:
                    break
                if r['image_sha256'] in used:
                    continue
                used.add(r['image_sha256'])
                chosen[split].append(r)
                need -= 1
            if need:
                raise ValueError(f'cannot fill {split} cell {cell}')
    return chosen, cells


def describe(rows):
    return {'n': len(rows), 'images': len({r['image_sha256'] for r in rows}),
            'sub_task_type': dict(sorted(Counter(r['sub_task_type'] for r in rows).items())),
            'answer_letter': dict(sorted(Counter(chr(65 + r['answer']) for r in rows).items())),
            'answer_by_sub_task': {s: dict(sorted(Counter(chr(65 + r['answer']) for r in rows if r['sub_task_type'] == s).items()))
                                   for s in sorted({r['sub_task_type'] for r in rows})},
            'option_count': dict(sorted(Counter(len(r['options']) for r in rows).items())),
            'image_pixels': {'median': sorted(r['image_size'][0] * r['image_size'][1] for r in rows)[len(rows) // 2],
                             'max': max(r['image_size'][0] * r['image_size'][1] for r in rows)}}


def freeze(source, out, data_json_sha256, image_meta):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    rows = {}
    for split in ('train', 'test'):
        raw = (Path(source) / f'{split}-data.json').read_bytes()
        if sha256(raw) != data_json_sha256[split]:
            raise ValueError('official data.json changed: ' + split)
        recs = [r for r in json.loads(raw) if r['task_type'] == TASK]
        for r in recs:
            member = image_member(split, r)
            meta = image_meta[f"{split}/{member.rsplit('/', 1)[1]}"]
            r.update(uid=f"{split}:{r['id']}", source_split=split, image_member=member, image_sha256=meta['sha256'],
                     image_size=meta['size'], image_mode=meta['mode'])
        if len({r['uid'] for r in recs}) != len(recs):
            raise ValueError('duplicate ids')
        rows[split] = recs
    test_hashes = {r['image_sha256'] for r in rows['test']}
    chosen, cells = select(rows['train'], test_hashes)
    splits = {'search': chosen['search'], 'validation': chosen['validation'], 'test': rows['test']}
    for name, items in splits.items():
        items.sort(key=lambda r: r['uid'])
        (out / f'{name}.jsonl').write_bytes(('\n'.join(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in items) + '\n').encode())
    h = {s: {r['image_sha256'] for r in v} for s, v in splits.items()}
    if h['search'] & h['validation'] or (h['search'] | h['validation']) & h['test'] or len(h['search']) != 200 or len(h['validation']) != 200:
        raise ValueError('image overlap or non-unique images in SEARCH/VALIDATION')
    per_file = {r['image_member']: r['image_sha256'] for r in rows['train']}
    identical_files = sum(n - 1 for n in Counter(per_file.values()).values())
    multi_question_images = sum(n > 1 for n in Counter(r['image_member'] for r in rows['train']).values())
    manifest = {'schema': 'omnispatial-perspective-taking-v1', 'repository': REPO, 'repository_commit': REPO_COMMIT,
                'official_blobs': OFFICIAL_BLOBS, 'dataset': DATASET, 'dataset_revision': DATASET_REVISION,
                'zip_lfs_sha256': {s: {'file': f, 'sha256': hsh} for s, (f, hsh) in ZIPS.items()},
                'data_json_sha256': data_json_sha256, 'task_type': TASK,
                'official_counts': {s: describe(rows[s]) for s in rows},
                'official_train_questions_per_image': dict(sorted(Counter(Counter(r['image_sha256'] for r in rows['train']).values()).items())),
                'official_train_images_with_multiple_questions': multi_question_images,
                'official_train_byte_identical_extra_image_files': identical_files,
                'official_train_images_identical_to_test': len({r['image_sha256'] for r in rows['train']} & test_hashes),
                'splits': {s: describe(v) for s, v in splits.items()},
                'cell_targets': {f'{s}|{chr(65 + a)}': n for (s, a), n in sorted(cells.items())},
                'split_rule': select.__doc__.strip() + ' ' + targets.__doc__.strip(), 'split_seed': SPLIT_SEED,
                'unused_train_records': len(rows['train']) - 400,
                'split_jsonl_sha256': {s: sha256((out / f'{s}.jsonl').read_bytes()) for s in splits},
                'source_metadata': 'OmniSpatial records carry no image-source field; stratification uses sub_task_type and answer position only.',
                'images_committed': False, 'model_outputs_used': False}
    (out / 'manifest.json').write_bytes((json.dumps(manifest, indent=2, ensure_ascii=False) + '\n').encode())
    (out / '.gitattributes').write_bytes(b'* -text\n')
    return manifest

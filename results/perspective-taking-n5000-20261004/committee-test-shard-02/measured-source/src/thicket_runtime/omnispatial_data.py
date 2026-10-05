"""Official OmniSpatial Complex Spatial Logic data, prompt and direct scorer.

The prompt and scorer reproduce the official deterministic multiple-choice path
of vlms_eval/qwenvl_eval.py at OmniSpatial commit 208bac2 with
--prompt_type none --eval_type direct. The prompt strings are the vendored,
blob-verified official system_prompts.py. Images go through the official
qwen_vl_utils 0.0.8 fetch_image (RGB conversion + smart_resize) on the GPU host.
"""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import random

REPO = 'https://github.com/qizekun/OmniSpatial'
REPO_COMMIT = '208bac258d8671c6857b3be2834997936ba7fc51'
OFFICIAL_BLOBS = {'vlms_eval/system_prompts.py': '38fa3ebb87f4b02559d0d763e32d4fea8482796d',
                  'vlms_eval/qwenvl_eval.py': 'b25906caaf9c2d4ffef8d20a0052870ce147e678'}
DATASET = 'qizekun/OmniSpatial'
DATASET_REVISION = '6691f3288bb1ff207d6ead4d841b505de08a6fd8'
ZIPS = {'train': ('OmniSpatial-train.zip', '6f3878c1cd839e4395dcd70c313228782e67a5d018488071890622aa88f1e198'),
        'test': ('OmniSpatial-test.zip', '6c2cc57a62e7ef6219c26593017e18a9fbc00b82fd2e86f31c01f42e8b97c988')}
TASK = 'Complex_Logic'
SPLIT_SEED = 20261004
VALIDATION_FRACTION = 0.4
PROMPTS_PATH = Path(__file__).resolve().parents[2] / 'third_party/omnispatial/system_prompts.py'


def blob(data):
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def official_prompts():
    data = PROMPTS_PATH.read_bytes()
    if blob(data) != OFFICIAL_BLOBS['vlms_eval/system_prompts.py']:
        raise ValueError('vendored official system_prompts.py changed')
    spec = importlib.util.spec_from_file_location('_omnispatial_system_prompts', PROMPTS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_prompt(record):
    """qwenvl_eval.py: SYS_PROMPTS['none'] + '\\n' + FORMAT_PROMPTS['direct'] + '\\n\\n' + question + options."""
    p = official_prompts()
    prompt = p.SYS_PROMPTS['none'] + '\n' + p.FORMAT_PROMPTS['direct'] + '\n\n' + record['question']
    for i, option in enumerate(record['options']):
        prompt += f'\n{chr(65 + i)}. {option}'
    return prompt


def score(text, answer):
    """Official direct scorer: first character of the stripped, upper-cased response."""
    pred = text.strip().upper()[:1]
    return {'prediction': pred, 'correct': pred == chr(65 + answer),
            'valid_letter': pred in ('A', 'B', 'C', 'D')}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def image_member(split, record):
    """qwenvl_eval.py: os.path.join(dataset, task_type, f"{id.split('_')[0]}.png")."""
    return f"OmniSpatial-{split}/{record['task_type']}/{record['id'].split('_')[0]}.png"


def stratified_split(records):
    """60/40 search/validation over identical-image groups (byte-identical PNGs never
    straddle splits). Each group is represented by its smallest uid; strata =
    (sub_task_type, answer) of that representative. Inside a stratum, groups are
    shuffled with the committed seed, then ordered by question text (spreads
    templates), and every k with floor((k+1)*0.4) > floor(k*0.4) goes to validation
    (systematic 40% sampling). All group members inherit the assignment."""
    rng = random.Random(SPLIT_SEED)
    groups = {}
    for r in sorted(records, key=lambda r: r['uid']):
        groups.setdefault(r['image_sha256'], []).append(r)
    strata = {}
    for members in sorted(groups.values(), key=lambda g: g[0]['uid']):
        rep = members[0]
        strata.setdefault((rep['sub_task_type'], rep['answer']), []).append(members)
    membership = {}
    for key in sorted(strata):
        items = strata[key]
        rng.shuffle(items)
        items.sort(key=lambda g: g[0]['question'])  # stable: shuffled order kept inside each template
        for k, members in enumerate(items):
            side = 'validation' if int((k + 1) * VALIDATION_FRACTION) > int(k * VALIDATION_FRACTION) else 'search'
            for r in members:
                membership[r['uid']] = side
    return membership


def describe(rows):
    return {'n': len(rows), 'sub_task_type': dict(sorted(Counter(r['sub_task_type'] for r in rows).items())),
            'answer_letter': dict(sorted(Counter(chr(65 + r['answer']) for r in rows).items())),
            'answer_by_sub_task': {s: dict(sorted(Counter(chr(65 + r['answer']) for r in rows if r['sub_task_type'] == s).items()))
                                   for s in sorted({r['sub_task_type'] for r in rows})},
            'option_count': dict(sorted(Counter(len(r['options']) for r in rows).items())),
            'distinct_question_texts': len({r['question'] for r in rows}),
            'visual_tokens_at_official_resolution': {'min': min(r['visual_tokens'] for r in rows),
                                                     'median': sorted(r['visual_tokens'] for r in rows)[len(rows) // 2],
                                                     'max': max(r['visual_tokens'] for r in rows),
                                                     'sum': sum(r['visual_tokens'] for r in rows)},
            'media': 'single PNG image per QA pair; no video'}


def freeze(source, out, data_json_sha256, image_meta):
    """source/{split}-data.json are the official data.json files; image_meta maps
    '{split}/{n}.png' -> {sha256, size, mode, tokens} for every Complex_Logic image."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    rows = {}
    for split in ('train', 'test'):
        raw = (Path(source) / f'{split}-data.json').read_bytes()
        if sha256(raw) != data_json_sha256[split]:
            raise ValueError('official data.json changed: ' + split)
        cl = [r for r in json.loads(raw) if r['task_type'] == TASK]
        for r in cl:
            member = image_member(split, r)
            meta = image_meta[f"{split}/{member.rsplit('/', 1)[1]}"]
            r.update(uid=f"{split}:{r['id']}", source_split=split, image_member=member, image_sha256=meta['sha256'],
                     image_size=meta['size'], image_mode=meta['mode'], visual_tokens=meta['tokens'])
        if len({r['uid'] for r in cl}) != len(cl):
            raise ValueError('duplicate ids')
        rows[split] = cl
    membership = stratified_split(rows['train'])
    splits = {'search': [r for r in rows['train'] if membership[r['uid']] == 'search'],
              'validation': [r for r in rows['train'] if membership[r['uid']] == 'validation'],
              'test': rows['test']}
    hashes = {name: {r['image_sha256'] for r in items} for name, items in splits.items()}
    if hashes['search'] & hashes['validation'] or (hashes['search'] | hashes['validation']) & hashes['test']:
        raise ValueError('identical image shared across splits')
    for name, items in splits.items():
        items.sort(key=lambda r: r['uid'])
        (out / f'{name}.jsonl').write_bytes(('\n'.join(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in items) + '\n').encode())
    manifest = {'schema': 'omnispatial-complex-logic-v1', 'repository': REPO, 'repository_commit': REPO_COMMIT,
                'official_blobs': OFFICIAL_BLOBS, 'dataset': DATASET, 'dataset_revision': DATASET_REVISION,
                'zip_lfs_sha256': {s: {'file': f, 'sha256': h} for s, (f, h) in ZIPS.items()},
                'data_json_sha256': data_json_sha256, 'task_type': TASK,
                'official_counts': {s: describe(rows[s]) for s in rows},
                'splits': {s: describe(v) for s, v in splits.items()},
                'split_rule': stratified_split.__doc__.strip(), 'split_seed': SPLIT_SEED,
                'split_jsonl_sha256': {s: sha256((out / f'{s}.jsonl').read_bytes()) for s in splits},
                'image_sha256_count': sum(len(v) for v in splits.values()),
                'duplicate_image_groups': {s: sum(n > 1 for n in Counter(r['image_sha256'] for r in v).values()) for s, v in splits.items()},
                'audit_findings': ['Official train Complex_Logic contains only Geometric_Reasoning (no Pattern_Recognition).',
                                   'Official test Complex_Logic has 229/252 QA pairs with options drawn inside the image (empty options list); 2 have two options.',
                                   'Official train contains byte-identical images under different ids; identical-image groups are kept within one split.',
                                   'Train questions come from a small set of templated puzzle texts; test questions are IQ-test style. No test question text occurs in train.'],
                'images_committed': False, 'images_source': 'pinned zips; every image verified by sha256 before use',
                'model_outputs_used': False}
    (out / 'manifest.json').write_bytes((json.dumps(manifest, indent=2, ensure_ascii=False) + '\n').encode())
    (out / '.gitattributes').write_bytes(b'* -text\n')
    return manifest


def load(root, split):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_bytes())
    path = root / f'{split}.jsonl'
    if sha256(path.read_bytes()) != manifest['split_jsonl_sha256'][split]:
        raise ValueError('frozen split changed: ' + split)
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]

"""Freeze fresh official test prompts, excluding all previous example inputs."""
import hashlib
import io
import json
from pathlib import Path
import re
import urllib.request

from freeze_gsm8k import REVISION, SUFFIX


def main():
    import pyarrow.parquet as pq
    root = Path('examples/complementarity_gsm8k')
    if root.exists():
        raise FileExistsError(root)
    previous, inputs = set(), {}
    for path in sorted(Path('examples').rglob('*.jsonl')):
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for line in path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            if 'prompt' in row:
                previous.add(row['prompt'])
    url = f'https://huggingface.co/datasets/openai/gsm8k/resolve/{REVISION}/main/test-00000-of-00001.parquet'
    raw = urllib.request.urlopen(url, timeout=120).read()
    if hashlib.sha256(raw).hexdigest() != 'ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59':
        raise ValueError('pinned official test parquet changed')
    source = pq.read_table(io.BytesIO(raw)).to_pylist()
    records, excluded = [], []
    for i in range(40, len(source)):
        row = source[i]
        prompt = row['question'] + ' ' + SUFFIX + '\n'
        if prompt in previous:
            excluded.append(i)
            continue
        records.append({'id': f'gsm8k-main-test-{i:04d}', 'split': 'test', 'source_index': i,
                        'prompt': prompt, 'answer': re.search(r'#### (\-?[0-9\.\,]+)', row['answer']).group(1).replace(',', ''),
                        'reference_solution': row['answer']})
        if len(records) == 300:
            break
    if len(records) != 300:
        raise ValueError('fewer than 300 fresh official test prompts; do not silently change the gate')
    data = ('\n'.join(json.dumps(r, ensure_ascii=False) for r in records) + '\n').encode()
    meta = {'dataset': 'openai/gsm8k', 'revision': REVISION, 'license': 'MIT', 'split': 'main/test',
            'source_url': url, 'source_rows': len(source), 'source_parquet_sha256': hashlib.sha256(raw).hexdigest(),
            'indices': [r['source_index'] for r in records], 'count': len(records),
            'excluded_duplicate_prompt_indices': excluded, 'previous_inputs_sha256': inputs,
            'frozen_jsonl_sha256': hashlib.sha256(data).hexdigest(),
            'selection_rule': 'first 300 fresh formatted prompts at test indices >=40; no outcome filtering',
            'upstream_commit': '4000d34fb5b69a3121cf1d2c564aa0be5a6a41ca',
            'format': 'same pinned upstream preprocessing and non-instruct format_prompt as adaptive study'}
    root.mkdir()
    (root / 'heldout300.jsonl').write_bytes(data)
    (root / 'provenance.json').write_bytes((json.dumps(meta, indent=2) + '\n').encode())
    (root / '.gitattributes').write_bytes(b'* -text\n')
    print(json.dumps({'count': len(records), 'first': records[0]['source_index'], 'last': records[-1]['source_index'],
                      'excluded': excluded, 'sha256': meta['frozen_jsonl_sha256']}))


if __name__ == '__main__':
    main()

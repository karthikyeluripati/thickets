"""Record dataset-only v2 validation and preservation of all prior research."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

from thicket_runtime.cli import revision_info
from thicket_runtime.line_tracing import load_split, sha

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--out',type=Path,default=Path('results/visual-line-tracing-v2-20261003'))
p.add_argument('--reviewed-image',action='append',required=True)
a = p.parse_args()
if (a.out/'validation.json').exists() or (a.out/'sha256.json').exists():
    raise FileExistsError('validation artifacts are immutable')
dataset = Path('examples/visual-line-tracing-v2')
manifest = json.loads((dataset/'manifest.json').read_bytes())
rows = {s:load_split(dataset,s) for s in ('selection','heldout')}
assert [len(v) for v in rows.values()] == [150,500]
assert not {r['seed'] for r in rows['selection']} & {r['seed'] for r in rows['heldout']}
all_rows = rows['selection']+rows['heldout']
assert len({r['image_sha256'] for r in all_rows}) == 650
assert sha('src/thicket_runtime/line_tracing_v2.py') == manifest['generator_sha256']
assert sha('src/thicket_runtime/line_tracing.py') == manifest['shared_v1_module_sha256']
old_rows = sum([load_split('examples/visual-line-tracing-v1',s) for s in rows],[])
assert not {r['image_sha256'] for r in all_rows} & {r['image_sha256'] for r in old_rows}
reviewed = [r for r in all_rows if r['image'] in a.reviewed_image]
assert len(reviewed) == len(a.reviewed_image)
assert {r['style']['stages'] for r in reviewed} == {1,2,3,4,5}
old = json.loads(Path('experiments/visual_line_tracing_protocol.json').read_bytes())
new = json.loads(Path('experiments/visual_line_tracing_v2_protocol.json').read_bytes())
protocol_changes = {k for k in old.keys() | new.keys() if old.get(k)!=new.get(k)}
assert protocol_changes == {'schema','parent_commit','dataset'}
protected = ['src/thicket_runtime/line_tracing.py','examples/visual-line-tracing-v1',
             'experiments/visual_line_tracing_protocol.json','docs/VISUAL_LINE_TRACING_PROTOCOL.md',
             'docs/VISUAL_THICKETS_LINE_TRACING_2026-10-03.md']
verified = {}
for index in sorted(Path('results').glob('*/sha256.json')):
    if index.parent.resolve() == a.out.resolve():
        continue
    entries = json.loads(index.read_bytes())
    for name, expected in entries.items():
        assert sha(index.parent/name) == expected, str(index.parent/name)
    verified[index.parent.as_posix()] = len(entries)
    protected.append(index.parent.as_posix())
parent = 'd037be5542530466d23b20c037166c4338543cab'
assert not subprocess.check_output(['git','diff','--name-only',parent,'--',*protected]).strip()
main = subprocess.check_output(['git','rev-parse','main'],text=True).strip()
assert main == 'deb414253139cc2559d19cdfe7e6b4786e7c40db'
sources = ['src/thicket_runtime/line_tracing_v2.py','scripts/freeze_line_tracing.py',
           'scripts/validate_line_tracing_v2.py','tests/test_line_tracing_v2.py',
           'experiments/visual_line_tracing_v2_protocol.json']
result = {'status':'dataset_validated_no_model_evaluation','source':revision_info(),
          'command':[sys.executable,*sys.argv],
          'freeze_command':'PYTHONPATH=src python scripts/freeze_line_tracing.py --version v2 --out examples/visual-line-tracing-v2',
          'test_command':[sys.executable,'-m','pytest','-q'],
          'test_log_sha256':sha(a.out/'cpu-tests.txt'),
          'python':sys.version,'pillow':importlib.metadata.version('Pillow'),
          'dataset_manifest_sha256':sha(dataset/'manifest.json'),
          'source_sha256':{name:sha(name) for name in sources},
          'within_v2_splits_disjoint':True,'v1_v2_image_hashes_disjoint':True,
          'v1_v2_seed_pairing_intentional':True,'images':len(all_rows),
          'visually_reviewed_images':[{'image':r['image'],'sha256':r['image_sha256'],'swaps':r['style']['stages'],'answer':r['answer']} for r in reviewed],
          'protocol_changed_fields':sorted(protocol_changes),'gpu_experiment_run':False,
          'prior_indexed_artifacts_verified':verified,'protected_changes':[], 'main_ref':main}
(a.out/'validation.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
index = {f.relative_to(a.out).as_posix():sha(f) for f in sorted(a.out.rglob('*')) if f.is_file()}
(a.out/'sha256.json').write_bytes((json.dumps(index,indent=2)+'\n').encode())
print({'images':len(all_rows),'prior_artifacts_verified':sum(verified.values()),'model_evaluation_run':False})

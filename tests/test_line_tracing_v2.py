"""Validate easier geometry without changing the frozen v1 task or inference."""
from collections import Counter
import json
from pathlib import Path

from PIL import Image
import pytest

from thicket_runtime.line_tracing import LEVELS as V1_LEVELS, PROMPT, load_split, sha
from thicket_runtime.line_tracing_v2 import LEVELS, draw_example, feasible_cells, freeze_dataset, topology

ROOT = Path('examples/visual-line-tracing-v2')


def test_v2_graph_labels_swap_ranges_balance_and_disjointness():
    splits = {s: load_split(ROOT, s) for s in ('selection', 'heldout')}
    a, b = splits.values()
    assert [len(a), len(b)] == [150, 500]
    assert not {r['seed'] for r in a} & {r['seed'] for r in b}
    assert len({r['image_sha256'] for r in a+b}) == 650
    for split, rows in splits.items():
        old = load_split('examples/visual-line-tracing-v1', split)
        for row, previous in zip(rows, old):
            assert all(row[k] == previous[k] for k in ('id','split','seed','difficulty','prompt','width','height','paths'))
            level, graph, spec = row['difficulty'], row['topology'], row['style']
            assert row['prompt'] == PROMPT
            assert spec['stages'] in LEVELS[level]['stages']
            assert len(graph['stages']) == spec['stages']
            assert spec['spacing'] == V1_LEVELS[level]['spacing']
            assert spec['wiggle'] == V1_LEVELS[level]['wiggle']
            lanes, visits = list(range(4)), [0]*4
            for crossing in graph['stages']:
                k = crossing['lane']
                pair = lanes[k:k+2]
                assert pair == crossing['crossing_paths'] and crossing['over_path'] in pair
                for path in pair:
                    visits[path] += 1
                lanes[k:k+2] = reversed(pair)
            target = graph['target_path']
            assert lanes == graph['final_lane_to_path']
            assert lanes[int(row['answer'])-1] == target
            assert graph['target_end_lane'] == int(row['answer'])-1
            assert visits == graph['path_crossings']
            assert visits[target] == graph['target_crossings'] >= spec['minimum_target_crossings']
        for level, conf in LEVELS.items():
            counts = Counter(r['style']['stages'] for r in rows if r['difficulty'] == level)
            assert set(counts) == set(conf['stages'])
            assert max(counts.values())-min(counts.values()) <= 1
            for n in conf['stages']:
                cells = Counter((r['topology']['target_path'],r['topology']['target_end_lane'])
                                for r in rows if r['difficulty']==level and r['style']['stages']==n)
                assert set(cells) == set(feasible_cells(level,n))
                assert max(cells.values())-min(cells.values()) <= 1


def test_v2_topology_and_all_swap_count_renders_reproduce():
    rows = load_split(ROOT,'selection') + load_split(ROOT,'heldout')
    seen = set()
    for row in rows:
        g, n = row['topology'], row['style']['stages']
        assert topology(row['seed'],row['difficulty'],g['target_path'],g['target_end_lane'],n) == g
        if n not in seen:
            with Image.open(ROOT/row['image']) as saved:
                assert draw_example(row).tobytes() == saved.convert('RGB').tobytes()
            seen.add(n)
    assert seen == {1,2,3,4,5}


def test_v2_renderer_is_identical_on_v1_geometry_and_v1_source_is_preserved():
    from thicket_runtime import line_tracing
    v1 = Path('examples/visual-line-tracing-v1')
    manifest = json.loads((v1/'manifest.json').read_bytes())
    assert sha(line_tracing.__file__) == manifest['generator_sha256']
    for row in load_split(v1,'selection')[:3]:
        with Image.open(v1/row['image']) as saved:
            assert draw_example(row).tobytes() == saved.convert('RGB').tobytes()


def test_v2_does_not_change_inference_or_scoring_protocol():
    old = json.loads(Path('experiments/visual_line_tracing_protocol.json').read_bytes())
    new = json.loads(Path('experiments/visual_line_tracing_v2_protocol.json').read_bytes())
    assert new['dataset'] == ROOT.as_posix()
    assert new['schema'] == 'visual-line-tracing-feasibility-v2'
    for field in ('schema','parent_commit','dataset'):
        old.pop(field)
        new.pop(field)
    assert old == new


def test_v2_rejects_impossible_geometry_and_existing_output(tmp_path):
    assert feasible_cells('easy',1) == ((0,1),(1,0),(1,2),(2,1),(2,3),(3,2))
    with pytest.raises(ValueError,match='unreachable'):
        topology(1,'easy',0,3,1)
    with pytest.raises(ValueError,match='outside'):
        topology(1,'hard',0,3,10)
    with pytest.raises(FileExistsError):
        freeze_dataset(tmp_path)

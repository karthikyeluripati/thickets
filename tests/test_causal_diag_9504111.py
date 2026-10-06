"""CPU tests for the 9504111 causal diagnostic: partition, manifests, exact hybrids, score contrasts, permutations."""
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import causal_diag_9504111 as c  # noqa: E402

SHAPES = Path('results/paper-analysis/causal-diagnostic/hf_tensor_shapes_0c351dd.json')


@pytest.mark.skipif(not SHAPES.exists(), reason='pinned tensor inventory not present')
def test_partition_exhaustive_hf_and_vllm():
    names = list(json.loads(SHAPES.read_text()))
    assert len(names) == 750
    g = c.partition(names)
    assert sum(len(v) for v in g.values()) == 750
    assert {k: len(v) for k, v in g.items()} == {'vision': 351, 'embed': 1, 'lm_q1': 99, 'lm_q2': 99, 'lm_q3': 99, 'lm_q4': 99, 'final_norm_head': 2}
    v = c.partition(c.hf_to_vllm_names(names))
    assert sum(len(x) for x in v.values()) == 642  # matches the tensor count observed in the forensic vLLM run
    assert all(any('deepstack' in n for n in v['vision']) for _ in [0])
    assert v['final_norm_head'] == sorted(['language_model.lm_head.weight', 'language_model.model.norm.weight'])


def test_assign_group_rules():
    assert c.assign_group('visual.blocks.3.attn.qkv.weight') == 'vision'
    assert c.assign_group('visual.deepstack_merger_list.0.linear_fc1.weight') == 'vision'
    assert c.assign_group('language_model.model.layers.8.mlp.gate_up_proj.weight') == 'lm_q1'
    assert c.assign_group('language_model.model.layers.9.self_attn.qkv_proj.weight') == 'lm_q2'
    assert c.assign_group('language_model.model.layers.35.post_attention_layernorm.weight') == 'lm_q4'
    assert c.assign_group('language_model.model.layers.3.input_layernorm.weight') == 'lm_q1'  # layer norms stay in their block
    with pytest.raises(ValueError):
        c.assign_group('language_model.model.layers.36.mlp.down_proj.weight')
    with pytest.raises(ValueError):
        c.assign_group('something.unexpected.weight')
    with pytest.raises(ValueError):
        c.partition(['visual.a.weight', 'visual.a.weight'])


def _rows():
    rows, k = [], 0
    for ph, cells in (('RERANK', {'repair': 20, 'regression': 4, 'both_wrong_diff': 12, 'both_correct': 75, 'both_wrong_same': 89}),
                      ('TEST', {'repair': 23, 'regression': 38, 'both_wrong_diff': 20, 'both_correct': 221, 'both_wrong_same': 259})):
        for t, n in cells.items():
            for i in range(n):
                g = 'A'; b, cd = {'repair': ('B', 'A'), 'regression': ('A', 'B'), 'both_wrong_diff': ('B', 'C'),
                                  'both_correct': ('A', 'A'), 'both_wrong_same': ('B', 'B')}[t]
                rows.append({'uid': f'{ph}:{k}', 'phase': ph, 'subtask': ['Allocentric', 'Egocentric', 'Hypothetical'][i % 3], 'gold': g, 'base': b, 'cand': cd})
                k += 1
    return rows


def test_manifest_deterministic_disjoint_and_half():
    m1, m2 = c.build_manifest(_rows()), c.build_manifest(_rows())
    assert m1['localization'] == m2['localization'] and m1['mechanism_check'] == m2['mechanism_check']
    assert not set(m1['localization']) & set(m1['mechanism_check'])
    cells = m1['cells']
    for k in ('RERANK|repair', 'RERANK|regression', 'TEST|repair', 'TEST|regression', 'TEST|both_wrong_diff'):
        assert abs(cells[k]['localization'] - cells[k]['mechanism_check']) <= 1
        assert cells[k]['localization'] + cells[k]['mechanism_check'] == cells[k]['total']
    assert cells['TEST|both_correct']['localization'] == cells['TEST|both_correct']['mechanism_check'] == 6
    assert 60 <= len(m1['localization']) <= 100


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = torch.nn.Linear(4, 4)
        self.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(2)])


class _Runner:
    def __init__(self, m):
        self.model = m


class _Worker:
    def __init__(self):
        torch.manual_seed(0)
        m = _Model().to(torch.bfloat16)
        self.model_runner = _Runner(m)
        self._base_weights = {n: p.detach().clone() for n, p in m.named_parameters()}
        for p in m.parameters():  # emulate an in-place perturbation in BF16
            p.data.add_(0.002 * torch.randn_like(p))
        self._cand = None


def test_hybrids_are_exact_copies_and_restore():
    w = _Worker()
    c._snapshot_candidate(w)
    names = c._param_names(w)
    cand = {n: t.clone() for n, t in w._cand.items()}
    group = [n for n in names if n.startswith('layers.0.')]
    k, bad = c._set_state(w, group)  # insertion: base everywhere, candidate in group
    assert bad == 0
    for n, p in w.model_runner.model.named_parameters():
        assert torch.equal(p.data, cand[n] if n in group else w._base_weights[n])
    k, bad = c._set_state(w, [n for n in names if n not in group])  # removal
    for n, p in w.model_runner.model.named_parameters():
        assert torch.equal(p.data, w._base_weights[n] if n in group else cand[n])
    c._set_state(w, [])
    assert all(torch.equal(p.data, w._base_weights[n]) for n, p in w.model_runner.model.named_parameters())
    c._set_state(w, names)
    assert all(torch.equal(p.data, cand[n]) for n, p in w.model_runner.model.named_parameters())


def test_scores_contrast_margin():
    sc, bound = c.letter_scores({32: -0.1, 33: -2.5, 34: -4.0, 99: -5.0})
    assert sc == {'A': -0.1, 'B': -2.5, 'C': -4.0, 'D': None} and bound == -5.0
    assert c.contrast(sc, 'B', 'A') == pytest.approx(-2.4)
    assert c.contrast(sc, 'D', 'A') is None and c.contrast(sc, 'A', 'A') == 0.0
    assert c.margin(sc, 'A') == pytest.approx(2.4) and c.margin(sc, 'B') == pytest.approx(-2.4)


def test_cyclic_permutations_map_back_to_content():
    opts, gold = ['left', 'right', 'front', 'behind'], 2
    seen_positions = set()
    for order in c.cyclic_orders():
        new, g2, back = c.permute_example(opts, gold, order)
        assert new[g2] == opts[gold]
        assert back['ABCD'[g2]] == gold
        seen_positions.add(g2)
    assert seen_positions == {0, 1, 2, 3}  # each content appears at every position
    assert c.order_sensitive(['A and B', 'none of the above', 'x', 'y']) and not c.order_sensitive(opts)

"""Tests for folded-gradient geometry: real name mapping, packed-row fold identity, first-order sanity."""
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import geometry_lib as G  # noqa: E402

VNAMES = Path('results/paper-analysis/causal-diagnostic/session1/parameter_groups_vllm.json')
HSHAPES = Path('results/paper-analysis/causal-diagnostic/hf_tensor_shapes_0c351dd.json')


@pytest.mark.skipif(not (VNAMES.exists() and HSHAPES.exists()), reason='recorded inventories missing')
def test_real_mapping_exhaustive_and_sizes():
    v = [n for g in json.loads(VNAMES.read_text()).values() for n in g]
    h = {k: x['shape'] for k, x in json.loads(HSHAPES.read_text()).items()}
    m = G.build_mapping(v, h)
    assert len(m) == 642 and sum(len(s['parts']) for s in m.values()) == 750
    assert max(s['numel'] for s in m.values()) == G.N_MAX
    qkv = m['language_model.model.layers.0.self_attn.qkv_proj.weight']
    assert [p[0].split('.')[-2] for p in qkv['parts']] == ['q_proj', 'k_proj', 'v_proj'] and qkv['numel'] == 6144 * 4096
    assert [p[1] for p in qkv['parts']] == [0, 4096 * 4096, 5120 * 4096]
    gu = m['language_model.model.layers.35.mlp.gate_up_proj.weight']
    assert gu['numel'] == 24576 * 4096 and gu['parts'][1][1] == 12288 * 4096
    # shape groups: 19 distinct vLLM numels/shapes as recorded in the forensic audit
    assert len({s['numel'] for s in m.values()}) <= 19


def _toy():
    """Two 'layers' with packed qkv (q 4 rows, k 2, v 2; 3 cols) and a shared-shape out proj (3x3), plus a vector."""
    hf = {}
    for l in range(2):
        hf[f'model.language_model.layers.{l}.self_attn.q_proj.weight'] = [4, 3]
        hf[f'model.language_model.layers.{l}.self_attn.k_proj.weight'] = [2, 3]
        hf[f'model.language_model.layers.{l}.self_attn.v_proj.weight'] = [2, 3]
        hf[f'model.language_model.layers.{l}.self_attn.o_proj.weight'] = [3, 3]
    hf['model.visual.blocks.0.norm1.weight'] = [5]
    v = [f'language_model.model.layers.{l}.self_attn.{t}' for l in range(2) for t in ('qkv_proj.weight', 'o_proj.weight')] + ['visual.blocks.0.norm1.weight']
    return v, hf


def test_fold_identity_matches_per_tensor_dot():
    v, hf = _toy()
    m = G.build_mapping(v, hf)
    torch.manual_seed(0)
    N = max(s['numel'] for s in m.values())
    eps = torch.randn(N, dtype=torch.float64)
    grads = {h: torch.randn(*s, dtype=torch.float64) for h, s in hf.items()}
    # per-tensor: each vLLM tensor's noise is eps prefix; HF parts are row slices of it
    direct = 0.
    for vn, spec in m.items():
        noise = eps[: spec['numel']]
        for h, off, n in spec['parts']:
            direct += float((grads[h].reshape(-1) * noise[off: off + n]).sum())
    F = G.fold_into(torch.zeros(N, dtype=torch.float64), grads, m)
    assert float(F @ eps) == pytest.approx(direct, rel=1e-12)


def test_first_order_prediction_on_toy_network():
    """For a small smooth function of replicated-noise parameters, sigma*<fold(grad), eps> predicts the change."""
    v, hf = _toy()
    m = G.build_mapping(v, hf)
    torch.manual_seed(1)
    params = {h: torch.randn(*s, dtype=torch.float64, requires_grad=True) for h, s in hf.items()}
    x = torch.randn(3, dtype=torch.float64)

    def f(P):
        h0 = torch.tanh(P['model.language_model.layers.0.self_attn.q_proj.weight'] @ x).sum()
        h1 = torch.tanh(P['model.language_model.layers.1.self_attn.o_proj.weight'] @ x).sum()
        k = (P['model.language_model.layers.0.self_attn.k_proj.weight'] ** 2).sum() + P['model.visual.blocks.0.norm1.weight'].sum()
        return h0 * h1 + k
    y = f(params); y.backward()
    N = max(s['numel'] for s in m.values()); eps = torch.randn(N, dtype=torch.float64); sigma = 1e-4
    F = G.fold_into(torch.zeros(N, dtype=torch.float64), {h: p.grad for h, p in params.items()}, m)
    pert = {}
    for vn, spec in m.items():
        noise = eps[: spec['numel']]
        for h, off, n in spec['parts']:
            pert[h] = params[h].detach() + sigma * noise[off: off + n].view(*hf[h])
    actual = float(f(pert) - y.detach())
    assert float(sigma * (F @ eps)) == pytest.approx(actual, rel=1e-3)


def test_hf_parts_for_vllm_names():
    assert G.hf_parts_for_vllm('visual.patch_embed.proj.weight') == ['model.visual.patch_embed.proj.weight']
    assert G.hf_parts_for_vllm('language_model.lm_head.weight') == ['lm_head.weight']
    assert G.hf_parts_for_vllm('language_model.model.embed_tokens.weight') == ['model.language_model.embed_tokens.weight']
    with pytest.raises(ValueError):
        G.hf_parts_for_vllm('something.else')

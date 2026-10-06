"""Folded-gradient geometry for RandOpt per-tensor-reseeded perturbations (pure, CPU-testable parts).

The pinned RandOpt worker draws, for every vLLM parameter p, noise = randn(p.shape, bf16, cuda, Generator.manual_seed(seed))
and applies p += sigma * noise (BF16). With CUDA Philox, a draw of n values equals the first n values of a longer draw
(forensic Phase G), so every tensor's noise is eps[:numel].view(shape) for ONE stream eps = randn(N_MAX) per seed.

First order: delta_t = sigma * eps[:n_t].view(shape_t); <grad, delta> = sigma * <fold(grad), eps>, where
fold(grad)[i] = sum over vLLM tensors t with numel_t > i of flatten(grad_t)[i] (vLLM packed layout).
HF exposes q/k/v and gate/up separately; they are reassembled into vLLM packed row order before flattening.
"""
import re

N_MAX = 151936 * 4096  # numel of embed_tokens / lm_head = 622,329,856

# vLLM packed tensor -> ordered HF parts (row blocks), for language layers
PACKED = {'self_attn.qkv_proj.weight': ('self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight'),
          'self_attn.qkv_proj.bias': ('self_attn.q_proj.bias', 'self_attn.k_proj.bias', 'self_attn.v_proj.bias'),
          'mlp.gate_up_proj.weight': ('mlp.gate_proj.weight', 'mlp.up_proj.weight'),
          'mlp.gate_up_proj.bias': ('mlp.gate_proj.bias', 'mlp.up_proj.bias')}


def hf_parts_for_vllm(vname):
    """vLLM parameter name -> list of HF parameter names whose row-concatenation (in order) is that vLLM tensor."""
    if vname.startswith('visual.'):
        for packed, parts in PACKED.items():  # e.g. Qwen2.5-VL vision MLP gate_up_proj (weight and bias)
            if vname.endswith(packed):
                return ['model.' + vname[: -len(packed)] + p for p in parts]
        return ['model.' + vname]
    if vname == 'language_model.lm_head.weight':
        return ['lm_head.weight']
    m = re.match(r'language_model\.model\.(.*)$', vname)
    if not m:
        raise ValueError(f'unmapped vLLM name {vname}')
    rest = m.group(1)
    for packed, parts in PACKED.items():
        if rest.endswith(packed):
            prefix = rest[: -len(packed)]
            return ['model.language_model.' + prefix + p for p in parts]
    return ['model.language_model.' + rest]


def build_mapping(vllm_names, hf_shapes):
    """Returns {vname: {'parts': [(hf_name, row_offset_elements, numel)], 'numel': n}} and verifies exhaustiveness."""
    import math
    used, out = set(), {}
    for v in vllm_names:
        parts, off = [], 0
        for h in hf_parts_for_vllm(v):
            if h not in hf_shapes:
                raise ValueError(f'missing HF tensor {h} for {v}')
            n = math.prod(hf_shapes[h])
            parts.append((h, off, n)); off += n; used.add(h)
        out[v] = {'parts': parts, 'numel': off}
    missing = set(hf_shapes) - used
    if missing:
        raise ValueError(f'HF tensors not covered: {sorted(missing)[:5]}')
    return out


def fold_into(F, grads_by_hf, mapping):
    """Accumulate fold(grad) into F (1-D tensor of length >= max numel). grads_by_hf: {hf_name: grad tensor}.
    Element order of a packed tensor = rows of part 1, then part 2, ... (row-major), matching vLLM storage."""
    for v, spec in mapping.items():
        for h, off, n in spec['parts']:
            g = grads_by_hf.get(h)
            if g is None:
                continue
            F[off: off + n].add_(g.reshape(-1))  # in-place mixed-dtype add: no full fp32 temporary
    return F


def noise_for(eps, shape):
    import math
    return eps[: math.prod(shape)].view(*shape)


def hf_slots(mapping):
    """HF parameter name -> (stream offset, numel, vLLM tensor name) for streaming folds."""
    return {h: (off, n, v) for v, spec in mapping.items() for h, off, n in spec['parts']}


class StreamingFolder:
    """Folds each parameter's gradient into a target stream buffer as soon as it is accumulated, then frees it.
    Optional: accumulates per-group dot products sigma*<grad_t, eps[off:off+n]> for one fixed eps (no extra buffers)."""

    def __init__(self, named_params, mapping, group_of_vllm=None):
        self.slots = hf_slots(mapping); self.target = None; self.eps = None; self.sigma = 0.; self.partials = None
        self.group_of_vllm = group_of_vllm or {}
        self.handles = [p.register_post_accumulate_grad_hook(self._hook(n)) for n, p in named_params.items()]

    def _hook(self, name):
        off, n, v = self.slots[name]

        def hook(p):
            g = p.grad.reshape(-1)
            if self.target is not None:
                self.target[off: off + n].add_(g)
            if self.eps is not None:
                grp = self.group_of_vllm[v]
                self.partials[grp] = self.partials.get(grp, 0.) + float(self.sigma * (g.float() @ self.eps[off: off + n].float()))
            p.grad = None
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()

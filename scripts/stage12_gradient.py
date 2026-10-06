"""Stage 2 prediction (results/paper-analysis/stage12/plan_lock.md): fold the gradient of the mean answer-content tilt
T-bar (fp32 contrast at BASE, HF) into non-vision and vision stream vectors; predict each frozen perturbation's tilt.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))
import geometry_lib as G  # noqa: E402
import causal_diag_9504111 as cd  # noqa: E402

IDS = [32, 33, 34, 35]
LET = 'ABCD'


def is_front(text):
    t = text.lower()
    return 'front' in t or 'forward' in t  # same rule as answer_prior_analysis.feats


def tilt_weights(options):
    F = [is_front(o) for o in options]; nf, nn = sum(F), len(F) - sum(F)
    return [(1.0 / nf if f else -1.0 / nn) for f in F]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--vllm-check', type=Path, default=Path('results/paper-analysis/geometry-gpu-a/gpu/vllm_stream_check.json'))
    ap.add_argument('--inputs', type=Path, default=Path('results/paper-analysis/stage12/frozen_inputs.json'))
    a = ap.parse_args()
    import importlib.util
    import torch
    from huggingface_hub import snapshot_download
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    a.out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    fin = json.loads(a.inputs.read_text()); probe = fin['probe_uids']
    ex = {}
    for s in ('search', 'validation'):
        for l in (cd.DATA / f'{s}.jsonl').read_text(encoding='utf-8').splitlines():
            r = json.loads(l); ex[r['uid']] = r
    path = snapshot_download(cd.MODEL, revision=cd.REV, allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja'])
    model = Qwen3VLForConditionalGeneration.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda()
    proc = AutoProcessor.from_pretrained(path)
    spec = importlib.util.spec_from_file_location('sp', 'third_party/omnispatial/system_prompts.py'); sp = importlib.util.module_from_spec(spec); spec.loader.exec_module(sp)
    params = dict(model.named_parameters())
    mapping = G.build_mapping(list(json.loads(a.vllm_check.read_text())['names_shapes']), {n: list(p.shape) for n, p in params.items()})
    slots = G.hf_slots(mapping); W = model.lm_head.weight

    def inputs(r):
        t = sp.SYS_PROMPTS['none'] + '\n' + sp.FORMAT_PROMPTS['direct'] + '\n\n' + r['question']
        for i, o in enumerate(r['options']):
            t += f'\n{LET[i]}. {o}'
        text = proc.apply_chat_template([{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': t}]}], tokenize=False, add_generation_prompt=True)
        img = Image.open(a.images / r['source_split'] / r['image_member'].rsplit('/', 1)[1]).convert('RGB')
        return {k: v.cuda() for k, v in proc(text=[text], images=[img], return_tensors='pt').items()}
    F = {'NOV': torch.zeros(G.N_MAX, dtype=torch.float32, device='cuda'), 'V': torch.zeros(G.N_MAX, dtype=torch.float32, device='cuda')}

    def hook_for(name):
        off, n, v = slots[name]; tgt = F['V' if cd.assign_group(v) == 'vision' else 'NOV']

        def hook(p):
            tgt[off: off + n].add_(p.grad.reshape(-1)); p.grad = None
        return hook
    for n, p in params.items():
        p.requires_grad_(True); p.register_post_accumulate_grad_hook(hook_for(n))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False}); model.config.use_cache = False; model.train()
    tbar = []
    for u in probe:
        r = ex[u]; w = torch.tensor(tilt_weights(r['options']), dtype=torch.float32, device='cuda')
        h = model.model(**inputs(r)).last_hidden_state[0, -1]
        T = (h.float() @ W[IDS].float().T) @ w / len(probe)
        tbar.append(float(T) * len(probe)); T.backward()
    gmin = (time.time() - t0) / 60
    buf = torch.empty(G.N_MAX, dtype=torch.bfloat16, device='cuda'); preds = []
    for p_ in fin['perturbations']:
        g = torch.Generator(device='cuda'); g.manual_seed(int(p_['seed'])); torch.randn(G.N_MAX, out=buf, generator=g); e = buf.float()
        preds.append({'candidate_id': p_['candidate_id'], 'seed': p_['seed'], 'role': p_['role'],
                      'pred_NOV': float(p_['sigma'] * (F['NOV'] @ e)), 'pred_V': float(p_['sigma'] * (F['V'] @ e))}); del e
    out = {'base_item_tilt_fp32': dict(zip(probe, tbar)), 'norm_F_NOV': float(F['NOV'].norm()), 'norm_F_V': float(F['V'].norm()),
           'minutes_gradient': gmin, 'minutes_total': (time.time() - t0) / 60, 'predictions': preds}
    (a.out / 'predictions.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ('predictions', 'base_item_tilt_fp32')}), flush=True)


if __name__ == '__main__':
    main()

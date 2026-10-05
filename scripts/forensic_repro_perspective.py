"""Phase Q (+ optional Phase R) clean-process GPU reproduction for the forensic audit.

Standalone: does not import the study runtime. Drives vllm.LLM directly with the pinned
upstream RandOpt WorkerExtension (apply_perturbation / reset_to_base_weights), rebuilds
the official direct prompt from the vendored official system_prompts.py, and compares
every raw response to the stored study outputs (exact text equality).

Phase R (--mask-diagnostic) is POST_HOC_MECHANISTIC_DIAGNOSTIC and never changes any
original decision.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('results/perspective-taking-n5000-20261004')
DATA = Path('examples/omnispatial-perspective-taking')
MODEL, REV = 'Qwen/Qwen3-VL-8B-Instruct', '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
LETTERS = 'ABCD'


def load(path):
    raw = Path(path).read_bytes()
    return json.loads(gzip.decompress(raw) if str(path).endswith('.gz') else raw)


def rows(split):
    return [json.loads(l) for l in (DATA / f'{split}.jsonl').read_text(encoding='utf-8').splitlines()]


def official_prompt(r):
    spec = importlib.util.spec_from_file_location('sp', 'third_party/omnispatial/system_prompts.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    text = m.SYS_PROMPTS['none'] + '\n' + m.FORMAT_PROMPTS['direct'] + '\n\n' + r['question']
    for i, o in enumerate(r['options']):
        text += f'\n{LETTERS[i]}. {o}'
    return text


def stored(seed):
    """Stored raw outputs for a candidate (or base when seed is None) on validation and test."""
    if seed is None:
        b = load(ROOT / 'baseline/base.json.gz')
        return {s: [o['text'] for o in b[s]['outputs']] for s in ('validation', 'test')}
    out = {}
    for d in sorted(ROOT.iterdir()):
        if not (d / 'candidates').is_dir(): continue
        for f in (d / 'candidates').glob('*.json.gz'):
            raw = load(f)
            if raw['candidate']['seed'] == seed:
                for s in ('validation', 'test'):
                    if s in raw['splits']: out[s] = [o['text'] for o in raw['splits'][s]['outputs']]
    return out


def rng_checks(worker):
    """Phase G on the actual GPU: same-seed noise for equal shapes, and prefix behaviour across sizes."""
    import torch
    res = {}
    dev = next(worker.model_runner.model.parameters()).device
    def noise(shape):
        g = torch.Generator(device=dev); g.manual_seed(9504111)
        return torch.randn(shape, dtype=torch.bfloat16, device=dev, generator=g)
    a, b = noise((4096, 4096)), noise((4096, 4096))
    res['equal_shape_identical_noise'] = bool(torch.equal(a, b))
    small, large = noise((1000,)).float(), noise((4096 * 4096,)).float()
    res['prefix_1000_of_16M_equal'] = bool(torch.equal(small, large[:1000]))
    res['prefix_1000_of_16M_fraction_equal'] = float((small == large[:1000]).float().mean())
    flat, mat = noise((4096 * 4096,)).float(), noise((4096, 4096)).float().reshape(-1)
    res['same_numel_different_shape_identical'] = bool(torch.equal(flat, mat))
    layout = [(n, tuple(p.shape)) for n, p in worker.model_runner.model.named_parameters()]
    from collections import Counter
    c = Counter(s for _, s in layout)
    res['tensors'] = len(layout); res['distinct_shapes'] = len(c)
    res['largest_identical_noise_groups'] = sorted(((v, list(k)) for k, v in c.items()), reverse=True)[:8]
    return res


def apply_mask(worker, seed, sigma, mode):
    """Phase R: reset to base, then add the upstream per-tensor noise only to the masked block."""
    import torch
    for name, p in worker.model_runner.model.named_parameters():
        p.data.copy_(worker._base_weights[name])
        visual = name.startswith(('visual.', 'model.visual.'))
        if (mode == 'vision_only' and visual) or (mode == 'language_only' and not visual) or mode == 'all':
            g = torch.Generator(device=p.device); g.manual_seed(int(seed))
            p.data.add_(float(sigma) * torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=g))
    torch.cuda.synchronize()
    return True


def state_hash(worker):
    import torch
    h = hashlib.sha256()
    for n, p in worker.model_runner.model.named_parameters():
        h.update(n.encode()); h.update(p.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--images', type=Path, required=True)
    ap.add_argument('--upstream', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--extra-seeds', default='9500931,9502763,9501287,9502542')
    ap.add_argument('--mask-diagnostic', action='store_true')
    a = ap.parse_args()
    if a.out.exists(): raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    sys.path.insert(0, str(a.upstream))
    os.environ.update({'VLLM_ENABLE_V1_MULTIPROCESSING': '0', 'PERTURB_VISUAL': '1', 'OMP_NUM_THREADS': '1'})
    from huggingface_hub import snapshot_download
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    path = snapshot_download(MODEL, revision=REV, allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja'])
    llm = LLM(model=path, tokenizer=path, dtype='bfloat16', tensor_parallel_size=1, distributed_executor_backend='uni',
              worker_extension_cls='utils.worker_extn.WorkerExtension', enforce_eager=True, enable_prefix_caching=False,
              gpu_memory_utilization=.5, max_model_len=16384, max_num_seqs=64, max_num_batched_tokens=16384,
              limit_mm_per_prompt={'image': 1, 'video': 0}, mm_processor_cache_gb=0, seed=0, disable_log_stats=True)
    rpc = lambda f, *args: llm.collective_rpc(f, args=args)[0]
    rpc('store_base_weights')
    proc = AutoProcessor.from_pretrained(path)
    sets = {s: rows(s) for s in ('validation', 'test')}
    reqs = {}
    for s, rs in sets.items():
        reqs[s] = []
        for r in rs:
            img = Image.open(a.images / r['source_split'] / r['image_member'].rsplit('/', 1)[1]).convert('RGB')
            prompt = proc.apply_chat_template([{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': official_prompt(r)}]}],
                                              tokenize=False, add_generation_prompt=True)
            reqs[s].append({'prompt': prompt, 'multi_modal_data': {'image': img}})
    sp = SamplingParams(temperature=0, seed=0, max_tokens=16)
    base_hash = rpc(state_hash)
    report = {'model': MODEL, 'revision': REV, 'base_state_sha256': base_hash, 'rng_checks': rpc(rng_checks), 'runs': []}

    def evaluate(label, seed, sigma, mode=None):
        t = time.time()
        if seed is None: rpc('reset_to_base_weights')
        elif mode is None: rpc('apply_perturbation', seed, sigma)
        else: rpc(apply_mask, seed, sigma, mode)
        st = rpc(state_hash)
        res = {'label': label, 'seed': seed, 'sigma': sigma, 'mask': mode or ('none' if seed is None else 'all (upstream apply_perturbation)'), 'state_sha256': st}
        want = stored(seed) if mode in (None, 'all') else None
        for s in ('validation', 'test'):
            outs = [o.outputs[0].text for o in llm.generate(reqs[s], sp, use_tqdm=False)]
            gold = [LETTERS[r['answer']] for r in sets[s]]
            pred = [((o.strip()[:1]).upper() if o.strip() else '') for o in outs]
            res[s] = {'correct': sum(p == g for p, g in zip(pred, gold)), 'n': len(gold)}
            if want is not None and s in want:
                res[s]['raw_text_identical_to_study'] = outs == want[s]
                res[s]['mismatched_questions'] = sum(x != y for x, y in zip(outs, want[s]))
            res[s]['predictions'] = pred
        rpc('reset_to_base_weights')
        res['seconds'] = time.time() - t
        report['runs'].append(res)
        print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != 'predictions'}) for k, v in res.items()}), flush=True)

    evaluate('base', None, 0.)
    evaluate('candidate_9504111_run1', 9504111, .002)
    evaluate('candidate_9504111_run2', 9504111, .002)
    for s in [int(x) for x in a.extra_seeds.split(',') if x]:
        evaluate(f'candidate_{s}', s, .002 if s not in (9501594, 9502542) else .001)
    evaluate('base_after', None, 0.)
    report['base_state_restored'] = rpc(state_hash) == base_hash
    if a.mask_diagnostic:
        report['POST_HOC_MECHANISTIC_DIAGNOSTIC'] = True
        for mode in ('language_only', 'vision_only', 'all'):
            evaluate(f'diagnostic_9504111_{mode}', 9504111, .002, mode)
    (a.out / 'report.json').write_text(json.dumps(report, indent=1))
    print('DONE', report['base_state_restored'])


if __name__ == '__main__':
    main()

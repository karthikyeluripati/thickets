"""OS runner (results/paper-analysis/os-prompt-control/plan_lock.md): the OmniSpatial benchmark's own three evaluation
prompts on the BASE model and on the selected winner 9504111, with M1's engine (vLLM 0.11 in-process, eager, pinned
RandOpt worker, PERTURB_VISUAL=1), fingerprints and exact base restore.
Prompts (third_party/omnispatial/system_prompts.py): direct = SYS 'none' + FORMAT 'direct', answer = first generated
token (max_tokens 1, exactly M1); zeroshot_cot / manual_cot = SYS 'zeroshot_cot' / 'manual_cot' + FORMAT 're', greedy up
to 2048 tokens, answer = the last 'Answer: X' line (X in A-D), else a parse failure (scored wrong).
Arms: BASE x 3 prompts on SEARCH200 (prompt selection); BASE x 3 and WINNER x 3 on the M1 hold-out (600)."""
import argparse
import gzip
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, 'src')
import causal_diag_9504111 as cd  # noqa: E402

BASE_ID = '5bc2499cb980429deee42ccabf91358e85a699f25b67ab5f7bdbf196457c8485'
CONFIG = {'direct': ('none', 'direct'), 'zeroshot_cot': ('zeroshot_cot', 're'), 'manual_cot': ('manual_cot', 're')}
ANS = re.compile(r'Answer:\s*\(?\s*([ABCD])\b')


def _restored_exact(worker):
    import torch
    return sum(int(not torch.equal(p.data, worker._base_weights[n].to(p.device))) for n, p in worker.model_runner.model.named_parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', type=Path, default=Path('/workspace/os-images')); ap.add_argument('--upstream', type=Path, default=Path('/workspace/RandOpt'))
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    sys.path.insert(0, str(a.upstream))
    os.environ.update({'VLLM_ENABLE_V1_MULTIPROCESSING': '0', 'PERTURB_VISUAL': '1', 'OMP_NUM_THREADS': '4'})
    from huggingface_hub import snapshot_download
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from thicket_runtime import fast_state
    a.out.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location('sp', 'third_party/omnispatial/system_prompts.py'); sp = importlib.util.module_from_spec(spec); spec.loader.exec_module(sp)
    sets = {'search': [json.loads(l) for l in open('examples/omnispatial-perspective-taking/search.jsonl', encoding='utf-8') if l.strip()],
            'holdout': [json.loads(l) for l in open('results/paper-analysis/m1/pod/holdout.jsonl', encoding='utf-8') if l.strip()]}
    if a.limit: sets = {k: v[: a.limit] for k, v in sets.items()}
    win = json.loads(Path('results/paper-analysis/m1/frozen_candidates.json').read_text())['candidates'][0]
    assert win['seed'] == 9504111
    path = snapshot_download(cd.MODEL, revision=cd.REV, allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja'])
    llm = LLM(model=path, tokenizer=path, dtype='bfloat16', tensor_parallel_size=1, distributed_executor_backend='uni',
              worker_extension_cls='utils.worker_extn.WorkerExtension', enforce_eager=True, enable_prefix_caching=False,
              gpu_memory_utilization=.5, max_model_len=16384, max_num_seqs=64, max_num_batched_tokens=16384, max_logprobs=20,
              limit_mm_per_prompt={'image': 1, 'video': 0}, mm_processor_cache_gb=0, seed=0, disable_log_stats=True)
    rpc = lambda f, *args: llm.collective_rpc(f, args=args)[0]
    rpc('store_base_weights'); proc = AutoProcessor.from_pretrained(path)

    def req(rows, cfg):
        sysk, fmtk = CONFIG[cfg]; out = []
        for r in rows:
            t = sp.SYS_PROMPTS[sysk] + '\n' + sp.FORMAT_PROMPTS[fmtk] + '\n\n' + r['question']
            for i, o in enumerate(r['options']):
                t += f'\n{cd.LETTERS[i]}. {o}'
            p = proc.apply_chat_template([{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': t}]}], tokenize=False, add_generation_prompt=True)
            out.append({'prompt': p, 'multi_modal_data': {'image': Image.open(a.images / r['source_split'] / r['image_member'].rsplit('/', 1)[1]).convert('RGB')}})
        return out

    def run(model, split, cfg):
        f = a.out / f'{model}_{split}_{cfg}.json.gz'
        if f.exists(): return
        rows = sets[split]
        spp = SamplingParams(temperature=0, seed=0, max_tokens=1, logprobs=20) if cfg == 'direct' else SamplingParams(temperature=0, seed=0, max_tokens=2048)
        recs = []
        for r, o in zip(rows, llm.generate(req(rows, cfg), spp, use_tqdm=False)):
            g = o.outputs[0]
            if cfg == 'direct':
                first = g.text.strip()[:1].upper(); parsed = first if first in cd.LETTERS else ''
            else:
                m = ANS.findall(g.text); parsed = m[-1] if m else ''
            recs.append({'uid': r['uid'], 'parsed': parsed, 'correct': parsed == cd.LETTERS[r['answer']], 'text': g.text,
                         'finish': g.finish_reason, 'ntok': len(g.token_ids)})
        f.write_bytes(gzip.compress(json.dumps(recs).encode()))
        print(json.dumps({'model': model, 'split': split, 'prompt': cfg, 'correct': sum(x['correct'] for x in recs), 'n': len(recs),
                          'parse_fail': sum(x['parsed'] == '' for x in recs)}), flush=True)
    if rpc(fast_state.fingerprint) != BASE_ID: raise SystemExit('base fingerprint mismatch')
    for cfg in CONFIG: run('base', 'search', cfg)
    for cfg in CONFIG: run('base', 'holdout', cfg)
    rpc('apply_perturbation', win['seed'], win['sigma'])
    if rpc(fast_state.fingerprint) != win['expected_state_id']: rpc('reset_to_base_weights'); raise SystemExit('winner fingerprint mismatch')
    for cfg in CONFIG: run('winner', 'holdout', cfg)
    rpc('reset_to_base_weights')
    rb = rpc(_restored_exact); print(json.dumps({'restore_mismatch': rb}), flush=True)
    if rb: raise SystemExit('restore mismatch')
    print('OS_DONE', flush=True)


if __name__ == '__main__':
    main()

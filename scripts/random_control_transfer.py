"""Random-control transfer for 9504111: 12 frozen sigma=0.002 audit candidates on RERANK200 + TEST561 (GPU).

Original engine (vLLM 0.11 in-process, uni executor, eager, BF16), pinned upstream RandOpt WorkerExtension
apply_perturbation / reset_to_base_weights (exact snapshot restore), original fingerprint
thicket_runtime.fast_state.fingerprint ('state-sha256-tree-v1'), official direct prompt, greedy, 16 tokens,
first-character parser, first-token A-D raw log-probabilities. Writes results after every candidate; an in-script
budget guard skips remaining candidates before the cap (partial results kept). Stops on any fidelity mismatch.
"""
import argparse
import gzip
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, 'src')
import causal_diag_9504111 as cd  # noqa: E402

ROOT = Path('results/perspective-taking-n5000-20261004')
BASE_ID = '5bc2499cb980429deee42ccabf91358e85a699f25b67ab5f7bdbf196457c8485'


def _restored_exact(worker):
    import torch
    return sum(int(not torch.equal(p.data, worker._base_weights[n].to(p.device))) for n, p in worker.model_runner.model.named_parameters())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--images', type=Path, required=True); ap.add_argument('--upstream', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--usd-per-hour', type=float, required=True)
    ap.add_argument('--pod-start-epoch', type=float, required=True); ap.add_argument('--cap-usd', type=float, required=True)
    ap.add_argument('--controls', default='results/paper-analysis/random-control-transfer/frozen_controls.json')
    a = ap.parse_args()
    sys.path.insert(0, str(a.upstream))
    os.environ.update({'VLLM_ENABLE_V1_MULTIPROCESSING': '0', 'PERTURB_VISUAL': '1', 'OMP_NUM_THREADS': '1'})
    from huggingface_hub import snapshot_download
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from thicket_runtime import fast_state
    a.out.mkdir(parents=True, exist_ok=True)
    spent = lambda: (time.time() - a.pod_start_epoch) / 3600 * a.usd_per_hour
    log = {'usd_per_hour': a.usd_per_hour, 'cap_usd': a.cap_usd, 'steps': []}

    def note(step, **kw):
        kw.update(step=step, spent_usd=round(spent(), 3), t=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        log['steps'].append(kw); (a.out / 'cost_log.json').write_text(json.dumps(log, indent=1)); print(json.dumps(kw)[:500], flush=True)

    ctrl = json.loads(Path(a.controls).read_text())
    rows = {}
    for s, ph in (('validation', 'RERANK'), ('test', 'TEST')):
        rows[ph] = [json.loads(l) for l in (cd.DATA / f'{s}.jsonl').read_text(encoding='utf-8').splitlines()]
    spec = importlib.util.spec_from_file_location('sp', 'third_party/omnispatial/system_prompts.py')
    sp = importlib.util.module_from_spec(spec); spec.loader.exec_module(sp)
    path = snapshot_download(cd.MODEL, revision=cd.REV, allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja'])
    llm = LLM(model=path, tokenizer=path, dtype='bfloat16', tensor_parallel_size=1, distributed_executor_backend='uni',
              worker_extension_cls='utils.worker_extn.WorkerExtension', enforce_eager=True, enable_prefix_caching=False,
              gpu_memory_utilization=.5, max_model_len=16384, max_num_seqs=64, max_num_batched_tokens=16384, max_logprobs=20,
              limit_mm_per_prompt={'image': 1, 'video': 0}, mm_processor_cache_gb=0, seed=0, disable_log_stats=True)
    rpc = lambda f, *args: llm.collective_rpc(f, args=args)[0]
    rpc('store_base_weights')
    proc = AutoProcessor.from_pretrained(path)

    def prompt(r):
        t = sp.SYS_PROMPTS['none'] + '\n' + sp.FORMAT_PROMPTS['direct'] + '\n\n' + r['question']
        for i, o in enumerate(r['options']):
            t += f'\n{cd.LETTERS[i]}. {o}'
        return proc.apply_chat_template([{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': t}]}], tokenize=False, add_generation_prompt=True)

    req = {ph: [{'prompt': prompt(r), 'multi_modal_data': {'image': Image.open(a.images / r['source_split'] / r['image_member'].rsplit('/', 1)[1]).convert('RGB')}}
                for r in rs] for ph, rs in rows.items()}
    spp = SamplingParams(temperature=0, seed=0, max_tokens=16, logprobs=20)

    def run(ph):
        out = []
        for r, o in zip(rows[ph], llm.generate(req[ph], spp, use_tqdm=False)):
            g = o.outputs[0]
            lp0 = {int(k): float(v.logprob) for k, v in (g.logprobs[0] if g.logprobs else {}).items()}
            sc, bound = cd.letter_scores(lp0)
            out.append({'uid': r['uid'], 'text': g.text, 'parsed': g.text.strip()[:1].upper() if g.text.strip()[:1].upper() in cd.LETTERS else '',
                        'first_token': int(g.token_ids[0]) if g.token_ids else None, 'letter_logprobs': sc, 'top20_floor': bound})
        missing = sum(any(v is None for v in x['letter_logprobs'].values()) for x in out)
        return out, missing

    def stored_texts(cid, split):
        for d in sorted(ROOT.iterdir()):
            f = d / 'candidates' / f'{cid}.json.gz'
            if f.exists() and '.failed' not in d.name:
                raw = json.loads(gzip.decompress(f.read_bytes()))
                if split in raw['splits']:
                    return {o['uid']: o['text'] for o in raw['splits'][split]['outputs']}
        return None

    note('model_loaded')
    fb = rpc(fast_state.fingerprint); note('base_fingerprint', ok=fb == BASE_ID, fingerprint=fb)
    if fb != BASE_ID: raise SystemExit('INSTRUMENTATION_MISMATCH base fingerprint')
    sb = json.loads(gzip.decompress((ROOT / 'baseline/base.json.gz').read_bytes()))
    s1 = {x['uid']: x for x in json.loads(gzip.decompress((cd.OUT / 'session1/base_full.json.gz').read_bytes()))}
    base = {}
    for ph, split in (('RERANK', 'validation'), ('TEST', 'test')):
        out, miss = run(ph); base[ph] = out
        st = {o['uid']: o['text'] for o in sb[split]['outputs']}
        fid = {'text_identical': sum(x['text'] == st[x['uid']] for x in out), 'scores_identical_to_session1': sum(x['letter_logprobs'] == s1[x['uid']]['letter_logprobs'] for x in out),
               'n': len(out), 'missing_scores': miss}
        note(f'base_fidelity_{ph}', **fid)
        if fid['text_identical'] != len(out) or miss: raise SystemExit(f'INSTRUMENTATION_MISMATCH base {ph}')
    (a.out / 'base.json.gz').write_bytes(gzip.compress(json.dumps(base).encode()))
    per_cand_minutes = None
    for i, c in enumerate(ctrl['candidates']):
        t0 = time.time()
        if per_cand_minutes and spent() + per_cand_minutes * 1.15 / 60 * a.usd_per_hour + 0.10 > a.cap_usd:
            note('STOP_BUDGET_before_candidate', index=i, seed=c['seed']); break
        rpc('apply_perturbation', c['seed'], c['sigma'])
        fp = rpc(fast_state.fingerprint)
        rec = {'candidate': c, 'fingerprint': fp, 'fingerprint_ok': fp == c['expected_state_id']}
        if not rec['fingerprint_ok']:
            note('FINGERPRINT_MISMATCH', seed=c['seed'], got=fp); rpc('reset_to_base_weights'); raise SystemExit('fingerprint mismatch')
        for ph, split in (('RERANK', 'validation'), ('TEST', 'test')):
            out, miss = run(ph); rec[ph] = out
            st = stored_texts(c['candidate_id'], split)
            rec[f'{ph}_stored_available'] = st is not None
            rec[f'{ph}_text_identical_to_stored'] = sum(x['text'] == st[x['uid']] for x in out) if st else None
            rec[f'{ph}_missing_scores'] = miss
        rpc('reset_to_base_weights'); rec['base_restored_mismatches'] = rpc(_restored_exact)
        (a.out / f"control_{c['seed']}.json.gz").write_bytes(gzip.compress(json.dumps(rec).encode()))
        per_cand_minutes = (time.time() - t0) / 60
        note(f"control_{c['seed']}", fingerprint_ok=True, rerank_identical=rec['RERANK_text_identical_to_stored'], test_identical=rec['TEST_text_identical_to_stored'],
             missing=rec['RERANK_missing_scores'] + rec['TEST_missing_scores'], restored_mismatches=rec['base_restored_mismatches'], minutes=round(per_cand_minutes, 2))
        if rec['RERANK_text_identical_to_stored'] != 200 or (rec['TEST_stored_available'] and rec['TEST_text_identical_to_stored'] != 561) \
                or rec['RERANK_missing_scores'] or rec['TEST_missing_scores'] or rec['base_restored_mismatches']:
            note('STOP_FIDELITY', seed=c['seed']); raise SystemExit('fidelity mismatch or missing scores')
    note('DONE')


if __name__ == '__main__':
    main()

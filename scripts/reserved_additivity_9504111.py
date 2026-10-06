"""Reserved-example additivity check for 9504111 (prepared; runs ONLY after explicit approval).

On the SAME 82 MECHANISM-CHECK examples from example_manifest.json: BASE and CANDIDATE fidelity passes (must match
the stored session-1 outputs exactly), then the seven single-group INSERTION hybrids built by copying exact BF16
values (same engine, prompt, processor, cache controls, verification and budget guard as session 1). No removals.
The frozen parameter-free predictor (scripts/margin_additivity.py: additive_prediction, coefficient 1 per group) is
applied afterwards on CPU without retuning.

Usage (GPU): python scripts/reserved_additivity_9504111.py --images DIR --upstream DIR --out DIR --usd-per-hour X --pod-start-epoch T --cap-usd C
"""
import argparse
import gzip
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))
import causal_diag_9504111 as cd  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--images', type=Path, required=True); ap.add_argument('--upstream', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--usd-per-hour', type=float, required=True)
    ap.add_argument('--pod-start-epoch', type=float, required=True); ap.add_argument('--cap-usd', type=float, required=True)
    ap.add_argument('--manifest', default=str(cd.OUT / 'example_manifest.json'))
    a = ap.parse_args()
    import importlib.util, os  # noqa: E401
    sys.path.insert(0, str(a.upstream))
    os.environ.update({'VLLM_ENABLE_V1_MULTIPROCESSING': '0', 'PERTURB_VISUAL': '1', 'OMP_NUM_THREADS': '1'})
    from huggingface_hub import snapshot_download
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    a.out.mkdir(parents=True, exist_ok=True)
    spent = lambda: (time.time() - a.pod_start_epoch) / 3600 * a.usd_per_hour
    log = {'usd_per_hour': a.usd_per_hour, 'cap_usd': a.cap_usd, 'steps': []}

    def note(step, **kw):
        kw.update(step=step, spent_usd=round(spent(), 3), t=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        log['steps'].append(kw); (a.out / 'cost_log.json').write_text(json.dumps(log, indent=1)); print(json.dumps(kw)[:500], flush=True)

    def guard(m):
        if spent() + m / 60 * a.usd_per_hour > a.cap_usd:
            note('STOP_BUDGET', next_minutes=m); raise SystemExit('budget stop')

    man = json.loads(Path(a.manifest).read_text()); mech = man['mechanism_check']
    rows = {}
    for s in ('validation', 'test'):
        for l in (cd.DATA / f'{s}.jsonl').read_text(encoding='utf-8').splitlines():
            r = json.loads(l); rows[r['uid']] = r
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

    req = {u: {'prompt': prompt(rows[u]), 'multi_modal_data': {'image': Image.open(a.images / rows[u]['source_split'] / rows[u]['image_member'].rsplit('/', 1)[1]).convert('RGB')}} for u in mech}
    spp = SamplingParams(temperature=0, seed=0, max_tokens=16, logprobs=20)

    def run(name):
        t0 = time.time(); out = []
        for u, o in zip(mech, llm.generate([req[u] for u in mech], spp, use_tqdm=False)):
            g = o.outputs[0]
            lp0 = {int(k): float(v.logprob) for k, v in (g.logprobs[0] if g.logprobs else {}).items()}
            sc, bound = cd.letter_scores(lp0)
            out.append({'uid': u, 'text': g.text, 'parsed': g.text.strip()[:1].upper() if g.text.strip()[:1].upper() in cd.LETTERS else '',
                        'first_token': int(g.token_ids[0]) if g.token_ids else None, 'letter_logprobs': sc, 'top20_floor': bound})
        if any(v is None for x in out for v in x['letter_logprobs'].values()):
            (a.out / f'{name}.json.gz').write_bytes(gzip.compress(json.dumps(out).encode()))
            note('STOP_SCORE_COVERAGE', condition=name); raise SystemExit('letter score missing: stop, no imputation')
        (a.out / f'{name}.json.gz').write_bytes(gzip.compress(json.dumps(out).encode()))
        note(f'run:{name}', n=len(out), seconds=round(time.time() - t0, 1))
        return out

    note('model_loaded')
    if rpc(cd._state_hash) != cd.BASE_STATE: note('INSTRUMENTATION_MISMATCH_base'); raise SystemExit(1)
    guard(1); base = run('base_mech')
    rpc('apply_perturbation', cd.SEED, cd.SIGMA)
    if rpc(cd._state_hash) != cd.CAND_STATE: note('INSTRUMENTATION_MISMATCH_candidate'); raise SystemExit(1)
    guard(1); cand = run('candidate_mech')
    sb = {x['uid']: x for x in json.loads(gzip.decompress((cd.OUT / 'session1/base_full.json.gz').read_bytes()))}
    sc_ = {x['uid']: x for x in json.loads(gzip.decompress((cd.OUT / 'session1/candidate_full.json.gz').read_bytes()))}
    fid = {'base_text_identical': sum(x['text'] == sb[x['uid']]['text'] for x in base), 'cand_text_identical': sum(x['text'] == sc_[x['uid']]['text'] for x in cand),
           'base_scores_identical': sum(x['letter_logprobs'] == sb[x['uid']]['letter_logprobs'] for x in base),
           'cand_scores_identical': sum(x['letter_logprobs'] == sc_[x['uid']]['letter_logprobs'] for x in cand), 'n': len(mech)}
    note('fidelity', **fid)
    if fid['base_text_identical'] != len(mech) or fid['cand_text_identical'] != len(mech):
        raise SystemExit('INSTRUMENTATION_MISMATCH: fidelity')
    rpc(cd._snapshot_candidate)
    names = rpc(cd._param_names); groups = cd.partition(names)
    for g in cd.GROUPS:
        guard(1)
        k, bad = rpc(cd._set_state, groups[g])
        if bad: note('HYBRID_VERIFY_FAIL', group=g, mismatches=bad); raise SystemExit(1)
        run(f'insertion_{g}_mech')
    k, bad = rpc(cd._set_state, [])
    note('final_restore', mismatches=bad, mismatches_vs_base=rpc(cd._mismatch, 'base'))
    note('DONE')


if __name__ == '__main__':
    main()

"""Create the immutable Perspective-Taking N=5000 protocol before any Qwen3-VL inference."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess

from thicket_runtime.line_tracing import sha
from thicket_runtime.omnispatial_data import DATASET, DATASET_REVISION, OFFICIAL_BLOBS, REPO, REPO_COMMIT, ZIPS, build_prompt, load, sha256
from thicket_runtime.perspective import density_audit, ident, manifest, verify_inputs
from thicket_runtime.visual_runtime import read, write

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--max-model-len', type=int, required=True)
p.add_argument('--max-num-batched-tokens', type=int, required=True)
a = p.parse_args()
DATA = 'examples/omnispatial-perspective-taking'
DATA_COMMIT = subprocess.check_output(['git', 'log', '-1', '--format=%H', '--', DATA], text=True).strip()
parent = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
old = read('experiments/omnispatial_visual_expert_protocol.json')
seeds = set(old['previous_candidate_seeds']) | set(range(9100000, 9100400))
for path in Path('results').rglob('*'):
    if path.is_file() and (path.suffix == '.json' or path.name.endswith('.json.gz')):
        raw = path.read_bytes()
        if path.suffix == '.gz': raw = gzip.decompress(raw)
        seeds.update(int(s) for s in re.findall(rb'"seed"\s*:\s*(\d+)', raw))
files = len(subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', DATA_COMMIT, '--', DATA], text=True).split())
sets = {s: load(DATA, s) for s in ('search', 'validation', 'test')}
proto = {
    'schema': 'perspective-taking-n5000-v1', 'parent_commit': parent,
    'question': 'At full Neural-Thickets/RandOpt population scale, does Qwen3-VL-8B contain a nearby model with materially and transferably stronger visual perspective-taking ability?',
    'benchmark': {'repository': REPO, 'repository_commit': REPO_COMMIT, 'official_blobs': OFFICIAL_BLOBS, 'dataset': DATASET,
                  'dataset_revision': DATASET_REVISION, 'zips': {s: {'file': f, 'lfs_sha256': h} for s, (f, h) in ZIPS.items()},
                  'task_type': 'Perspective_Taking (all sub-task types; no selection by model performance)',
                  'split_use': 'official train -> SEARCH200 + VALIDATION200 (rest unused); official test Perspective_Taking -> FINAL TEST (all 561)'},
    'dataset': DATA, 'dataset_pin': {'commit': DATA_COMMIT, 'manifest_sha256': sha256((Path(DATA)/'manifest.json').read_bytes()), 'files': files},
    'model': 'Qwen/Qwen3-VL-8B-Instruct', 'model_revision': '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',
    'processor_revision': '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b', 'tokenizer_revision': '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',
    'chat_template_revision': '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',
    'upstream_commit': old['upstream_commit'], 'dtype': 'bfloat16', 'tensor_parallel_size': 1, 'temperature': 0, 'sampling_seed': 0,
    'max_tokens': 16,
    'perturbation': 'Unchanged pinned RandOpt worker apply_perturbation/reset_to_base_weights with PERTURB_VISUAL=1: all parameters including vision, native tensor layout, randopt-per-tensor-v1, exact snapshot anchoring (no add/subtract restoration). Identity = base_id + seed + sigma + RNG contract + parameter mask.',
    'required_versions': {'torch': '2.8.0+cu128', 'vllm': '0.11.0', 'ray': '2.49.2', 'transformers': '4.57.1', 'numpy': '2.1.2',
                          'Pillow': '12.3.0', 'tokenizers': '0.22.2', 'huggingface-hub': '0.36.2'},
    'engine': {'gpu_memory_utilization': .5, 'max_model_len': a.max_model_len, 'max_num_seqs': 64,
               'max_num_batched_tokens': a.max_num_batched_tokens, 'enforce_eager': True, 'prefix_caching': False, 'mm_processor_cache_gb': 0,
               'image_processing': 'pinned Qwen3-VL processor defaults (patch 16, merge 2, 65536..16777216 pixels)'},
    'evaluation': {'path': 'official OmniSpatial vlms_eval direct path: --prompt_type none --eval_type direct',
                   'prompt': "SYS_PROMPTS['none'] + '\\n' + FORMAT_PROMPTS['direct'] + '\\n\\n' + question + '\\nA. ..\\nB. ..\\nC. ..\\nD. ..'; user message [image, text]; pinned Qwen3-VL chat template with generation prompt",
                   'prompt_source': 'vendored third_party/omnispatial/system_prompts.py, git blob ' + OFFICIAL_BLOBS['vlms_eval/system_prompts.py'],
                   'image_preprocessing': 'PIL open + convert("RGB") as in official qwen_vl_utils.fetch_image; resizing by the pinned Qwen3-VL processor; RGB pixel and prompt hashes recorded per image and required equal in every phase',
                   'scorer': 'official direct: prediction = response.strip().upper()[:1]; correct iff equals chr(65+answer); no fallback letter; non A-D first character is invalid and incorrect',
                   'decoding': 'greedy, temperature 0 (overrides the checkpoint sampling defaults), max_tokens 16; the direct scorer reads only the first non-space character',
                   'frozen': 'Fixed before any nonzero candidate inference; never modified.',
                   'prompt_sha256_by_split': {s: hashlib.sha256(json.dumps([build_prompt(r) for r in rows], ensure_ascii=False).encode()).hexdigest() for s, rows in sets.items()}},
    'populations': {s: {'n': len(rows), 'sha256': ident(rows)} for s, rows in sets.items()},
    'candidates': {'count': 5000, 'seed_start': 9500000, 'sigma_mixture': [.00025, .0005, .001, .002], 'shard_size': 100,
                   'rule': 'candidate i=0..4999: seed 9500000+i, sigma sigma_mixture[i % 4], shard i // 100; exactly 1250 per sigma; manifest fixed before inference; no calibration, early stop, reallocation, extra sigma or population change',
                   'retries': 'a failed shard is kept as .failed-N and rerun entirely; only complete audited shard directories enter analysis; each manifest index appears exactly once'},
    'density_audit': {'seed': 20261006, 'per_sigma': 125, 'indices': None,
                      'rule': '500 manifest indices (125 per sigma) drawn before any output; evaluated on VALIDATION200 regardless of SEARCH score; never used for selection; reported with Wilson 95% intervals as validation-level prevalence, not final transferable density'},
    'search': {'top_k': 50, 'ranking': "SEARCH correct count descending, then SHA256('visual-rank-v1:'+candidate_id) ascending",
               'freeze': 'complete 5000 ranking, top-50 recipes, scores and fingerprints committed before any VALIDATION candidate inference',
               'no_early_stop': 'never stop on population mean, degradation or one sigma'},
    'validation': {'ranking': 'VALIDATION correct desc, SEARCH correct desc, then the frozen hash',
                   'freeze': 'rank 1, top 5 and top 10 with recipes, scores and fingerprints committed before any TEST candidate inference',
                   'shards': 'operational: [top 50] + [density audit not in top 50] split into shards of 35 across single-GPU processes'},
    'test': {'candidates': 'frozen validation top 10 only; no other candidate after test outputs are visible'},
    'gate': {'expert_gain': .05, 'positive_sub_tasks': 2, 'answer_share_limit': .8,
             'expert': 'TRANSFERABLE_PERSPECTIVE_EXPERT: test gain >= +5 pp; paired bootstrap 95% lower bound > 0; strictly positive gain in >= 2 of the 3 Perspective-Taking sub-task types; no true-answer letter contributes >= 80% of positive net correct-count gains; all controls pass',
             'primary': 'GO_VISUAL_NEURAL_THICKET_PERSPECTIVE_TAKING if validation rank 1 is a TRANSFERABLE_PERSPECTIVE_EXPERT',
             'replicated': 'GO_VISUAL_NEURAL_THICKET_PERSPECTIVE_TAKING_REPLICATED if >=3 of top 10 reach +3 pp, >=1 reaches +5 pp, mean top-10 gain >= +3 pp, and a +5 pp candidate passes the expert checks',
             'otherwise': 'NO_GO_VISUAL_NEURAL_THICKET_PERSPECTIVE_N5000; stop, no rescue',
             'answer_bias': 'concentration = max positive per-true-letter net gain / sum of positive net gains; >= 0.80 fails; validity criterion only, never reranks',
             'ensemble': 'top-5 and top-10 majority vote (invalid ignored, ties to first in frozen validation order); >= +7 pp is secondary evidence only'},
    'statistics': {'bootstrap_replicates': 10000, 'bootstrap_seed': 20261003,
                   'method': 'paired question percentile bootstrap, common PCG64 int64 indices per split, 2.5/97.5 linear percentiles; exact two-sided McNemar'},
    'preflight': {'seed': 5200001, 'sigma': .002,
                  'scope': 'operational only, inside the baseline phase: load, store exact base, zero perturbation, base repeat, one nonzero perturbation (no generation) to prove fast controls equal the original controls, exact restore, memory record'},
    'controls': {'zero': 'zero perturbation reproduces base outputs on every evaluated split in every phase',
                 'repeat': 'first candidate of every shard and of test reconstructed again with identical state and outputs',
                 'restore': 'bitwise exact restoration after every candidate (parameters and buffers)',
                 'fingerprint': 'state-sha256-tree-v1 (per-tensor SHA256 in a thread pool, then SHA256 over the canonical leaf list); base also recorded with the original flat SHA256',
                 'fast_drift': 'per-tensor bitwise equality on GPU with fallback to the original per-value comparison; proven equal to the original drift on base and on a perturbed state in the baseline phase',
                 'fingerprints_across_phases': 'validation and test reject a candidate before generation unless its fingerprint equals its search fingerprint',
                 'encoding': 'encoder cache cleared before each generation, every image freshly encoded, unique multimodal UUIDs, prefix and multimodal processor caches disabled',
                 'data': 'pinned zips by LFS sha256, every image by sha256; per-image RGB pixel and prompt hashes equal the baseline in every phase',
                 'locks': 'dataset, protocol, baseline, search and validation locks committed before the phases that consume them; analysis checks commit times',
                 'operational': 'any runtime/control failure is operational, never a scientific NO-GO'},
    'cost_discipline': 'Throughput is measured from the first audited search shards and reported; it never changes the protocol.',
    'no_base_capability_gate': True,
    'out_of_scope': 'No N=10000, new sigmas, other model, other subset, prompt change, BO, CMA-ES, Modular Norm RandOpt, transfer-aware ranking or localization.',
    'previous_candidate_seeds': sorted(seeds)}
proto['density_audit']['indices'] = density_audit(proto)
if {r['seed'] for r in manifest(proto)} & seeds: raise ValueError('candidate seed reuse')
verify_inputs(proto)
path = Path('experiments/perspective_taking_n5000_protocol.json')
write(path, proto)
print({'protocol_sha256': sha(path), 'dataset_commit': DATA_COMMIT, 'prior_seeds_excluded': len(seeds),
       'populations': {s: v['n'] for s, v in proto['populations'].items()}})

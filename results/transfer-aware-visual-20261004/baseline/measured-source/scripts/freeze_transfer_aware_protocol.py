"""Create the immutable transfer-aware first-bridge protocol before any 7B inference."""
import gzip
import json
from pathlib import Path
import re
import subprocess

from thicket_runtime.line_tracing import sha
from thicket_runtime.transfer_aware import candidate_plan, verify_inputs
from thicket_runtime.transfer_aware_data import FOLDS, load_folds
from thicket_runtime.visual_runtime import read, write
from thicket_runtime.visual_v21 import population_id

DATASET = 'examples/visual-line-tracing-v1-transfer-aware'
DATASET_COMMIT = 'dd89aae0c5ababab38d339f260920749750d57b0'
parent = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
old = read('experiments/visual_line_tracing_final_protocol.json')
seeds = set(old['previous_candidate_seeds']) | set(range(6100000, 6100300))
for path in Path('results').rglob('*'):
    if path.is_file() and (path.suffix == '.json' or path.name.endswith('.json.gz')):
        raw = path.read_bytes()
        if path.suffix == '.gz': raw = gzip.decompress(raw)
        seeds.update(int(s) for s in re.findall(rb'"seed"\s*:\s*(\d+)', raw))
files = len(subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', DATASET_COMMIT, '--', DATASET], text=True).split())
p = {'schema': 'transfer-aware-visual-search-v1', 'parent_commit': parent,
     'question': 'Given the exact same population of nearby VLM weight perturbations, can a transfer-aware ranking rule identify candidates that generalize better than standard RandOpt ranking?',
     'dataset': DATASET, 'dataset_pin': {'commit': DATASET_COMMIT, 'manifest_sha256': sha(Path(DATASET)/'manifest.json'), 'files': files},
     'model': 'Qwen/Qwen2.5-VL-7B-Instruct', 'model_revision': 'cc594898137f460bfe9f0759e9844b3ce807cfb5',
     'processor_revision': 'cc594898137f460bfe9f0759e9844b3ce807cfb5', 'tokenizer_revision': 'cc594898137f460bfe9f0759e9844b3ce807cfb5',
     **{k: old[k] for k in ('upstream_commit', 'dtype', 'tensor_parallel_size', 'temperature', 'sampling_seed', 'max_tokens',
                            'max_model_len', 'max_num_seqs', 'max_num_batched_tokens', 'gpu_memory_utilization', 'image_pixels',
                            'perturbation', 'cache_contract', 'required_versions')},
     'populations': {}, 'folds': {},
     'candidates': {'count': 400, 'seed_start': 7300000, 'sigma_mixture': [.00025, .0005, .001, .002],
                    'rule': 'candidate i=0..399: seed 7300000+i, sigma sigma_mixture[i % 4]; exactly 100 per sigma; fixed before any output; no calibration, no population-mean stop, no change after results',
                    'shard_size': 50, 'shards': 'eight execution shards of 50 consecutive candidates; operational only, each with its own zero/repeat controls',
                    'reconstruction': 'every candidate from the same exact base snapshot via unchanged pinned RandOpt apply_perturbation/reset_to_base_weights'},
     'ranking': {'hash': "SHA256('visual-rank-v1:' + candidate_id) ascending (existing deterministic tie-break)",
                 'VANILLA': 'pooled search300 correct count descending, then hash',
                 'TRANSFER_MIN': 'transfer_score = min(fold_A_gain, fold_B_gain, fold_C_gain) with fold gain = candidate fold accuracy - base fold accuracy; descending, then mean fold gain descending, then hash',
                 'no_tuning': 'No lambda, variance penalty, or alternative score; both rules rank the identical 400 audited records'},
     'validation': {'top_k': 20, 'examples': 300, 'union': 'VANILLA top-20 then TRANSFER_MIN top-20 not already present; each evaluated once on validation300',
                    'method_metrics': 'over each method top-20 in search order: best gain, top-5 mean, top-10 mean, counts > base, >=1, >=3, >=5 pp (top-20 and top-10)',
                    'transfer_correlation': 'Pearson and Spearman over the union (pooled search gain vs validation gain; transfer score vs validation gain) and within each method top-20; descriptive only'},
     'validation_gate': {'a_best_gain': .03, 'a_margin': .02, 'b_count': 3, 'b_margin': 2,
                         'rule': 'Pass if A or B. A: best TRANSFER_MIN top-20 validation gain >= +3 pp AND >= 2 pp above best VANILLA top-20 validation gain. B: >=3 of TRANSFER_MIN top-10 (search order) have validation gain >= +3 pp AND that count exceeds the VANILLA top-10 count by >= 2. Otherwise NO_GO_TRANSFER_AWARE_VISUAL_SEARCH and stop: no final test, no other score, no BO/CMA-ES.',
                         'wording_resolution': 'The request says BOTH conditions hold, then lists A OR B; its later text (if neither A nor B / if A or B passes) is applied: either condition passes.'},
     'final_selection': {'rule': "For each method, order its top-20 by validation gain descending, then that method's frozen search rank. Best = first; top-5 = first five (includes best). Deduplicated union, committed before test inference.",
                         'primary': "The method's best validation candidate is its frozen primary final candidate."},
     'final_gate': {'expert_gain': .05, 'label_concentration_limit': .8, 'beat_margin': .02, 'secondary_count': 3, 'secondary_mean': .03,
                    'expert': 'Genuine visual expert: test gain >= +5 pp; paired bootstrap 95% lower bound > 0; strictly positive gain in >= 2 of easy/medium/hard; no true-answer class contributes >= 80% of positive per-class net correct-count gains. Runtime controls checked globally.',
                    'strong': 'GO_TRANSFER_AWARE_VISUAL_SEARCH if the TRANSFER_MIN primary is a genuine expert, all controls pass, AND (the VANILLA primary is not a genuine expert OR TRANSFER_MIN primary test gain - VANILLA primary test gain >= 2 pp).',
                    'secondary': 'PROMISING_BUT_NOT_EXPERT if strong GO fails AND >=3 of TRANSFER_MIN final top-5 reach >= +3 pp AND their mean gain >= +3 pp AND NOT (VANILLA has >=3 of its top-5 at >= +3 pp with mean >= +3 pp). Not proof of the paper.',
                    'otherwise': 'NO_GO_TRANSFER_AWARE_VISUAL_SEARCH and stop.',
                    'table': 'Best final-test gain = max over the method top-5; top-5 final mean over the method top-5; genuine expert found = any of the method top-5.'},
     'statistics': {'bootstrap_replicates': 10000, 'bootstrap_seed': 20261003,
                    'method': 'paired question percentile bootstrap; common PCG64 int64 indices per split; 2.5/97.5 linear percentiles; exact two-sided McNemar reported, descriptive only'},
     'controls': {'zero': 'Zero perturbation reproduces base outputs on every evaluated split in every phase; baseline repeat equals base.',
                  'repeat': 'First candidate of every search shard, validation and test phase is reconstructed again; identical state fingerprint and outputs.',
                  'all_candidates': 'Exact bitwise snapshot before and after every candidate; parameters and buffers.',
                  'fingerprints': 'Validation and test reconstruct each candidate and reject before generation unless its state fingerprint equals the search fingerprint.',
                  'encoder': 'Encoder cache cleared before every generation; every image freshly encoded; prefix and multimodal processor caches disabled.',
                  'locks': 'Dataset/folds, protocol, baseline, ranking and validation locks are git-committed before the phase that consumes them; analysis checks commit times.',
                  'operational_vs_scientific': 'Any control/runtime failure is operational and never a scientific decision.'},
     'budget': {'baseline': '1600 base + 1600 zero + 1600 base repeat', 'search': '400x300 + 8x(300 zero + 300 repeat)',
                'validation': '<=40x300 + 300 zero + 300 repeat', 'test': '<=12x1000 + 1000 zero + 1000 repeat'},
     'no_base_capability_gate': True, 'no_early_scientific_stop': True,
     'out_of_scope': 'No BO, CMA-ES, modular-norm RandOpt, 3B/32B scaling, new optimizer, geometry change, sigma change, population increase, prompt/benchmark/model change, TRANSFER_MIN tuning, or test labels for selection.',
     'previous_candidate_seeds': sorted(seeds)}
sets = {s: [json.loads(l) for l in (Path(DATASET)/(s+'.jsonl')).read_text().splitlines()] for s in ('search', 'validation', 'test')}
p['populations'] = {s: {'n': len(r), 'sha256': population_id(r)} for s, r in sets.items()}
fold_ids = load_folds(DATASET)
p['folds'] = {f: {'n': len(fold_ids[f]), 'sha256': __import__('hashlib').sha256(json.dumps(fold_ids[f]).encode()).hexdigest()} for f in FOLDS}
if set(s for s, _ in candidate_plan(p)) & seeds: raise ValueError('candidate seed reuse')
verify_inputs(p)
path = Path('experiments/transfer_aware_visual_protocol.json')
write(path, p)
print({'protocol_sha256': sha(path), 'dataset_files': files, 'prior_seeds_excluded': len(seeds),
       'populations': {s: v['n'] for s, v in p['populations'].items()}})

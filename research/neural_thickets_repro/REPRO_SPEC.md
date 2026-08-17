# Reproduction Specification — Neural Thickets §5.2 (Qwen2.5-VL-3B-Instruct + GQA)

Target: arXiv:2603.12228 ("Neural Thickets: Diverse Task Experts Are Dense Around Pretrained Weights",
Yulu Gan & Phillip Isola, ICML 2026 Spotlight), §5.2 and Appendix E.3.

Published result: Base **56.6%** → RandOpt (N=5000, K=50) **69.0%** on GQA.

**No upstream source code or paper text is copied into this document.** Each row describes behavior in
our own words and cites where it was verified: a `paper §/Appendix` reference, or an
`upstream_file:function_name` reference into `github.com/sunrainyg/RandOpt` (pinned commit
`536df0a308f3990b6270c991fbb96bd0b779a58e`, see `external/EXTERNAL_COMMIT.txt`). The upstream repo has
no declared license (`license: None` per the GitHub API), so its code is never vendored or transcribed
here or anywhere in this repo — it is only read, described, and invoked as an external subprocess.

## Specification table

| Item | Published / official value | Source | Status |
|---|---|---|---|
| Model checkpoint | `Qwen/Qwen2.5-VL-3B-Instruct` | paper §5.2 | Confirmed |
| Model revision | commit `66285546d2b821cf421d4f5eb2576359d3770cd3` (HF hub `main` as of this check) | fetched directly from HF hub `config.json`/index ETag, this session | Confirmed as of check date; **not** pinned by the paper or upstream repo — we are choosing to pin it ourselves for reproducibility |
| Model architecture / class | `Qwen2_5_VLForConditionalGeneration`, `tie_word_embeddings=true` (no separate `lm_head.weight` tensor; it is tied to `model.embed_tokens.weight`) | fetched `config.json` + `model.safetensors.index.json` from HF hub, this session | Confirmed |
| Dataset | GQA | paper §5.2 | Confirmed |
| Dataset source | HuggingFace `lmms-lab-encoder/GQA` (renamed from `lmms-lab/GQA` -- confirmed via HTTP redirect from the old name; `upstream data/README.md` still cites the pre-rename name), or an author-provided Google Drive "all-in-one" archive | `upstream data/README.md`; HF hub redirect, this session | Confirmed as *a* source; see next row for our own resolution of the exact subset |
| Dataset revision / exact subset | **Resolved by documented reproduction assumption** (not upstream-confirmed): test split = `testdev_balanced` config (12,578 questions over 398 unique images, verified by actually generating it this session); selection split = first `selection_set_size` rows of `train_balanced` (943,000 questions total, verified this session). Upstream never showed how its own `train.parquet`/`testdev.parquet` were produced -- these are the standard, commonly-cited GQA splits and the most plausible match for upstream's own file names ("testdev.parquet" most plausibly means testdev-balanced, the split GQA papers conventionally report one headline number against), not a confirmed match to upstream's actual (unpublished) prep step. | `src/neural_thickets_repro/prepare_gqa_data.py`, validated against the real dataset this session (see script docstring for exact HF configs used) | Resolved-by-assumption, clearly distinct from Confirmed |
| Selection split | referred to as "training set" by upstream, loaded from a `train.parquet`-shaped file | `upstream_file:data_handlers/gqa.py:GQAHandler.load_data`, `default_train_path` | Confirmed path/role; exact subset resolved-by-assumption above |
| Test split | referred to as "testdev", loaded from a `testdev.parquet`-shaped file | `upstream_file:data_handlers/gqa.py:GQAHandler.load_data`, `default_test_path` | Confirmed path/role; exact subset resolved-by-assumption above |
| Selection-set size | 200 (first 200 examples of the selection/training split) | paper Appendix E.3 ("first 200 samples from the training set as the RandOpt training set"), matches `upstream_file:randopt.py` CLI default `--train_samples 200` | Confirmed — paper and code agree |
| Test-set size | 12,578 (all of testdev_balanced, no subsampling) | `upstream_file:randopt.py` CLI default `--test_samples None`; exact count confirmed by generating the split this session | Confirmed count under the resolved-by-assumption split choice above |
| Prompt template | An instruction telling the model to look at the image, answer the given question, reason step by step, and place the final short answer inside a boxed/delimited span | `upstream_file:data_handlers/gqa.py:GQAHandler.load_data` (described, not transcribed) | Confirmed |
| Chat formatting | Applied via the tokenizer's chat template when the model name indicates an instruct/chat model (true for Qwen2.5-VL-3B-**Instruct**) | `upstream_file:randopt.py:main` (`format_prompt`) | Confirmed |
| Image message format | Multimodal chat message with an image content block followed by the text prompt block, in the format vLLM/Qwen2.5-VL expects | `upstream_file:data_handlers/gqa.py:GQAHandler.load_data` | Confirmed |
| Image preprocessing | Not explicitly configured upstream; relies on vLLM/Qwen2.5-VL's own default image handling (resize/max-pixels policy) | absence checked in `upstream_file:data_handlers/gqa.py`, `upstream_file:core/engine.py` | **UNRESOLVED** — no explicit resolution/max-pixels setting found anywhere upstream |
| Answer normalization / extraction | Boxed-answer extraction takes priority; falls back to pattern-based extraction from the response text; normalization lowercases, strips punctuation/articles, singularizes, and maps a fixed synonym table (e.g. sofa↔couch) before comparison; a final whole-word scan over the response is used as a last-resort match | `upstream_file:utils/reward_score/gqa.py` (`extract_boxed`, `extract_answer`, `compute_score`), `upstream_file:data_handlers/gqa.py` (`_normalize_answer`, `_singularize`, `_canonicalize`, `_match_answer`, `_whole_word_search`) — behavior described, not transcribed | Confirmed (behavior read and understood; exact regex/word-list content deliberately not copied into this repo — see `external/` for the authoritative source) |
| Baseline decoding | Greedy decoding (temperature 0), fixed generation seed, `max_tokens=256` for GQA | `upstream_file:randopt.py:main` (`SamplingParams`), `upstream_file:data_handlers/gqa.py` (`default_max_tokens`) | Confirmed |
| Perturbation target | Language-model component only | paper §5.2 ("we perturb the language model while keeping the visual encoder frozen") | Confirmed |
| Perturbation scope (exact) | Every named parameter is perturbed **except** ones whose name is prefixed `visual.` or `model.visual.` — i.e. token embeddings and any LM head weights (here tied to embeddings, so N/A as a separate tensor) **are** perturbed; only the vision tower is excluded, unless an explicit opt-in override is set | `upstream_file:utils/worker_extn.py:WorkerExtension._should_perturb` | Confirmed against upstream code, and cross-checked against the real checkpoint's actual tensor names (fetched from HF hub `model.safetensors.index.json` this session): 100% of vision-tower tensors are prefixed `visual.` (`visual.blocks.*`, `visual.merger.*`, `visual.patch_embed.*`), everything else is `model.embed_tokens.*` / `model.layers.*` / `model.norm` — the `visual.` rule cleanly separates them with no ambiguous tensor names |
| Vision encoder | Frozen (bitwise unchanged across candidate perturbations) | paper §5.2, `upstream_file:utils/worker_extn.py:_should_perturb` | Confirmed |
| Perturbation formula | θ' = θ + σε, ε ~ N(0, I), generated fresh per parameter tensor from a seeded generator, added in-place in the parameter's native dtype (no fp32 accumulation in this code path); restoration re-seeds identically and subtracts the same noise rather than restoring from a stored weight copy | paper §4 (formula); `upstream_file:utils/worker_extn.py:perturb_self_weights`/`restore_self_weights` (mechanism) | Confirmed |
| N (population size) | 5000 | paper §5.2 | Confirmed |
| K (ensemble size) | 50 | paper §5.2 | Confirmed |
| Candidate (seed, sigma) assignment | For a run of size N: N unique seeds drawn without replacement from a large integer range; one sigma per candidate drawn **with** replacement from the configured sigma set; both draws come from one global-seed-derived RNG stream | `upstream_file:randopt.py:run_sampling` | Confirmed as the *mechanism*; the exact sigma set used for this experiment is unresolved (next row) |
| Sigma / sigma set | **UNRESOLVED — no single confirmed value or set for this exact experiment.** Three non-equivalent candidates found: (1) the code's global CLI default set; (2) the value used in the upstream example launch scripts (for a different model/dataset, Olmo-3-7B/countdown, not Qwen-VL/GQA); (3) a single value `σ=0.005` mentioned in paper Appendix E.3, in a sentence about "landscape analysis" whose scope relative to the headline GQA result is not clearly stated in the fetched text | paper Appendix E.3; `upstream_file:randopt.py` CLI default; `upstream_file:scripts/local_run.sh` / `scripts/single_node.sh` | **UNRESOLVED — first-class open reproduction variable.** Treated as a set of labeled sensitivity-analysis configs in Gate 2, never silently resolved to one value and presented as confirmed. |
| Perturbation distribution | Gaussian, i.i.d. per-weight, standard normal scaled by σ | paper §4 | Confirmed |
| Precision | bfloat16 (upstream CLI default) | `upstream_file:randopt.py` CLI `--precision` default | Confirmed as upstream default; not independently confirmed this is what was used for this specific published run |
| Selection metric | Mean 0/1 reward on the selection set (same reward function used for final correctness, not a proxy metric) | `upstream_file:data_handlers/base.py:DatasetHandler.postprocess_outputs`, `upstream_file:data_handlers/gqa.py:GQAHandler.compute_reward` | Confirmed |
| Top-K selection rule | Sort all N scored candidates by selection-set score, descending; take the top K as a prefix of that sorted list | `upstream_file:randopt.py:main` | Confirmed |
| Voting procedure | Majority vote: for each test example, take the most-common normalized answer among the top-K models' (non-empty) predictions; an example where every model produced an empty/unparseable answer counts as incorrect (not excluded from the denominator) | `upstream_file:randopt.py:run_ensemble_evaluation` | Confirmed |
| Tie-breaking rule | Not an explicit rule upstream — it is a consequence of Python's `collections.Counter.most_common`, which preserves first-insertion order among tied counts; since models are inserted in selection-score-descending order, ties are implicitly resolved in favor of the higher-selection-score model | `upstream_file:randopt.py:run_ensemble_evaluation` | Confirmed (as an implementation consequence, not a documented design choice) |
| Multimodal engine flag | Upstream's engine launch call for the GQA path does not appear to pass an explicit multimodal-enable flag to vLLM, despite GQA prompts containing image content | `upstream_file:randopt.py:main` vs. `upstream_file:core/engine.py:launch_engines` (`multimodal` parameter) | **UNRESOLVED** — cannot tell from reading alone whether this is intentional (the pinned vLLM version auto-detects multimodal models) or a latent gap; only resolvable by testing on GPU hardware, tracked in `src/neural_thickets_repro/vlm_adapter.py` |
| Weight-reconstruction script model class | Upstream's standalone single-expert reconstruction script loads the checkpoint via a causal-LM-only model class, which is not the correct class for this VLM checkpoint (`Qwen2_5_VLForConditionalGeneration`) | `upstream_file:utils/repro_seed.py` vs. checkpoint `config.json` `architectures` field | **Confirmed discrepancy** — documented and corrected only in our own `src/neural_thickets_repro/vlm_adapter.py`, never by editing the external clone |

## Sigma — resolution plan (see Pipeline gates in the plan)

Gate 2 (small-scale RandOpt on real GPU hardware) will run each candidate sigma set below as a labeled,
independent sensitivity configuration and record all results side by side. No candidate is presented as
"the" reproduced sigma unless the paper's ambiguity is independently resolved first (e.g. by re-reading
the full paper with more budget, or by locating an author clarification). The candidates, as of this
writing:

1. `sigma_default = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01]` — upstream CLI default.
2. `sigma_example_scripts = [0.0005, 0.001, 0.002]` — upstream `scripts/local_run.sh` / `scripts/single_node.sh`, used there for Olmo-3-7B/countdown, not confirmed for Qwen-VL/GQA.
3. `sigma_appendix_e3 = [0.005]` — paper Appendix E.3's single stated value, ambiguous scope.

`configs/gqa_repro.yaml` leaves `randopt.sigmas` as `null` with a comment pointing here.

## Explicitly out of scope for this phase

Layer localization, module-wise RandOpt, construction datasets, low-rank search, CoRP, Iterative RandOpt,
routing, distillation, continual learning, and any other WACV-extension work. This document and the
accompanying Gate 0 scaffold cover reproduction of the published §5.2 result only.

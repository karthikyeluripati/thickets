# Scripts index

Scripts stay in this one flat folder on purpose: their paths are written into the committed plan locks
(`results/paper-analysis/*/plan_lock.md`), and several import each other (`sys.path` points here). Prefixes match
the experiment folders in `results/README.md`. Run everything from the repository root. GPU scripts ran on rented
pods via `pod_jobqueue.sh`; analysis scripts are CPU-only.

## Claim 1: what weight search buys
| Experiment | GPU (run on a pod) | CPU analysis |
|---|---|---|
| C, C3B (GSM8K same-run; RandOpt's own `randopt.py`) | – | `c_prep_gsm8k.py`, `c_analysis.py` |
| G1 (frozen GQA items) | `g1_eval.py`, `g1_gradient.py`, `g1_images.py` | `g1_check.py`, `g1_analysis.py` |
| G2, G2R, GD (GQA same-run) | `g2_prep.py`, `g2_randopt_gqa.py` (`--pop-seed`, `--prompt`) | `g2_analysis.py`, `g2r_analysis.py`, `gd_analysis.py` |
| O1 (GSM8K same-run, OLMo-2-1B; RandOpt's own `randopt.py`) | – | `o1_analysis.py` (uses `c_prep_gsm8k.py`, `p2_sc.py`) |
| GB (GSM8K search under the boxed prompt) | `gsm_randopt_fast.py` | `gb_analysis.py` |
| O2 (OLMo prompt control) | `o2_eval.py` (`--prompt randopt/plain/boxed`) | `o2_analysis.py` |
| Q2 (Qwen prompt control) | `o2_eval.py` | `q2_analysis.py` |
| RV (reviewer round: public templates, Qwen boxed searches, base print, temperature) | `o2_eval.py`, `g3_eval.py` (new templates, `--temperature`), `gsm_randopt_fast.py`, `rv_choose.py`, `rv_gates.py`, `rv_resume_q3.py`; pod: `pod_rv_common.sh`, `rv_deploy.sh`, `rv_watch.sh` | `rv_analysis.py`, `gb_analysis.py` |
| PS (prompt selection, all rows) | `o2_eval.py --split select`, `g3_eval.py --split selection` | `ps_analysis.py` |
| G3, G4 (budgets; prompt controls) | `g3_eval.py` (`--prompt cot/direct/short`) | `g3_analysis.py`, `g4_analysis.py` |
| GQA shift (exploratory) | – | `gqa_shift_analysis.py` |
| Robustness of Claim 1 (exploratory) | – | `robustness_checks.py` |
| Selection gain vs test gain (exploratory) | – | `transfer_analysis.py` |
| Prompt-specific experts (exploratory) | – | `expert_specificity.py` |
| P0–P3 (pilots; SC vs published RandOpt) | `p0_hf.py`, `p1_sample.py`, `p2_sc.py` | `p0_analysis.py`, `p1_analysis.py`, `p2_analysis.py`, `p3_analysis.py` |

## Claim 2: the first-order tilt law
| Experiment | GPU | CPU analysis |
|---|---|---|
| Stage 1+2 | `stage12_measure.py`, `stage12_gradient.py`, `stage12_position_check.py` | `stage12_analysis.py` |
| Stage 3 | `stage3_gradient.py`, `stage3_localize.py` | `stage3_analysis.py` |
| R1 | `r1_extract_images.py`, `r1_measure.py`, `r1_gradient.py` | `r1_analysis.py` |
| R2, R2b | `r2_measure.py`, `r2_gradient.py`, `r2_vllm_check.py` | `r2_analysis.py`, `r2b_analysis.py` |
| I5 | `i5_gradient.py` | `i5_analysis.py` |
| S1, S1-7B | `s1_a_olmo.py`, `s1_b_winner.py` | `s1_analysis.py` |
| GPU-A (vision weights) | `geometry_extract_images.py`, `geometry_gpu_a.py`, `geometry_gpu_a2.py`, `geometry_vllm_check.py` | `geometry_analysis.py`, `geometry_analysis_a1.py` |
| Theory checks | – | `theory_check.py` |

## Claim 3: OmniSpatial case study
| Experiment | GPU | CPU analysis |
|---|---|---|
| M1 | `m1_prepare_images.py`, `m1_measure.py` | `m1_analysis.py` |
| C1 | – | `c1_analysis.py` |
| OS (prompt control for the winner) | `os_prep.py`, `os_prompt_eval.py` | `os_analysis.py` |
| Answer priors (post-hoc) | – | `answer_prior_analysis.py`, `answer_prior_bias_model.py`, `answer_prior_checks.py`, `answer_prior_goldsplit.py` |

## Shared modules (imported by other scripts)
`geometry_lib.py` (13 scripts), `r1_common.py`, `r2_common.py`, `stage12_gradient.py` (`tilt_weights`, `is_front`),
`stage3_gradient.py` (`block_of`, `GROUPS`).

## Pod session scripts
`pod_jobqueue.sh` (current: guard with hard cost cap, idle stop and stop-on-retrieved, setup, ordered job queue);
`pod_rv_common.sh` (shared launcher: layered worker launch, retry, gates, search row), `rv_deploy.sh`, `rv_watch.sh`;
`pod_g1.sh`, `pod_stage12.sh`, `pod_geometry_a.sh`, `pod_geometry_a2.sh`, `pod_geometry_a2_s2c.sh` (earlier sessions).

## Superseded (split mismatch; kept so the records reproduce)
`causal_diag_9504111.py` (also a shared module: many runners import its image and prompt helpers) and
`selection_vs_specificity.py` (produces the CSV Stage 3 reads; imports `selection_vs_shared.py`).

Figures and paper tables: `paper/figures/scripts/` (`thickets_or_tilts.py`, `build_master.py`).

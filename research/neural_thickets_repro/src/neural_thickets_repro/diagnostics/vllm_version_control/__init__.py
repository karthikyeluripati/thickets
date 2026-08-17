"""Final controlled causal test: vLLM 0.27.1 (current) vs vLLM 0.11.0 (paper-era, per
docker/Dockerfile_vllm) on an identical fixed 200-example sample. Diagnosis only.

Three scripts, run in this order:
  1. select_fixed_sample.py   -- pins the 200 examples once (question IDs + metadata),
                                  so both environments evaluate exactly the same set.
  2. generate_predictions.py -- run TWICE, once per vLLM environment (0.27.1's existing
                                  venv, and an isolated 0.11.0 venv/container) -- identical
                                  script, identical args, only the installed vLLM differs.
  3. compare_results.py      -- environment-agnostic (no vLLM/GPU needed): scores both
                                  prediction sets with march-era scoring and applies the
                                  pre-agreed decision rule.
"""

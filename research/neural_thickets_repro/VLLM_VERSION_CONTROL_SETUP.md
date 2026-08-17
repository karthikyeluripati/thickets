# Final controlled causal test: vLLM 0.27.1 vs 0.11.0 (paper-era)

This is the last planned diagnostic step for the residual −2.41pp Gate 1 gap. It compares
the current working environment (vLLM 0.27.1) against the paper repo's own pinned
`vllm/vllm-openai:v0.11.0` (`docker/Dockerfile_vllm`, unchanged March 2026 → pinned HEAD) on
an identical fixed 200-example sample, then applies a pre-agreed statistical decision rule.

**Isolation approach: a fresh Python venv, not Docker.** RunPod pods commonly run without
privileged/nested-container access, so Docker-in-Docker for the `vllm/vllm-openai:v0.11.0`
image may simply not be available. A separate venv gives equivalent isolation for this
purpose (`generate_predictions.py` is deliberately standalone — it imports nothing from our
package, only `transformers`/`vllm`/`Pillow`/`huggingface_hub` — so it drops into the new
venv without needing our package or `requirements-gpu.txt` installed there at all) and
**cannot touch or downgrade the existing 0.27.1 environment**, which lives in a different
venv/site-packages entirely. If you do have working GPU-passthrough Docker-in-Docker, the
Docker route is noted as an alternative at the bottom — functionally equivalent, just more
setup risk on a typical pod.

**GPU is shared and not meant to run both versions concurrently** — check `nvidia-smi` is
clear of other processes before starting each generation run.

## 1. Pin the fixed 200-example sample (once, in the normal 0.27.1 environment)

```bash
cd /workspace/thickets/research/neural_thickets_repro
git pull origin neural-thickets-repro-gate1-prep
pip install -e .

python -m neural_thickets_repro.diagnostics.vllm_version_control.select_fixed_sample \
    --sample-size 200 --seed 42 \
    --out results/gate1_diagnosis/vllm_version_control/fixed_200.json
```

## 2. Generate under vLLM 0.27.1 (current environment — no changes needed here)

```bash
python src/neural_thickets_repro/diagnostics/vllm_version_control/generate_predictions.py \
    --fixed-sample results/gate1_diagnosis/vllm_version_control/fixed_200.json \
    --model-name Qwen/Qwen2.5-VL-3B-Instruct \
    --revision 66285546d2b821cf421d4f5eb2576359d3770cd3 \
    --max-tokens 256 --seed 42 --label vllm0271 \
    --out-dir results/gate1_diagnosis/vllm_version_control
```

## 3. Isolated venv for vLLM 0.11.0 — build once

```bash
nvidia-smi   # confirm no other GPU process is currently running

python3 -m venv /workspace/venv_vllm0110
source /workspace/venv_vllm0110/bin/activate

pip install "vllm==0.11.0"
pip install pillow huggingface_hub   # in case not already pulled in transitively

# sanity check the pin actually took (this venv only, does not affect the main environment):
python -c "import vllm; print(vllm.__version__)"   # must print 0.11.0
```

## 4. Generate under vLLM 0.11.0 (isolated venv — same script, run as a plain file, no package install needed)

```bash
# still inside: source /workspace/venv_vllm0110/bin/activate
python /workspace/thickets/research/neural_thickets_repro/src/neural_thickets_repro/diagnostics/vllm_version_control/generate_predictions.py \
    --fixed-sample /workspace/thickets/research/neural_thickets_repro/results/gate1_diagnosis/vllm_version_control/fixed_200.json \
    --model-name Qwen/Qwen2.5-VL-3B-Instruct \
    --revision 66285546d2b821cf421d4f5eb2576359d3770cd3 \
    --max-tokens 256 --seed 42 --label vllm0110 \
    --out-dir /workspace/thickets/research/neural_thickets_repro/results/gate1_diagnosis/vllm_version_control

deactivate   # back to the normal environment for step 5
```

**If vLLM 0.11.0's API doesn't accept the same `LLM(...)`/`SamplingParams(...)`/
`multi_modal_data` shapes as 0.27.1** (plausible across a 16-minor-version gap), that is a
mechanical library-compatibility issue, not a reproduction-behavior change — fix the call
signature only (e.g. an older multimodal input kwarg name), do not touch the prompt text,
image content, decoding parameters, or scoring. Note whatever had to change in
`GATE1_DIAGNOSIS.md` when you report back.

## 5. Compare (back in the normal 0.27.1 environment — no GPU needed for this step)

```bash
python -m neural_thickets_repro.diagnostics.vllm_version_control.compare_results \
    --predictions-a results/gate1_diagnosis/vllm_version_control/predictions_vllm0271.jsonl \
    --predictions-b results/gate1_diagnosis/vllm_version_control/predictions_vllm0110.jsonl \
    --label-a vllm0271 --label-b vllm0110
```

Prints accuracy for both, the paired delta, predictions-changed count, both discordant
counts, the McNemar exact p-value, and a `RECOMMENDATION` field:
`RUN_FULL_BASELINE_UNDER_B` or `STOP_AND_ACCEPT_GATE1_AS_PAPER_FAITHFUL_WITH_UNRECOVERABLE_RUNTIME_DISCREPANCY`.

## Decision rule (fixed before any numbers exist)

vLLM 0.11.0 counts as a **meaningful improvement** only if it is directionally better AND
the McNemar exact two-sided p-value on the 200-example paired sample is < 0.05. Not a raw
percentage-point threshold — at n=200 a few-point swing is not reliably distinguishable
from sampling noise, and an arbitrary cutoff would just be a different kind of tuning.

- **Significant improvement** → run the full 12,578-example baseline under vLLM 0.11.0
  (same isolated venv), same as `eval_base_image_aware.py` but pointed at that venv's
  Python interpreter.
- **Not significant** → stop. Classify the Gate 1 reconstruction as a paper-faithful
  reproduction of the released method with a documented, unrecoverable runtime-version
  discrepancy (the paper repo itself never pinned exact versions either). Gate 1 accepted
  for the purpose of continuing the research; Gate 2 prep (not launch) begins.

## Docker alternative (only if you have working GPU-passthrough Docker-in-Docker)

```bash
docker run --rm --gpus all \
    -v /workspace/thickets:/workspace/thickets \
    -v /workspace/hf_cache:/root/.cache/huggingface \
    vllm/vllm-openai:v0.11.0 \
    python /workspace/thickets/research/neural_thickets_repro/src/neural_thickets_repro/diagnostics/vllm_version_control/generate_predictions.py \
        --fixed-sample /workspace/thickets/research/neural_thickets_repro/results/gate1_diagnosis/vllm_version_control/fixed_200.json \
        --model-name Qwen/Qwen2.5-VL-3B-Instruct \
        --revision 66285546d2b821cf421d4f5eb2576359d3770cd3 \
        --max-tokens 256 --seed 42 --label vllm0110 \
        --out-dir /workspace/thickets/research/neural_thickets_repro/results/gate1_diagnosis/vllm_version_control
```

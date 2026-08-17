# Final controlled causal test: vLLM 0.27.1 vs 0.11.0 (paper-era, official Docker image)

This is the last planned diagnostic step for the residual −2.41pp Gate 1 gap. It compares
the current working environment (vLLM 0.27.1) against the **official
`vllm/vllm-openai:v0.11.0` Docker image** — the exact base image `docker/Dockerfile_vllm`
pins, unchanged between the March 2026 commit and the pinned HEAD — on an identical fixed
200-example sample, then applies a pre-agreed statistical decision rule.

**Isolation: the official Docker image**, matching the RandOpt repo's own documented
environment more faithfully than a pip-only venv would (the image pins the surrounding
CUDA/system libraries too, not just the `vllm` package version). `generate_predictions.py`
is deliberately standalone — it imports nothing from our package, only
`transformers`/`vllm`/`Pillow`/`huggingface_hub`, all of which vLLM's own image already
bundles — so it runs inside the container unmodified, no extra install step expected.
**Cannot touch or downgrade the existing 0.27.1 environment**: nothing about running a
separate container writes to the host's Python environment.

**GPU is shared and not meant to run both versions concurrently** — check `nvidia-smi` is
clear of other processes before starting each generation run, and don't run the container
while a 0.27.1 process still holds GPU memory.

## 0. Sanity-check Docker + GPU passthrough before relying on it mid-run

```bash
docker --version
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```
If this fails (no privileged access, no nested containers on this pod), stop here and say
so rather than debugging Docker itself as part of this reproduction task — that's an
infrastructure question, not a reproduction one.

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

## 3. Fidelity gate — MANDATORY, must PASS before touching vLLM 0.11.0

`generate_predictions.py` is a standalone harness (deliberately not importing our package,
so it runs inside the isolated 0.11.0 container). Before trusting its 0.27.1 output as one
arm of the version comparison, prove it actually reproduces the already-validated
`eval_base_image_aware.py` run (which completed successfully over all 12,578 examples) on
the same 200 IDs. A harness bug here would be indistinguishable from a real vLLM-version
effect if left unchecked.

```bash
python -m neural_thickets_repro.diagnostics.vllm_version_control.fidelity_gate \
    --fixed-sample results/gate1_diagnosis/vllm_version_control/fixed_200.json \
    --new-predictions results/gate1_diagnosis/vllm_version_control/predictions_vllm0271.jsonl \
    --baseline-predictions results/base_image_aware/predictions.jsonl
```

Reports: IDs matched, raw-prediction exact-match count/rate, march-score accuracy from the
old full-baseline records (this 200-subset) vs. from the new helper, correctness
agreement count/rate, and up to 20 concrete disagreements. Exits 0 with `GATE RESULT: PASS`
only if ≥98% raw-text exact match AND ≥98% correctness agreement — greedy decoding on
identical model/precision/prompt/image/seed within the same vLLM version should be
deterministic, so anything short of that means the harness is doing something different,
not sampling noise. **If it prints `FAIL`, stop — do not run the Docker step below until
this passes**, and explain (don't paper over) any remaining disagreements it reports.

## 4. Generate under the official vLLM 0.11.0 Docker image

Mount the repo (and the HF cache, so the already-downloaded model snapshot is reused
instead of a redundant multi-GB re-download) at **identical paths** inside the container --
`fixed_200.json`'s image paths are absolute host paths under
`/workspace/thickets/...`, so mounting at the same path means they resolve correctly
inside the container with no rewriting.

`--entrypoint python3` is required: the `vllm/vllm-openai` image's default entrypoint
launches the OpenAI-compatible server, not a plain Python interpreter -- overriding it is
the standard way to run an arbitrary script in that image. `--ipc=host` is vLLM's own
documented requirement for Docker (its multiprocessing/tensor ops need more shared memory
than Docker's small default).

```bash
docker run --rm --gpus all --ipc=host \
    --entrypoint python3 \
    -v /workspace/thickets:/workspace/thickets \
    -v /workspace/hf_cache:/root/.cache/huggingface \
    -e HF_HOME=/root/.cache/huggingface \
    vllm/vllm-openai:v0.11.0 \
    /workspace/thickets/research/neural_thickets_repro/src/neural_thickets_repro/diagnostics/vllm_version_control/generate_predictions.py \
        --fixed-sample /workspace/thickets/research/neural_thickets_repro/results/gate1_diagnosis/vllm_version_control/fixed_200.json \
        --model-name Qwen/Qwen2.5-VL-3B-Instruct \
        --revision 66285546d2b821cf421d4f5eb2576359d3770cd3 \
        --max-tokens 256 --seed 42 --label vllm0110 \
        --out-dir /workspace/thickets/research/neural_thickets_repro/results/gate1_diagnosis/vllm_version_control
```

**If vLLM 0.11.0's Python API doesn't accept the same `LLM(...)`/`SamplingParams(...)`/
`multi_modal_data` shapes as 0.27.1** (plausible across a 16-minor-version gap), that's a
mechanical library-compatibility issue, not a reproduction-behavior change -- fix the call
signature only (e.g. an older multimodal input kwarg name), never the prompt text, image
content, decoding parameters, or scoring. Note whatever had to change in
`GATE1_DIAGNOSIS.md` when you report back. If a bundled dependency is missing inside the
image (unlikely -- `transformers`/`Pillow`/`huggingface_hub` all ship with vLLM's own
image), `docker run` with an interactive shell and `pip install <package>` first, same
container/image, still isolated from the host.

## 5. Compare (back in the normal 0.27.1 environment — no GPU/Docker needed for this step)

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

- **Significant improvement** → run the full 12,578-example baseline under the same
  `vllm/vllm-openai:v0.11.0` container (same `docker run` shape as step 4, pointed at all
  12,578 examples instead of the fixed 200 -- an image-aware equivalent of
  `eval_base_image_aware.py` for that environment).
- **Not significant** → stop. Classify the Gate 1 reconstruction as a paper-faithful
  reproduction of the released method with a documented, unrecoverable runtime-version
  discrepancy (the paper repo itself never pinned exact versions either, only the Docker
  base-image tag). Gate 1 accepted for the purpose of continuing the research; Gate 2 prep
  (not launch) begins.

## Fallback: isolated venv, only if Docker-in-Docker isn't available on this pod

If step 0's sanity check fails, `generate_predictions.py` also runs fine in a plain venv --
it has no dependency on Docker specifically:

```bash
python3 -m venv /workspace/venv_vllm0110 && source /workspace/venv_vllm0110/bin/activate
pip install "vllm==0.11.0" pillow huggingface_hub
python /workspace/thickets/research/neural_thickets_repro/src/neural_thickets_repro/diagnostics/vllm_version_control/generate_predictions.py \
    --fixed-sample /workspace/thickets/research/neural_thickets_repro/results/gate1_diagnosis/vllm_version_control/fixed_200.json \
    --model-name Qwen/Qwen2.5-VL-3B-Instruct --revision 66285546d2b821cf421d4f5eb2576359d3770cd3 \
    --max-tokens 256 --seed 42 --label vllm0110 \
    --out-dir /workspace/thickets/research/neural_thickets_repro/results/gate1_diagnosis/vllm_version_control
deactivate
```
This pins the `vllm` package version but not the surrounding system libraries the official
image provides — less faithful to "the RandOpt repo's documented environment" than Docker,
so prefer Docker whenever it's actually available on the pod.

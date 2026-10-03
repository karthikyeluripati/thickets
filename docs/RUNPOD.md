# RunPod: first measurement session

## Resource scope

No GPU was available in the development environment. GPU tests and LLM experiments
are still pending. Start with **one CUDA GPU**, not a cluster. For a 0.5B–1.5B dense
model and a small batch, a 24 GB memory budget gives substantial room for this
pilot. A 48 GB budget gives more headroom for 3B-class tests and snapshots. These
are planning budgets, not a guarantee for every model/context. Model weights plus
base snapshot alone cost approximately `2 * parameters * bytes_per_parameter`;
KV cache, activations, library workspaces and per-parameter noise add to that.
Do not run the whole large-model sweep initially.

Use a CUDA-enabled PyTorch image compatible with its driver. The only environment
validated initially is Python 3.13.5 / PyTorch 2.10.0+cpu. Dependency bounds are not a
claim that every allowed GPU/framework version has been tested. Preserve the
container image/version and `pip freeze` for any reported run.

## Get the source

If the branch is published:

```bash
git clone --branch research/candidate-runtime-baseline \
  https://github.com/karthikyeluripati/thickets.git
cd thickets
```

Otherwise upload/extract the delivered source ZIP, or apply the delivered patch
on a local clone of main as described in `DELIVERY.md`. No remote branch existed
at initial delivery: GitHub branch creation was denied by the integration.

## Preflight and synthetic smoke

```bash
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available())'
pip install -e '.[test]'
pytest -q
bash scripts/run_gpu_pilot.sh runs/first-gpu
```

The shell script will not start if CUDA is unavailable, and it refuses an existing
output directory. It runs a modest **synthetic** model, not a real language model.
Review CUDA tests, reset/candidate gates and trace integrity first.

## Real LLM lifecycle

Install the optional HF dependencies and use a local model snapshot or an immutable
40-character model revision. Start with a small non-gated model already available
to you. No Hugging Face token is required for the synthetic smoke; do not paste
private tokens or SSH private keys into the conversation or repository.

```bash
pip install -e '.[hf,test]'
pip freeze > runs/first-gpu/pip-freeze-hf.txt
thicket-profile --workload hf --device cuda:0 --dtype bfloat16 \
  --model MODEL_ID --revision FULL_40_CHARACTER_COMMIT \
  --data examples/arithmetic_smoke.jsonl --max-new-tokens 32 \
  --candidates 8 --repeats 3 --out runs/llm-32
```

Then use a representative frozen scoring set and a longer budget, keeping all
other settings fixed. The four example prompts are not sufficient research data.
Inspect generated lengths for natural-length runs; add `--fixed-length` only for
an explicitly labeled fixed-token microbenchmark. Prefix/KV caches are not shared
across candidates. Avoid concurrent workloads on the measured GPU.

A diagnostic operator trace uses `--trace --candidates 2 --repeats 1 --warmup 0`.
Use separate output paths and do not mix its throughput with coarse runs.

## What to return

Preserve the run directories, GPU/driver information, package versions and original
command. `manifest.json`, `candidates.json`, per-strategy reports and `summary.json`
are needed to audit a result; optional traces attribute operator costs.

The next scientific decision is based on measured state-overhead fractions and
correctness, not on an assumed benefit of virtual weights.

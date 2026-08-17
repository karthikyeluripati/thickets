# Running Gate 1 on RunPod (1x L40S 48GB)

This machine (local dev box) has no GPU and 13-15GB free disk, so none of this can execute
here. Everything below is meant to run **on the RunPod pod**, not locally. I do not have
direct shell/SSH access to your pod, so these are the exact commands for you (or an agent
running on the pod) to execute — I can't run them for you from here.

**Scope: Gate 1 baseline only.** Do not run `run_randopt.py` (Gate 2/3) until Gate 1 is
reviewed and accepted.

## Update: disk-check fix (pull this before continuing)

`validate_env`'s disk check had a real bug: `check_disk()` called `shutil.disk_usage(path.anchor)`,
and `.anchor` collapses any POSIX path to `/` regardless of mount points further down the
tree — so it was reporting free space on the container's root filesystem instead of the
actual persistent `/workspace` volume the repo/data/cache live under. Fixed to check the
real path directly (`shutil.disk_usage` correctly resolves through mount points when given
an actual path, not its anchor). `validate_env` now also checks disk space at `HF_HOME`
separately from the repo root (they can be different filesystems), and prints an advisory
filesystem-consistency line. The required free-space threshold (`hardware.min_free_disk_gb`,
25GB) is unchanged.

Since your GQA prep already completed (testdev: 12,578/398, selection: 200/154 — both
plausible counts, matching what I validated locally at small scale), **do not re-run
`prepare_gqa_data.py`**. The `PyGILState_Release` abort you saw happened at interpreter
shutdown, after the script's own `Done.` print — that's consistent with a third-party C
extension misbehaving during teardown, not with the actual parquet/image writing having
failed (that work was already complete and flushed to disk by the time `Done.` printed).
Verify the artifacts directly instead (step below) rather than inferring anything from the
shutdown exception.

```bash
cd /workspace/thickets   # or wherever you cloned it
git fetch origin
git checkout neural-thickets-repro-gate1-prep
git pull origin neural-thickets-repro-gate1-prep
cd research/neural_thickets_repro
pip install -e .   # re-run if this is a fresh shell and the editable install didn't persist

# 1. Verify the already-prepared GQA data is actually intact (does NOT re-download anything)
python -m neural_thickets_repro.verify_gqa_data --config configs/gqa_repro.yaml
# Exit code 1 / "VERIFICATION FAILED" means something really is missing/corrupt -- only
# THEN consider re-running prepare_gqa_data.py, and only for the specific split that failed.

# 2. Re-run validate_env with the fix
python -m neural_thickets_repro.validate_env --config configs/gqa_repro.yaml
# Gate 1 should now read FEASIBLE if /workspace actually has >= 25GB free.
```

If `verify_gqa_data` passes and `validate_env` shows Gate 1 FEASIBLE, skip straight to
section 7 (Gate 1 baseline) below — sections 0-6 describe the full first-time setup and are
otherwise unchanged.

## 0. Get the code onto `/workspace`

Pick one:

**Option A — git (recommended, repo already has a remote):**
```bash
# on your LOCAL machine, review then push what Gate 0 + this prep pass produced:
git add research/ .gitignore
git status   # review before committing -- nothing else should be staged
git commit -m "Add Neural Thickets §5.2 reproduction Gate 0 scaffold + Gate 1 prep"
git push origin main

# on the POD:
git clone git@github.com:karthikyeluripati/thickets.git /workspace/thickets
# (needs your SSH key or a PAT added to the pod -- if that's not set up, use Option B)
cd /workspace/thickets/research/neural_thickets_repro
```

**Option B — direct transfer (no push needed):**
```bash
# from your LOCAL machine, if the pod exposes SSH:
rsync -avz --exclude='.venv' --exclude='__pycache__' \
    "c:/Users/karth/OneDrive/Desktop/projects/thickets/research" \
    root@<pod-ip>:/workspace/thickets/
```

I did **not** run the `git add`/`commit`/`push` above myself — those are yours to run when
ready (committing/pushing wasn't something you'd asked for yet).

## 1. Python environment

RunPod reports PyTorch 2.8.0+cu128 preinstalled — **don't let a blind `pip install` clobber
that CUDA-matched build.** Install the lightweight deps first, then vLLM/Ray separately with
a verification step in between:

```bash
cd /workspace/thickets/research/neural_thickets_repro
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"  # sanity baseline

pip install transformers accelerate qwen-vl-utils pillow datasets huggingface-hub pandas pyarrow pyyaml pytest

pip install "ray[default]>=2.0.0"
pip install "vllm>=0.10.0"

# STOP AND CHECK -- if torch got downgraded or CUDA broke, don't proceed, report back instead:
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import vllm; print('vllm', vllm.__version__)"
python -c "import ray; print('ray', ray.__version__)"

pip install -e .   # installs neural_thickets_repro so `python -m neural_thickets_repro.X` works
```

## 2. Point HF cache at persistent storage

```bash
export HF_HOME=/workspace/hf_cache
mkdir -p "$HF_HOME"
```
(Add this to your shell profile or the pod's env so it survives across commands.)

## 3. Clone the external RandOpt repo (pinned commit, never committed to this repo)

```bash
python external/setup_external_repo.py
```
Confirms/clones `external/RandOpt/` at commit `536df0a308f3990b6270c991fbb96bd0b779a58e`.

## 4. Prepare the real GQA data

Resolves `REPRO_SPEC.md`'s dataset-provenance row: testdev_balanced (12,578 questions / 398
images, ~66MB) as the test split, first 200 rows of train_balanced as the selection set.
Already validated end-to-end against the real HF dataset during this session (small scale) —
this is the same script, just running the full 200-row selection set instead of 5.

```bash
python -m neural_thickets_repro.prepare_gqa_data --config configs/gqa_repro.yaml
python -m neural_thickets_repro.verify_gqa_data --config configs/gqa_repro.yaml
```

Always run `verify_gqa_data` right after prep, regardless of whether the prep script exited
cleanly — it checks the actual parquet row counts/columns and that every referenced image
exists and isn't corrupt, rather than trusting the prep script's own exit behavior.

## 5. Validate the environment

```bash
python -m neural_thickets_repro.validate_env --config configs/gqa_repro.yaml
```
Gate 1 should now read **FEASIBLE**. If it doesn't, the reason is printed — fix that before
continuing rather than forcing past it.

## 6. Drift sanity check (small, real-model — NOT the full audit)

Separate tool, does not touch the reproduction algorithm. For Gate 1, only a small 1/10-cycle
real-model sanity check is required — the full 1/10/100/1000/5000-cycle audit is potentially
expensive on the 3B VLM and is not a Gate-1 requirement, so it's deferred (the tool still
supports it later via `--cycles 1 10 100 1000 5000`, unchanged):

```bash
python -m neural_thickets_repro.diagnostics.perturb_restore_drift \
    --model-name Qwen/Qwen2.5-VL-3B-Instruct --dtype bfloat16 --sigma 0.001 \
    --cycles 1 10 --check-predictions --device cuda \
    --out results/drift_audit_real_model_1_10.json
```

A CPU-only run against the Gate-0 synthetic dummy model (bf16, full 1-1000 cycle sweep)
already showed relative-norm drift growing from ~0.05% (1 cycle) to ~0.77% (1000 cycles) —
confirming the concern is real at bf16 precision, not just a float32 curiosity. This 1/10-cycle
real-model check is just confirming the mechanism behaves sanely (nonzero but small drift,
vision encoder untouched) before Gate 1 proceeds — not a full characterization.

## 7. Gate 1 baseline (population_size=0 — no perturbation, no RandOpt)

```bash
python -m neural_thickets_repro.eval_base --config configs/gqa_repro.yaml
```

This subprocess-invokes `external/RandOpt/randopt.py --population_size 0` unmodified (see
`eval_base.py` / `REPRO_SPEC.md` for why `population_size=0` is a zero-modification way to
get base-only accuracy from the official script).

**If this fails or clearly mishandles the image inputs** (garbage/text-only-looking
completions, vLLM errors about missing pixel values, etc.), that's the empirical answer to
the open "was the missing `multimodal=True` a bug" question. Try the monkeypatch launcher
next, which forces `multimodal=True` without ever editing the external clone:

```bash
python external/multimodal_patch_launcher.py \
    --dataset gqa --model_name Qwen/Qwen2.5-VL-3B-Instruct --precision bfloat16 \
    --train_samples 200 --population_size 0 --max_tokens 256 --global_seed 42 \
    --experiment_dir results/base
```

Record whichever way it goes (worked unmodified / needed the patch / neither worked) in
`REPRO_SPEC.md` and `src/neural_thickets_repro/vlm_adapter.py`'s `MULTIMODAL_FIX_NOTES`.

## 8. What to bring back

- `results/base/*/results.json` (or wherever `--experiment_dir results/base` landed it) —
  base_test_accuracy specifically.
- `results/drift_audit_real_model_1_10.json`.
- `verify_gqa_data`'s output (pass/fail, and full JSON if it failed).
- `validate_env`'s output, including the HF_HOME line and both disk(repo_root)/disk(hf_home) numbers.
- Console output/errors from step 7 (Gate 1 baseline), especially if the patch launcher was needed.

I'll compare the baseline number against 56.6% using the agreed thresholds (≤1pp
proceed/document, 1-3pp investigate, >3pp hard stop) and write up the Gate 1 report from
what you bring back — **not** from a run I haven't actually seen.

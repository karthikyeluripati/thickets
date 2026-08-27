"""Perturb-restore drift diagnostic.

SEPARATE from the reproduction algorithm: this tool measures a property of the
perturb/restore mechanism, it does not change how candidates are perturbed, restored, or
selected. It reuses perturb_cpu.perturb/restore (the exact mechanism REPRO_SPEC.md
describes and Gate 0 unit-tested) so the measurement reflects what the real pipeline
actually does -- but running this script never affects a real RandOpt run's behavior, and
neither eval_base.py nor run_randopt.py import anything from this module.

Motivation: restoration is regenerate-noise-and-subtract, not restore-from-a-stored-clone
(see REPRO_SPEC.md "Perturbation formula"). Gate 0 testing (tests/test_perturb_cpu.py)
already showed a single perturb->restore cycle is only exact to ~1 ULP at float32, not
bitwise. The practical question for a real N=5000-candidate run: each candidate is
evaluated relative to whatever the *current* weights are after the previous candidate's
restore, not by reloading fresh original weights from disk -- so does that per-cycle
rounding error compound over many cycles, and if so, does it become large enough to matter
(in absolute terms, relative to weight norms, and in terms of whether the nominally
unperturbed model's actual outputs change)?

Usage (on GPU hardware, real checkpoint, real bf16 precision):
    python -m neural_thickets_repro.diagnostics.perturb_restore_drift \
        --model-name Qwen/Qwen2.5-VL-3B-Instruct --dtype bfloat16 --sigma 0.001 \
        --cycles 1 10 100 1000 5000 --check-predictions --device cuda \
        --out results/drift_audit_real_model.json

Usage (CPU-only illustrative run against the Gate-0 synthetic dummy model -- NOT the real
checkpoint, clearly labeled as such in the output):
    python -m neural_thickets_repro.diagnostics.perturb_restore_drift --dummy --dtype bfloat16
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

import torch
import torch.nn as nn

from ..perturb_cpu import DEFAULT_VISUAL_PREFIXES, perturb, restore, should_perturb
from ..thicket.memory_bounded_ops import DEFAULT_CHUNK_ELEMENTS, chunked_abs_stats, chunked_squared_l2_diff_sum, chunked_squared_l2_sum

REPO_ROOT = Path(__file__).resolve().parents[3]


def measure_drift(
    model: nn.Module,
    original_state: Dict[str, torch.Tensor],
    param_filter: Optional[Callable[[str], bool]] = None,
    *, chunk_elements: int = DEFAULT_CHUNK_ELEMENTS,
) -> dict:
    """param_filter, if given, restricts the measurement to only the named parameters it
    accepts (e.g. an in-scope or out-of-scope subset for scope_isolation_gpu_check.py) --
    default None measures every parameter, exactly as before this option existed (existing
    callers/tests unaffected).

    MEMORY-BOUNDED (this repair pass, same root cause as thicket.perturbation.
    apply_anatomical_relative_l2 -- see thicket/memory_bounded_ops.py's own docstring): the
    original implementation computed `diff = (p.detach().float() - orig.float())`, a FULL
    float32 materialization per parameter tensor -- called here on the OUT-OF-region complement,
    which for the anatomical anatomy track (vision/connector/language, unlike whole_model, whose
    out-of-region set is empty by construction) can itself be most of a 7B+ model. Every
    reduction below now goes through the same chunked float64 accumulators used there, with a
    peak temporary bounded by `chunk_elements` regardless of parameter size. `n_differing`'s
    native-dtype exact comparison (`p.detach() != orig`) was already memory-cheap (no float
    upcast) and is unchanged.
    """
    max_abs = 0.0
    sum_abs = 0.0
    n_elems = 0
    sq_diff_sum = 0.0
    orig_sq_sum = 0.0
    n_differing = 0
    for name, p in model.named_parameters():
        if param_filter is not None and not param_filter(name):
            continue
        orig = original_state[name]
        p_max_abs, p_sum_abs = chunked_abs_stats(p.detach(), orig, chunk_elements=chunk_elements)
        max_abs = max(max_abs, p_max_abs)
        sum_abs += p_sum_abs
        n_elems += p.numel()
        sq_diff_sum += chunked_squared_l2_diff_sum(p.detach(), orig, chunk_elements=chunk_elements)
        orig_sq_sum += chunked_squared_l2_sum(orig, chunk_elements=chunk_elements)
        n_differing += int((p.detach() != orig).sum().item())
    mean_abs = sum_abs / n_elems if n_elems else 0.0
    rel_norm = (sq_diff_sum**0.5) / (orig_sq_sum**0.5) if orig_sq_sum > 0 else 0.0
    fraction_differing = n_differing / n_elems if n_elems else 0.0
    return {
        "max_abs_drift": max_abs,
        "mean_abs_drift": mean_abs,
        "relative_norm_drift": rel_norm,
        "fraction_elements_differing": fraction_differing,
    }


def check_vision_frozen(
    model: nn.Module,
    original_state: Dict[str, torch.Tensor],
    visual_prefixes=DEFAULT_VISUAL_PREFIXES,
) -> bool:
    for name, p in model.named_parameters():
        if not should_perturb(name, visual_prefixes):
            if not torch.equal(p.detach(), original_state[name]):
                return False
    return True


def run_drift_audit(
    model: nn.Module,
    sigma: float,
    cycle_checkpoints: List[int],
    seed_offset: int = 0,
    visual_prefixes=DEFAULT_VISUAL_PREFIXES,
    on_checkpoint: Optional[Callable[[int], Dict]] = None,
) -> List[dict]:
    """Runs max(cycle_checkpoints) perturb->restore cycles, each with a distinct seed
    (mirroring distinct candidates in a real N-candidate run), and records drift relative
    to the pre-cycling weights at each requested checkpoint cycle count.
    """
    original_state = {k: v.clone() for k, v in model.state_dict().items()}
    checkpoint_set: Set[int] = set(cycle_checkpoints)
    max_cycles = max(cycle_checkpoints)
    results = []

    start = time.time()
    for cycle in range(1, max_cycles + 1):
        seed = seed_offset + cycle
        perturb(model, seed=seed, sigma=sigma, visual_prefixes=visual_prefixes)
        restore(model, seed=seed, sigma=sigma, visual_prefixes=visual_prefixes)
        if cycle in checkpoint_set:
            entry = measure_drift(model, original_state)
            entry["cycle"] = cycle
            entry["elapsed_seconds"] = time.time() - start
            if on_checkpoint is not None:
                entry["predictions"] = on_checkpoint(cycle)
            results.append(entry)
    return results


DEFAULT_PROMPTS = [
    "What is the capital of France?",
    "Complete this sentence: The sky is",
    "Name three colors.",
]


def check_predictions_stable(model, processor, prompts: List[str], max_new_tokens: int = 16) -> List[str]:
    """Greedy-decode a handful of fixed text-only prompts. Text-only (no image) is
    sufficient for THIS diagnostic's purpose -- checking whether tiny rounding drift in the
    weights changes generation output at all -- it is not a GQA accuracy check.
    """
    model.eval()
    outputs = []
    with torch.no_grad():
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            if hasattr(processor, "apply_chat_template"):
                text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            else:
                text = prompt
            inputs = processor(text=text, return_tensors="pt").to(model.device)
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            outputs.append(processor.decode(gen[0], skip_special_tokens=True))
    return outputs


def _load_dummy_model(dtype: torch.dtype) -> nn.Module:
    """Loads the Gate-0 synthetic dummy model (tests/conftest.py's DummyVLM) -- NOT the
    real checkpoint. Used only for a CPU-only illustrative run of this tool.
    """
    sys.path.insert(0, str(REPO_ROOT / "tests"))
    from conftest import DummyVLM  # type: ignore

    torch.manual_seed(0)
    return DummyVLM().to(dtype)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=None, help="HF model name/path; omit and pass --dummy to use the Gate-0 synthetic model instead")
    parser.add_argument("--dummy", action="store_true", help="use the Gate-0 synthetic dummy model (CPU-only illustrative run, NOT the real checkpoint)")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--sigma", type=float, default=0.001)
    parser.add_argument("--cycles", type=int, nargs="+", default=[1, 10, 100, 1000, 5000])
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--check-predictions", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    if not args.dummy and not args.model_name:
        print("Pass --model-name <checkpoint> or --dummy.", file=sys.stderr)
        return 1

    dtype = getattr(torch, args.dtype)
    processor = None

    if args.dummy:
        model = _load_dummy_model(dtype)
        model_label = "gate0_synthetic_dummy_model (NOT the real checkpoint -- illustrative only)"
    else:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        model = AutoModelForImageTextToText.from_pretrained(args.model_name, torch_dtype=dtype).to(args.device)
        if args.check_predictions:
            processor = AutoProcessor.from_pretrained(args.model_name)
        model_label = args.model_name

    on_checkpoint = None
    if args.check_predictions:
        if args.dummy:
            print("--check-predictions is not supported with --dummy (no chat-capable processor for the synthetic model); skipping.", file=sys.stderr)
        else:
            def on_checkpoint(cycle):  # noqa: ANN001
                return check_predictions_stable(model, processor, DEFAULT_PROMPTS)

    pre_audit_state = {k: v.clone() for k, v in model.state_dict().items()}

    checkpoints = sorted(set(args.cycles))
    drift_results = run_drift_audit(
        model, sigma=args.sigma, cycle_checkpoints=checkpoints,
        seed_offset=args.seed_offset, on_checkpoint=on_checkpoint,
    )

    vision_frozen = check_vision_frozen(model, pre_audit_state)

    report = {
        "model": model_label,
        "dtype": args.dtype,
        "sigma": args.sigma,
        "cycle_checkpoints": checkpoints,
        "drift_by_cycle": drift_results,
        "vision_encoder_unchanged_after_all_cycles": vision_frozen,
        "note": (
            "SCAFFOLD/DIAGNOSTIC TOOL, kept separate from the reproduction algorithm. "
            "Does not alter perturb_cpu.py, eval_base.py, or run_randopt.py behavior."
        ),
    }

    print(json.dumps(report, indent=2, default=str))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

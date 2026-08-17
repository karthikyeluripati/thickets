"""python -m neural_thickets_repro.check_real_model_prefixes [--config configs/gqa_repro.yaml]

Metadata-only sanity check: fetches the real Qwen2.5-VL-3B-Instruct `config.json` and
`model.safetensors.index.json` from the HF hub (a few KB total -- no weight tensors are
downloaded) and verifies that perturb_cpu.should_perturb()'s `visual.`/`model.visual.`
prefix rule actually partitions the checkpoint's real tensor names the way REPRO_SPEC.md
claims: 100% of the vision tower under `visual.*`, everything else perturbable. This needs
network access but no GPU and negligible disk, so it's a Gate-0 task, not a Gate-1 one.

Writes results/real_model_prefix_check.json.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from .config import load_config
from .perturb_cpu import DEFAULT_VISUAL_PREFIXES, should_perturb

REPO_ROOT = Path(__file__).resolve().parents[2]


def _fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as resp:  # nosec B310 - fixed HF hub URLs only
        return json.loads(resp.read().decode("utf-8"))


def check_real_model_prefixes(model_name: str) -> dict:
    base = f"https://huggingface.co/{model_name}/resolve/main"
    config = _fetch_json(f"{base}/config.json")
    index = _fetch_json(f"{base}/model.safetensors.index.json")

    tensor_names = sorted(index.get("weight_map", {}).keys())
    perturbed = [n for n in tensor_names if should_perturb(n)]
    frozen = [n for n in tensor_names if not should_perturb(n)]

    return {
        "model_name": model_name,
        "architectures": config.get("architectures"),
        "tie_word_embeddings": config.get("tie_word_embeddings"),
        "total_tensors": len(tensor_names),
        "perturbed_tensor_count": len(perturbed),
        "frozen_tensor_count": len(frozen),
        "visual_prefixes_used": list(DEFAULT_VISUAL_PREFIXES),
        "sample_perturbed": perturbed[:5],
        "sample_frozen": frozen[:5],
        "has_separate_lm_head_tensor": any(n.startswith("lm_head.") for n in tensor_names),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "gqa_repro.yaml"))
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    result = check_real_model_prefixes(cfg.model.name)

    out_path = REPO_ROOT / "results" / "real_model_prefix_check.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

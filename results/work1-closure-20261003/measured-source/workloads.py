"""Synthetic plumbing workload and an optional real causal-LM workload."""
import hashlib
import json
import re
from pathlib import Path

import torch


class SyntheticWorkload:
    """Dense MLP forward pass. Not LLM inference and not an accuracy benchmark."""
    def __init__(self, device: torch.device, dtype: torch.dtype,
                 width: int = 128, depth: int = 3, batch: int = 4):
        if min(width, depth, batch) < 1:
            raise ValueError("synthetic dimensions must be positive")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(314159)
            layers = []
            for _ in range(depth):
                layers.extend([torch.nn.Linear(width, width, bias=False), torch.nn.Tanh()])
            self.model = torch.nn.Sequential(*layers).to(device=device, dtype=dtype).eval()
            self.inputs = torch.randn(batch, width).to(device=device, dtype=dtype)
        self.metadata = {"kind": "synthetic-mlp", "width": width, "depth": depth,
                         "batch": batch, "score_is_task_quality": False,
                         "input_sha256": hashlib.sha256(
                             self.inputs.cpu().view(torch.uint8).numpy().tobytes()).hexdigest()}

    @torch.inference_mode()
    def infer(self) -> torch.Tensor:
        return self.model(self.inputs)

    def score(self, output: torch.Tensor) -> float:
        # A checksum-like scalar only; it is explicitly NOT an accuracy reward.
        return float(output.float().mean().cpu().item())


class HFWorkload:
    """One-device greedy generation, with fresh KV state for every candidate.

    Optional and untested without transformers/model weights. A revision is
    required for remote models; trust_remote_code is deliberately disabled.
    """
    def __init__(self, model_id: str, revision: str | None, data: str,
                 device: torch.device, dtype: torch.dtype, max_new_tokens: int,
                 fixed_length: bool = False):
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        local = Path(model_id).is_dir()
        if not local and (revision is None or re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None):
            raise ValueError("remote HF models require a full 40-character --revision commit SHA")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("install the optional dependency: pip install -e '.[hf]'") from exc
        raw = Path(data).read_bytes()
        rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
        if not rows or any(not isinstance(r.get("prompt"), str)
                           or not isinstance(r.get("answer"), (str, int)) for r in rows):
            raise ValueError("JSONL requires nonempty prompt/answer records")
        self.answers = [str(r["answer"]).strip() for r in rows]
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, trust_remote_code=False, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("tokenizer needs a pad token or EOS token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, torch_dtype=dtype,
            trust_remote_code=False, attn_implementation="eager").to(device).eval()
        self.inputs = self.tokenizer([r["prompt"] for r in rows], padding=True,
                                     return_tensors="pt").to(device)
        self.prompt_width = self.inputs["input_ids"].shape[1]
        self.gen = {"do_sample": False, "num_beams": 1, "use_cache": True,
                    "max_new_tokens": max_new_tokens,
                    "pad_token_id": self.tokenizer.pad_token_id}
        if fixed_length:
            self.gen["min_new_tokens"] = max_new_tokens
        import transformers
        self.metadata = {"kind": "huggingface-greedy", "model": model_id,
                         "revision": revision, "transformers": transformers.__version__,
                         "data_sha256": hashlib.sha256(raw).hexdigest(),
                         "examples": len(rows), "generation": self.gen,
                         "attention": "eager", "fresh_kv_per_call": True,
                         "score_is_task_quality": True,
                         "scorer": "last-signed-integer-exact-match-v1",
                         "fixed_length_microbenchmark": fixed_length,
                         "prompt_format": "raw-text-no-chat-template"}

    @torch.inference_mode()
    def infer(self) -> torch.Tensor:
        output = self.model.generate(**self.inputs, **self.gen)
        return output[:, self.prompt_width:]

    def score(self, output: torch.Tensor) -> float:
        texts = self.tokenizer.batch_decode(output.detach().cpu(), skip_special_tokens=True)
        correct = 0
        for text, expected in zip(texts, self.answers):
            values = re.findall(r"[-+]?\d+", text)
            correct += bool(values and values[-1] == expected)
        return correct / len(self.answers)

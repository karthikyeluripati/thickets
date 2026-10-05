"""Two EXISTING candidate-state strategies. No new kernel is claimed here.

The legacy arithmetic mirrors the pinned RandOpt worker's tensor-local RNG,
native-dtype noise and materialized scaling. It is not a complete vLLM replica.
"""
from contextlib import contextmanager, nullcontext
import hashlib
import random
from typing import Iterator

import numpy as np
import torch

from .candidate import CandidateSpec, canonical_json

STRATEGIES = ("legacy-add-subtract", "snapshot-copy")


def state_hash(model: torch.nn.Module) -> str:
    """Hash parameters AND buffers, including layout metadata, outside timing."""
    digest = hashlib.sha256()
    entries = [("parameter:" + n, t) for n, t in model.named_parameters()]
    entries += [("buffer:" + n, t) for n, t in model.named_buffers()]
    for name, tensor in entries:
        t = tensor.detach().cpu().contiguous()
        digest.update(canonical_json([name, list(t.shape), str(t.dtype)]).encode())
        digest.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class WeightState:
    """Single-device, dense floating-point, eval-only reference executor.

    Base snapshots stay on the model device, as in the inspected upstream.
    Legacy mode intentionally retains add/subtract drift within a trace.
    Snapshot mode reanchors BEFORE each candidate and resets afterward.
    """

    def __init__(self, model: torch.nn.Module, strategy: str,
                 *, cleanup_cache: bool = True, detail: bool = False):
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy: {strategy}")
        if model.training:
            raise ValueError("model must be in eval mode")
        params = dict(model.named_parameters())
        if not params or any(not p.is_floating_point() or p.is_sparse for p in params.values()):
            raise ValueError("expected dense floating-point model parameters")
        devices = {p.device for p in params.values()}
        if len(devices) != 1 or next(iter(devices)).type not in ("cpu", "cuda"):
            raise ValueError("this pilot supports one CPU or CUDA device only")
        if any(p.is_meta or not p.is_contiguous() for p in params.values()):
            raise ValueError("meta/non-contiguous parameters are not supported")
        # Deduplicated tied Parameter objects are supported. Different parameter
        # objects sharing storage are rejected rather than silently double-added.
        pointers = [p.untyped_storage().data_ptr() for p in params.values() if p.numel()]
        if len(set(pointers)) != len(pointers):
            raise ValueError("distinct parameter objects sharing storage are unsupported")
        self.model, self.params = model, params
        self.strategy, self.cleanup_cache, self.detail = strategy, cleanup_cache, detail
        self.device = next(iter(devices))
        self.base_id = state_hash(model)
        self.base = {n: p.detach().clone() for n, p in params.items()}
        self.buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
        self.active: CandidateSpec | None = None

    def region(self, name: str):
        return torch.profiler.record_function(f"thicket/{name}") if self.detail else nullcontext()

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _boundary(self) -> None:
        with self.region("sync_and_cache_boundary"):
            self.synchronize()
            if self.device.type == "cuda" and self.cleanup_cache:
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()

    def _global_seed(self, candidate: CandidateSpec) -> None:
        # Match upstream side effects for the legacy RNG contract. In this pilot
        # inference is greedy; randomized inference needs a separate RNG contract.
        if candidate.rng == "randopt-per-tensor-v1":
            random.seed(candidate.seed)
            np.random.seed(candidate.seed)
            torch.manual_seed(candidate.seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(candidate.seed)

    def noise(self, name: str, p: torch.Tensor, candidate: CandidateSpec) -> torch.Tensor:
        gen = torch.Generator(device=p.device)
        gen.manual_seed(candidate.tensor_seed(name))
        return torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)

    @torch.no_grad()
    def _change(self, candidate: CandidateSpec, sign: int, phase: str) -> None:
        self._global_seed(candidate)
        for name, p in self.params.items():
            with self.region(f"{phase}/noise"):
                noise = self.noise(name, p, candidate)
            with self.region(f"{phase}/scale"):
                delta = (sign * candidate.sign * candidate.sigma) * noise
            with self.region(f"{phase}/add"):
                p.add_(delta)
            del delta, noise
        self._boundary()

    @torch.no_grad()
    def rebase(self) -> None:
        for name, p in self.params.items():
            p.copy_(self.base[name])
        for name, b in self.model.named_buffers():
            b.copy_(self.buffers[name])
        self.active = None
        self.synchronize()

    @torch.no_grad()
    def apply(self, candidate: CandidateSpec) -> None:
        if candidate.base_id != self.base_id:
            raise ValueError("candidate is bound to different base weights")
        if self.active is not None:
            raise RuntimeError("finish/reset the active candidate first")
        if self.strategy == "snapshot-copy":
            with self.region("apply/copy_base"):
                self.rebase()
        self.active = candidate
        try:
            self._change(candidate, 1, "apply")
        except BaseException:
            self.rebase()
            raise

    @torch.no_grad()
    def reset(self) -> None:
        if self.active is None:
            raise RuntimeError("no active candidate")
        candidate = self.active
        try:
            if self.strategy == "legacy-add-subtract":
                self._change(candidate, -1, "restore")
                self.active = None
            else:
                with self.region("restore/copy_base"):
                    self.rebase()
                self._boundary()
        except BaseException:
            self.rebase()
            raise

    @contextmanager
    def using(self, candidate: CandidateSpec) -> Iterator[None]:
        self.apply(candidate)
        try:
            yield
        except BaseException:
            # Recover to the actual base after failures, not approximate undo.
            self.rebase()
            raise
        else:
            self.reset()

    @torch.no_grad()
    def drift(self) -> dict:
        """Full audit, never part of reported candidate timing."""
        changed, count, max_abs = 0, 0, 0.0
        finite = True
        for name, p in self.params.items():
            current, reference = p.reshape(-1), self.base[name].reshape(-1)
            count += p.numel()
            # Bound audit scratch memory rather than converting an entire LLM
            # embedding matrix to float64 in one allocation.
            for offset in range(0, p.numel(), 1_000_000):
                cur = current[offset:offset + 1_000_000]
                ref = reference[offset:offset + 1_000_000]
                changed += int((cur != ref).sum().item())
                diff = (cur.double() - ref.double()).abs()
                chunk_finite = bool(torch.isfinite(diff).all().item())
                finite = finite and chunk_finite
                if chunk_finite and diff.numel():
                    max_abs = max(max_abs, float(diff.max().item()))
        buffer_changed, buffer_count = 0, 0
        for name, b in self.model.named_buffers():
            reference = self.buffers[name]
            if b.shape != reference.shape:
                buffer_changed += max(b.numel(), reference.numel(), 1)
            else:
                buffer_changed += int((b != reference).sum().item())
            buffer_count += reference.numel()
            if b.is_floating_point():
                finite = finite and bool(torch.isfinite(b).all().item())
        return {"changed_values": changed, "parameter_values": count,
                "changed_fraction": changed / max(1, count),
                "changed_buffer_values": buffer_changed, "buffer_values": buffer_count,
                "max_abs": max_abs if finite else None, "finite": finite,
                "exact_base": finite and changed == 0 and buffer_changed == 0}

    @property
    def parameter_bytes(self) -> int:
        return sum(p.numel() * p.element_size() for p in self.params.values())

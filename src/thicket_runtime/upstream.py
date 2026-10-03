"""Opt-in adapter for the original, hash-pinned RandOpt weight operations.

This runs the worker methods against the loaded PyTorch model, NOT the whole
vLLM/Ray pipeline. No upstream source is vendored or downloaded automatically.
"""
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

from .execution import WeightState

UPSTREAM_COMMIT = "4000d34fb5b69a3121cf1d2c564aa0be5a6a41ca"
WORKER_BLOB = "3b672d6636364845474be02e01cce764b6e18af3"


def load_worker(root: str):
    path = Path(root).resolve() / "utils" / "worker_extn.py"
    content = path.read_bytes()
    blob = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
    if blob != WORKER_BLOB:
        raise ValueError(f"upstream worker mismatch: {blob}; expected {WORKER_BLOB}")
    spec = importlib.util.spec_from_file_location("_pinned_randopt_worker", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import pinned worker")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WorkerExtension


class UpstreamWeightState(WeightState):
    def __init__(self, model, strategy, *, upstream_root, cleanup_cache=True, detail=False):
        if strategy != "legacy-add-subtract" or not cleanup_cache:
            raise ValueError("pinned adapter supports the unmodified legacy cleanup path only")
        worker_cls = load_worker(upstream_root)
        super().__init__(model, strategy, cleanup_cache=cleanup_cache, detail=detail)
        self.worker = worker_cls()
        self.worker.model_runner = SimpleNamespace(model=model)

    def _change(self, candidate, sign, phase):
        if candidate.rng != "randopt-per-tensor-v1":
            raise ValueError("pinned worker requires its original RNG contract")
        fn = self.worker.perturb_self_weights if sign == 1 else self.worker.restore_self_weights
        with self.region(f"{phase}/original_upstream_worker"):
            fn(candidate.seed, candidate.sigma, candidate.sign == -1)

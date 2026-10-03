"""Candidate identity is a recipe PLUS an explicit base and execution contract."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math

RNG_SCHEMES = ("randopt-per-tensor-v1", "named-tensor-v1")


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class CandidateSpec:
    base_id: str
    seed: int
    sigma: float
    rng: str = "randopt-per-tensor-v1"
    sign: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.base_id, str) or not self.base_id.strip():
            raise ValueError("base_id must identify the actual base weights")
        if type(self.seed) is not int or not 0 <= self.seed < 2**31:
            raise ValueError("seed must be an integer in [0, 2**31)")
        if isinstance(self.sigma, bool) or not isinstance(self.sigma, (int, float)):
            raise ValueError("sigma must be numeric")
        if not math.isfinite(self.sigma) or self.sigma < 0:
            raise ValueError("sigma must be finite and nonnegative")
        if self.rng not in RNG_SCHEMES:
            raise ValueError(f"unsupported RNG contract: {self.rng}")
        if type(self.sign) is not int or self.sign not in (-1, 1):
            raise ValueError("sign must be -1 or 1")
        object.__setattr__(self, "sigma", float(self.sigma))

    @property
    def candidate_id(self) -> str:
        return hashlib.sha256(canonical_json(asdict(self)).encode()).hexdigest()

    def tensor_seed(self, name: str) -> int:
        if self.rng == "randopt-per-tensor-v1":
            # Deliberately reproduce upstream's reseeding for EACH parameter.
            # Equal-shape tensors consequently receive identical noise.
            return self.seed
        payload = f"named-tensor-v1:{self.seed}:{name}".encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63)

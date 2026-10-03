import gzip
import json

import pytest

from thicket_runtime.evaluation_matrix import write_gzip


def test_raw_trace_compression_is_lossless_deterministic_and_never_overwrites(tmp_path):
    value = {"candidate_id": "test", "outputs": [{"token_ids": [0, 1, 99999], "text": "answer\n42", "reward": 1.0}]}
    a, b = tmp_path / "a.json.gz", tmp_path / "b.json.gz"
    write_gzip(a, value)
    write_gzip(b, value)
    assert a.read_bytes() == b.read_bytes()
    assert json.loads(gzip.decompress(a.read_bytes())) == value
    with pytest.raises(FileExistsError):
        write_gzip(a, value)
    with pytest.raises(ValueError):
        write_gzip(tmp_path / "invalid.json.gz", {"reward": float("nan")})

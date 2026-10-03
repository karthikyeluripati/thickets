"""Freeze the first 32 training rows before inference, using upstream formatting."""
import hashlib
import io
import json
from pathlib import Path
import re
import urllib.request

REVISION = "740312add88f781978c0658806c59bc2815b9866"
URL = f"https://huggingface.co/datasets/openai/gsm8k/resolve/{REVISION}/main/train-00000-of-00001.parquet"
SUFFIX = 'Let\'s think step by step and output the final answer after "####".'


def main():
    import pyarrow.parquet as pq
    root = Path(__file__).resolve().parents[1]
    target = root / "examples/gsm8k_train32.jsonl"
    provenance = target.with_suffix(".provenance.json")
    if target.exists() or provenance.exists():
        raise FileExistsError("frozen workload already exists; refusing overwrite")
    raw = urllib.request.urlopen(URL, timeout=120).read()
    rows = pq.read_table(io.BytesIO(raw)).slice(0, 32).to_pylist()
    records = []
    for i, row in enumerate(rows):
        answer = re.search(r"#### (\-?[0-9\.\,]+)", row["answer"]).group(1).replace(",", "")
        records.append({"id": f"gsm8k-main-train-{i:04d}", "source_index": i,
                        "prompt": row["question"] + " " + SUFFIX + "\n",
                        "answer": answer, "reference_solution": row["answer"]})
    data = ("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n").encode()
    target.write_bytes(data)
    meta = {"source_url": URL, "dataset_revision": REVISION, "license": "MIT",
            "selection": "first 32 main/train rows, indices 0..31; frozen before inference",
            "source_parquet_sha256": hashlib.sha256(raw).hexdigest(),
            "frozen_jsonl_sha256": hashlib.sha256(data).hexdigest(),
            "upstream_commit": "4000d34fb5b69a3121cf1d2c564aa0be5a6a41ca",
            "prompt_preprocessor_blob": "1656cdbc896a8f14fc7e09705d36335f52165533",
            "base_model_prompt_format": "randopt.py format_prompt (non-instruct branch)",
            "scorer_blob": "85783e641d512fb95510aad3182217ed605725ba"}
    provenance.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

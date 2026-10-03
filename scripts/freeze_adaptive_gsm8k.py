"""Freeze disjoint selection/test prompts before exhaustive candidate collection."""
import hashlib
import io
import json
from pathlib import Path
import re
import urllib.request

from freeze_gsm8k import REVISION, SUFFIX


def main():
    import pyarrow.parquet as pq
    root = Path(__file__).resolve().parents[1] / "examples/adaptive_gsm8k"
    root.mkdir(exist_ok=False)
    provenance = {"dataset": "openai/gsm8k", "revision": REVISION, "license": "MIT",
                  "upstream_commit": "4000d34fb5b69a3121cf1d2c564aa0be5a6a41ca",
                  "selection_rule": "train indices 32..231; test indices 0..39; no outcome-based filtering",
                  "format": "pinned RandOpt preprocessing and non-instruct format_prompt",
                  "sources": {}}
    for split, start, count, name in (("train", 32, 200, "selection200"), ("test", 0, 40, "heldout40")):
        url = f"https://huggingface.co/datasets/openai/gsm8k/resolve/{REVISION}/main/{split}-00000-of-00001.parquet"
        raw = urllib.request.urlopen(url, timeout=120).read()
        rows = pq.read_table(io.BytesIO(raw)).slice(start, count).to_pylist()
        records = [{"id": f"gsm8k-main-{split}-{i:04d}", "split": split, "source_index": i,
                    "prompt": row["question"] + " " + SUFFIX + "\n",
                    "answer": re.search(r"#### (\-?[0-9\.\,]+)", row["answer"]).group(1).replace(",", ""),
                    "reference_solution": row["answer"]} for i, row in enumerate(rows, start)]
        data = ("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n").encode()
        (root / (name + ".jsonl")).write_bytes(data)
        provenance["sources"][name] = {"url": url, "parquet_sha256": hashlib.sha256(raw).hexdigest(),
                                       "jsonl_sha256": hashlib.sha256(data).hexdigest(), "count": count}
    (root / "provenance.json").write_bytes((json.dumps(provenance, indent=2) + "\n").encode())
    (root / ".gitattributes").write_bytes(b"# Frozen input bytes are hash-pinned.\n* -text\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()

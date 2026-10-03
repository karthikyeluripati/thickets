"""Bounded six-process vLLM/Ray closure matrix; never overwrite run directories.

Run with the isolated vLLM environment. The original-worker HF controls should
use the original HF environment, as documented in the Work 1 report.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream-root", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=False)
    jobs = [("fixed-32-b4", ["--fixed-length", "--max-new-tokens", "32"]),
            ("fixed-128-b4", ["--fixed-length", "--max-new-tokens", "128"]),
            ("natural-128-b8", ["--data", "examples/search_word_problems.jsonl",
                                "--max-new-tokens", "128", "--stop", "\nQuestion:"])]
    records = []
    # Alternate order across independent process invocations within each workload.
    for name, extra in jobs:
        for replicate, order in enumerate(("legacy-first", "snapshot-first"), 1):
            run = f"{name}-process-{replicate}"
            command = [sys.executable, "-u", "-m", "thicket_runtime.vllm_profile",
                       "--upstream-root", args.upstream_root,
                       "--strategy-order", order, "--repeats", "3", "--warmup", "2",
                       "--out", str(root / run), *extra]
            record = {"name": run, "command": command, "timeout_seconds": 900}
            records.append(record)
            def save():
                (root / "commands.json").write_text(json.dumps(records, indent=2) + "\n")
            save()
            print("START", run, flush=True)
            with (root / (run + ".log")).open("w") as log:
                try:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                            timeout=record["timeout_seconds"])
                    record["returncode"] = result.returncode
                except subprocess.TimeoutExpired:
                    record["timed_out"] = True
            save()
            if record.get("returncode") != 0:
                raise SystemExit(f"{run} failed/timed out; inspect logs and Ray cleanup before continuing")
            print("COMPLETE", run, flush=True)


if __name__ == "__main__":
    main()

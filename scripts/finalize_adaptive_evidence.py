"""Check historical and new evidence, then freeze a root artifact checksum index."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from thicket_runtime.cli import revision_info, write_json


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    root = args.root
    if (root / "sha256.json").exists() or (root / "integrity.json").exists():
        raise FileExistsError("finalized evidence must not be overwritten")
    checked = {}
    for folder in [Path("results") / name for name in (
            "llm-bounded-20261003", "work1-closure-20261003", "shared-speculative-20261003")] + [root / "matrix-01"]:
        index = json.loads((folder / "sha256.json").read_bytes())
        for name, expected in index.items():
            assert sha(folder / name) == expected, (folder, name)
        checked[str(folder)] = len(index)
    manifest = json.loads((root / "matrix-01/manifest.json").read_bytes())
    source = manifest["source"]["commit"]
    measured = root / "matrix-01/measured-source"
    source_files = list(f for f in measured.rglob("*") if f.is_file())
    for f in source_files:
        blob = subprocess.check_output(["git", "show", source + ":" + f.relative_to(measured).as_posix()])
        assert blob == f.read_bytes(), str(f)
    original_protocol = subprocess.check_output(["git", "show", source + ":experiments/adaptive_evaluation_protocol.json"])
    assert json.loads(original_protocol) == json.loads((root / "matrix-01/protocol.json").read_bytes())
    assert hashlib.sha256(original_protocol).hexdigest() == manifest["protocol_sha256"]
    main_ref = subprocess.check_output(["git", "rev-parse", "main"], text=True).strip()
    assert main_ref == "deb414253139cc2559d19cdfe7e6b4786e7c40db"
    old_paths = ["results/llm-bounded-20261003", "results/work1-closure-20261003",
                 "results/shared-speculative-20261003", "docs/LLM_PILOT_2026-10-03.md",
                 "docs/WORK1_CLOSURE_2026-10-03.md", "docs/SHARED_SPECULATIVE_FEASIBILITY_2026-10-03.md"]
    changed = subprocess.check_output(["git", "diff", "--name-only", "92ef27d7a03abb66e928cdb7807b5bf87f6ec213", "--", *old_paths])
    assert not changed.strip(), "historical evidence changed"
    audit = json.loads((root / "report-audit.json").read_bytes())
    assert audit["all_accounting_checks_pass"]
    write_json(root / "integrity.json", {"source": revision_info(), "command": [sys.executable, *sys.argv],
        "checksum_files_verified": checked, "mismatches": 0, "main_ref": main_ref,
        "historical_tracked_changes": [], "measured_source_commit": source,
        "measured_source_files_match_git_blobs": len(source_files), "frozen_protocol_matches_collection": True,
        "accounting_trials_verified": sum(v["audited_trials"] for v in audit["populations"].values())})
    files = sorted(f for f in root.rglob("*") if f.is_file() and f != root / "sha256.json")
    write_json(root / "sha256.json", {f.relative_to(root).as_posix(): sha(f) for f in files})
    print(json.dumps({"indexed_files": len(files), "all_integrity_checks_pass": True}))


if __name__ == "__main__":
    main()

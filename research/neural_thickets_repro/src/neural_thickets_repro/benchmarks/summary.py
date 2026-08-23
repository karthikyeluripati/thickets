"""Master summary generation across all completed benchmark cards -- reads card.json files
already written by card.write_card() under a results root and writes summary.md (compact
table) + summary.json.

MISSING-CAPABILITY ROBUSTNESS (this repair pass): a real multi-capability baseline run was
observed with a capability's row silently absent from summary.md even though the run was
described as covering all N configured capabilities -- with the code AS IT STOOD, iterating
`cards` (whatever `find_cards()` happened to discover on disk) can never itself distinguish
"this capability's card.json was never written" from "there was nothing to report" -- the
table simply has as many rows as cards were found, with no signal that fewer were found than
expected. `expected_capabilities` (when the caller knows in advance which capabilities were
supposed to run, e.g. run_baseline_characterization.py's own config list) makes this an
observable, reported fact instead of a silent gap: build_summary_table() renders an explicit
"MISSING" row for any expected capability with no card, and build_summary_json() lists them
in `missing_capabilities` and folds them into `all_pass=False`. This guarantees every
attempted capability appears as a row -- either with real data or an explicit MISSING marker
-- never simply absent.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional


def find_cards(results_root: "str | Path") -> List[Path]:
    return sorted(Path(results_root).glob("*/card.json"))


def load_cards(results_root: "str | Path") -> List[dict]:
    return [json.loads(p.read_text()) for p in find_cards(results_root)]


def _capability_order(cards: List[dict], expected_capabilities: Optional[List[str]]) -> List[str]:
    """Every expected capability, in the given order, FOLLOWED by any real card whose
    capability wasn't in that list (never silently dropping a real card either -- e.g. a
    capability run outside the caller's own expected set).
    """
    found = [c["capability"] for c in cards]
    if expected_capabilities is None:
        return found
    order = list(expected_capabilities)
    for capability in found:
        if capability not in order:
            order.append(capability)
    return order


def build_summary_table(cards: List[dict], expected_capabilities: Optional[List[str]] = None) -> str:
    columns = ["Capability", "Dataset", "N", "Base", "Repeat", "Image integrity", "Parser", "Visual sanity", "Status"]
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join("-" * (len(c) + 2) for c in columns) + "|"
    lines = [header, separator]

    cards_by_capability = {c["capability"]: c for c in cards}
    for capability in _capability_order(cards, expected_capabilities):
        card = cards_by_capability.get(capability)
        if card is None:
            # Expected to run, but no card.json was found -- an explicit, visible row, never
            # a silently-absent one.
            lines.append("| " + " | ".join([capability, "MISSING", "-", "-", "-", "-", "-", "-", "MISSING"]) + " |")
            continue

        base = card["base_metrics"].get("primary_metric")
        repeat_metrics = card.get("repeat_metrics") or {}
        repeat = repeat_metrics.get("primary_metric")
        integrity = card["integrity"]
        parser_rate = card["base_metrics"].get("parser_failure_rate", 0.0)
        sanity = card.get("image_sanity")

        image_integrity_str = f"{integrity['n_valid_images']}/{integrity['n_loaded']}"
        base_str = f"{base:.3f}" if base is not None else "N/A"
        repeat_str = f"{repeat:.3f}" if repeat is not None else "N/A"
        parser_str = f"{parser_rate:.1%}"
        sanity_str = f"{sanity['correct_minus_shuffled']:+.3f}" if sanity else "N/A"

        row_values = [
            card["capability"], card["dataset"], str(card["subset_size"]),
            base_str, repeat_str, image_integrity_str, parser_str, sanity_str, card["status"],
        ]
        lines.append("| " + " | ".join(row_values) + " |")

    return "\n".join(lines)


def build_summary_json(cards: List[dict], expected_capabilities: Optional[List[str]] = None) -> dict:
    status_counts = {"PASS": 0, "FAIL": 0, "NEEDS_REVIEW": 0}
    for card in cards:
        status_counts[card["status"]] = status_counts.get(card["status"], 0) + 1

    found_capabilities = {c["capability"] for c in cards}
    missing_capabilities = [c for c in (expected_capabilities or []) if c not in found_capabilities]

    return {
        "n_capabilities": len(cards),
        "n_expected_capabilities": len(expected_capabilities) if expected_capabilities is not None else len(cards),
        "missing_capabilities": missing_capabilities,
        "status_counts": status_counts,
        "all_pass": (
            len(cards) > 0 and not missing_capabilities
            and status_counts.get("FAIL", 0) == 0 and status_counts.get("NEEDS_REVIEW", 0) == 0
        ),
        "cards": cards,
    }


def write_summary(results_root: "str | Path", expected_capabilities: Optional[List[str]] = None) -> dict:
    cards = load_cards(results_root)
    table = build_summary_table(cards, expected_capabilities)
    summary_json = build_summary_json(cards, expected_capabilities)

    root = Path(results_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.md").write_text(table)
    (root / "summary.json").write_text(json.dumps(summary_json, indent=2))
    return summary_json

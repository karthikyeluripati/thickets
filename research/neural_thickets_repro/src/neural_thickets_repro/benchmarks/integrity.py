"""Dataset integrity validation for the Capability Benchmark Gate. "Evaluation completing"
is never itself sufficient for a benchmark to pass -- this module makes the underlying
requested/loaded/valid counts explicit and flags structurally invalid examples loudly rather
than silently dropping them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

from .base import Example


@dataclass
class IntegrityReport:
    n_requested: int
    n_loaded: int
    n_valid_images: int
    n_valid_targets: int
    n_duplicate_ids: int
    n_missing_targets: int
    n_invalid_images: int
    duplicate_ids: List[str] = field(default_factory=list)
    invalid_image_ids: List[str] = field(default_factory=list)
    missing_target_ids: List[str] = field(default_factory=list)
    parser_failures: int = 0
    parser_failure_rate: float = 0.0

    @property
    def passed(self) -> bool:
        return self.n_duplicate_ids == 0 and self.n_missing_targets == 0 and self.n_invalid_images == 0

    def to_dict(self) -> dict:
        return {
            "n_requested": self.n_requested, "n_loaded": self.n_loaded,
            "n_valid_images": self.n_valid_images, "n_valid_targets": self.n_valid_targets,
            "n_duplicate_ids": self.n_duplicate_ids, "n_missing_targets": self.n_missing_targets,
            "n_invalid_images": self.n_invalid_images,
            "duplicate_ids": self.duplicate_ids, "invalid_image_ids": self.invalid_image_ids,
            "missing_target_ids": self.missing_target_ids,
            "parser_failures": self.parser_failures, "parser_failure_rate": self.parser_failure_rate,
            "passed": self.passed,
        }


def _is_valid_image(image) -> bool:
    if image is None:
        return False
    size = getattr(image, "size", None)
    if size is None:
        return False
    width, height = size
    return width > 0 and height > 0


def validate_examples(examples: Sequence[Example], n_requested: int, require_images: bool = True) -> IntegrityReport:
    seen_counts = {}
    invalid_image_ids = []
    missing_target_ids = []

    for ex in examples:
        seen_counts[ex.example_id] = seen_counts.get(ex.example_id, 0) + 1
        if require_images and not _is_valid_image(ex.image):
            invalid_image_ids.append(ex.example_id)
        if ex.target is None:
            missing_target_ids.append(ex.example_id)

    duplicate_ids = sorted(eid for eid, count in seen_counts.items() if count > 1)
    n_loaded = len(examples)
    n_invalid_images = len(invalid_image_ids)
    n_missing_targets = len(missing_target_ids)

    return IntegrityReport(
        n_requested=n_requested,
        n_loaded=n_loaded,
        n_valid_images=(n_loaded - n_invalid_images) if require_images else n_loaded,
        n_valid_targets=n_loaded - n_missing_targets,
        n_duplicate_ids=len(duplicate_ids),
        n_missing_targets=n_missing_targets,
        n_invalid_images=n_invalid_images,
        duplicate_ids=duplicate_ids,
        invalid_image_ids=invalid_image_ids,
        missing_target_ids=missing_target_ids,
    )


def merge_parser_failure_stats(report: IntegrityReport, parser_failures: int, n: int) -> IntegrityReport:
    report.parser_failures = parser_failures
    report.parser_failure_rate = parser_failures / n if n > 0 else 0.0
    return report

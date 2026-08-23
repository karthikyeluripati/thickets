"""Shared chat-message construction for the net-new (non-GQA-derived) adapters. Content-block
shape (an image block followed by a text block) matches this project's own established
convention (REPRO_SPEC.md's "Image message format" row). The two GQA-derived adapters do NOT
use this -- they pass through GQAHandler's own `d["messages"]` unchanged.
"""
from __future__ import annotations

from typing import List


def build_image_text_messages(instruction_text: str) -> List[dict]:
    return [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": instruction_text},
        ],
    }]

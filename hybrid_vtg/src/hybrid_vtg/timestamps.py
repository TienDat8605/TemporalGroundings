"""Robust parsing and coordinate conversion for model-produced timestamps."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from .types import Component


_SPAN = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*(?:seconds?|secs?|sec|s)?\s*"
    r"(?:to|until|through|and|[-–—,])\s*"
    r"(-?\d+(?:\.\d+)?)\s*(?:seconds?|secs?|sec|s)?",
    flags=re.IGNORECASE,
)


def parse_timestamp(text: str) -> tuple[float, float]:
    """Return the last valid span, favoring an explicit JSON object when present."""
    for match in reversed(re.findall(r"\{[^{}]+\}", text, flags=re.DOTALL)):
        try:
            value = json.loads(match)
        except json.JSONDecodeError:
            continue
        if "start" in value and "end" in value:
            return float(value["start"]), float(value["end"])
    spans = [(float(start), float(end)) for start, end in _SPAN.findall(text)]
    if not spans:
        raise ValueError(f"model response contains no timestamp span: {text!r}")
    return spans[-1]


def normalize_timestamp(
    interval: Sequence[float],
    component: Component,
    video_duration: float,
    mode: str = "absolute",
) -> tuple[float, float]:
    start, end = float(interval[0]), float(interval[1])
    if mode not in {"absolute", "relative", "auto"}:
        raise ValueError("timestamp mode must be absolute, relative, or auto")
    if mode == "relative" or (
        mode == "auto" and component.start > component.duration and 0 <= start < end <= component.duration + 1
    ):
        start, end = start + component.start, end + component.start
    start = max(component.start, min(component.end, video_duration, start))
    end = max(component.start, min(component.end, video_duration, end))
    if end <= start:
        raise ValueError(f"invalid timestamp after coordinate normalization: {(start, end)}")
    return round(start, 3), round(end, 3)

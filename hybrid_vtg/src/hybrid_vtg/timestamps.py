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


def _json_objects(text: str) -> list[dict]:
    values = []
    for match in reversed(re.findall(r"\{[^{}]+\}", text, flags=re.DOTALL)):
        try:
            value = json.loads(match)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            values.append(value)
    return values


def parse_timestamp(text: str) -> tuple[float, float]:
    """Return the last valid span, favoring an explicit JSON object when present."""
    for value in _json_objects(text):
        if "start" in value and "end" in value:
            return float(value["start"]), float(value["end"])
    spans = [(float(start), float(end)) for start, end in _SPAN.findall(text)]
    if not spans:
        raise ValueError(f"model response contains no timestamp span: {text!r}")
    return spans[-1]


def parse_intervals(text: str) -> tuple[tuple[float, float], ...]:
    """Parse ordered, deduplicated intervals for set-valued grounding."""
    try:
        value = json.loads(text.strip())
    except (json.JSONDecodeError, TypeError, AttributeError):
        value = None
    candidates = []
    if isinstance(value, list):
        if len(value) == 2 and all(isinstance(item, (int, float)) for item in value):
            candidates = [value]
        else:
            candidates = [item for item in value if isinstance(item, list) and len(item) == 2]
    if not candidates:
        candidates = [list(map(float, match)) for match in _SPAN.findall(text)]
    if not candidates:
        candidates = [
            [float(start), float(end)]
            for start, end in re.findall(
                r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]",
                text,
            )
        ]
    unique = {
        (float(interval[0]), float(interval[1]))
        for interval in candidates
        if float(interval[1]) > float(interval[0])
    }
    return tuple(sorted(unique))


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

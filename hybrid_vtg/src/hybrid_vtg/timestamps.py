"""Robust parsing and coordinate conversion for model-produced timestamps."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

from .types import Component


_SPAN = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*(?:seconds?|secs?|sec|s)?\s*"
    r"(?:to|until|through|and|[-–—,])\s*"
    r"(-?\d+(?:\.\d+)?)\s*(?:seconds?|secs?|sec|s)?",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class GroundingResponse:
    """Parsed frozen-model decision before timestamp normalization."""

    present: bool
    interval: tuple[float, float] | None
    confidence: float


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


def parse_grounding_response(text: str) -> GroundingResponse:
    """Parse presence, confidence, and an optional timestamp span.

    Explicit negative responses are preserved instead of forcing a timestamp.
    Legacy timestamp-only responses remain positive for compatibility with old
    checkpoints and already generated outputs.
    """
    for value in _json_objects(text):
        raw_present = value.get("present")
        if raw_present is not None and not isinstance(raw_present, bool):
            raise ValueError(f"model response has non-boolean presence: {raw_present!r}")
        present = bool(raw_present) if raw_present is not None else (
            "start" in value and "end" in value
        )
        default_confidence = 1.0 if present else 0.0
        confidence = float(value.get("confidence", default_confidence))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"model response confidence must be in [0, 1]: {confidence}")
        if not present:
            return GroundingResponse(False, None, confidence)
        if "start" in value and "end" in value:
            return GroundingResponse(
                True, (float(value["start"]), float(value["end"])), confidence,
            )
        raise ValueError(f"positive model response contains no timestamp span: {text!r}")
    return GroundingResponse(True, parse_timestamp(text), 1.0)


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

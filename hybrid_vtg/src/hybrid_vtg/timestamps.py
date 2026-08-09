"""Robust parsing and coordinate conversion for model-produced timestamps."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from collections.abc import Sequence

from .types import Component


_SPAN = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*(?:seconds?|secs?|sec|s)?\s*"
    r"(?:to|until|through|and|[-–—,])\s*"
    r"(-?\d+(?:\.\d+)?)\s*(?:seconds?|secs?|sec|s)?",
    flags=re.IGNORECASE,
)
_NUMBER = r"-?\d+(?:\.\d+)?"
_OBJECT_SPAN = re.compile(
    rf'["\']?(?:start|start_time)["\']?\s*:\s*({_NUMBER}).{{0,48}}?'
    rf'["\']?(?:end|end_time)["\']?\s*:\s*({_NUMBER})',
    flags=re.IGNORECASE | re.DOTALL,
)
_BRACKET_SPAN = re.compile(rf"\[\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\]")
_FENCED_JSON = re.compile(r"```(?:json)?\s*(.*?)```", flags=re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class IntervalParseResult:
    intervals: tuple[tuple[float, float], ...]
    status: str


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
    return parse_intervals_detailed(text).intervals


def _json_candidates(text: str) -> list[object]:
    sources = [text.strip(), *_FENCED_JSON.findall(text)]
    left, right = text.find("["), text.rfind("]")
    if left >= 0 and right > left:
        sources.append(text[left:right + 1])
    values = []
    for source in sources:
        try:
            values.append(json.loads(source))
        except (json.JSONDecodeError, TypeError, AttributeError):
            continue
    return values


def _intervals_from_json(value: object) -> list[tuple[float, float]] | None:
    if not isinstance(value, list):
        return None
    if not value:
        return []
    items: list[object]
    if len(value) == 2 and all(isinstance(item, (int, float)) for item in value):
        items = [value]
    else:
        items = value
    output = []
    for item in items:
        if isinstance(item, list) and len(item) == 2:
            start, end = item
        elif isinstance(item, dict) and "start" in item and "end" in item:
            start, end = item["start"], item["end"]
        elif isinstance(item, dict) and "start_time" in item and "end_time" in item:
            start, end = item["start_time"], item["end_time"]
        else:
            continue
        try:
            output.append((float(start), float(end)))
        except (TypeError, ValueError):
            continue
    return output


def _ordered_unique(intervals: Sequence[Sequence[float]]) -> tuple[tuple[float, float], ...]:
    return tuple(sorted({
        (float(interval[0]), float(interval[1]))
        for interval in intervals
        if len(interval) == 2 and float(interval[1]) > float(interval[0])
    }))


def parse_intervals_detailed(text: str) -> IntervalParseResult:
    """Parse Qwen's JSON variants and recover complete spans from truncated output."""
    if not isinstance(text, str):
        raise TypeError("model response must be text")
    for value in reversed(_json_candidates(text)):
        candidates = _intervals_from_json(value)
        if candidates is None:
            continue
        intervals = _ordered_unique(candidates)
        if intervals:
            return IntervalParseResult(intervals, "valid_json")
        if value == []:
            return IntervalParseResult((), "explicit_empty")

    for pattern in (_OBJECT_SPAN, _BRACKET_SPAN, _SPAN):
        intervals = _ordered_unique(pattern.findall(text))
        if intervals:
            return IntervalParseResult(intervals, "recovered")
    raise ValueError(f"model response contains no interval set: {text!r}")


def consolidate_intervals(
    intervals: Sequence[Sequence[float]],
    *,
    duration: float,
    duplicate_iou: float = 0.8,
    merge_gap: float = 1.0,
) -> tuple[tuple[float, float], ...]:
    """Normalize the shared OMTG duplicate suppression and gap merge policy."""
    if duration <= 0:
        raise ValueError("duration must be positive")
    cleaned = sorted(
        (
            max(0.0, float(start)),
            min(float(duration), float(end)),
        )
        for start, end in intervals
        if float(end) > float(start) and float(start) < duration and float(end) > 0
    )

    def iou(first: tuple[float, float], second: tuple[float, float]) -> float:
        overlap = max(0.0, min(first[1], second[1]) - max(first[0], second[0]))
        union = first[1] - first[0] + second[1] - second[0] - overlap
        return overlap / union if union > 0 else 0.0

    deduplicated: list[tuple[float, float]] = []
    for candidate in cleaned:
        duplicate = next(
            (index for index, kept in enumerate(deduplicated) if iou(candidate, kept) >= duplicate_iou),
            None,
        )
        if duplicate is None:
            deduplicated.append(candidate)
        elif candidate[1] - candidate[0] < deduplicated[duplicate][1] - deduplicated[duplicate][0]:
            deduplicated[duplicate] = candidate

    merged: list[list[float]] = []
    for start, end in sorted(deduplicated):
        if merged and start <= merged[-1][1] + merge_gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return tuple((round(start, 3), round(end, 3)) for start, end in merged)


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

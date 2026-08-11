"""Shared temporal interval parsing and deterministic consolidation."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence

from .contracts import ScoredSpan

_PAIR = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:,|-|to)\s*(-?\d+(?:\.\d+)?)")


def parse_spans(text: str) -> tuple[ScoredSpan, ...]:
    values: list[Sequence[float]] = []
    try:
        decoded = json.loads(text.strip())
        if isinstance(decoded, dict):
            decoded = decoded.get("spans", decoded.get("timestamps", []))
        if isinstance(decoded, list):
            values = [value for value in decoded if isinstance(value, (list, tuple)) and len(value) >= 2]
    except (json.JSONDecodeError, TypeError):
        values = []
    if not values:
        values = [(float(start), float(end)) for start, end in _PAIR.findall(text)]
    return tuple(
        ScoredSpan(float(value[0]), float(value[1]), 1.0 / (index + 1))
        for index, value in enumerate(values)
        if float(value[1]) > float(value[0])
    )


def temporal_iou(first: ScoredSpan, second: ScoredSpan) -> float:
    intersection = max(0.0, min(first.end, second.end) - max(first.start, second.start))
    union = first.end - first.start + second.end - second.start - intersection
    return intersection / union if union > 0 else 0.0


def temporal_nms(
    spans: Iterable[ScoredSpan],
    threshold: float = 0.7,
    maximum: int | None = None,
) -> tuple[ScoredSpan, ...]:
    kept: list[ScoredSpan] = []
    for candidate in sorted(spans, key=lambda value: (-value.score, value.start, value.end)):
        if any(temporal_iou(candidate, prior) > threshold for prior in kept):
            continue
        kept.append(candidate)
        if maximum is not None and len(kept) >= maximum:
            break
    return tuple(kept)


def consolidate_spans(
    spans: Iterable[ScoredSpan],
    duration: float,
    *,
    duplicate_iou: float = 0.8,
    merge_gap: float = 3.0,
) -> tuple[ScoredSpan, ...]:
    clipped = [value for span in spans if (value := span.clipped(duration)) is not None]
    deduplicated = list(temporal_nms(clipped, duplicate_iou))
    merged: list[ScoredSpan] = []
    for span in sorted(deduplicated, key=lambda value: (value.start, value.end)):
        if merged and span.start <= merged[-1].end + merge_gap:
            previous = merged[-1]
            merged[-1] = ScoredSpan(
                previous.start,
                max(previous.end, span.end),
                max(previous.score, span.score),
            )
        else:
            merged.append(span)
    # A final NMS pass drops any spans that still overlap after merging (e.g. a
    # dense sweep of adjacent chunks that merge into a few near-identical spans).
    return temporal_nms(merged, duplicate_iou)

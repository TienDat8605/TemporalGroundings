"""High-FPS deterministic boundary refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class BoundaryRefinementConfig:
    radius_seconds: float = 2.0
    evidence_window_seconds: float = 0.5
    continuity_weight: float = 0.25
    minimum_gain: float = 0.01


@dataclass(frozen=True)
class RefinedInterval:
    original: tuple[float, float]
    refined: tuple[float, float]
    start_gain: float
    end_gain: float
    start_accepted: bool
    end_accepted: bool


def _local_mean(values: np.ndarray, center: int, radius: int, side: str) -> float:
    if side == 'left':
        selected = values[max(0, center - radius):center]
    else:
        selected = values[center:min(len(values), center + radius)]
    return float(selected.mean()) if len(selected) else float(values[center])


def boundary_scores(
    timestamps: Sequence[float],
    evidence: Sequence[float],
    continuity: Sequence[float] | None,
    *,
    window_seconds: float,
    continuity_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(timestamps, dtype=float)
    values = np.asarray(evidence, dtype=float)
    if len(times) != len(values) or len(times) == 0:
        raise ValueError('timestamps and evidence must be equal non-empty vectors')
    if np.any(np.diff(times) < 0):
        raise ValueError('timestamps must be sorted')
    change = np.zeros(len(times), dtype=float) if continuity is None else np.asarray(continuity, dtype=float)
    if len(change) != len(times):
        raise ValueError('continuity must match timestamps')
    step = float(np.median(np.diff(times))) if len(times) > 1 else window_seconds
    radius = max(1, int(round(window_seconds / max(step, 1e-6))))
    start_scores = np.empty(len(times), dtype=float)
    end_scores = np.empty(len(times), dtype=float)
    for index in range(len(times)):
        left = _local_mean(values, index, radius, 'left')
        right = _local_mean(values, index, radius, 'right')
        start_scores[index] = right - left + continuity_weight * change[index]
        end_scores[index] = left - right + continuity_weight * change[index]
    return start_scores, end_scores


def _best_near(
    times: np.ndarray,
    scores: np.ndarray,
    original: float,
    radius: float,
) -> tuple[float, float]:
    allowed = np.flatnonzero((times >= original - radius) & (times <= original + radius))
    if len(allowed) == 0:
        return original, 0.0
    order = np.lexsort((times[allowed], np.abs(times[allowed] - original), -scores[allowed]))
    best = int(allowed[order[0]])
    nearest = int(np.argmin(np.abs(times - original)))
    return float(times[best]), float(scores[best] - scores[nearest])


def refine_interval(
    interval: Sequence[float],
    timestamps: Sequence[float],
    evidence: Sequence[float],
    continuity: Sequence[float] | None = None,
    *,
    duration: float,
    component: Sequence[float] | None = None,
    config: BoundaryRefinementConfig = BoundaryRefinementConfig(),
) -> RefinedInterval:
    start, end = float(interval[0]), float(interval[1])
    if not 0 <= start < end <= duration + 1e-6:
        raise ValueError('interval must be ordered and inside the video')
    times = np.asarray(timestamps, dtype=float)
    start_scores, end_scores = boundary_scores(
        times,
        evidence,
        continuity,
        window_seconds=config.evidence_window_seconds,
        continuity_weight=config.continuity_weight,
    )
    candidate_start, start_gain = _best_near(times, start_scores, start, config.radius_seconds)
    candidate_end, end_gain = _best_near(times, end_scores, end, config.radius_seconds)
    start_accepted = start_gain >= config.minimum_gain
    end_accepted = end_gain >= config.minimum_gain
    refined_start = candidate_start if start_accepted else start
    refined_end = candidate_end if end_accepted else end
    if component is not None:
        refined_start = max(float(component[0]), refined_start)
        refined_end = min(float(component[1]), refined_end)
    refined_start = min(duration, max(0.0, refined_start))
    refined_end = min(duration, max(0.0, refined_end))
    if refined_end <= refined_start:
        refined_start, refined_end = start, end
        start_accepted = end_accepted = False
    return RefinedInterval(
        original=(start, end),
        refined=(round(refined_start, 3), round(refined_end, 3)),
        start_gain=start_gain,
        end_gain=end_gain,
        start_accepted=start_accepted,
        end_accepted=end_accepted,
    )

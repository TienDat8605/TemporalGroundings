"""Analytic temporal proposal scoring for the training-free fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ProposalScorerConfig:
    motion_weight: float = 0.25
    median_width: int = 5
    threshold_mad_scale: float = 0.5
    minimum_duration: float = 0.25
    merge_gap: float = 0.5
    exterior_seconds: float = 1.0
    maximum_proposals: int = 8


@dataclass(frozen=True)
class Proposal:
    start: float
    end: float
    score: float
    interior: float
    contrast: float


def evidence_curve(
    query_scores: np.ndarray,
    motion_scores: np.ndarray,
    mask: np.ndarray,
    motion_weight: float = 0.25,
) -> np.ndarray:
    if query_scores.shape != motion_scores.shape or query_scores.shape != mask.shape:
        raise ValueError('query, motion, and mask shapes must match')
    if query_scores.ndim != 3:
        raise ValueError('scores must have shape (time, height, width)')
    selected = np.asarray(mask, dtype=bool)
    output = np.zeros(query_scores.shape[0], dtype=np.float32)
    for index in range(len(output)):
        valid = selected[index]
        if valid.any():
            output[index] = float(
                np.mean(query_scores[index][valid])
                + motion_weight * np.mean(motion_scores[index][valid])
            )
    return output


def median_filter(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return np.asarray(values, dtype=np.float32).copy()
    if width % 2 == 0:
        width += 1
    radius = width // 2
    padded = np.pad(np.asarray(values, dtype=np.float32), (radius, radius), mode='edge')
    return np.asarray([
        np.median(padded[index:index + width]) for index in range(len(values))
    ], dtype=np.float32)


def robust_threshold(values: np.ndarray, scale: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return 0.0
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    return median + scale * max(mad, 1e-6)


def _runs(active: np.ndarray) -> list[tuple[int, int]]:
    output = []
    start = None
    for index, value in enumerate(active.tolist() + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            output.append((start, index - 1))
            start = None
    return output


def _merge_runs(
    runs: list[tuple[int, int]],
    timestamps: np.ndarray,
    gap: float,
) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in runs:
        if merged and timestamps[start] - timestamps[merged[-1][1]] <= gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [tuple(value) for value in merged]


def propose_intervals(
    timestamps: Sequence[float],
    evidence: Sequence[float],
    *,
    cardinality: str,
    route_score: float = 0.0,
    config: ProposalScorerConfig = ProposalScorerConfig(),
) -> list[Proposal]:
    times = np.asarray(timestamps, dtype=float)
    values = median_filter(np.asarray(evidence, dtype=float), config.median_width)
    if times.ndim != 1 or values.ndim != 1 or len(times) != len(values) or len(times) == 0:
        raise ValueError('timestamps and evidence must be equal non-empty vectors')
    if np.any(np.diff(times) < 0):
        raise ValueError('timestamps must be sorted')
    threshold = robust_threshold(values, config.threshold_mad_scale)
    runs = _merge_runs(_runs(values >= threshold), times, config.merge_gap)
    step = float(np.median(np.diff(times))) if len(times) > 1 else config.minimum_duration
    proposals = []
    for start_index, end_index in runs:
        start = float(times[start_index])
        end = float(times[end_index] + step)
        if end - start < config.minimum_duration:
            continue
        interior = float(values[start_index:end_index + 1].mean())
        left = values[(times >= start - config.exterior_seconds) & (times < start)]
        right = values[(times > times[end_index]) & (times <= end + config.exterior_seconds)]
        exterior_values = np.concatenate([left, right])
        exterior = float(exterior_values.mean()) if len(exterior_values) else float(np.median(values))
        contrast = interior - exterior
        score = interior + contrast + float(route_score)
        proposals.append(Proposal(start, end, score, interior, contrast))
    proposals.sort(key=lambda item: (-item.score, item.start, item.end))
    if cardinality == 'single':
        return proposals[:1]
    if cardinality != 'multi':
        raise ValueError(f'unknown cardinality: {cardinality}')
    return sorted(proposals[:config.maximum_proposals], key=lambda item: (item.start, item.end))

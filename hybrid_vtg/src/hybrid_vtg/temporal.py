"""Training-free multi-scale temporal search and budgeted routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .coarse_encoder import normalize_rows
from .config import CoarseConfig, ProposalConfig
from .index import CoarseIndex
from .types import Component, TemporalRoute


@dataclass(frozen=True)
class Candidate:
    index: int
    start: float
    end: float
    scale: float
    mean_score: float = 0.0
    peak_score: float = 0.0
    score: float = 0.0
    left_uncertainty: float = 0.0
    right_uncertainty: float = 0.0

    @property
    def duration(self) -> float:
        return self.end - self.start


def multiscale_candidates(duration: float, scales: Sequence[float], stride_ratio: float) -> list[Candidate]:
    if duration <= 0 or not 0 < stride_ratio <= 1:
        raise ValueError("invalid duration or stride ratio")
    intervals: set[tuple[float, float, float]] = set()
    for scale in sorted({float(value) for value in scales if value > 0}):
        if duration <= scale:
            intervals.add((0.0, duration, scale))
            continue
        stride = scale * stride_ratio
        for start in np.arange(0.0, duration, stride):
            end = min(duration, float(start) + scale)
            intervals.add((round(float(start), 6), round(end, 6), scale))
            if end >= duration:
                break
        intervals.add((round(duration - scale, 6), duration, scale))
    ordered = sorted(intervals, key=lambda item: (item[0], item[1], item[2]))
    return [Candidate(index, start, end, scale) for index, (start, end, scale) in enumerate(ordered)]


def score_candidates(
    candidates: Sequence[Candidate],
    index: CoarseIndex,
    query_embedding: np.ndarray,
    mean_weight: float,
) -> list[Candidate]:
    if not 0 <= mean_weight <= 1:
        raise ValueError("mean_weight must be in [0, 1]")
    features = normalize_rows(index.features)
    query = normalize_rows(np.asarray(query_embedding, dtype=np.float32).reshape(1, -1))[0]
    if features.shape[1] != len(query):
        raise ValueError("query and visual feature dimensions differ")
    frame_scores = features @ query
    output = []
    score_floor = float(np.median(frame_scores))
    score_scale = max(float(np.percentile(frame_scores, 90) - score_floor), 1e-6)
    boundary_samples = max(1, int(round(index.fps)))

    def endpoint_uncertainty(indices: np.ndarray, *, left: bool) -> float:
        boundary = int(indices[0] if left else indices[-1])
        if (left and boundary == 0) or (not left and boundary == len(frame_scores) - 1):
            return 0.0
        inside_indices = indices[:boundary_samples] if left else indices[-boundary_samples:]
        if left:
            outside = frame_scores[max(0, boundary - boundary_samples):boundary]
        else:
            outside = frame_scores[boundary + 1:min(len(frame_scores), boundary + 1 + boundary_samples)]
        inside_score = float(frame_scores[inside_indices].mean())
        outside_score = float(outside.mean()) if len(outside) else score_floor
        boundary_relevance = np.clip((inside_score - score_floor) / score_scale, 0.0, 1.0)
        outward_drop = np.clip((inside_score - outside_score) / score_scale, 0.0, 1.0)
        return float(boundary_relevance * (1.0 - outward_drop))

    for item in candidates:
        mask = (index.timestamps >= item.start) & (index.timestamps < item.end)
        if item.end >= index.duration - 1e-6:
            mask |= index.timestamps == index.timestamps[-1]
        if mask.any():
            pooled = normalize_rows(features[mask].mean(axis=0, keepdims=True))[0]
            mean_score = float(pooled @ query)
            peak_score = float(frame_scores[mask].max())
        else:
            mean_score = peak_score = -1.0
        score = mean_weight * mean_score + (1.0 - mean_weight) * peak_score
        indices = np.flatnonzero(mask)
        left_uncertainty = endpoint_uncertainty(indices, left=True) if len(indices) else 0.0
        right_uncertainty = endpoint_uncertainty(indices, left=False) if len(indices) else 0.0
        output.append(Candidate(
            item.index, item.start, item.end, item.scale, mean_score, peak_score, score,
            left_uncertainty, right_uncertainty,
        ))
    return output


def interval_boundary_quality(
    index: CoarseIndex,
    query_embedding: np.ndarray,
    interval: tuple[float, float],
    component: Component,
    config: ProposalConfig,
) -> dict[str, float]:
    """Rerank a grounded interval by transitions and evidence concentration."""
    features = normalize_rows(index.features)
    query = normalize_rows(np.asarray(query_embedding, dtype=np.float32).reshape(1, -1))[0]
    values = features @ query
    times = index.timestamps
    start, end = interval
    context = config.context_seconds

    def mean_between(lower: float, upper: float, fallback: float) -> float:
        selected = values[(times >= lower) & (times < upper)]
        return float(selected.mean()) if len(selected) else fallback

    inside = values[(times >= start) & (times <= end)]
    component_values = values[(times >= component.start) & (times <= component.end)]
    if not len(inside):
        return {
            "score": -1.0, "boundary_contrast": -1.0,
            "start_contrast": -1.0, "end_contrast": -1.0,
            "tightness": -1.0, "start_confidence": 0.0,
            "end_confidence": 0.0, "boundary_confidence": 0.0,
        }
    inside_mean = float(inside.mean())
    component_mean = float(component_values.mean()) if len(component_values) else inside_mean
    left_inside = mean_between(start, min(end, start + context), inside_mean)
    right_inside = mean_between(max(start, end - context), end + 1e-9, inside_mean)
    left_outside = mean_between(max(component.start, start - context), start, component_mean)
    right_outside = mean_between(end, min(component.end, end + context) + 1e-9, component_mean)
    start_contrast = left_inside - left_outside
    end_contrast = right_inside - right_outside
    boundary_contrast = (start_contrast + end_contrast) / 2.0
    tightness = inside_mean - component_mean

    component_indices = np.flatnonzero((times >= component.start) & (times <= component.end))
    possible_starts = []
    possible_ends = []
    for candidate_index in component_indices:
        candidate_time = float(times[candidate_index])
        center = float(values[candidate_index])
        before = mean_between(max(component.start, candidate_time - context), candidate_time, center)
        after = mean_between(candidate_time, min(component.end, candidate_time + context) + 1e-9, center)
        possible_starts.append(after - before)
        possible_ends.append(before - after)

    def confidence(value: float, candidates: list[float]) -> float:
        distribution = np.asarray(candidates, dtype=float)
        # A flat or non-positive transition is not a confident semantic boundary,
        # even though a conventional percentile rank would assign tied values 1.0.
        if value <= 0 or len(distribution) < 2 or float(np.ptp(distribution)) <= 1e-8:
            return 0.0
        return float(np.mean(distribution <= value))

    start_confidence = confidence(start_contrast, possible_starts)
    end_confidence = confidence(end_contrast, possible_ends)
    weight_sum = config.boundary_contrast_weight + config.tightness_weight
    score = (
        config.boundary_contrast_weight * boundary_contrast
        + config.tightness_weight * tightness
    ) / weight_sum
    return {
        "score": float(score),
        "boundary_contrast": float(boundary_contrast),
        "start_contrast": float(start_contrast),
        "end_contrast": float(end_contrast),
        "tightness": float(tightness),
        "start_confidence": start_confidence,
        "end_confidence": end_confidence,
        "boundary_confidence": min(start_confidence, end_confidence),
    }


def union_seconds(intervals: Sequence[tuple[float, float]]) -> float:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        elif end > start:
            merged.append([start, end])
    return sum(end - start for start, end in merged)


def _expanded(item: Candidate, duration: float, config: CoarseConfig) -> tuple[float, float]:
    base = max(config.minimum_halo_seconds, 1.0 / config.fps, item.scale * config.halo_scale_ratio)
    left = min(config.maximum_halo_seconds, base * (1.0 + item.left_uncertainty))
    right = min(config.maximum_halo_seconds, base * (1.0 + item.right_uncertainty))
    return max(0.0, item.start - left), min(duration, item.end + right)


def _merged_intervals(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + 1e-9:
            merged[-1][1] = max(merged[-1][1], end)
        elif end > start:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _components(
    candidates: Sequence[Candidate], selected: Sequence[int], duration: float, config: CoarseConfig,
) -> tuple[Component, ...]:
    expanded = sorted((*_expanded(candidates[index], duration, config), index) for index in selected)
    merged: list[dict] = []
    for start, end, index in expanded:
        if merged and start <= merged[-1]["end"] + 1e-9:
            merged[-1]["end"] = max(end, merged[-1]["end"])
            merged[-1]["indices"].append(index)
            merged[-1]["score"] = max(candidates[index].score, merged[-1]["score"])
        else:
            merged.append({"start": start, "end": end, "indices": [index], "score": candidates[index].score})
    return tuple(Component(
        start=round(item["start"], 6), end=round(item["end"], 6),
        score=float(item["score"]), source_candidates=tuple(item["indices"]),
    ) for item in merged)


def select_candidates(candidates: Sequence[Candidate], duration: float, config: CoarseConfig) -> TemporalRoute:
    if not candidates:
        raise ValueError("at least one temporal candidate is required")
    order = sorted(range(len(candidates)), key=lambda i: (-candidates[i].score, candidates[i].duration, i))
    scores = np.asarray([item.score for item in candidates])
    confidence = float(scores.max() - np.median(scores))
    fallback = confidence < config.low_confidence_margin
    selected: list[int] = []
    intervals: list[tuple[float, float]] = []

    def try_add(index: int) -> bool:
        if index in selected:
            return False
        item = candidates[index]
        expanded = _expanded(item, duration, config)
        before, after = union_seconds(intervals), union_seconds([*intervals, expanded])
        proposed = _merged_intervals([*intervals, expanded])
        if selected and after - before < config.minimum_uncovered_seconds:
            return False
        if len(proposed) > config.maximum_components or after > config.union_budget_seconds + 1e-9:
            return False
        selected.append(index)
        intervals.append(expanded)
        return True

    if fallback:
        try_add(order[0])
        centers = np.linspace(0.0, duration, config.maximum_components + 2)[1:-1]
        shortest = min(item.scale for item in candidates)
        uniform_pool = [index for index, item in enumerate(candidates) if item.scale == shortest]
        for center in centers:
            index = min(
                uniform_pool,
                key=lambda i: (abs((candidates[i].start + candidates[i].end) / 2 - center), i),
            )
            try_add(index)
    else:
        for index in order:
            try_add(index)
    if not selected:
        raise ValueError("temporal budget cannot fit any halo-expanded candidate")
    components = _components(candidates, selected, duration, config)
    return TemporalRoute(
        components=components,
        selected_candidates=tuple(selected),
        confidence_margin=confidence,
        low_confidence_fallback=fallback,
        retained_union_seconds=union_seconds([(item.start, item.end) for item in components]),
    )


def route(index: CoarseIndex, query_embedding: np.ndarray, config: CoarseConfig) -> tuple[TemporalRoute, list[Candidate]]:
    candidates = multiscale_candidates(index.duration, config.scales, config.stride_ratio)
    scored = score_candidates(candidates, index, query_embedding, config.mean_weight)
    return select_candidates(scored, index.duration, config), scored

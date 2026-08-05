"""Training-free multi-scale temporal search and budgeted routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .coarse_encoder import normalize_rows
from .config import CoarseConfig
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
        output.append(Candidate(item.index, item.start, item.end, item.scale, mean_score, peak_score, score))
    return output


def interval_evidence_score(
    index: CoarseIndex,
    query_embedding: np.ndarray,
    interval: tuple[float, float],
    mean_weight: float,
) -> float:
    """Score a grounded sub-interval using the same frozen evidence as routing."""
    features = normalize_rows(index.features)
    query = normalize_rows(np.asarray(query_embedding, dtype=np.float32).reshape(1, -1))[0]
    mask = (index.timestamps >= interval[0]) & (index.timestamps <= interval[1])
    if not mask.any():
        return -1.0
    pooled = normalize_rows(features[mask].mean(axis=0, keepdims=True))[0]
    mean_score = float(pooled @ query)
    peak_score = float((features[mask] @ query).max())
    return mean_weight * mean_score + (1.0 - mean_weight) * peak_score


def interval_iou(first: tuple[float, float], second: tuple[float, float]) -> float:
    overlap = max(0.0, min(first[1], second[1]) - max(first[0], second[0]))
    union = first[1] - first[0] + second[1] - second[0] - overlap
    return overlap / union if union > 0 else 0.0


def union_seconds(intervals: Sequence[tuple[float, float]]) -> float:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        elif end > start:
            merged.append([start, end])
    return sum(end - start for start, end in merged)


def _expanded(item: Candidate, duration: float, halo: float) -> tuple[float, float]:
    return max(0.0, item.start - halo), min(duration, item.end + halo)


def _components(
    candidates: Sequence[Candidate], selected: Sequence[int], duration: float, halo: float,
) -> tuple[Component, ...]:
    expanded = sorted((*_expanded(candidates[index], duration, halo), index) for index in selected)
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

    def try_add(index: int) -> None:
        if index in selected or len(selected) >= config.maximum_candidates:
            return
        item = candidates[index]
        expanded = _expanded(item, duration, config.halo_seconds)
        overlap = max((interval_iou((item.start, item.end), (candidates[p].start, candidates[p].end))
                       for p in selected), default=0.0)
        before, after = union_seconds(intervals), union_seconds([*intervals, expanded])
        if overlap >= config.nms_iou and after - before < config.minimum_uncovered_seconds:
            return
        if after <= config.union_budget_seconds + 1e-9:
            selected.append(index)
            intervals.append(expanded)

    if fallback:
        try_add(order[0])
        centers = np.linspace(0.0, duration, config.maximum_candidates + 2)[1:-1]
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
    components = _components(candidates, selected, duration, config.halo_seconds)
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

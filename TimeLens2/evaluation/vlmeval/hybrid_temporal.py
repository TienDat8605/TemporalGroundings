"""Training-free coarse-to-fine temporal retrieval utilities."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from vlmeval.omtg_search import Window


@dataclass(frozen=True)
class CoarseIndex:
    video_path: str
    video_hash: str
    checkpoint: str
    fps: float
    timestamps: np.ndarray
    features: np.ndarray

    def validate(self) -> None:
        if self.timestamps.ndim != 1:
            raise ValueError('timestamps must be one-dimensional')
        if self.features.ndim != 2 or self.features.shape[0] != len(self.timestamps):
            raise ValueError('features must have shape (num_timestamps, hidden_size)')
        if len(self.timestamps) == 0 or not np.isfinite(self.timestamps).all():
            raise ValueError('coarse index must contain finite timestamps')
        if not np.isfinite(self.features).all():
            raise ValueError('coarse index must contain finite features')
        if np.any(np.diff(self.timestamps) < 0):
            raise ValueError('timestamps must be sorted')


@dataclass(frozen=True)
class Candidate:
    index: int
    start: float
    end: float
    scale: float
    source: str
    mean_score: float = 0.0
    peak_score: float = 0.0
    score: float = 0.0

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def window(self) -> Window:
        return Window(self.start, self.end)


@dataclass(frozen=True)
class Component:
    start: float
    end: float
    source_candidates: tuple[int, ...]
    score: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class TemporalRoute:
    selected: tuple[int, ...]
    components: tuple[Component, ...]
    confidence_margin: float
    low_confidence_fallback: bool
    retained_union_seconds: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_rows(features: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    value = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(value, axis=-1, keepdims=True)
    return value / np.maximum(norms, eps)


def save_index(path: Path, index: CoarseIndex) -> None:
    index.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        'video_path': index.video_path,
        'video_hash': index.video_hash,
        'checkpoint': index.checkpoint,
        'fps': index.fps,
    }
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(
            handle,
            metadata=json.dumps(metadata, sort_keys=True),
            timestamps=index.timestamps.astype(np.float64),
            features=index.features.astype(np.float32),
        )
    temporary.replace(path)


def load_index(path: Path) -> CoarseIndex:
    with np.load(path, allow_pickle=False) as value:
        metadata = json.loads(str(value['metadata']))
        index = CoarseIndex(
            **metadata,
            timestamps=np.asarray(value['timestamps'], dtype=np.float64),
            features=np.asarray(value['features'], dtype=np.float32),
        )
    index.validate()
    return index


def index_cache_key(
    video_hash: str,
    checkpoint: str,
    fps: float,
    width: int,
    instruction: str,
) -> str:
    payload = json.dumps(
        {
            'video_hash': video_hash,
            'checkpoint': checkpoint,
            'fps': fps,
            'width': width,
            'instruction': instruction,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def multiscale_candidates(
    duration: float,
    scales: Sequence[float] = (8.0, 16.0, 32.0, 64.0),
    stride_ratio: float = 0.5,
    content_windows: Iterable[Window] = (),
) -> list[Candidate]:
    if duration <= 0:
        raise ValueError('duration must be positive')
    if not 0 < stride_ratio <= 1:
        raise ValueError('stride_ratio must be in (0, 1]')
    raw: list[tuple[float, float, float, str]] = []
    for scale in sorted({float(value) for value in scales if value > 0}):
        if duration <= scale:
            raw.append((0.0, duration, scale, 'multiscale'))
            continue
        stride = max(1e-3, scale * stride_ratio)
        start = 0.0
        while start < duration:
            end = min(duration, start + scale)
            if end - start > 1e-6:
                raw.append((start, end, scale, 'multiscale'))
            if end >= duration:
                break
            start += stride
        tail_start = max(0.0, duration - scale)
        raw.append((tail_start, duration, scale, 'multiscale'))
    for window in content_windows:
        start = max(0.0, min(duration, float(window.start)))
        end = max(start, min(duration, float(window.end)))
        if end > start:
            raw.append((start, end, end - start, 'content'))
    unique = sorted(
        {(round(a, 6), round(b, 6), round(s, 6), source) for a, b, s, source in raw},
        key=lambda value: (value[0], value[1], value[3]),
    )
    return [Candidate(i, a, b, scale, source) for i, (a, b, scale, source) in enumerate(unique)]


def score_candidates(
    candidates: Sequence[Candidate],
    index: CoarseIndex,
    query_embedding: np.ndarray,
    mean_weight: float = 0.5,
) -> list[Candidate]:
    index.validate()
    if not 0 <= mean_weight <= 1:
        raise ValueError('mean_weight must be in [0, 1]')
    features = normalize_rows(index.features)
    query = normalize_rows(np.asarray(query_embedding, dtype=np.float32).reshape(1, -1))[0]
    if features.shape[1] != query.shape[0]:
        raise ValueError('query and visual feature dimensions differ')
    similarities = features @ query
    output = []
    for candidate in candidates:
        mask = (index.timestamps >= candidate.start) & (index.timestamps < candidate.end)
        if candidate.end >= index.timestamps[-1]:
            mask |= index.timestamps == index.timestamps[-1]
        values = similarities[mask]
        if len(values):
            pooled = normalize_rows(features[mask].mean(axis=0, keepdims=True))[0]
            mean_score = float(pooled @ query)
            peak_score = float(values.max())
        else:
            mean_score = peak_score = -1.0
        score = mean_weight * mean_score + (1.0 - mean_weight) * peak_score
        output.append(Candidate(
            **{key: value for key, value in asdict(candidate).items()
               if key not in ('mean_score', 'peak_score', 'score')},
            mean_score=mean_score,
            peak_score=peak_score,
            score=float(score),
        ))
    return output


def interval_iou(first: tuple[float, float], second: tuple[float, float]) -> float:
    intersection = max(0.0, min(first[1], second[1]) - max(first[0], second[0]))
    union = first[1] - first[0] + second[1] - second[0] - intersection
    return intersection / union if union > 0 else 0.0


def union_seconds(intervals: Iterable[tuple[float, float]]) -> float:
    merged: list[list[float]] = []
    for start, end in sorted((float(a), float(b)) for a, b in intervals if b > a):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged)


def _halo(candidate: Candidate, duration: float, seconds: float) -> tuple[float, float]:
    return max(0.0, candidate.start - seconds), min(duration, candidate.end + seconds)


def _merge_components(
    candidates: Sequence[Candidate],
    selected: Sequence[int],
    duration: float,
    halo_seconds: float,
) -> tuple[Component, ...]:
    expanded = sorted(
        [(*_halo(candidates[index], duration, halo_seconds), index) for index in selected],
        key=lambda item: (item[0], item[1], item[2]),
    )
    merged: list[dict] = []
    for start, end, index in expanded:
        if merged and start <= merged[-1]['end'] + 1e-9:
            merged[-1]['end'] = max(merged[-1]['end'], end)
            merged[-1]['indices'].append(index)
            merged[-1]['score'] = max(merged[-1]['score'], candidates[index].score)
        else:
            merged.append({'start': start, 'end': end, 'indices': [index], 'score': candidates[index].score})
    return tuple(Component(
        start=round(item['start'], 6),
        end=round(item['end'], 6),
        source_candidates=tuple(item['indices']),
        score=float(item['score']),
    ) for item in merged)


def select_candidates(
    candidates: Sequence[Candidate],
    duration: float,
    *,
    union_budget_seconds: float,
    maximum_candidates: int = 8,
    nms_iou: float = 0.7,
    minimum_uncovered_seconds: float = 1.0,
    halo_seconds: float = 2.0,
    low_confidence_margin: float = 0.05,
) -> TemporalRoute:
    if not candidates:
        raise ValueError('at least one candidate is required')
    if union_budget_seconds <= 0 or maximum_candidates <= 0:
        raise ValueError('budgets must be positive')
    order = sorted(
        range(len(candidates)),
        key=lambda i: (-candidates[i].score, candidates[i].duration, candidates[i].start, i),
    )
    scores = np.asarray([candidate.score for candidate in candidates], dtype=float)
    confidence = float(scores.max() - np.median(scores))
    fallback = confidence < low_confidence_margin
    selected: list[int] = []
    selected_intervals: list[tuple[float, float]] = []
    for index in order:
        if len(selected) >= maximum_candidates:
            break
        candidate = candidates[index]
        expanded = _halo(candidate, duration, halo_seconds)
        overlap = max(
            (interval_iou((candidate.start, candidate.end),
                          (candidates[prior].start, candidates[prior].end)) for prior in selected),
            default=0.0,
        )
        before = union_seconds(selected_intervals)
        after = union_seconds([*selected_intervals, expanded])
        added = after - before
        if overlap >= nms_iou and added < minimum_uncovered_seconds:
            continue
        if after > union_budget_seconds + 1e-9:
            continue
        selected.append(index)
        selected_intervals.append(expanded)
    if not selected:
        raise ValueError(
            'temporal union budget cannot fit any halo-expanded candidate; '
            'increase the budget or reduce window scales/halo'
        )
    if fallback and len(selected) < maximum_candidates:
        centers = np.linspace(0.0, duration, maximum_candidates + 2)[1:-1]
        uniform_order = sorted(
            range(len(candidates)),
            key=lambda i: (min(abs((candidates[i].start + candidates[i].end) / 2 - c) for c in centers), i),
        )
        for index in uniform_order:
            if index in selected or len(selected) >= maximum_candidates:
                continue
            expanded = _halo(candidates[index], duration, halo_seconds)
            if union_seconds([*selected_intervals, expanded]) <= union_budget_seconds + 1e-9:
                selected.append(index)
                selected_intervals.append(expanded)
    components = _merge_components(candidates, selected, duration, halo_seconds)
    return TemporalRoute(
        selected=tuple(selected),
        components=components,
        confidence_margin=confidence,
        low_confidence_fallback=fallback,
        retained_union_seconds=union_seconds((item.start, item.end) for item in components),
    )

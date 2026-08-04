"""Deterministic spatial-token scoring and budgeted retention."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class SpatialPruningConfig:
    keep_ratio: float = 0.25
    query_weight: float = 0.35
    motion_weight: float = 0.25
    uniqueness_weight: float = 0.20
    boundary_weight: float = 0.20
    query_quota: float = 0.10
    motion_quota: float = 0.08
    uniqueness_quota: float = 0.06
    boundary_quota: float = 0.06
    minimum_tokens_per_frame: int = 1
    spatial_cells: tuple[int, int] = (2, 2)
    boundary_top_fraction: float = 0.25

    def validate(self) -> None:
        if not 0 < self.keep_ratio <= 1:
            raise ValueError('keep_ratio must be in (0, 1]')
        weights = (self.query_weight, self.motion_weight, self.uniqueness_weight, self.boundary_weight)
        if any(value < 0 for value in weights) or sum(weights) <= 0:
            raise ValueError('score weights must be non-negative and not all zero')
        quotas = (self.query_quota, self.motion_quota, self.uniqueness_quota, self.boundary_quota)
        if any(not 0 <= value <= 1 for value in quotas):
            raise ValueError('quotas must be in [0, 1]')
        if self.minimum_tokens_per_frame < 1:
            raise ValueError('minimum_tokens_per_frame must be positive')


@dataclass(frozen=True)
class SpatialSignals:
    query: np.ndarray
    motion: np.ndarray
    uniqueness: np.ndarray
    boundary: np.ndarray
    combined: np.ndarray
    motion_fallback: np.ndarray


@dataclass(frozen=True)
class SpatialPruningResult:
    mask: np.ndarray
    kept_indices: np.ndarray
    signals: SpatialSignals
    budget: int
    protected_count: int


@dataclass(frozen=True)
class MergedSpatialTokens:
    embeddings: np.ndarray
    frame_indices: np.ndarray
    grid_yx: np.ndarray
    source_indices: tuple[tuple[int, ...], ...]


def _normalize_features(features: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    value = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(value, axis=-1, keepdims=True)
    return value / np.maximum(norms, eps)


def _stable_top(values: np.ndarray, count: int, allowed: np.ndarray | None = None) -> np.ndarray:
    flat = np.asarray(values, dtype=float).reshape(-1)
    indices = np.arange(len(flat)) if allowed is None else np.flatnonzero(allowed.reshape(-1))
    if count <= 0 or len(indices) == 0:
        return np.empty(0, dtype=np.int64)
    order = np.lexsort((indices, -flat[indices]))
    return indices[order[:min(count, len(indices))]].astype(np.int64)


def percentile_ranks(values: np.ndarray, component_ids: Sequence[int]) -> np.ndarray:
    value = np.asarray(values, dtype=float)
    if value.ndim != 3:
        raise ValueError('values must have shape (time, height, width)')
    if len(component_ids) != value.shape[0]:
        raise ValueError('one component id is required per frame')
    output = np.zeros_like(value, dtype=np.float32)
    component_ids = np.asarray(component_ids)
    for component in sorted(set(component_ids.tolist())):
        frame_mask = component_ids == component
        selected = np.nan_to_num(value[frame_mask].reshape(-1), nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
        order = np.argsort(selected, kind='stable')
        ranks = np.empty(len(selected), dtype=np.float32)
        if len(selected) == 1 or np.all(selected == selected[0]):
            ranks[:] = 0.0
        else:
            position = 0
            while position < len(order):
                end = position + 1
                while end < len(order) and selected[order[end]] == selected[order[position]]:
                    end += 1
                average_rank = ((position + end - 1) / 2) / (len(order) - 1)
                ranks[order[position:end]] = average_rank
                position = end
        output[frame_mask] = ranks.reshape(output[frame_mask].shape)
    return output


def query_relevance(features: np.ndarray, query_embedding: np.ndarray) -> np.ndarray:
    normalized = _normalize_features(features)
    query = _normalize_features(np.asarray(query_embedding).reshape(1, -1))[0]
    if normalized.shape[-1] != len(query):
        raise ValueError('query and token dimensions differ')
    return np.einsum('thwd,d->thw', normalized, query).astype(np.float32)


def _gray(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
    return cv2.resize(image.astype(np.float32) / 255.0, size, interpolation=cv2.INTER_AREA)


def _affine(previous: np.ndarray, current: np.ndarray, grid_size: tuple[int, int]) -> tuple[np.ndarray, bool]:
    width, height = grid_size
    before = _gray(previous, (max(32, width * 8), max(32, height * 8)))
    after = _gray(current, before.shape[::-1])
    transform = np.eye(2, 3, dtype=np.float32)
    try:
        _, transform = cv2.findTransformECC(
            after,
            before,
            transform,
            cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5),
        )
        scale_x = width / before.shape[1]
        scale_y = height / before.shape[0]
        transform[0, 2] *= scale_x
        transform[1, 2] *= scale_y
        return transform, False
    except cv2.error:
        return np.eye(2, 3, dtype=np.float32), True


def motion_importance(
    features: np.ndarray,
    component_ids: Sequence[int],
    rgb_frames: Sequence[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    normalized = _normalize_features(features)
    time, height, width, hidden = normalized.shape
    components = np.asarray(component_ids)
    output = np.zeros((time, height, width), dtype=np.float32)
    fallback = np.zeros(time, dtype=bool)
    for index in range(1, time):
        if components[index] != components[index - 1]:
            continue
        previous = normalized[index - 1]
        if rgb_frames is not None:
            transform, failed = _affine(rgb_frames[index - 1], rgb_frames[index], (width, height))
            fallback[index] = failed
            warped = np.empty_like(previous)
            for channel in range(hidden):
                warped[..., channel] = cv2.warpAffine(
                    previous[..., channel],
                    transform,
                    (width, height),
                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_REPLICATE,
                )
            warped = _normalize_features(warped)
        else:
            warped = previous
            fallback[index] = True
        output[index] = 1.0 - np.einsum('hwd,hwd->hw', normalized[index], warped)
    return np.maximum(output, 0.0), fallback


def uniqueness_importance(features: np.ndarray, component_ids: Sequence[int]) -> np.ndarray:
    normalized = _normalize_features(features)
    time, height, width, _ = normalized.shape
    components = np.asarray(component_ids)
    maximum = np.full((time, height, width), -1.0, dtype=np.float32)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        shifted = np.roll(normalized, shift=(dy, dx), axis=(1, 2))
        similarity = np.einsum('thwd,thwd->thw', normalized, shifted)
        if dy < 0:
            similarity[:, dy:, :] = -1
        elif dy > 0:
            similarity[:, :dy, :] = -1
        if dx < 0:
            similarity[:, :, dx:] = -1
        elif dx > 0:
            similarity[:, :, :dx] = -1
        maximum = np.maximum(maximum, similarity)
    for offset in (-1, 1):
        shifted = np.roll(normalized, offset, axis=0)
        similarity = np.einsum('thwd,thwd->thw', normalized, shifted)
        valid = components == np.roll(components, offset)
        if offset < 0:
            valid[offset:] = False
        else:
            valid[:offset] = False
        maximum = np.maximum(maximum, np.where(valid[:, None, None], similarity, -1.0))
    return np.clip(1.0 - maximum, 0.0, 2.0)


def boundary_importance(
    query_scores: np.ndarray,
    component_ids: Sequence[int],
    top_fraction: float = 0.25,
) -> np.ndarray:
    time, height, width = query_scores.shape
    count = max(1, int(math.ceil(height * width * top_fraction)))
    sorted_scores = np.sort(query_scores.reshape(time, -1), axis=1)
    evidence = sorted_scores[:, -count:].mean(axis=1)
    components = np.asarray(component_ids)
    boundary = np.zeros(time, dtype=np.float32)
    for index in range(time):
        if index > 0 and components[index] == components[index - 1]:
            boundary[index] += abs(float(evidence[index] - evidence[index - 1]))
        if index + 1 < time and components[index] == components[index + 1]:
            boundary[index] += abs(float(evidence[index + 1] - evidence[index]))
    return np.broadcast_to(boundary[:, None, None], (time, height, width)).copy()


def _spatial_anchors(shape: tuple[int, int, int], cells: tuple[int, int]) -> list[int]:
    time, height, width = shape
    output = []
    for t in range(time):
        for row in np.array_split(np.arange(height), cells[0]):
            for col in np.array_split(np.arange(width), cells[1]):
                if len(row) and len(col):
                    y, x = int(row[len(row) // 2]), int(col[len(col) // 2])
                    output.append(np.ravel_multi_index((t, y, x), shape))
    return output


def prune_spatial_tokens(
    features: np.ndarray,
    query_embedding: np.ndarray,
    component_ids: Sequence[int],
    *,
    rgb_frames: Sequence[np.ndarray] | None = None,
    config: SpatialPruningConfig = SpatialPruningConfig(),
) -> SpatialPruningResult:
    config.validate()
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 4:
        raise ValueError('features must have shape (time, height, width, hidden)')
    time, height, width, _ = features.shape
    if len(component_ids) != time:
        raise ValueError('one component id is required per frame')
    if rgb_frames is not None and len(rgb_frames) != time:
        raise ValueError('one RGB frame is required per feature frame')
    query = query_relevance(features, query_embedding)
    motion, fallback = motion_importance(features, component_ids, rgb_frames)
    uniqueness = uniqueness_importance(features, component_ids)
    boundary = boundary_importance(query, component_ids, config.boundary_top_fraction)
    q_rank = percentile_ranks(query, component_ids)
    m_rank = percentile_ranks(motion, component_ids)
    u_rank = percentile_ranks(uniqueness, component_ids)
    b_rank = percentile_ranks(boundary, component_ids)
    weight_sum = config.query_weight + config.motion_weight + config.uniqueness_weight + config.boundary_weight
    combined = (
        config.query_weight * q_rank
        + config.motion_weight * m_rank
        + config.uniqueness_weight * u_rank
        + config.boundary_weight * b_rank
    ) / weight_sum
    total = time * height * width
    budget = max(time * config.minimum_tokens_per_frame, int(math.ceil(total * config.keep_ratio)))
    budget = min(total, budget)
    mandatory: set[int] = set()
    protected: set[int] = set()
    for signal, ratio in (
        (q_rank, config.query_quota),
        (m_rank, config.motion_quota),
        (u_rank, config.uniqueness_quota),
        (b_rank, config.boundary_quota),
    ):
        if float(np.ptp(signal)) > 1e-12:
            protected.update(_stable_top(signal, int(math.ceil(budget * ratio))).tolist())
    for frame in range(time):
        offset = frame * height * width
        local = _stable_top(combined[frame], config.minimum_tokens_per_frame)
        mandatory.update((offset + local).tolist())
    protected.update(_spatial_anchors((time, height, width), config.spatial_cells))
    protected.update(mandatory)
    if len(protected) > budget:
        optional = protected - mandatory
        remaining = budget - len(mandatory)
        allowed = np.zeros(total, dtype=bool)
        allowed[list(optional)] = True
        kept_set = set(mandatory)
        kept_set.update(_stable_top(combined, remaining, allowed).tolist())
        kept = np.asarray(sorted(kept_set), dtype=np.int64)
    else:
        kept_set = set(protected)
        for index in _stable_top(combined, total):
            kept_set.add(int(index))
            if len(kept_set) >= budget:
                break
        kept = np.asarray(sorted(kept_set), dtype=np.int64)
    mask = np.zeros(total, dtype=bool)
    mask[kept] = True
    mask = mask.reshape(time, height, width)
    signals = SpatialSignals(q_rank, m_rank, u_rank, b_rank, combined.astype(np.float32), fallback)
    return SpatialPruningResult(mask, kept, signals, budget, min(len(protected), budget))


def merge_redundant_background(
    features: np.ndarray,
    result: SpatialPruningResult,
    *,
    similarity_threshold: float = 0.95,
    maximum_motion_rank: float = 0.25,
    maximum_uniqueness_rank: float = 0.25,
) -> MergedSpatialTokens:
    """Merge only discarded, low-motion background within 2x2 frame cells.

    Retained tokens pass through unchanged. Discarded tokens are summarized
    only when a same-frame 2x2 cell contains at least two mutually similar,
    low-motion, low-uniqueness tokens. Every output records its source indices.
    """
    value = np.asarray(features, dtype=np.float32)
    if value.ndim != 4 or result.mask.shape != value.shape[:3]:
        raise ValueError('features and pruning mask shapes differ')
    normalized = _normalize_features(value)
    time, height, width, _ = value.shape
    embeddings = []
    frame_indices = []
    coordinates = []
    sources: list[tuple[int, ...]] = []

    def append_group(frame: int, members: list[tuple[int, int]]) -> None:
        flat = tuple(
            int(np.ravel_multi_index((frame, y, x), (time, height, width)))
            for y, x in members
        )
        embeddings.append(np.mean([value[frame, y, x] for y, x in members], axis=0))
        frame_indices.append(frame)
        coordinates.append((
            float(np.mean([y for y, _ in members])),
            float(np.mean([x for _, x in members])),
        ))
        sources.append(flat)

    for frame in range(time):
        for y in range(height):
            for x in range(width):
                if result.mask[frame, y, x]:
                    append_group(frame, [(y, x)])
        for y0 in range(0, height, 2):
            for x0 in range(0, width, 2):
                candidates = [
                    (y, x)
                    for y in range(y0, min(height, y0 + 2))
                    for x in range(x0, min(width, x0 + 2))
                    if not result.mask[frame, y, x]
                    and result.signals.motion[frame, y, x] <= maximum_motion_rank
                    and result.signals.uniqueness[frame, y, x] <= maximum_uniqueness_rank
                ]
                if len(candidates) < 2:
                    continue
                reference = normalized[frame, candidates[0][0], candidates[0][1]]
                similar = [
                    member for member in candidates
                    if float(reference @ normalized[frame, member[0], member[1]]) >= similarity_threshold
                ]
                if len(similar) >= 2:
                    append_group(frame, similar)
    hidden = value.shape[-1]
    return MergedSpatialTokens(
        embeddings=np.asarray(embeddings, dtype=np.float32).reshape(-1, hidden),
        frame_indices=np.asarray(frame_indices, dtype=np.int64),
        grid_yx=np.asarray(coordinates, dtype=np.float32).reshape(-1, 2),
        source_indices=tuple(sources),
    )

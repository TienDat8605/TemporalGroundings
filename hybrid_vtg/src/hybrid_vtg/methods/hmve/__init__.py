"""Hierarchical multi-view evidence (HMVE)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from ...contracts import (
    GroundingContext,
    Method,
    ModelBackend,
    Prediction,
    Sample,
    TemporalEvidence,
)
from ...media import uniform_timestamps


@dataclass(frozen=True)
class Corridor:
    start: float
    end: float
    score: float


@dataclass(frozen=True)
class BoundaryBand:
    start: float
    end: float
    role: str
    score: float


def _temporal_scores(evidence: TemporalEvidence, scores) -> list[tuple[float, float, int]]:
    grouped: dict[float, list[tuple[float, int]]] = {}
    for index, (timestamp, score) in enumerate(zip(evidence.timestamps, scores.tolist())):
        grouped.setdefault(round(timestamp, 6), []).append((float(score), index))
    return [(timestamp, max(values)[0], max(values)[1]) for timestamp, values in sorted(grouped.items())]


def propose_corridors(
    temporal_scores: list[tuple[float, float, int]],
    duration: float,
    maximum: int = 4,
) -> tuple[Corridor, ...]:
    ranked = sorted(temporal_scores, key=lambda value: (-value[1], value[0]))
    centers: list[tuple[float, float]] = []
    for timestamp, score, _ in ranked:
        if all(abs(timestamp - prior) >= 8.0 for prior, _ in centers):
            centers.append((timestamp, score))
        if len(centers) == maximum:
            break
    if not centers:
        centers = [(duration / 2.0, 0.0)]
    corridors = []
    for center, score in centers:
        start = max(0.0, center - 8.0)
        end = min(duration, center + 8.0)
        start = max(0.0, end - 16.0)
        corridors.append(Corridor(start, end, score))
    return tuple(sorted(corridors, key=lambda value: value.start))


def propose_boundary_bands(
    temporal_scores: list[tuple[float, float, int]],
    corridors: tuple[Corridor, ...],
    duration: float,
    radius: float = 2.0,
) -> tuple[BoundaryBand, ...]:
    """Find query-relevance rises and falls without generating intermediate spans."""
    if radius <= 0:
        raise ValueError("boundary radius must be positive")
    bands = []
    for corridor in corridors:
        values = [value for value in temporal_scores if corridor.start <= value[0] <= corridor.end]
        if len(values) < 2:
            centers = ((corridor.start, "start", 0.0), (corridor.end, "end", 0.0))
        else:
            rises = [(values[index][0], values[index][1] - values[index - 1][1]) for index in range(1, len(values))]
            falls = [(values[index][0], values[index][1] - values[index + 1][1]) for index in range(len(values) - 1)]
            start_time, start_score = max(rises, key=lambda value: (value[1], -value[0]))
            end_time, end_score = max(falls, key=lambda value: (value[1], value[0]))
            centers = ((start_time, "start", start_score), (end_time, "end", end_score))
        for center, role, score in centers:
            start = max(0.0, center - radius)
            end = min(duration, center + radius)
            if end > start:
                bands.append(BoundaryBand(start, end, role, float(score)))
    return tuple(sorted(bands, key=lambda value: (value.start, value.end, value.role)))


def observation_timestamps(windows, fps: float, minimum_per_window: int = 2) -> tuple[float, ...]:
    """Build one chronological batch for a logical observation pass."""
    if fps <= 0:
        raise ValueError("observation FPS must be positive")
    values: dict[float, float] = {}
    for window in windows:
        count = max(minimum_per_window, math.ceil((window.end - window.start) * fps))
        for timestamp in uniform_timestamps(window.start, window.end, count):
            values.setdefault(round(timestamp, 6), timestamp)
    timestamps = [values[key] for key in sorted(values)]
    if len(timestamps) % 2:
        # Qwen temporal tubelets consume pairs. Repeating the final real frame
        # keeps a single logical encoder call without inventing a new time.
        timestamps.append(timestamps[-1])
    return tuple(timestamps)


def pack_evidence(evidence: TemporalEvidence, scores, target: int, anchor_indices: set[int]) -> TemporalEvidence:
    import torch
    import torch.nn.functional as functional

    target = min(max(target, len(anchor_indices)), evidence.size)
    normalized = functional.normalize(evidence.embeddings.float(), dim=-1, eps=1e-6)
    selected = set(anchor_indices)
    ranked = torch.argsort(scores.float(), descending=True, stable=True).tolist()
    for index in ranked:
        if index in selected:
            continue
        duplicate = any(
            abs(evidence.timestamps[index] - evidence.timestamps[prior]) <= 0.51
            and float(torch.dot(normalized[index], normalized[prior]).item()) >= 0.98
            for prior in selected
        )
        if not duplicate:
            selected.add(index)
        if len(selected) == target:
            break
    if len(selected) < target:
        for index in ranked:
            selected.add(index)
            if len(selected) == target:
                break
    ordered = sorted(selected, key=lambda index: (evidence.timestamps[index], index))
    return evidence.select(ordered)


class HMVE(Method):
    name = "hmve"

    def __init__(
        self,
        scout_fps: float = 0.5,
        detail_fps: float = 1.0,
        boundary_fps: float = 2.0,
        boundary_radius: float = 2.0,
        retention_ratio: float = 0.125,
    ) -> None:
        self.scout_fps = scout_fps
        self.detail_fps = detail_fps
        self.boundary_fps = boundary_fps
        self.boundary_radius = boundary_radius
        self.retention_ratio = retention_ratio

    def run(self, sample: Sample, model: ModelBackend, cache_dir: Path) -> Prediction:
        del cache_dir
        self.validate_model(model)
        scout_frames = max(2, math.ceil(sample.duration * self.scout_fps))
        if model.maximum_evidence_units is not None:
            # Leave at least half of a bounded temporal model's capacity for
            # detailed corridor evidence instead of filling it with anchors.
            scout_frames = min(scout_frames, max(2, model.maximum_evidence_units // 2))
        if scout_frames % 2:
            scout_frames = scout_frames - 1 if scout_frames > 2 else 2
        scout = model.encode(sample, uniform_timestamps(0.0, sample.duration, scout_frames))
        scout_scores = model.query_scores(scout, sample.query)
        temporal = _temporal_scores(scout, scout_scores)
        corridors = propose_corridors(temporal, sample.duration)

        # Pass 1: encode every medium-detail corridor in one chronological batch.
        corridor_timestamps = observation_timestamps(corridors, self.detail_fps)
        corridor_evidence = model.encode(sample, corridor_timestamps)
        corridor_scores = model.query_scores(corridor_evidence, sample.query)
        refined_temporal = _temporal_scores(corridor_evidence, corridor_scores)

        # Pass 2: revisit directional relevance rises/falls at high temporal detail.
        boundary_bands = propose_boundary_bands(
            refined_temporal,
            corridors,
            sample.duration,
            self.boundary_radius,
        )
        boundary_timestamps = observation_timestamps(boundary_bands, self.boundary_fps)
        boundary_evidence = model.encode(sample, boundary_timestamps)

        dense_reference_frames = max(2, math.ceil(sample.duration * self.boundary_fps))
        units_per_frame = scout.size / max(scout.source_frames, 1)
        target = max(1, round(dense_reference_frames * units_per_frame * self.retention_ratio))
        target = max(target, len(temporal))
        if model.maximum_evidence_units is not None:
            target = min(target, model.maximum_evidence_units)

        merged = TemporalEvidence.concatenate([scout, corridor_evidence, boundary_evidence])
        merged_scores = model.query_scores(merged, sample.query)

        anchor_indices = set()
        scout_counts: dict[float, int] = {}
        for timestamp in scout.timestamps:
            key = round(timestamp, 6)
            scout_counts[key] = scout_counts.get(key, 0) + 1
        for timestamp, _, _ in temporal:
            candidates = [index for index, value in enumerate(merged.timestamps) if abs(value - timestamp) < 1e-5]
            if candidates:
                scout_candidates = candidates[: scout_counts[round(timestamp, 6)]]
                anchor_indices.add(max(scout_candidates, key=lambda index: float(merged_scores[index])))
        compact = pack_evidence(merged, merged_scores, target, anchor_indices)
        prediction = model.predict(sample, compact, GroundingContext(0.0, sample.duration))
        return Prediction(
            prediction.spans,
            prediction.raw_output,
            {
                **prediction.telemetry,
                "scout_frames": scout_frames,
                "corridor_frames": corridor_evidence.source_frames,
                "boundary_frames": boundary_evidence.source_frames,
                "detail_frames": corridor_evidence.source_frames + boundary_evidence.source_frames,
                "encoder_calls": 3,
                "query_scoring_calls": 3,
                "llm_or_fusion_calls": 1,
                "corridors": [corridor.__dict__ for corridor in corridors],
                "boundary_bands": [band.__dict__ for band in boundary_bands],
                "passes": [
                    {"id": 0, "role": "global-scout", "frames": scout.source_frames},
                    {"id": 1, "role": "corridor-refinement", "frames": corridor_evidence.source_frames},
                    {"id": 2, "role": "boundary-refinement", "frames": boundary_evidence.source_frames},
                ],
                "created_evidence": merged.size,
                "retained_evidence": compact.size,
                "scout_anchors": len(anchor_indices),
                "absolute_timestamps_preserved": True,
            },
        )

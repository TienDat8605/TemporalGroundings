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
        detail_fps: float = 2.0,
        retention_ratio: float = 0.125,
    ) -> None:
        self.scout_fps = scout_fps
        self.detail_fps = detail_fps
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

        dense_reference_frames = max(2, math.ceil(sample.duration * self.detail_fps))
        units_per_frame = scout.size / max(scout.source_frames, 1)
        target = max(1, round(dense_reference_frames * units_per_frame * self.retention_ratio))
        target = max(target, len(temporal))
        if model.maximum_evidence_units is not None:
            target = min(target, model.maximum_evidence_units)

        detail_frame_budget = max(2 * len(corridors), math.ceil(target / max(units_per_frame, 1e-6)))
        per_corridor, remainder = divmod(detail_frame_budget, len(corridors))
        detailed_blocks = []
        for index, corridor in enumerate(corridors):
            count = max(2, per_corridor + int(index < remainder))
            if count % 2:
                count += 1
            detailed_blocks.append(
                model.encode(
                    sample,
                    uniform_timestamps(corridor.start, corridor.end, count),
                )
            )
        merged = TemporalEvidence.concatenate([scout, *detailed_blocks])
        merged_scores = model.query_scores(merged, sample.query)

        anchor_indices = set()
        for timestamp, _, scout_index in temporal:
            candidates = [index for index, value in enumerate(merged.timestamps) if abs(value - timestamp) < 1e-5]
            if candidates:
                anchor_indices.add(max(candidates, key=lambda index: float(merged_scores[index])))
        compact = pack_evidence(merged, merged_scores, target, anchor_indices)
        prediction = model.predict(sample, compact, GroundingContext(0.0, sample.duration))
        return Prediction(
            prediction.spans,
            prediction.raw_output,
            {
                **prediction.telemetry,
                "scout_frames": scout_frames,
                "detail_frames": sum(value.source_frames for value in detailed_blocks),
                "encoder_calls": 1 + len(detailed_blocks),
                "llm_or_fusion_calls": 1,
                "corridors": [corridor.__dict__ for corridor in corridors],
                "created_evidence": merged.size,
                "retained_evidence": compact.size,
                "scout_anchors": len(anchor_indices),
                "absolute_timestamps_preserved": True,
            },
        )

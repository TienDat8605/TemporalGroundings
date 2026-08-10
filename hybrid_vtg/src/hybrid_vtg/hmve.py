"""Deterministic Phase-A controller and evidence accumulator for HMVE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from .tpsa import patch_query_relevance, percentile_rank


@dataclass(frozen=True)
class ObservationCorridor:
    start: float
    end: float
    score: float
    center: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class EvidenceUnit:
    pass_id: int
    absolute_time: float
    grid_height: int
    grid_width: int
    source_height: int
    source_width: int
    embeddings: torch.Tensor
    query_relevance: torch.Tensor
    source_observation: int

    @property
    def token_count(self) -> int:
        return int(self.embeddings.shape[0])


@dataclass(frozen=True)
class EvidenceSelection:
    local_indices: tuple[torch.LongTensor, ...]
    target_tokens: int
    anchor_tokens: int
    retained_by_pass: dict[int, int]
    redundant_coarse_tokens: int

    @property
    def actual_tokens(self) -> int:
        return sum(int(value.numel()) for value in self.local_indices)


def query_token_embeddings(
    embedding_layer: Any,
    query_ids: torch.LongTensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Return normalized valid text-token embeddings in the visual projection space."""
    if query_ids.ndim == 2:
        if query_ids.shape[0] != 1:
            raise ValueError("HMVE Phase A supports one query per controller invocation")
        query_ids = query_ids[0]
    embedded = embedding_layer(query_ids)
    valid = (
        attention_mask[0].bool()
        if attention_mask is not None and attention_mask.ndim == 2
        else attention_mask.bool() if attention_mask is not None
        else torch.ones_like(query_ids, dtype=torch.bool)
    )
    if not bool(valid.any()):
        raise ValueError("HMVE requires at least one valid query token")
    return F.normalize(embedded[valid].float(), dim=-1, eps=1e-6)


def estimate_vision_transformer_tflops(vision_config: Any, grid_thw: torch.Tensor) -> float:
    """Estimate vision-transformer core FLOPs with full per-unit attention.

    Qwen may use cheaper window attention in some blocks, so the attention term
    is a conservative full-attention estimate. Patch embedding and the final
    merger are intentionally excluded and the telemetry names this estimate
    accordingly.
    """
    if grid_thw.ndim != 2 or grid_thw.shape[1] != 3:
        raise ValueError("vision grids must have shape [observations, 3]")
    depth = int(vision_config.depth)
    hidden = int(vision_config.hidden_size)
    intermediate = int(vision_config.intermediate_size)
    grids = grid_thw.long()
    tokens = grids.prod(dim=-1)
    token_count = int(tokens.sum().item())
    spatial_tokens = grids[:, 1] * grids[:, 2]
    attention_pairs = int((grids[:, 0] * spatial_tokens * spatial_tokens).sum().item())
    projection_and_mlp = 8 * token_count * hidden**2 + 6 * token_count * hidden * intermediate
    attention = 4 * attention_pairs * hidden
    return depth * (projection_and_mlp + attention) / 1e12


def unit_times(metadata: Any, temporal_units: int, temporal_patch_size: int = 2) -> list[float]:
    """Map projected temporal units to absolute source-video seconds."""
    fps = float(metadata.fps if hasattr(metadata, "fps") else metadata["fps"])
    indices = list(
        metadata.frames_indices
        if hasattr(metadata, "frames_indices") else metadata["frames_indices"]
    )
    if fps <= 0 or not indices:
        raise ValueError("HMVE video metadata requires FPS and absolute frame indices")
    needed = temporal_units * temporal_patch_size
    indices.extend([indices[-1]] * max(0, needed - len(indices)))
    return [
        0.5 * (indices[offset] + indices[offset + temporal_patch_size - 1]) / fps
        for offset in range(0, needed, temporal_patch_size)
    ]


def _merge_corridors(
    corridors: Sequence[ObservationCorridor], duration: float,
) -> tuple[ObservationCorridor, ...]:
    if not corridors:
        return ()
    ordered = sorted(corridors, key=lambda value: (value.start, value.end, -value.score))
    merged: list[ObservationCorridor] = []
    for value in ordered:
        if merged and value.start <= merged[-1].end:
            previous = merged[-1]
            preferred = value if value.score > previous.score else previous
            merged[-1] = ObservationCorridor(
                previous.start,
                min(duration, max(previous.end, value.end)),
                max(previous.score, value.score),
                preferred.center,
            )
        else:
            merged.append(value)
    return tuple(merged)


def propose_corridors(
    times: torch.Tensor,
    relevance: torch.Tensor,
    duration: float,
    *,
    maximum_corridors: int,
    minimum_seconds: float,
    margin_seconds: float,
    nms_seconds: float,
    minimum_total_seconds: float = 0.0,
) -> tuple[ObservationCorridor, ...]:
    """Select several deterministic, disjoint query-relevant temporal corridors."""
    if duration <= 0 or times.ndim != 1 or relevance.shape != times.shape or times.numel() == 0:
        raise ValueError("invalid HMVE corridor inputs")
    times = times.float().clamp(0, duration)
    relevance = relevance.float()
    flat = bool(torch.allclose(relevance, relevance[:1], rtol=0.0, atol=1e-8))
    if flat:
        count = min(maximum_corridors, int(times.numel()))
        positions = torch.linspace(0, times.numel() - 1, count).round().long().tolist()
    else:
        order = torch.argsort(relevance, descending=True, stable=True).tolist()
        positions = []
        for index in order:
            center = float(times[index].item())
            if all(abs(center - float(times[previous].item())) >= nms_seconds for previous in positions):
                positions.append(index)
                if len(positions) == maximum_corridors:
                    break
    if not positions:
        positions = [int(torch.argmax(relevance).item())]

    base_width = minimum_seconds + 2.0 * margin_seconds
    required = min(duration, max(minimum_total_seconds, base_width * len(positions)))
    width = max(base_width, required / len(positions))
    corridors = []
    for index in positions:
        center = float(times[index].item())
        start = max(0.0, center - width / 2.0)
        end = min(duration, start + width)
        start = max(0.0, end - width)
        corridors.append(ObservationCorridor(
            start=start,
            end=end,
            score=float(relevance[index].item()),
            center=center,
        ))
    return _merge_corridors(corridors, duration)


def _prototype_index(unit: EvidenceUnit) -> int:
    normalized = F.normalize(unit.embeddings.float(), dim=-1, eps=1e-6)
    summary = F.normalize(normalized.mean(dim=0), dim=0, eps=1e-6)
    return int(torch.matmul(normalized, summary).argmax().item())


def _spatial_cell(unit: EvidenceUnit, index: int) -> tuple[float, float]:
    row, column = divmod(index, unit.grid_width)
    return (
        (row + 0.5) / max(unit.grid_height, 1),
        (column + 0.5) / max(unit.grid_width, 1),
    )


def select_evidence(
    units: Sequence[EvidenceUnit],
    target_tokens: int,
    *,
    deduplication_similarity: float = 0.98,
) -> EvidenceSelection:
    """Preserve scout anchors, replace redundant coarse detail, and pack exactly."""
    if not units or target_tokens <= 0:
        raise ValueError("HMVE requires evidence units and a positive final budget")
    anchors: set[tuple[int, int]] = {
        (unit_index, _prototype_index(unit))
        for unit_index, unit in enumerate(units) if unit.pass_id == 0
    }
    if target_tokens < len(anchors):
        raise ValueError("HMVE final budget cannot preserve every scout anchor")
    total_created = sum(unit.token_count for unit in units)
    if target_tokens > total_created:
        raise ValueError(
            f"HMVE created {total_created} evidence tokens but requires {target_tokens}"
        )

    redundant: set[tuple[int, int]] = set()
    detailed = [(i, unit) for i, unit in enumerate(units) if unit.pass_id > 0]
    coarse = [(i, unit) for i, unit in enumerate(units) if unit.pass_id == 0]
    for coarse_index, coarse_unit in coarse:
        for token_index in range(coarse_unit.token_count):
            key = (coarse_index, token_index)
            if key in anchors:
                continue
            coarse_cell = _spatial_cell(coarse_unit, token_index)
            coarse_embedding = F.normalize(
                coarse_unit.embeddings[token_index].float(), dim=0, eps=1e-6,
            )
            for _, detailed_unit in detailed:
                if abs(detailed_unit.absolute_time - coarse_unit.absolute_time) > 0.51:
                    continue
                distances = []
                for detailed_index in range(detailed_unit.token_count):
                    cell = _spatial_cell(detailed_unit, detailed_index)
                    distances.append((
                        (cell[0] - coarse_cell[0]) ** 2 + (cell[1] - coarse_cell[1]) ** 2,
                        detailed_index,
                    ))
                if not distances:
                    continue
                _, nearest = min(distances)
                detailed_embedding = F.normalize(
                    detailed_unit.embeddings[nearest].float(), dim=0, eps=1e-6,
                )
                if float(torch.dot(coarse_embedding, detailed_embedding).item()) >= deduplication_similarity:
                    redundant.add(key)
                    break

    selected = set(anchors)
    candidates = []
    for unit_index, unit in enumerate(units):
        ranked = percentile_rank(unit.query_relevance.float())
        for token_index in range(unit.token_count):
            key = (unit_index, token_index)
            if key in selected or key in redundant:
                continue
            candidates.append((
                -float(ranked[token_index].item()),
                -unit.pass_id,
                unit.absolute_time,
                token_index,
                unit_index,
            ))
    candidates.sort()
    for _, _, _, token_index, unit_index in candidates[:target_tokens - len(selected)]:
        selected.add((unit_index, token_index))

    if len(selected) < target_tokens:
        recovery = []
        for unit_index, unit in enumerate(units):
            for token_index in range(unit.token_count):
                key = (unit_index, token_index)
                if key not in selected:
                    recovery.append((unit.absolute_time, unit.pass_id, token_index, unit_index))
        recovery.sort()
        for _, _, token_index, unit_index in recovery[:target_tokens - len(selected)]:
            selected.add((unit_index, token_index))

    local = []
    retained_by_pass: dict[int, int] = {}
    for unit_index, unit in enumerate(units):
        indices = sorted(token for owner, token in selected if owner == unit_index)
        tensor = torch.tensor(indices, device=unit.embeddings.device, dtype=torch.long)
        local.append(tensor)
        retained_by_pass[unit.pass_id] = retained_by_pass.get(unit.pass_id, 0) + len(indices)
    result = EvidenceSelection(
        local_indices=tuple(local),
        target_tokens=target_tokens,
        anchor_tokens=len(anchors),
        retained_by_pass=retained_by_pass,
        redundant_coarse_tokens=len(redundant),
    )
    if result.actual_tokens != target_tokens:
        raise RuntimeError(
            f"HMVE budget error: retained {result.actual_tokens}, expected {target_tokens}"
        )
    return result


def evidence_unit(
    *,
    pass_id: int,
    absolute_time: float,
    grid_height: int,
    grid_width: int,
    source_height: int,
    source_width: int,
    embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    source_observation: int,
) -> EvidenceUnit:
    relevance = patch_query_relevance(
        embeddings.reshape(1, embeddings.shape[0], embeddings.shape[1]), query_embeddings,
    ).reshape(-1)
    return EvidenceUnit(
        pass_id=pass_id,
        absolute_time=absolute_time,
        grid_height=grid_height,
        grid_width=grid_width,
        source_height=source_height,
        source_width=source_width,
        embeddings=embeddings,
        query_relevance=relevance,
        source_observation=source_observation,
    )

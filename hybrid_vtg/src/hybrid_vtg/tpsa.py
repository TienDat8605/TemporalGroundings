"""Training-free timeline-preserving spatial allocation for video grounding."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import torch
import torch.nn.functional as F

from .config import SpatialAllocatorConfig


TPSA_POLICIES = ("tpsa_query", "tpsa_motion", "tpsa_boundary")


def _synchronize(tensor: torch.Tensor) -> None:
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)


@dataclass(frozen=True)
class BoundaryBand:
    role: str
    center_frame: int
    center_seconds: float
    start_frame: int
    end_frame: int
    start_seconds: float
    end_seconds: float
    score: float


@dataclass
class SpatialAllocation:
    """Indices and audit data for one dense ``[T,P,D]`` video."""

    keep_indices: torch.LongTensor
    prototype_indices: torch.LongTensor
    start_indices: torch.LongTensor
    end_indices: torch.LongTensor
    adaptive_indices: torch.LongTensor
    query_core_indices: torch.LongTensor
    refinement_indices: torch.LongTensor
    frame_allocation: torch.LongTensor
    query_relevance: torch.Tensor
    novelty: torch.Tensor
    start_evidence: torch.Tensor
    end_evidence: torch.Tensor
    start_bands: tuple[BoundaryBand, ...]
    end_bands: tuple[BoundaryBand, ...]
    target_tokens: int
    original_tokens: int
    latencies: dict[str, float]

    @property
    def actual_tokens(self) -> int:
        return int(self.keep_indices.numel())

    def stats(self) -> dict[str, Any]:
        return {
            "target_retained_tokens": self.target_tokens,
            "actual_retained_tokens": self.actual_tokens,
            "original_visual_tokens": self.original_tokens,
            "effective_retention_ratio": self.actual_tokens / max(self.original_tokens, 1),
            "per_frame_allocation": [int(value) for value in self.frame_allocation.tolist()],
            "prototype_tokens": int(self.prototype_indices.numel()),
            "start_boundary_tokens": int(self.start_indices.numel()),
            "end_boundary_tokens": int(self.end_indices.numel()),
            "adaptive_tokens": int(self.adaptive_indices.numel()),
            "protected_query_tokens": int(self.query_core_indices.numel()),
            "auxiliary_refinement_tokens": int(self.refinement_indices.numel()),
            "start_boundary_bands": [asdict(value) for value in self.start_bands],
            "end_boundary_bands": [asdict(value) for value in self.end_bands],
            **self.latencies,
        }


def percentile_rank(values: torch.Tensor) -> torch.Tensor:
    """Return deterministic within-video ordinal percentiles; flat input maps to zero."""
    flat = values.float().reshape(-1)
    if flat.numel() <= 1 or bool(torch.allclose(flat, flat[:1], rtol=0.0, atol=1e-8)):
        return torch.zeros_like(values, dtype=torch.float32)
    _, inverse, counts = torch.unique(
        flat, sorted=True, return_inverse=True, return_counts=True,
    )
    counts_f = counts.to(flat.dtype)
    before = torch.cumsum(counts_f, dim=0) - counts_f
    average_rank = before + 0.5 * (counts_f - 1.0)
    tied = average_rank[inverse]
    tied = (tied - tied.min()) / (tied.max() - tied.min()).clamp_min(1e-8)
    return tied.reshape(values.shape)


def _rank_mean(*values: torch.Tensor) -> torch.Tensor:
    return torch.stack([percentile_rank(value) for value in values], dim=0).mean(dim=0)


def effective_temporal_fps(decoded_fps: float, temporal_patch_size: int) -> float:
    """Convert decoded-frame FPS to the post-encoder temporal-tubelet rate."""
    if decoded_fps <= 0 or temporal_patch_size <= 0:
        raise ValueError("decoded FPS and temporal patch size must be positive")
    return decoded_fps / temporal_patch_size


def patch_query_relevance(
    visual_tokens: torch.Tensor,
    query_tokens: torch.Tensor,
    query_attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Maximum cosine similarity to any valid query token for every visual patch."""
    if query_tokens.ndim == 1:
        query_tokens = query_tokens.unsqueeze(0)
    if query_tokens.ndim != 2 or query_tokens.shape[-1] != visual_tokens.shape[-1]:
        raise ValueError(
            "query embeddings must have shape [L,D] matching visual token dimension "
            f"{visual_tokens.shape[-1]}, got {tuple(query_tokens.shape)}"
        )
    if query_attention_mask is not None:
        valid = query_attention_mask.reshape(-1).bool()
        if valid.numel() != query_tokens.shape[0]:
            raise ValueError("query attention mask length does not match query embeddings")
        query_tokens = query_tokens[valid]
    if query_tokens.shape[0] == 0:
        raise ValueError("at least one valid query-token embedding is required")
    visual = F.normalize(visual_tokens.float(), dim=-1, eps=1e-6)
    query = F.normalize(query_tokens.float().to(visual.device), dim=-1, eps=1e-6)
    return torch.matmul(visual, query.transpose(0, 1)).max(dim=-1).values


def frame_relevance_curve(patch_relevance: torch.Tensor, top_fraction: float = 0.10) -> torch.Tensor:
    count = max(1, min(patch_relevance.shape[1], int(round(patch_relevance.shape[1] * top_fraction))))
    return torch.topk(patch_relevance, k=count, dim=1).values.mean(dim=1)


def _transition_from_source(
    source: torch.Tensor,
    destination: torch.Tensor,
    height: int,
    width: int,
    radius: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Camera-compensated source-patch transition evidence for one frame pair."""
    source_n = F.normalize(source.float(), dim=-1, eps=1e-6)
    destination_n = F.normalize(destination.float(), dim=-1, eps=1e-6)
    source_grid = source_n.reshape(height, width, -1)
    destination_grid = destination_n.reshape(height, width, -1)
    offsets = [
        (row_offset, column_offset)
        for row_offset in range(-radius, radius + 1)
        for column_offset in range(-radius, radius + 1)
    ]
    # Nearest offsets come first, so argmax deterministically prefers the
    # zero-displacement correspondence when cosine similarities tie.
    offsets.sort(key=lambda value: (value[0] ** 2 + value[1] ** 2, value[0], value[1]))
    similarities = source_n.new_full((len(offsets), height, width), -torch.inf)
    for offset_index, (row_offset, column_offset) in enumerate(offsets):
        source_row_start, source_row_end = max(0, -row_offset), min(height, height - row_offset)
        source_col_start, source_col_end = max(0, -column_offset), min(width, width - column_offset)
        destination_row_start = source_row_start + row_offset
        destination_row_end = source_row_end + row_offset
        destination_col_start = source_col_start + column_offset
        destination_col_end = source_col_end + column_offset
        similarities[
            offset_index, source_row_start:source_row_end, source_col_start:source_col_end
        ] = (
            source_grid[source_row_start:source_row_end, source_col_start:source_col_end]
            * destination_grid[
                destination_row_start:destination_row_end,
                destination_col_start:destination_col_end,
            ]
        ).sum(dim=-1)
    best_similarity, best_offset = similarities.max(dim=0)
    offset_values = source_n.new_tensor(offsets)
    displacement = offset_values[best_offset.reshape(-1)]
    evidence_appearance = (1.0 - best_similarity.reshape(-1)).clamp_min(0.0)
    evidence_appearance = torch.where(
        evidence_appearance <= 1e-5,
        torch.zeros_like(evidence_appearance),
        evidence_appearance,
    )
    camera_motion = displacement.median(dim=0).values
    local_motion = (displacement - camera_motion).norm(dim=-1)
    return evidence_appearance, local_motion


def feature_transition_maps(
    visual_tokens: torch.Tensor,
    height: int,
    width: int,
    radius: int = 2,
) -> torch.Tensor:
    """Bidirectional feature-native novelty aligned to each frame's patch grid."""
    frames, patches, _ = visual_tokens.shape
    if height * width != patches:
        raise ValueError(
            f"merged grid mismatch: H*W={height}*{width}={height * width}, but P={patches}; "
            "pass Qwen's post-merge grid dimensions explicitly"
        )
    if frames == 1:
        return visual_tokens.new_zeros((1, patches), dtype=torch.float32)
    incoming: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * frames
    outgoing: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * frames
    for frame in range(frames - 1):
        outgoing[frame] = _transition_from_source(
            visual_tokens[frame], visual_tokens[frame + 1], height, width, radius,
        )
        incoming[frame + 1] = _transition_from_source(
            visual_tokens[frame + 1], visual_tokens[frame], height, width, radius,
        )
    appearance_output = []
    motion_output = []
    for frame in range(frames):
        if incoming[frame] is None:
            value = outgoing[frame]
        elif outgoing[frame] is None:
            value = incoming[frame]
        else:
            value = (
                0.5 * (incoming[frame][0] + outgoing[frame][0]),
                0.5 * (incoming[frame][1] + outgoing[frame][1]),
            )
        if value is not None:
            appearance_output.append(value[0])
            motion_output.append(value[1])
    appearance = torch.stack(appearance_output, dim=0)
    local_motion = torch.stack(motion_output, dim=0)
    appearance_rank = percentile_rank(appearance)
    motion_rank = percentile_rank(local_motion)
    appearance_rank[appearance <= 1e-5] = 0
    motion_rank[local_motion <= 1e-5] = 0
    return 0.5 * (appearance_rank + motion_rank)


def directional_boundary_evidence(
    relevance: torch.Tensor,
    novelty: torch.Tensor,
    fps: float,
    window_seconds: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Positive relevance rises are starts; positive falls are ends."""
    if fps <= 0:
        raise ValueError("FPS must be positive")
    frames = int(relevance.numel())
    width = max(1, int(round(window_seconds * fps)))
    start = relevance.new_zeros(frames, dtype=torch.float32)
    end = relevance.new_zeros(frames, dtype=torch.float32)
    novelty_rank = percentile_rank(novelty)
    for frame in range(1, frames):
        before = relevance[max(0, frame - width):frame]
        after = relevance[frame:min(frames, frame + width)]
        if before.numel() == 0 or after.numel() == 0:
            continue
        change = after.float().mean() - before.float().mean()
        start[frame] = change.clamp_min(0.0) * novelty_rank[frame]
        end[frame] = (-change).clamp_min(0.0) * novelty_rank[frame]
    return start, end


def select_boundary_bands(
    evidence: torch.Tensor,
    role: str,
    fps: float,
    nms_seconds: float,
    expansion_seconds: float,
    maximum_bands: int,
) -> tuple[BoundaryBand, ...]:
    frames = int(evidence.numel())
    positive = evidence[evidence > 0].float()
    if positive.numel() == 0:
        return ()
    if positive.numel() > 1 and bool(torch.allclose(positive, positive[:1], rtol=0.0, atol=1e-8)):
        return ()
    confidence_floor = float(positive.min().item())
    separation = max(1, int(round(nms_seconds * fps)))
    expansion = max(0, int(round(expansion_seconds * fps)))
    order = torch.argsort(evidence, descending=True, stable=True).tolist()
    centers: list[int] = []
    for center in order:
        if float(evidence[center].item()) < confidence_floor:
            break
        if all(abs(center - previous) >= separation for previous in centers):
            centers.append(center)
            if len(centers) == maximum_bands:
                break
    centers.sort()
    return tuple(
        BoundaryBand(
            role=role,
            center_frame=center,
            center_seconds=center / fps,
            start_frame=max(0, center - expansion),
            end_frame=min(frames - 1, center + expansion),
            start_seconds=max(0, center - expansion) / fps,
            end_seconds=min(frames - 1, center + expansion) / fps,
            score=float(evidence[center].item()),
        )
        for center in centers
    )


def _allocate_capped(weights: torch.Tensor, total: int, capacity: torch.Tensor) -> torch.LongTensor:
    allocation = torch.zeros_like(capacity, dtype=torch.long)
    remaining = min(int(total), int(capacity.sum().item()))
    while remaining > 0:
        slack = capacity - allocation
        active = slack > 0
        if not bool(active.any()):
            break
        active_weights = weights.float().clone().clamp_min(0)
        active_weights[~active] = 0
        if bool(torch.allclose(active_weights, active_weights[:1], rtol=0.0, atol=1e-8)) or float(active_weights.sum()) <= 1e-8:
            active_weights = active.float()
        ideal = remaining * active_weights / active_weights.sum()
        addition = torch.minimum(torch.floor(ideal).long(), slack)
        if int(addition.sum().item()) == 0:
            candidates = torch.nonzero(active, as_tuple=False).flatten().tolist()
            candidates.sort(key=lambda index: (-float(ideal[index].item()), index))
            for index in candidates[:remaining]:
                addition[index] += 1
        allocation += addition
        consumed = int(addition.sum().item())
        remaining -= consumed
        if consumed == 0:
            break
    return allocation


def _prototype_indices(tokens: torch.Tensor) -> torch.LongTensor:
    normalized = F.normalize(tokens.float(), dim=-1, eps=1e-6)
    summaries = F.normalize(normalized.mean(dim=1), dim=-1, eps=1e-6)
    local = torch.matmul(normalized, summaries.unsqueeze(-1)).squeeze(-1).argmax(dim=1)
    frames = torch.arange(tokens.shape[0], device=tokens.device, dtype=torch.long)
    return frames * tokens.shape[1] + local


def _band_candidate_score(
    bands: tuple[BoundaryBand, ...],
    patch_score: torch.Tensor,
) -> torch.Tensor:
    if not bands:
        return torch.zeros_like(patch_score)
    frame_score = patch_score.new_zeros(patch_score.shape[0])
    eligible = torch.zeros(patch_score.shape[0], device=patch_score.device, dtype=torch.bool)
    for band in bands:
        eligible[band.start_frame:band.end_frame + 1] = True
        frame_score[band.start_frame:band.end_frame + 1] = torch.maximum(
            frame_score[band.start_frame:band.end_frame + 1],
            frame_score.new_full((band.end_frame - band.start_frame + 1,), band.score),
        )
    output = _rank_mean(patch_score, frame_score[:, None].expand_as(patch_score))
    output[~eligible] = 0
    return output


def _take_global(
    score: torch.Tensor,
    quota: int,
    selected: torch.BoolTensor,
    candidate_mask: torch.BoolTensor | None = None,
) -> torch.LongTensor:
    flat_score = score.reshape(-1).clone()
    eligible = (~selected) & torch.isfinite(flat_score) & flat_score.gt(0)
    if candidate_mask is not None:
        eligible &= candidate_mask.reshape(-1)
    candidates = torch.nonzero(eligible, as_tuple=False).flatten()
    if quota <= 0 or candidates.numel() == 0:
        return candidates[:0]
    order = torch.argsort(flat_score[candidates], descending=True, stable=True)
    chosen = candidates[order[:min(quota, int(candidates.numel()))]]
    selected[chosen] = True
    return chosen.sort().values


def _take_boundary_sides(
    bands: tuple[BoundaryBand, ...],
    score: torch.Tensor,
    quota: int,
    selected: torch.BoolTensor,
) -> torch.LongTensor:
    if quota <= 0 or not bands:
        return selected.new_empty((0,), dtype=torch.long)
    before = torch.zeros_like(score, dtype=torch.bool)
    after = torch.zeros_like(score, dtype=torch.bool)
    for band in bands:
        before[band.start_frame:band.center_frame] = True
        after[band.center_frame:band.end_frame + 1] = True
    first_quota = quota // 2
    before_indices = _take_global(score, first_quota, selected, before)
    after_indices = _take_global(score, quota - int(before_indices.numel()), selected, after)
    chosen = torch.cat((before_indices, after_indices))
    if chosen.numel() < quota:
        eligible = before | after
        remainder = _take_global(score, quota - int(chosen.numel()), selected, eligible)
        chosen = torch.cat((chosen, remainder))
    return chosen.sort().values


def _mmr_select(
    tokens: torch.Tensor,
    scores: torch.Tensor,
    count: int,
    unavailable: torch.BoolTensor,
    mmr_lambda: float,
) -> torch.LongTensor:
    candidates = torch.nonzero(~unavailable, as_tuple=False).flatten()
    count = min(count, int(candidates.numel()))
    if count <= 0:
        return candidates[:0]
    normalized = F.normalize(tokens.float(), dim=-1, eps=1e-6)
    relevance = percentile_rank(scores.float())
    selected: list[int] = []
    redundancy = relevance.new_zeros(relevance.shape)
    for _ in range(count):
        candidate_scores = mmr_lambda * relevance - (1.0 - mmr_lambda) * redundancy
        candidate_scores[unavailable] = -torch.inf
        if selected:
            candidate_scores[torch.tensor(selected, device=tokens.device)] = -torch.inf
        choice = int(torch.argmax(candidate_scores).item())
        selected.append(choice)
        redundancy = torch.maximum(redundancy, torch.matmul(normalized, normalized[choice]))
    return torch.tensor(selected, device=tokens.device, dtype=torch.long).sort().values


def _query_gated_score(
    query_score: torch.Tensor,
    auxiliary_score: torch.Tensor,
    bonus_fraction: float,
) -> torch.Tensor:
    query_rank = percentile_rank(query_score)
    if auxiliary_score.numel() == 0 or not bool((auxiliary_score > 0).any()):
        return query_rank
    auxiliary_rank = percentile_rank(auxiliary_score)
    auxiliary_rank[auxiliary_score <= 0] = 0
    return query_rank + bonus_fraction * query_rank * auxiliary_rank


def _select_per_frame(
    tokens: torch.Tensor,
    scores: torch.Tensor,
    frame_counts: torch.LongTensor,
    selected: torch.BoolTensor,
    mmr_lambda: float,
) -> torch.LongTensor:
    frames, patches, _ = tokens.shape
    parts = []
    for frame in range(frames):
        unavailable = selected[frame * patches:(frame + 1) * patches].clone()
        local = _mmr_select(
            tokens[frame], scores[frame], int(frame_counts[frame].item()),
            unavailable, mmr_lambda,
        )
        if local.numel():
            global_indices = local + frame * patches
            selected[global_indices] = True
            parts.append(global_indices)
    return torch.cat(parts).sort().values if parts else selected.new_empty((0,), dtype=torch.long)


class TimelinePreservingSpatialAllocator:
    """Pure-PyTorch TPSA selector; it never creates replacement visual tokens."""

    def __init__(self, config: SpatialAllocatorConfig) -> None:
        if config.spatial_policy not in TPSA_POLICIES:
            raise ValueError(f"TPSA requires one of {TPSA_POLICIES}, got {config.spatial_policy!r}")
        self.config = config

    def __call__(
        self,
        visual_tokens: torch.Tensor,
        merged_grid_height: int,
        merged_grid_width: int,
        query_token_embeddings: torch.Tensor,
        fps: float,
        retention_ratio: float | None = None,
        query_attention_mask: torch.Tensor | None = None,
    ) -> SpatialAllocation:
        if visual_tokens.ndim != 3:
            raise ValueError(f"visual tokens must have shape [T,P,D], got {tuple(visual_tokens.shape)}")
        frames, patches, _ = visual_tokens.shape
        if frames <= 0 or patches <= 0:
            raise ValueError("visual tokens must contain at least one frame and patch")
        if merged_grid_height * merged_grid_width != patches:
            raise ValueError(
                f"merged grid mismatch: H*W={merged_grid_height}*{merged_grid_width}="
                f"{merged_grid_height * merged_grid_width}, but P={patches}; pass Qwen's post-merge "
                "(H,W) metadata without reshaping or changing the token budget"
            )
        ratio = self.config.retention_ratio if retention_ratio is None else retention_ratio
        if not 0 <= ratio <= 1:
            raise ValueError("retention ratio must be in [0, 1]")
        original = frames * patches
        target = min(original, max(int(round(ratio * original)), frames))
        _synchronize(visual_tokens)
        overall_started = perf_counter()

        query_started = perf_counter()
        patch_relevance = patch_query_relevance(
            visual_tokens, query_token_embeddings, query_attention_mask,
        )
        frame_relevance = frame_relevance_curve(
            patch_relevance, self.config.relevance_top_fraction,
        )
        prototype = _prototype_indices(visual_tokens)
        _synchronize(visual_tokens)
        query_seconds = perf_counter() - query_started

        motion_started = perf_counter()
        if self.config.spatial_policy in {"tpsa_motion", "tpsa_boundary"}:
            novelty_map = feature_transition_maps(
                visual_tokens,
                merged_grid_height,
                merged_grid_width,
                self.config.motion_neighborhood_radius,
            )
            novelty = frame_relevance_curve(
                novelty_map,
                self.config.relevance_top_fraction,
            )
        else:
            novelty_map = torch.zeros_like(patch_relevance)
            novelty = torch.zeros_like(frame_relevance)
        _synchronize(visual_tokens)
        motion_seconds = perf_counter() - motion_started

        boundary_started = perf_counter()
        if self.config.spatial_policy == "tpsa_boundary":
            start_evidence, end_evidence = directional_boundary_evidence(
                frame_relevance,
                novelty,
                fps,
                self.config.boundary_window_seconds,
            )
            start_bands = select_boundary_bands(
                start_evidence, "start", fps, self.config.boundary_nms_seconds,
                self.config.boundary_expansion_seconds, self.config.maximum_boundary_bands,
            )
            end_bands = select_boundary_bands(
                end_evidence, "end", fps, self.config.boundary_nms_seconds,
                self.config.boundary_expansion_seconds, self.config.maximum_boundary_bands,
            )
        else:
            start_evidence = torch.zeros_like(frame_relevance)
            end_evidence = torch.zeros_like(frame_relevance)
            start_bands, end_bands = (), ()
        _synchronize(visual_tokens)
        boundary_seconds = perf_counter() - boundary_started

        selection_started = perf_counter()
        selected = torch.zeros(original, device=visual_tokens.device, dtype=torch.bool)
        selected[prototype] = True
        remaining = target - frames
        boundary_quota = (
            int(remaining * self.config.boundary_quota_fraction)
            if self.config.spatial_policy == "tpsa_boundary" else 0
        )
        query_patch_score = percentile_rank(patch_relevance)
        patch_selection_score = (
            query_patch_score
            if self.config.spatial_policy == "tpsa_query"
            else _query_gated_score(
                patch_relevance,
                novelty_map,
                self.config.motion_bonus_fraction,
            )
        )

        start_score = _band_candidate_score(start_bands, patch_selection_score)
        start = _take_boundary_sides(start_bands, start_score, boundary_quota, selected)
        end_score = _band_candidate_score(end_bands, patch_selection_score)
        end = _take_boundary_sides(end_bands, end_score, boundary_quota, selected)

        query_core = prototype[:0]
        if self.config.spatial_policy in {"tpsa_motion", "tpsa_boundary"}:
            query_is_flat = bool(torch.allclose(
                frame_relevance.float(), frame_relevance[:1].float(), rtol=0.0, atol=1e-8,
            ))
            query_core_total = (
                0 if query_is_flat
                else int(round(remaining * self.config.query_core_fraction))
            )
            query_capacity = (~selected.reshape(frames, patches)).sum(dim=1).long()
            query_frame_extra = _allocate_capped(
                percentile_rank(frame_relevance),
                query_core_total,
                query_capacity,
            )
            query_core = _select_per_frame(
                visual_tokens,
                query_patch_score,
                query_frame_extra,
                selected,
                self.config.mmr_lambda,
            )

        adaptive_total = target - int(selected.sum().item())
        capacity = (~selected.reshape(frames, patches)).sum(dim=1).long()
        if self.config.spatial_policy == "tpsa_query":
            frame_weight = percentile_rank(frame_relevance)
        else:
            frame_weight = _query_gated_score(
                frame_relevance,
                novelty,
                self.config.motion_bonus_fraction,
            )
        frame_extra = _allocate_capped(frame_weight, adaptive_total, capacity)
        refinement = _select_per_frame(
            visual_tokens,
            patch_selection_score,
            frame_extra,
            selected,
            self.config.mmr_lambda,
        )
        adaptive = torch.cat((query_core, refinement)).sort().values
        keep = torch.nonzero(selected, as_tuple=False).flatten().sort().values
        if keep.numel() != target:
            raise RuntimeError(f"TPSA budget error: selected {keep.numel()} tokens, expected {target}")
        frame_allocation = selected.reshape(frames, patches).sum(dim=1).long()
        if bool((frame_allocation < 1).any()):
            raise RuntimeError("TPSA invariant violated: a frame lost all real visual tokens")
        _synchronize(visual_tokens)
        selection_seconds = perf_counter() - selection_started
        return SpatialAllocation(
            keep_indices=keep,
            prototype_indices=prototype.sort().values,
            start_indices=start,
            end_indices=end,
            adaptive_indices=adaptive,
            query_core_indices=query_core,
            refinement_indices=(
                refinement if self.config.spatial_policy != "tpsa_query" else prototype[:0]
            ),
            frame_allocation=frame_allocation,
            query_relevance=frame_relevance,
            novelty=novelty,
            start_evidence=start_evidence,
            end_evidence=end_evidence,
            start_bands=start_bands,
            end_bands=end_bands,
            target_tokens=target,
            original_tokens=original,
            latencies={
                "query_allocation_seconds": query_seconds,
                "motion_allocation_seconds": motion_seconds,
                "boundary_allocation_seconds": boundary_seconds,
                "selection_seconds": selection_seconds,
                "allocator_seconds": perf_counter() - overall_started,
            },
        )

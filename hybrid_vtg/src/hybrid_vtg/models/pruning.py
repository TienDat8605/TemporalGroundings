"""Independent visual pruning policies for the Qwen evidence backend.

``mage_cell_plan`` is a clean-room, training-free adaptation of Mage-VL's
anchor/update principle.  It consumes motion/residual importance maps but does
not copy Mage-ViT or claim compatibility with its trained codec tokenizer.

``semvid_select`` adapts the role-aware selector from the official SemVID
Qwen3-VL implementation (Apache-2.0, upstream revision recorded in NOTICE.md).
It operates on already encoded evidence immediately before language prefill.
The implementation is modified to support a variable number of tokens per
timestamp, which is required when Mage-style encoder pruning is also enabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..contracts import TemporalEvidence


@dataclass(frozen=True)
class MageCellPlan:
    """Sparse complete-merger-cell layout for one processed video."""

    patch_indices: tuple[int, ...]
    cells_per_time: tuple[int, ...]
    selected_cells: tuple[tuple[int, int, int], ...]
    dense_cells: int
    target_cells: int


def _robust_unit(values: np.ndarray) -> np.ndarray:
    values = np.maximum(values.astype(np.float32, copy=False), 0.0)
    positive = values[values > 0]
    if positive.size == 0:
        return np.zeros_like(values)
    scale = float(np.percentile(positive, 95))
    if scale <= 1e-8:
        return np.zeros_like(values)
    return np.clip(values / scale, 0.0, 1.0)


def motion_residual_importance(
    frames: Sequence[Any],
    *,
    temporal_units: int,
    cell_height: int,
    cell_width: int,
) -> np.ndarray:
    """Return camera-compensated motion/residual importance per merger cell.

    This is the deterministic decoded-frame fallback for Mage-style selection.
    Dense optical flow approximates codec motion vectors; motion-compensated
    luminance error approximates codec residual energy.  Keeping this provider
    separate allows true exported codec maps to replace it later without
    changing the sparse-encoder contract.
    """
    if temporal_units <= 0 or cell_height <= 0 or cell_width <= 0:
        raise ValueError("importance dimensions must be positive")
    if not frames:
        raise ValueError("at least one frame is required")

    import cv2

    analysis_height = max(64, cell_height * 12)
    analysis_width = max(64, cell_width * 12)
    grays = []
    for frame in frames:
        rgb = np.asarray(frame.convert("RGB") if hasattr(frame, "convert") else frame)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        grays.append(cv2.resize(gray, (analysis_width, analysis_height), interpolation=cv2.INTER_AREA))

    per_frame = [np.zeros((cell_height, cell_width), dtype=np.float32)]
    grid_x, grid_y = np.meshgrid(
        np.arange(analysis_width, dtype=np.float32),
        np.arange(analysis_height, dtype=np.float32),
    )
    for previous, current in zip(grays, grays[1:]):
        flow = cv2.calcOpticalFlowFarneback(
            previous,
            current,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )
        global_motion = np.median(flow.reshape(-1, 2), axis=0)
        local_motion = np.linalg.norm(flow - global_motion.reshape(1, 1, 2), axis=-1)
        # Farneback estimates previous->current flow. Sampling the previous frame
        # at current-flow is a stable, inexpensive backward-warp approximation.
        warped = cv2.remap(
            previous,
            grid_x - flow[..., 0],
            grid_y - flow[..., 1],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        residual = np.abs(current.astype(np.float32) - warped.astype(np.float32))
        local_motion = _robust_unit(local_motion)
        residual = _robust_unit(residual)
        novelty = 0.5 * local_motion + 0.5 * residual
        per_frame.append(cv2.resize(novelty, (cell_width, cell_height), interpolation=cv2.INTER_AREA))

    groups = np.array_split(np.arange(len(per_frame)), temporal_units)
    output = []
    for group in groups:
        if group.size:
            output.append(np.max(np.stack([per_frame[int(index)] for index in group]), axis=0))
        else:
            output.append(np.zeros((cell_height, cell_width), dtype=np.float32))
    return np.stack(output).astype(np.float32)


def mage_cell_plan(
    importance: np.ndarray,
    *,
    merge_size: int,
    retention_ratio: float,
    anchor_stride: int = 8,
    minimum_cells_per_time: int = 1,
) -> MageCellPlan:
    """Select complete Qwen merger cells using Mage-style anchors and updates."""
    if importance.ndim != 3:
        raise ValueError("importance must have shape (time, cell_height, cell_width)")
    if not 0 < retention_ratio <= 1:
        raise ValueError("encoder retention ratio must be in (0, 1]")
    if merge_size <= 0 or anchor_stride <= 0 or minimum_cells_per_time <= 0:
        raise ValueError("merge size, anchor stride, and minimum cell count must be positive")

    temporal, cell_height, cell_width = importance.shape
    cells_per_dense_time = cell_height * cell_width
    dense_cells = temporal * cells_per_dense_time
    target = min(dense_cells, max(temporal * minimum_cells_per_time, round(dense_cells * retention_ratio)))

    selected: list[set[int]] = [set() for _ in range(temporal)]
    # Every temporal unit gets a content-dependent floor rather than becoming empty.
    for time in range(temporal):
        ranked = sorted(
            range(cells_per_dense_time),
            key=lambda cell: (-float(importance[time].reshape(-1)[cell]), cell),
        )
        selected[time].update(ranked[:minimum_cells_per_time])

    used = sum(len(values) for values in selected)
    # Dense anchors are admitted only when the exact global budget can still be
    # respected. The first unit is always considered first, followed periodically.
    for time in range(0, temporal, anchor_stride):
        missing = cells_per_dense_time - len(selected[time])
        if used + missing <= target:
            selected[time].update(range(cells_per_dense_time))
            used += missing

    candidates = []
    for time in range(temporal):
        for cell in range(cells_per_dense_time):
            if cell not in selected[time]:
                candidates.append((-float(importance[time].reshape(-1)[cell]), time, cell))
    for _, time, cell in sorted(candidates):
        if used >= target:
            break
        selected[time].add(cell)
        used += 1

    patch_height = cell_height * merge_size
    patch_width = cell_width * merge_size
    patches_per_dense_time = patch_height * patch_width
    patch_indices: list[int] = []
    selected_cells: list[tuple[int, int, int]] = []
    for time, cells in enumerate(selected):
        for cell in sorted(cells):
            row, column = divmod(cell, cell_width)
            # Qwen's processor stores each complete merger cell contiguously.
            start = time * patches_per_dense_time + cell * merge_size * merge_size
            patch_indices.extend(range(start, start + merge_size * merge_size))
            selected_cells.append((time, row, column))

    return MageCellPlan(
        patch_indices=tuple(patch_indices),
        cells_per_time=tuple(len(values) for values in selected),
        selected_cells=tuple(selected_cells),
        dense_cells=dense_cells,
        target_cells=target,
    )


def _allocate_with_caps(weights, total: int, caps):
    """Deterministically allocate an integer total under per-bin capacities."""
    import torch

    output = torch.zeros_like(caps, dtype=torch.long)
    remaining = min(int(total), int(caps.sum().item()))
    while remaining > 0:
        slack = (caps - output).clamp_min(0)
        active = slack > 0
        if not bool(active.any()):
            break
        current = weights.float().clamp_min(0) * active.float()
        if float(current.sum()) <= 1e-8:
            current = active.float()
        ideal = current / current.sum() * remaining
        add = torch.minimum(torch.floor(ideal).long(), slack)
        if int(add.sum()) == 0:
            fractions = ideal - torch.floor(ideal)
            candidates = torch.nonzero(active, as_tuple=False).flatten().tolist()
            candidates.sort(key=lambda index: (-float(fractions[index]), -float(current[index]), index))
            add[candidates[0]] = 1
        output += add
        remaining = min(total - int(output.sum()), int((caps - output).clamp_min(0).sum()))
    return output


def _mmr_select(relevance, normalized_embeddings, count: int, coefficient: float):
    import torch

    count = min(max(int(count), 0), int(relevance.numel()))
    if count == 0:
        return torch.empty((0,), device=relevance.device, dtype=torch.long)
    selected: list[int] = []
    available = torch.ones(relevance.numel(), device=relevance.device, dtype=torch.bool)
    redundancy = torch.zeros_like(relevance.float())
    for _ in range(count):
        score = coefficient * relevance.float() - (1.0 - coefficient) * redundancy
        score = score.masked_fill(~available, -1e9)
        index = int(torch.argmax(score).item())
        selected.append(index)
        available[index] = False
        similarity = torch.matmul(normalized_embeddings, normalized_embeddings[index])
        redundancy = torch.maximum(redundancy, similarity.float())
    return torch.tensor(selected, device=relevance.device, dtype=torch.long)


def semvid_select(
    evidence: TemporalEvidence,
    query_embeddings,
    *,
    retention_ratio: float,
    dense_evidence_units: int | None = None,
    frame_weight_alpha: float = 0.7,
    object_ratio: float = 0.6,
    mmr_lambda: float = 0.9,
    motion_query_beta: float = 0.5,
    query_token_max: int = 50,
) -> TemporalEvidence:
    """Apply SemVID role-aware selection to encoded temporal evidence.

    Defaults match the official Qwen3-VL SemVID configuration. Unlike the
    upstream ``(T,P,D)`` implementation, groups may have different capacities.
    The final target remains relative to the original dense encoder output when
    ``dense_evidence_units`` is supplied by Mage pruning.
    """
    import torch
    import torch.nn.functional as functional

    if not 0 < retention_ratio <= 1:
        raise ValueError("SemVID retention ratio must be in (0, 1]")
    if not evidence.timestamps:
        return evidence
    groups: list[list[int]] = []
    for index, timestamp in enumerate(evidence.timestamps):
        if groups and abs(evidence.timestamps[groups[-1][0]] - timestamp) < 1e-6:
            groups[-1].append(index)
        else:
            groups.append([index])

    device = evidence.embeddings.device
    embeddings = evidence.embeddings.float()
    query = query_embeddings.float().to(device)
    if query.ndim == 1:
        query = query.unsqueeze(0)
    if query.ndim != 2 or query.shape[1] != embeddings.shape[1]:
        raise ValueError("query embeddings must have shape (tokens, embedding_dim)")
    query_is_tokenwise = 1 < query.shape[0] <= query_token_max
    if query.shape[0] > query_token_max:
        query = query.mean(dim=0, keepdim=True)
    query = functional.normalize(query, dim=-1, eps=1e-6)
    globals_ = torch.stack([embeddings[indices].mean(dim=0) for indices in groups])
    globals_norm = functional.normalize(globals_, dim=-1, eps=1e-6)
    relation = torch.matmul(globals_norm, query.T)
    top = min(2, relation.shape[1])
    frame_relevance = torch.topk(relation, k=top, dim=1).values.mean(dim=1)
    motion_energy = torch.zeros(len(groups), device=device)
    if len(groups) > 1:
        motion_energy[1:] = (globals_norm[1:] - globals_norm[:-1]).norm(dim=-1)
        motion_energy[0] = motion_energy[1]
    weights = frame_weight_alpha * frame_relevance.relu() + (1.0 - frame_weight_alpha) * motion_energy

    dense = int(dense_evidence_units or evidence.size)
    target = min(evidence.size, max(len(groups), round(dense * retention_ratio)))
    capacities = torch.tensor([len(indices) for indices in groups], device=device, dtype=torch.long)
    budgets = torch.ones(len(groups), device=device, dtype=torch.long)
    budgets += _allocate_with_caps(weights, target - len(groups), capacities - 1)

    coordinates = evidence.metadata.get("cell_coordinates")
    coordinate_by_index = (
        {index: tuple(value) for index, value in enumerate(coordinates)}
        if isinstance(coordinates, list) and len(coordinates) == evidence.size
        else {}
    )
    coordinate_maps = []
    for indices in groups:
        coordinate_maps.append(
            {(coordinate_by_index[index][1], coordinate_by_index[index][2]): index for index in indices}
            if coordinate_by_index
            else {}
        )

    selected: list[int] = []
    role_by_index: dict[int, str] = {}
    for time, indices in enumerate(groups):
        count = min(int(budgets[time]), len(indices))
        tokens = embeddings[indices]
        normalized = functional.normalize(tokens, dim=-1, eps=1e-6)
        mean = globals_norm[time]
        proto_local = int(torch.argmax(torch.matmul(normalized, mean)).item())
        chosen = {proto_local}
        role_by_index[indices[proto_local]] = "context"
        extra = count - 1
        object_count = max(0, min(extra, round(extra * object_ratio)))

        if object_count:
            if query_is_tokenwise:
                similarities = torch.matmul(normalized, query.T)
                query_weights = similarities.max(dim=0).values.clamp_min(0)
                query_budgets = _allocate_with_caps(
                    query_weights,
                    object_count,
                    torch.full_like(query_weights, object_count, dtype=torch.long),
                )
                for query_index in torch.argsort(query_weights, descending=True, stable=True).tolist():
                    count_for_query = int(query_budgets[query_index])
                    candidates = torch.tensor(
                        [index for index in range(len(indices)) if index not in chosen],
                        device=device,
                        dtype=torch.long,
                    )
                    if count_for_query <= 0 or not candidates.numel():
                        continue
                    local = _mmr_select(
                        similarities[candidates, query_index],
                        normalized[candidates],
                        count_for_query,
                        mmr_lambda,
                    )
                    for value in candidates[local].tolist():
                        chosen.add(int(value))
                        role_by_index[indices[int(value)]] = "object"
            else:
                candidates = torch.tensor(
                    [index for index in range(len(indices)) if index not in chosen],
                    device=device,
                    dtype=torch.long,
                )
                relevance = torch.matmul(normalized[candidates], query[0])
                local = _mmr_select(relevance, normalized[candidates], object_count, mmr_lambda)
                for value in candidates[local].tolist():
                    chosen.add(int(value))
                    role_by_index[indices[int(value)]] = "object"

        motion_count = extra - object_count
        if motion_count:
            candidates = [index for index in range(len(indices)) if index not in chosen]
            motion_scores = []
            for local_index in candidates:
                global_index = indices[local_index]
                comparisons = []
                if coordinate_maps:
                    coordinate = coordinate_by_index.get(global_index)
                    key = (coordinate[1], coordinate[2]) if coordinate is not None else None
                    for neighbor in (time - 1, time + 1):
                        if key is not None and 0 <= neighbor < len(groups) and key in coordinate_maps[neighbor]:
                            comparisons.append(
                                (embeddings[global_index] - embeddings[coordinate_maps[neighbor][key]]).norm()
                            )
                if not comparisons:
                    for neighbor in (time - 1, time + 1):
                        if 0 <= neighbor < len(groups):
                            comparisons.append((embeddings[global_index] - globals_[neighbor]).norm())
                motion = torch.stack(comparisons).mean() if comparisons else embeddings.new_tensor(0.0)
                query_score = torch.matmul(normalized[local_index], query.T).max().clamp_min(0)
                motion_scores.append((motion, query_score))
            if motion_scores:
                motion_values = torch.stack([value[0] for value in motion_scores])
                query_values = torch.stack([value[1] for value in motion_scores])
                motion_values = motion_values / motion_values.max().clamp_min(1e-6)
                query_values = query_values / query_values.max().clamp_min(1e-6)
                score = motion_values
                if query_is_tokenwise:
                    score = (1.0 - motion_query_beta) * motion_values + motion_query_beta * query_values
                keep = torch.topk(score, k=min(motion_count, len(candidates))).indices.tolist()
                for candidate_index in keep:
                    value = candidates[int(candidate_index)]
                    chosen.add(value)
                    role_by_index[indices[value]] = "motion"

        if len(chosen) < count:
            remaining = [index for index in range(len(indices)) if index not in chosen]
            norms = tokens.norm(dim=-1)
            remaining.sort(key=lambda index: (-float(norms[index]), index))
            for value in remaining[: count - len(chosen)]:
                chosen.add(value)
                role_by_index[indices[value]] = "context"
        selected.extend(indices[index] for index in sorted(chosen))

    selected.sort(key=lambda index: (evidence.timestamps[index], index))
    compact = evidence.select(selected)
    compact.roles = tuple(role_by_index.get(index, "context") for index in selected)
    if coordinate_by_index:
        compact.metadata["cell_coordinates"] = [list(coordinate_by_index[index]) for index in selected]
    compact.metadata.update(
        {
            "post_pruning": "semvid",
            "semvid_upstream_revision": "432a76928817cdfba7d04c460ac475482cd7c3a4",
            "semvid_dense_evidence_units": dense,
            "semvid_input_evidence_units": evidence.size,
            "semvid_retained_evidence_units": compact.size,
            "semvid_retention_ratio": retention_ratio,
            "semvid_role_counts": {
                role: compact.roles.count(role) for role in ("context", "object", "motion")
            },
        }
    )
    return compact


__all__ = ["MageCellPlan", "mage_cell_plan", "motion_residual_importance", "semvid_select"]
